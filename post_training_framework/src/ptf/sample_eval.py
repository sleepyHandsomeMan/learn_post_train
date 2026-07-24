"""GRPO checkpoint 的固定随机 sample 评估与统计工具。"""

from __future__ import annotations

from dataclasses import fields
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from .config import ExperimentConfig
from .generation import build_eos_token_ids, load_generation_model
from .reward import GSM8KRewardConfig, compute_gsm8k_rule_reward
from .rl_dataset import load_rl_dataset


BINARY_METRICS = (
    "exact_match",
    "format_ok",
    "single_final_answer_ok",
    "terminated_by_eos",
    "reached_max_tokens_without_eos",
)


def _mean(values: list[float]) -> float:
    """计算均值，空列表返回0。"""
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], probability: float) -> float:
    """用线性插值计算分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """计算二项比例的Wilson 95%置信区间。"""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总逐回答sample记录，并为所有二项指标计算Wilson区间。"""
    summary: dict[str, Any] = {
        "responses": len(rows),
        "prompt_count": len({int(row["prompt_index"]) for row in rows}),
        "eval_seed_count": len({int(row["eval_seed"]) for row in rows}),
        "response_tokens_mean": round(
            _mean([float(row["response_token_count"]) for row in rows]), 6
        ),
        "reward_mean": round(_mean([float(row["reward"]) for row in rows]), 6),
        "metrics": {},
    }
    for key in BINARY_METRICS:
        successes = sum(1 for row in rows if bool(row[key]))
        lower, upper = wilson_interval(successes, len(rows))
        summary["metrics"][key] = {
            "successes": successes,
            "total": len(rows),
            "rate": round(successes / len(rows), 6) if rows else 0.0,
            "wilson_low": round(lower, 6),
            "wilson_high": round(upper, 6),
        }
    return summary


def paired_prompt_bootstrap(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """按prompt聚类bootstrap，估计candidate-control差值区间。"""
    if metric not in BINARY_METRICS:
        raise ValueError(f"不支持的配对指标: {metric}")
    control_map = {
        (int(row["eval_seed"]), int(row["prompt_index"]), int(row["return_index"])): row
        for row in control_rows
    }
    candidate_map = {
        (int(row["eval_seed"]), int(row["prompt_index"]), int(row["return_index"])): row
        for row in candidate_rows
    }
    if control_map.keys() != candidate_map.keys():
        missing_control = len(candidate_map.keys() - control_map.keys())
        missing_candidate = len(control_map.keys() - candidate_map.keys())
        raise ValueError(
            "C0/L1逐回答键不一致: "
            f"missing_control={missing_control}, missing_candidate={missing_candidate}"
        )

    prompt_deltas: dict[int, list[float]] = {}
    for key in sorted(control_map):
        prompt_index = key[1]
        delta = float(bool(candidate_map[key][metric])) - float(bool(control_map[key][metric]))
        prompt_deltas.setdefault(prompt_index, []).append(delta)
    prompt_means = {
        prompt_index: _mean(values) for prompt_index, values in prompt_deltas.items()
    }
    prompt_ids = sorted(prompt_means)
    observed = _mean([prompt_means[prompt_id] for prompt_id in prompt_ids])
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    for _ in range(samples):
        drawn = [rng.choice(prompt_ids) for _ in prompt_ids]
        bootstrap_values.append(_mean([prompt_means[prompt_id] for prompt_id in drawn]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "paired_responses": len(control_map),
        "prompt_clusters": len(prompt_ids),
        "delta": round(observed, 6),
        "ci_low": round(_quantile(bootstrap_values, alpha), 6),
        "ci_high": round(_quantile(bootstrap_values, 1.0 - alpha), 6),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _trim_generated_response_ids(
    response_ids: list[int], eos_token_id: int | None, pad_token_id: int | None
) -> list[int]:
    """按训练器相同规则裁掉eos/pad之后的尾部token。"""
    trimmed: list[int] = []
    for token_id in response_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            trimmed.append(token_id)
            break
        if pad_token_id is not None and token_id == pad_token_id:
            break
        trimmed.append(token_id)
    return trimmed


def _load_reward_config(checkpoint_dir: Path) -> GSM8KRewardConfig:
    """优先复用checkpoint保存的训练reward配置。"""
    state_path = checkpoint_dir / "trainer_state.json"
    raw: dict[str, Any] = {}
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("reward_config"), dict):
            raw = data["reward_config"]
    allowed = {field.name for field in fields(GSM8KRewardConfig)}
    return GSM8KRewardConfig(**{key: value for key, value in raw.items() if key in allowed})


def _sha256(path: Path) -> str:
    """计算文件SHA256，固定评估权重血缘。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    full = summary["full"]
    first10 = summary["first10"]
    lines = [
        f"# {summary['trial_id']} 扩大sample评估",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- adapter SHA256: `{summary['adapter_sha256']}`",
        f"- responses: {full['responses']}，prompts: {full['prompt_count']}，eval seeds: {full['eval_seed_count']}",
        f"- sampling: temperature={summary['sampling']['temperature']}, top-p={summary['sampling']['top_p']}, top-k={summary['sampling']['top_k']}",
        "",
        "## 全量eval100",
        "",
        "| 指标 | 成功数/总数 | 比例 | Wilson 95% CI |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "exact_match": "sample EM",
        "format_ok": "sample格式率",
        "single_final_answer_ok": "单一final率",
        "terminated_by_eos": "EOS率",
        "reached_max_tokens_without_eos": "截顶率",
    }
    for key in BINARY_METRICS:
        metric = full["metrics"][key]
        lines.append(
            f"| {labels[key]} | {metric['successes']}/{metric['total']} | "
            f"{metric['rate']:.4f} | [{metric['wilson_low']:.4f}, {metric['wilson_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 前10题复现切片",
            "",
            "| 指标 | 比例 | Wilson 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for key in BINARY_METRICS:
        metric = first10["metrics"][key]
        lines.append(
            f"| {labels[key]} | {metric['rate']:.4f} | "
            f"[{metric['wilson_low']:.4f}, {metric['wilson_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "> 前10题仅用于解释旧80条sample是否对题目子集或评估seed敏感；晋级判定只使用全量eval100。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_grpo_checkpoint_sample(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    trial_id: str,
    variant: str,
    training_seed: int,
    max_items: int,
    max_response_tokens: int,
    eval_batch_size: int,
    return_sequences: int,
    eval_seeds: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    max_prompt_length: int,
    format_instruction: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    """加载冻结GRPO LoRA并执行可复现sample评估。"""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    if return_sequences < 1 or eval_batch_size < 1 or max_items < 1:
        raise ValueError("return_sequences、eval_batch_size和max_items必须大于0")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "summary.md"
    run_config_path = output_dir / "run_config.json"
    cfg = ExperimentConfig.load(config_path)
    reward_config = _load_reward_config(checkpoint_dir)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    run_config = {
        "schema_version": 1,
        "trial_id": trial_id,
        "variant": variant,
        "training_seed": training_seed,
        "config_path": str(config_path.resolve()),
        "checkpoint": str(checkpoint_dir.resolve()),
        "adapter_sha256": _sha256(adapter_path),
        "max_items": max_items,
        "max_response_tokens": max_response_tokens,
        "eval_batch_size": eval_batch_size,
        "return_sequences": return_sequences,
        "eval_seeds": eval_seeds,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_prompt_length": max_prompt_length,
        "format_instruction": format_instruction,
        "enable_thinking": enable_thinking,
    }
    _write_json(run_config_path, run_config)

    started = time.time()
    model = None
    tokenizer = None
    evaluated_prompt_count = 0
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer = load_generation_model(
            cfg, model_kind="grpo", model_dir=checkpoint_dir
        )
        dataset = load_rl_dataset(
            parquet_path=cfg.path("dataset.eval_file"),
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            format_instruction=format_instruction,
            enable_thinking=enable_thinking,
        )
        prompts = [dataset[index] for index in range(min(max_items, len(dataset)))]
        evaluated_prompt_count = len(prompts)
        eos_token_id = build_eos_token_ids(tokenizer)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        device = next(model.parameters()).device
        with rows_path.open("w", encoding="utf-8", newline="") as rows_file:
            for eval_seed in eval_seeds:
                for start in range(0, len(prompts), eval_batch_size):
                    batch_prompts = prompts[start : start + eval_batch_size]
                    old_padding_side = tokenizer.padding_side
                    tokenizer.padding_side = "left"
                    try:
                        inputs = tokenizer(
                            [prompt.prompt_text for prompt in batch_prompts],
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        ).to(device)
                    finally:
                        tokenizer.padding_side = old_padding_side
                    torch.manual_seed(eval_seed + start)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(eval_seed + start)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_response_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            num_return_sequences=return_sequences,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                        )
                    prompt_width = inputs.input_ids.shape[-1]
                    for flat_index, item_ids in enumerate(output_ids):
                        prompt = batch_prompts[flat_index // return_sequences]
                        return_index = flat_index % return_sequences
                        response_ids = _trim_generated_response_ids(
                            item_ids[prompt_width:].tolist(), eos_token_id, pad_token_id
                        )
                        terminated_by_eos = (
                            eos_token_id is not None and eos_token_id in response_ids
                        )
                        reached_max = (
                            len(response_ids) >= max_response_tokens
                            and not terminated_by_eos
                        )
                        response = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        ).strip()
                        reward = compute_gsm8k_rule_reward(
                            response=response,
                            gold_answer=prompt.ground_truth,
                            config=reward_config,
                            response_token_count=len(response_ids),
                            max_response_tokens=max_response_tokens,
                            terminated_by_eos=terminated_by_eos,
                        )
                        row = {
                            "trial_id": trial_id,
                            "variant": variant,
                            "training_seed": training_seed,
                            "eval_seed": eval_seed,
                            "prompt_index": prompt.idx,
                            "return_index": return_index,
                            "question": prompt.question,
                            "gold_answer": prompt.ground_truth,
                            "pred_answer": reward["pred_answer"],
                            "response": response,
                            "response_token_count": len(response_ids),
                            "reward": reward["score"],
                            "exact_match": bool(reward["exact_match"]),
                            "format_ok": bool(reward["format_ok"]),
                            "single_final_answer_ok": bool(
                                reward["single_final_answer_ok"]
                            ),
                            "repeat_like": bool(reward["repeat_like"]),
                            "terminated_by_eos": terminated_by_eos,
                            "reached_max_tokens_without_eos": reached_max,
                        }
                        rows.append(row)
                        rows_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_file.flush()
                    print(
                        f"[{trial_id}] seed={eval_seed} prompts="
                        f"{min(start + len(batch_prompts), len(prompts))}/{len(prompts)} "
                        f"responses={len(rows)}",
                        flush=True,
                    )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_responses = len(eval_seeds) * evaluated_prompt_count * return_sequences
    if len(rows) != expected_responses:
        raise RuntimeError(
            f"评估回答数不完整: actual={len(rows)}, expected={expected_responses}"
        )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "trial_id": trial_id,
        "variant": variant,
        "training_seed": training_seed,
        "checkpoint": str(checkpoint_dir.resolve()),
        "adapter_sha256": run_config["adapter_sha256"],
        "sampling": {
            "eval_seeds": eval_seeds,
            "return_sequences": return_sequences,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_response_tokens": max_response_tokens,
            "eval_batch_size": eval_batch_size,
        },
        "full": summarize_sample_rows(rows),
        "first10": summarize_sample_rows(
            [row for row in rows if int(row["prompt_index"]) < 10]
        ),
        "seconds": round(time.time() - started, 3),
        "rows_path": str(rows_path.resolve()),
    }
    _write_json(summary_path, summary)
    _write_markdown(report_path, summary)
    return summary
