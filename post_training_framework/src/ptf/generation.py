"""模型加载、推理和评估主流程。"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .data import load_eval_samples
from .metrics import score_prediction, summarize_rows
from .prompting import DEFAULT_FORMAT_INSTRUCTION, render_generation_prompt
from .reports import build_eval_markdown, write_json, write_jsonl, write_markdown


def build_eos_token_ids(tokenizer: Any) -> int | None:
    """用 <|im_end|> 作为唯一停止标记。

    Qwen3 的 eos_token <|endoftext|> (id=151643) 在训练数据中从未出现，
    模型不会在该 token 处自然停止，不应作为 eos_token。
    只保留 <|im_end|> (id=151645)，这是训练数据中 assistant 回答的真实结束标记。
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    # 回退到 tokenizer 默认 eos（仅用于非 Qwen 模型）
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def load_tokenizer(tokenizer_dir: Path):
    """加载 tokenizer 并保证 pad_token 可用。"""
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_optional_path(raw_path: Any, cfg: ExperimentConfig) -> Path | None:
    """把 checkpoint/config 中的可选路径解析为绝对路径。"""
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cfg.workspace_root / path
    return path.resolve()


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    """读取可选 JSON 文件；不存在时返回空字典。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _load_grpo_adapter_model(
    cfg: ExperimentConfig,
    adapter_dir: Path,
    dtype: torch.dtype,
    device_map: str | None,
):
    """按训练链路加载 GRPO LoRA checkpoint: base + SFT merge + GRPO LoRA。"""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("加载 GRPO LoRA checkpoint 需要安装 peft。") from exc

    state = _load_json_if_exists(adapter_dir / "trainer_state.json")
    base_model_dir = _resolve_optional_path(state.get("base_model_dir"), cfg) or cfg.path("model.base_model_dir")
    sft_adapter_dir = _resolve_optional_path(state.get("sft_adapter_dir"), cfg)
    if sft_adapter_dir is None:
        sft_adapter_dir = _resolve_optional_path(cfg.get("sft.eval_adapter_dir"), cfg)

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if sft_adapter_dir is not None:
        if not sft_adapter_dir.exists():
            raise FileNotFoundError(f"GRPO checkpoint 需要的 SFT adapter 不存在: {sft_adapter_dir}")
        model = PeftModel.from_pretrained(model, str(sft_adapter_dir))
        model = model.merge_and_unload()

    return PeftModel.from_pretrained(model, str(adapter_dir))


def load_generation_model(
    cfg: ExperimentConfig,
    model_kind: str,
    adapter_dir: Path | None = None,
    model_dir: Path | None = None,
):
    """加载 base、LoRA SFT、GRPO LoRA checkpoint 或已导出的 GRPO HuggingFace 模型。"""
    base_model_dir = cfg.path("model.base_model_dir")
    generation_cfg = cfg.get("generation", {})
    dtype = torch.float16 if torch.cuda.is_available() and bool(generation_cfg.get("fp16", True)) else torch.float32
    device_map = generation_cfg.get("device_map", "auto") if torch.cuda.is_available() else None

    if model_kind == "grpo":
        if model_dir is None:
            raise ValueError("加载 GRPO 模型必须提供 model_dir。")
        model_dir = Path(model_dir)
        if (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists():
            tokenizer_dir = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model_dir
            tokenizer = load_tokenizer(tokenizer_dir)
            model = _load_grpo_adapter_model(cfg, adapter_dir=model_dir, dtype=dtype, device_map=device_map)
            model.eval()
            return model, tokenizer
        tokenizer_dir = model_dir
        load_model_dir = model_dir
    else:
        tokenizer_dir = adapter_dir if model_kind == "sft" and adapter_dir is not None else base_model_dir
        load_model_dir = base_model_dir

    tokenizer = load_tokenizer(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(load_model_dir),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if model_kind == "sft":
        if adapter_dir is None:
            raise ValueError("加载 SFT 模型必须提供 adapter_dir。")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("加载 LoRA adapter 需要安装 peft。") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
    prompt_mode: str,
    include_format_instruction: bool,
    format_instruction: str,
    enable_thinking: bool | None,
) -> str:
    """对单个问题生成完整原始输出。"""
    prompt_text = render_generation_prompt(
        tokenizer=tokenizer,
        question=question,
        prompt_mode=prompt_mode,
        format_instruction=format_instruction,
        include_format_instruction=include_format_instruction,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=build_eos_token_ids(tokenizer),
            repetition_penalty=1.0,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_batch(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    max_new_tokens: int,
    prompt_mode: str,
    include_format_instruction: bool,
    format_instruction: str,
    enable_thinking: bool | None,
) -> list[str]:
    """对一批问题并行生成完整原始输出。"""
    prompt_texts = [
        render_generation_prompt(
            tokenizer=tokenizer,
            question=question,
            prompt_mode=prompt_mode,
            format_instruction=format_instruction,
            include_format_instruction=include_format_instruction,
            enable_thinking=enable_thinking,
        )
        for question in questions
    ]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=build_eos_token_ids(tokenizer),
            repetition_penalty=1.0,
        )

    prompt_width = inputs["input_ids"].shape[-1]
    predictions: list[str] = []
    for item_ids in output_ids:
        new_tokens = item_ids[prompt_width:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return predictions


def evaluate_model(
    cfg: ExperimentConfig,
    model_kind: str,
    max_new_tokens: int | None = None,
    max_items: int | None = None,
    adapter_dir: Path | None = None,
    model_dir: Path | None = None,
    run_name: str | None = None,
    output_dir: Path | None = None,
    eval_batch_size: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """运行推理评估，并保存 JSONL、summary 和 Markdown。"""
    if model_kind not in {"base", "sft", "grpo"}:
        raise ValueError("model_kind 必须是 base、sft 或 grpo。")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size 必须 >= 1。")

    eval_file = cfg.path("dataset.eval_file")
    format_instruction = str(cfg.get("prompt.format_instruction", DEFAULT_FORMAT_INSTRUCTION))
    include_format_instruction = bool(cfg.get("prompt.include_format_instruction", True))
    prompt_mode = str(cfg.get(f"prompt.{model_kind}_prompt_mode", "plain" if model_kind == "base" else "chat"))
    enable_thinking = cfg.get(f"prompt.{model_kind}_enable_thinking", None if model_kind == "base" else False)

    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else cfg.get("generation.max_new_tokens", 160))
    max_items = int(max_items if max_items is not None else cfg.get("dataset.max_eval_items", 20))
    run_name = run_name or str(cfg.get(f"runs.{model_kind}_eval_name", f"{model_kind}_eval"))

    samples = load_eval_samples(eval_file, max_items=max_items, format_instruction=format_instruction)
    model, tokenizer = load_generation_model(cfg, model_kind=model_kind, adapter_dir=adapter_dir, model_dir=model_dir)

    rows: list[dict[str, Any]] = []
    started = time.time()
    tag = f"{run_name}_max{max_new_tokens}_full"

    for start in range(0, len(samples), eval_batch_size):
        batch_samples = samples[start : start + eval_batch_size]
        predictions = generate_batch(
            model=model,
            tokenizer=tokenizer,
            questions=[sample.question for sample in batch_samples],
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
            include_format_instruction=include_format_instruction,
            format_instruction=format_instruction,
            enable_thinking=enable_thinking,
        )
        if len(predictions) != len(batch_samples):
            raise RuntimeError(f"批量生成数量不匹配: samples={len(batch_samples)} predictions={len(predictions)}")
        for sample, prediction in zip(batch_samples, predictions):
            metric = score_prediction(prediction, sample.gold_answer)
            row = {
                "idx": sample.idx,
                "tag": tag,
                "model_kind": model_kind,
                "prompt_mode": prompt_mode,
                "include_format_instruction": include_format_instruction,
                "max_new_tokens": max_new_tokens,
                "eval_batch_size": eval_batch_size,
                "question": sample.question,
                "prediction": prediction,
                "gold": sample.gold,
                **metric,
            }
            rows.append(row)
            print(
                f"[{len(rows)}/{len(samples)}] "
                f"gold={row['gold_answer']} pred={row['pred_answer']} "
                f"em={row['exact_match']} format={row['format_ok']} "
                f"repeat={row['repeat_like']} chars={row['pred_chars']}"
            )

    summary = summarize_rows(rows, tag=tag, max_new_tokens=max_new_tokens)
    summary.update(
        {
            "model_kind": model_kind,
            "prompt_mode": prompt_mode,
            "include_format_instruction": include_format_instruction,
            "eval_batch_size": eval_batch_size,
            "eval_file": str(eval_file),
            "seconds": round(time.time() - started, 2),
        }
    )

    out_dir = output_dir or (cfg.ensure_experiment_dir() / "eval" / model_kind)
    jsonl_path = out_dir / f"{run_name}_eval_{max_items}_max{max_new_tokens}_full.jsonl"
    summary_path = out_dir / f"{run_name}_eval_{max_items}_max{max_new_tokens}_summary.json"
    md_path = out_dir / f"{run_name}_eval_{max_items}_max{max_new_tokens}_full_report.md"

    write_jsonl(jsonl_path, rows)
    write_json(summary_path, summary)
    write_markdown(md_path, build_eval_markdown(summary, rows, title=f"{run_name} Evaluation Report"))

    print("saved jsonl:", jsonl_path)
    print("saved summary:", summary_path)
    print("saved report:", md_path)
    return summary, rows, jsonl_path
