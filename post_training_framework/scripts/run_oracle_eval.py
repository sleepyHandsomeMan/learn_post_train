"""运行 oracle@k 采样评估。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import set_seed

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.config import ExperimentConfig
from ptf.data import load_eval_samples
from ptf.generation import build_eos_token_ids, load_generation_model
from ptf.metrics import score_prediction
from ptf.prompting import DEFAULT_FORMAT_INSTRUCTION, render_generation_prompt
from ptf.reports import write_json, write_jsonl, write_markdown


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 base/SFT/GRPO 的 oracle@k 采样评估。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_0d6b.json",
        help="实验配置文件路径。",
    )
    parser.add_argument("--model-kind", choices=["base", "sft", "grpo"], required=True, help="要评估的模型阶段。")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="SFT LoRA adapter 目录。")
    parser.add_argument("--model-dir", type=Path, default=None, help="GRPO checkpoint 或导出模型目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="评估结果输出目录。")
    parser.add_argument("--run-name", type=str, required=True, help="本次评估产物名称。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="生成最大新 token 数。")
    parser.add_argument("--max-items", type=int, default=None, help="最多评估多少条样本。")
    parser.add_argument("--oracle-k", type=int, default=8, help="每道题采样多少条候选回答。")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="一次并行多少道题；实际序列数约为 batch_size * oracle_k。")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度。")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling 的 top_p。")
    parser.add_argument("--top-k", type=int, default=50, help="top_k 采样候选数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段，例如 --set dataset.eval_file=datasets/gsm8k_grpo/eval_100.parquet。",
    )
    return parser.parse_args()


def _generate_oracle_batch(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    max_new_tokens: int,
    prompt_mode: str,
    include_format_instruction: bool,
    format_instruction: str,
    enable_thinking: bool | None,
    oracle_k: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[list[str]]:
    """对一批问题采样 oracle_k 条候选回答。"""
    prompts = [
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
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": oracle_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": build_eos_token_ids(tokenizer),
        "repetition_penalty": 1.0,
    }
    if top_k > 0:
        generate_kwargs["top_k"] = top_k

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    prompt_width = inputs["input_ids"].shape[-1]
    flat_predictions = [
        tokenizer.decode(item_ids[prompt_width:], skip_special_tokens=True).strip()
        for item_ids in output_ids
    ]

    grouped: list[list[str]] = []
    for start in range(0, len(flat_predictions), oracle_k):
        grouped.append(flat_predictions[start : start + oracle_k])
    return grouped


def _build_oracle_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """生成 oracle@k Markdown 报告。"""
    lines = [
        f"# {summary['run_name']} Oracle Evaluation",
        "",
        "## Summary",
        "",
        f"- model_kind: `{summary['model_kind']}`",
        f"- n: {summary['n']}",
        f"- oracle_k: {summary['oracle_k']}",
        f"- max_new_tokens: {summary['max_new_tokens']}",
        f"- temperature: {summary['temperature']}",
        f"- top_p: {summary['top_p']}",
        f"- top_k: {summary['top_k']}",
        f"- eval_batch_size: {summary['eval_batch_size']}",
        f"- oracle_exact_match: {summary['oracle_exact_match']:.4f}",
        f"- sample_exact_rate: {summary['sample_exact_rate']:.4f}",
        f"- avg_exact_count: {summary['avg_exact_count']:.4f}",
        f"- avg_format_rate_per_sample: {summary['avg_format_rate_per_sample']:.4f}",
        f"- avg_unique_pred_answer_count: {summary['avg_unique_pred_answer_count']:.4f}",
        f"- seconds: {summary['seconds']}",
        f"- peak_cuda_memory_mib: {summary['peak_cuda_memory_mib']}",
        "",
        "## Items",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### #{row['idx']}",
                "",
                f"- oracle_exact_match: {row['oracle_exact_match']}",
                f"- exact_count: {row['exact_count']}",
                f"- format_count: {row['format_count']}",
                f"- unique_pred_answer_count: {row['unique_pred_answer_count']}",
                f"- gold_answer: `{row['gold_answer']}`",
                f"- pred_answer_counts: `{row['pred_answer_counts']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """入口函数。"""
    args = parse_args()
    if args.oracle_k < 1:
        raise ValueError("--oracle-k 必须 >= 1。")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size 必须 >= 1。")

    set_seed(args.seed)
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    model_kind = args.model_kind
    adapter_dir = args.adapter_dir or (cfg.path("sft.eval_adapter_dir") if model_kind == "sft" else None)

    eval_file = cfg.path("dataset.eval_file")
    format_instruction = str(cfg.get("prompt.format_instruction", DEFAULT_FORMAT_INSTRUCTION))
    include_format_instruction = bool(cfg.get("prompt.include_format_instruction", True))
    prompt_mode = str(cfg.get(f"prompt.{model_kind}_prompt_mode", "plain" if model_kind == "base" else "chat"))
    enable_thinking = cfg.get(f"prompt.{model_kind}_enable_thinking", None if model_kind == "base" else False)
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else cfg.get("generation.max_new_tokens", 160))
    max_items = int(args.max_items if args.max_items is not None else cfg.get("dataset.max_eval_items", 20))

    samples = load_eval_samples(eval_file, max_items=max_items, format_instruction=format_instruction)
    model, tokenizer = load_generation_model(
        cfg,
        model_kind=model_kind,
        adapter_dir=adapter_dir,
        model_dir=args.model_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    rows: list[dict[str, Any]] = []
    all_exact = 0
    all_format = 0

    for start in range(0, len(samples), args.eval_batch_size):
        batch_samples = samples[start : start + args.eval_batch_size]
        grouped_predictions = _generate_oracle_batch(
            model=model,
            tokenizer=tokenizer,
            questions=[sample.question for sample in batch_samples],
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
            include_format_instruction=include_format_instruction,
            format_instruction=format_instruction,
            enable_thinking=enable_thinking,
            oracle_k=args.oracle_k,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        for sample, predictions in zip(batch_samples, grouped_predictions):
            candidate_rows = []
            pred_answers = []
            exact_count = 0
            format_count = 0
            for candidate_idx, prediction in enumerate(predictions):
                metric = score_prediction(prediction, sample.gold_answer)
                exact_count += int(bool(metric["exact_match"]))
                format_count += int(bool(metric["format_ok"]))
                pred_answers.append(metric["pred_answer"])
                candidate_rows.append(
                    {
                        "candidate_idx": candidate_idx,
                        "prediction": prediction,
                        **metric,
                    }
                )

            all_exact += exact_count
            all_format += format_count
            answer_counts = Counter(str(answer) for answer in pred_answers if answer is not None)
            row = {
                "idx": sample.idx,
                "model_kind": model_kind,
                "prompt_mode": prompt_mode,
                "include_format_instruction": include_format_instruction,
                "max_new_tokens": max_new_tokens,
                "oracle_k": args.oracle_k,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "question": sample.question,
                "gold": sample.gold,
                "gold_answer": sample.gold_answer,
                "oracle_exact_match": exact_count > 0,
                "exact_count": exact_count,
                "format_count": format_count,
                "unique_pred_answer_count": len(answer_counts),
                "pred_answer_counts": dict(answer_counts.most_common()),
                "candidates": candidate_rows,
            }
            rows.append(row)
            print(
                f"[{len(rows)}/{len(samples)}] "
                f"gold={sample.gold_answer} oracle={row['oracle_exact_match']} "
                f"exact_count={exact_count}/{args.oracle_k} format_count={format_count}/{args.oracle_k} "
                f"unique_answers={row['unique_pred_answer_count']}"
            )

    n = len(rows)
    total_candidates = n * args.oracle_k
    peak_mem_mib = 0.0
    if torch.cuda.is_available():
        peak_mem_mib = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)

    summary = {
        "run_name": args.run_name,
        "model_kind": model_kind,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "model_dir": str(args.model_dir) if args.model_dir is not None else None,
        "eval_file": str(eval_file),
        "n": n,
        "oracle_k": args.oracle_k,
        "max_new_tokens": max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "eval_batch_size": args.eval_batch_size,
        "oracle_exact_match": sum(1 for row in rows if row["oracle_exact_match"]) / n if n else 0.0,
        "avg_exact_count": all_exact / n if n else 0.0,
        "sample_exact_rate": all_exact / total_candidates if total_candidates else 0.0,
        "avg_format_rate_per_sample": all_format / total_candidates if total_candidates else 0.0,
        "avg_unique_pred_answer_count": sum(row["unique_pred_answer_count"] for row in rows) / n if n else 0.0,
        "seconds": round(time.time() - started, 2),
        "peak_cuda_memory_mib": peak_mem_mib,
    }

    default_dir = cfg.workspace_root / "eval_results" / f"{model_kind}_model" / args.run_name
    out_dir = args.output_dir or default_dir
    write_jsonl(out_dir / f"{args.run_name}.jsonl", rows)
    write_json(out_dir / f"{args.run_name}_summary.json", summary)
    write_markdown(out_dir / f"{args.run_name}_report.md", _build_oracle_report(summary, rows))
    print("saved jsonl:", out_dir / f"{args.run_name}.jsonl")
    print("saved summary:", out_dir / f"{args.run_name}_summary.json")
    print("saved report:", out_dir / f"{args.run_name}_report.md")


if __name__ == "__main__":
    main()
