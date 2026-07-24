"""运行 GRPO 模型推理评估。

支持两种输入：
1. train_grpo.py 保存的 GRPO LoRA checkpoint；
2. 已经 merge/export 后的完整 HuggingFace actor 目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.config import ExperimentConfig
from ptf.generation import evaluate_model
from ptf.reports import read_jsonl, write_json, write_jsonl, write_markdown
from ptf.reward import GSM8KRewardConfig, compute_gsm8k_rule_reward


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 GRPO actor 的 GSM8K 评估。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_1d7b.json",
        help="实验配置文件路径，主要用于读取 base model、eval_file 和 prompt 设置。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="GRPO LoRA checkpoint 目录，或 merge/export 后的完整 HuggingFace 模型目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="评估结果输出目录；默认写入 eval_results/grpo_model/<run_name>。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大新 token 数。")
    parser.add_argument("--max-items", type=int, default=20, help="最多评估多少条样本。")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="评估推理 batch size；RTX 4070 12GB 可先试 8。")
    parser.add_argument("--run-name", type=str, required=True, help="本次 GRPO 评估名称。")
    parser.add_argument("--overlong-chars", type=int, default=1200, help="rule reward 中判定过长输出的字符阈值。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段，例如 --set prompt.grpo_prompt_mode=chat。",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0


def build_reward_outputs(jsonl_path: Path, output_dir: Path, run_name: str, overlong_chars: int) -> None:
    """对 GRPO 评估 JSONL 额外计算 rule reward，并写出 reward 报告。"""
    rows = read_jsonl(jsonl_path)
    config = GSM8KRewardConfig(overlong_chars=overlong_chars)
    reward_rows: list[dict[str, Any]] = []

    for row in rows:
        reward_info = compute_gsm8k_rule_reward(row.get("prediction", ""), row.get("gold_answer"), config=config)
        reward_rows.append(
            {
                "idx": row.get("idx"),
                "question": row.get("question"),
                "gold_answer": row.get("gold_answer"),
                "pred_answer": row.get("pred_answer"),
                "prediction": row.get("prediction"),
                "reward_score": reward_info["score"],
                "reward_raw_score": reward_info["raw_score"],
                "reward_components": reward_info["components"],
                "exact_match": reward_info["exact_match"],
                "format_ok": reward_info["format_ok"],
                "single_final_answer_ok": reward_info["single_final_answer_ok"],
                "repeat_like": reward_info["repeat_like"],
                "overlong": reward_info["overlong"],
                "pred_chars": reward_info["pred_chars"],
            }
        )

    scores = [float(row["reward_score"]) for row in reward_rows]
    summary = {
        "run_name": run_name,
        "n": len(reward_rows),
        "avg_reward": _mean(scores),
        "min_reward": min(scores) if scores else 0.0,
        "max_reward": max(scores) if scores else 0.0,
        "exact_match_rate": _mean([1.0 if row["exact_match"] else 0.0 for row in reward_rows]),
        "format_rate": _mean([1.0 if row["format_ok"] else 0.0 for row in reward_rows]),
        "single_final_answer_rate": _mean([1.0 if row["single_final_answer_ok"] else 0.0 for row in reward_rows]),
        "repeat_like_rate": _mean([1.0 if row["repeat_like"] else 0.0 for row in reward_rows]),
        "overlong_rate": _mean([1.0 if row["overlong"] else 0.0 for row in reward_rows]),
    }

    reward_jsonl = output_dir / f"{run_name}_reward.jsonl"
    reward_summary = output_dir / f"{run_name}_reward_summary.json"
    reward_report = output_dir / f"{run_name}_reward_report.md"

    lines = [
        f"# {run_name} Rule Reward Report",
        "",
        "## Summary",
        "",
        f"- n: {summary['n']}",
        f"- avg_reward: {summary['avg_reward']:.4f}",
        f"- exact_match_rate: {summary['exact_match_rate']:.4f}",
        f"- format_rate: {summary['format_rate']:.4f}",
        f"- single_final_answer_rate: {summary['single_final_answer_rate']:.4f}",
        f"- repeat_like_rate: {summary['repeat_like_rate']:.4f}",
        f"- overlong_rate: {summary['overlong_rate']:.4f}",
        "",
        "## Items",
        "",
        "| idx | reward | gold | pred | exact | format | single | repeat | chars |",
        "|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in reward_rows:
        lines.append(
            "| {idx} | {reward:.3f} | {gold} | {pred} | {exact} | {fmt} | {single} | {repeat} | {chars} |".format(
                idx=row["idx"],
                reward=row["reward_score"],
                gold=row["gold_answer"],
                pred=row["pred_answer"],
                exact="yes" if row["exact_match"] else "no",
                fmt="yes" if row["format_ok"] else "no",
                single="yes" if row["single_final_answer_ok"] else "no",
                repeat="yes" if row["repeat_like"] else "no",
                chars=row["pred_chars"],
            )
        )

    write_jsonl(reward_jsonl, reward_rows)
    write_json(reward_summary, summary)
    write_markdown(reward_report, "\n".join(lines))
    print("saved reward jsonl:", reward_jsonl)
    print("saved reward summary:", reward_summary)
    print("saved reward report:", reward_report)


def main() -> None:
    """入口函数。"""
    args = parse_args()
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    output_dir = args.output_dir or (WORKSPACE_ROOT / "eval_results" / "grpo_model" / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, jsonl_path = evaluate_model(
        cfg,
        model_kind="grpo",
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        max_items=args.max_items,
        eval_batch_size=args.eval_batch_size,
        run_name=args.run_name,
        output_dir=output_dir,
    )
    build_reward_outputs(jsonl_path, output_dir, args.run_name, args.overlong_chars)


if __name__ == "__main__":
    main()
