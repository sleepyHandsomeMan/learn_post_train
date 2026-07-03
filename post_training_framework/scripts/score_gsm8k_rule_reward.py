"""离线验证 GSM8K rule reward。

输入已有 eval JSONL，输出逐条 reward JSONL、summary JSON 和 Markdown 报告。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.reports import read_jsonl, write_json, write_jsonl, write_markdown
from ptf.reward import GSM8KRewardConfig, compute_gsm8k_rule_reward


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对 eval JSONL 离线计算 GSM8K rule reward。")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="已有评估 JSONL。")
    parser.add_argument("--output-dir", type=Path, required=True, help="reward 结果输出目录。")
    parser.add_argument("--run-name", type=str, default=None, help="输出文件名前缀。")
    parser.add_argument("--overlong-chars", type=int, default=1200, help="超过该字符数视为过长。")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    """计算均值。"""
    return sum(values) / len(values) if values else 0.0


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 reward 与关键指标。"""
    scores = [float(row["reward_score"]) for row in rows]
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["reward_bucket"])].append(float(row["reward_score"]))

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0

    return {
        "n": len(rows),
        "avg_reward": _mean(scores),
        "min_reward": min(scores) if scores else 0.0,
        "max_reward": max(scores) if scores else 0.0,
        "exact_match_rate": rate("exact_match"),
        "format_rate": rate("format_ok"),
        "single_final_answer_rate": rate("single_final_answer_ok"),
        "repeat_like_rate": rate("repeat_like"),
        "overlong_rate": rate("overlong"),
        "avg_reward_by_bucket": {key: _mean(value) for key, value in sorted(by_bucket.items())},
    }


def classify_reward_bucket(reward_info: dict[str, Any]) -> str:
    """把样本分成便于人工检查的 reward 类别。"""
    if reward_info["first_hash_exact_match"] and not reward_info["repeat_like"]:
        return "correct_format_clean"
    if reward_info["fallback_exact_match"] and not reward_info["format_ok"]:
        return "correct_without_format"
    if reward_info["format_ok"] and not reward_info["exact_match"]:
        return "wrong_but_formatted"
    if reward_info["repeat_like"]:
        return "repeat_like"
    return "other_wrong"


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], title: str) -> str:
    """生成 reward 检查报告。"""
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- n: {summary['n']}",
        f"- avg_reward: {summary['avg_reward']:.4f}",
        f"- min_reward: {summary['min_reward']:.4f}",
        f"- max_reward: {summary['max_reward']:.4f}",
        f"- exact_match_rate: {summary['exact_match_rate']:.4f}",
        f"- format_rate: {summary['format_rate']:.4f}",
        f"- single_final_answer_rate: {summary['single_final_answer_rate']:.4f}",
        f"- repeat_like_rate: {summary['repeat_like_rate']:.4f}",
        f"- overlong_rate: {summary['overlong_rate']:.4f}",
        "",
        "## Avg Reward By Bucket",
        "",
        "| bucket | avg_reward |",
        "|---|---:|",
    ]
    for bucket, avg_reward in summary["avg_reward_by_bucket"].items():
        lines.append(f"| {bucket} | {avg_reward:.4f} |")

    lines.extend(
        [
            "",
            "## Items",
            "",
            "| idx | bucket | reward | gold | pred | exact | format | single | repeat | chars |",
            "|---:|---|---:|---:|---:|---|---|---|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {idx} | {bucket} | {reward:.3f} | {gold} | {pred} | {exact} | {fmt} | {single} | {repeat} | {chars} |".format(
                idx=row["idx"],
                bucket=row["reward_bucket"],
                reward=row["reward_score"],
                gold=row.get("gold_answer"),
                pred=row.get("pred_answer"),
                exact="yes" if row.get("exact_match") else "no",
                fmt="yes" if row.get("format_ok") else "no",
                single="yes" if row.get("single_final_answer_ok") else "no",
                repeat="yes" if row.get("repeat_like") else "no",
                chars=row.get("pred_chars"),
            )
        )

    lines.extend(["", "## Low Reward Samples", ""])
    for row in sorted(rows, key=lambda item: (item["reward_score"], item["idx"]))[:5]:
        lines.extend(
            [
                f"### #{row['idx']} reward={row['reward_score']:.3f} bucket={row['reward_bucket']}",
                "",
                f"- gold_answer: `{row.get('gold_answer')}`",
                f"- pred_answer: `{row.get('pred_answer')}`",
                f"- components: `{row.get('reward_components')}`",
                "",
                "```text",
                str(row.get("prediction", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """入口函数。"""
    args = parse_args()
    input_rows = read_jsonl(args.input_jsonl)
    run_name = args.run_name or args.input_jsonl.stem
    config = GSM8KRewardConfig(overlong_chars=args.overlong_chars)

    output_rows: list[dict[str, Any]] = []
    for row in input_rows:
        reward_info = compute_gsm8k_rule_reward(row.get("prediction", ""), row.get("gold_answer"), config=config)
        reward_bucket = classify_reward_bucket(reward_info)
        output_rows.append(
            {
                **row,
                "reward_score": reward_info["score"],
                "reward_raw_score": reward_info["raw_score"],
                "reward_components": reward_info["components"],
                "reward_bucket": reward_bucket,
                "reward_overlong": reward_info["overlong"],
            }
        )

    summary = build_summary(output_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{run_name}_reward.jsonl"
    summary_path = args.output_dir / f"{run_name}_reward_summary.json"
    md_path = args.output_dir / f"{run_name}_reward_report.md"

    write_jsonl(jsonl_path, output_rows)
    write_json(summary_path, summary)
    write_markdown(md_path, build_markdown(summary, output_rows, title=f"{run_name} GSM8K Rule Reward"))

    print("saved reward jsonl:", jsonl_path)
    print("saved reward summary:", summary_path)
    print("saved reward report:", md_path)
    print("summary:", summary)


if __name__ == "__main__":
    main()
