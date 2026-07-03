"""从评估 JSONL 生成 Markdown 报告。

这个脚本只读已有评估结果，不重新推理模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_rate(df: pd.DataFrame, column: str) -> tuple[int, int, float]:
    """统计布尔指标的命中数、总数和比例。"""
    total = len(df)
    hit = int(df[column].sum()) if total else 0
    rate = hit / total if total else 0.0
    return hit, total, rate


def yes_no(value: object) -> str:
    """把布尔值渲染成简短文本。"""
    return "yes" if bool(value) else "no"


def build_full_report(path: Path, title: str) -> str:
    """生成包含逐题原始输出的 Markdown 报告。"""
    df = pd.read_json(path, lines=True)
    exact_hit, exact_total, exact_rate = bool_rate(df, "exact_match")
    format_hit, format_total, format_rate = bool_rate(df, "format_ok")
    single_hit, single_total, single_rate = bool_rate(df, "single_final_answer_ok")
    repeat_hit, repeat_total, repeat_rate = bool_rate(df, "repeat_like")

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- source: `{path}`",
        f"- n: {len(df)}",
        f"- exact_match: {exact_hit}/{exact_total} ({exact_rate:.2%})",
        f"- format_rate: {format_hit}/{format_total} ({format_rate:.2%})",
        f"- single_final_answer_rate: {single_hit}/{single_total} ({single_rate:.2%})",
        f"- repeat_like_rate: {repeat_hit}/{repeat_total} ({repeat_rate:.2%})",
        f"- avg_chars: {df['pred_chars'].mean():.1f}",
        "",
        "## Items",
        "",
    ]

    for row in df.to_dict("records"):
        lines.extend(
            [
                f"### #{row.get('idx')}",
                "",
                f"- exact_match: {yes_no(row.get('exact_match'))}",
                f"- format_ok: {yes_no(row.get('format_ok'))}",
                f"- single_final_answer_ok: {yes_no(row.get('single_final_answer_ok'))}",
                f"- repeat_like: {yes_no(row.get('repeat_like'))}",
                f"- gold_answer: `{row.get('gold_answer')}`",
                f"- pred_answer: `{row.get('pred_answer')}`",
                f"- first_hash_answer: `{row.get('first_hash_answer')}`",
                f"- hash_count: {row.get('hash_count')}",
                f"- final_answer_count: {row.get('final_answer_count')}",
                f"- pred_chars: {row.get('pred_chars')}",
                "",
                "Question:",
                "",
                "```text",
                str(row.get("question", "")),
                "```",
                "",
                "Prediction:",
                "",
                "```text",
                str(row.get("prediction", "")),
                "```",
                "",
                "Gold:",
                "",
                "```text",
                str(row.get("gold", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_compare_report(named_paths: list[tuple[str, Path]]) -> str:
    """生成多个评估文件的指标对比报告。"""
    frames: dict[str, pd.DataFrame] = {}
    lines = [
        "# Qwen3-1.7B Base GSM8K Eval 对比报告",
        "",
        "## Summary",
        "",
        "| run | exact | format | single final | repeat-like | avg chars | correct idx |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for name, path in named_paths:
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        frames[name] = df
        exact_hit, exact_total, exact_rate = bool_rate(df, "exact_match")
        format_hit, format_total, format_rate = bool_rate(df, "format_ok")
        single_hit, single_total, single_rate = bool_rate(df, "single_final_answer_ok")
        repeat_hit, repeat_total, repeat_rate = bool_rate(df, "repeat_like")
        correct_idx = df.loc[df["exact_match"], "idx"].tolist()
        lines.append(
            f"| {name} | {exact_hit}/{exact_total} ({exact_rate:.0%}) | "
            f"{format_hit}/{format_total} ({format_rate:.0%}) | "
            f"{single_hit}/{single_total} ({single_rate:.0%}) | "
            f"{repeat_hit}/{repeat_total} ({repeat_rate:.0%}) | "
            f"{df['pred_chars'].mean():.1f} | {correct_idx} |"
        )

    old = frames.get("0.6B base max512")
    new = frames.get("1.7B base max512")
    if old is not None and new is not None:
        old_ok = set(old.loc[old["exact_match"], "idx"])
        new_ok = set(new.loc[new["exact_match"], "idx"])
        all_idx = set(new["idx"].tolist())
        lines.extend(
            [
                "",
                "## 0.6B Base max512 vs 1.7B Base max512",
                "",
                f"- improved idx: {sorted(new_ok - old_ok)}",
                f"- regressed idx: {sorted(old_ok - new_ok)}",
                f"- both correct idx: {sorted(old_ok & new_ok)}",
                f"- both wrong idx: {sorted(all_idx - old_ok - new_ok)}",
            ]
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 1.7B base 在 max512 下比 0.6B base 明显更强：exact match 从 5/20 提升到 11/20。",
            "- 1.7B base 在 max160 下仍只有 1/20，说明它需要更长的推理空间才能把算术链条走完。",
            "- base 模型仍不会遵守 GSM8K SFT 的 #### 最终答案格式：format_rate 仍为 0/20。",
            "- max512 下 1.7B 的 repeat-like 为 12/20，说明它虽然更会解题，但停止边界和输出风格仍需要 SFT 约束。",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从评估 JSONL 生成 Markdown 报告。")
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_results/base_model"))
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    eval_dir = args.eval_dir.resolve()

    max160 = eval_dir / "qwen3_1d7b_base_eval_20_max160_full.jsonl"
    max512 = eval_dir / "qwen3_1d7b_base_eval_20_max512_full.jsonl"
    (eval_dir / "qwen3_1d7b_base_eval_20_max160_full_report.md").write_text(
        build_full_report(max160, "Qwen3-1.7B Base GSM8K Eval max160"),
        encoding="utf-8",
    )
    (eval_dir / "qwen3_1d7b_base_eval_20_max512_full_report.md").write_text(
        build_full_report(max512, "Qwen3-1.7B Base GSM8K Eval max512"),
        encoding="utf-8",
    )

    compare_paths = [
        ("0.6B base max160", eval_dir / "base_eval_20_max160_full.jsonl"),
        ("0.6B base max512", eval_dir / "base_eval_20_max512_full.jsonl"),
        ("1.7B base max160", max160),
        ("1.7B base max512", max512),
        (
            "0.6B SFT max160",
            eval_dir.parent / "sft_model" / "sft_lora_len768_lr3e-5_ep1_run2_eval_20_max160_full.jsonl",
        ),
    ]
    (eval_dir / "qwen3_1d7b_base_compare_report.md").write_text(
        build_compare_report(compare_paths),
        encoding="utf-8",
    )

    print("saved reports to:", eval_dir)


if __name__ == "__main__":
    main()
