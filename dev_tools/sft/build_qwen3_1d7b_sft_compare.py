"""生成 Qwen3-1.7B base/SFT 的评估对比报告。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


YHY_DIR = Path(__file__).resolve().parents[2]
BASE_EVAL_DIR = YHY_DIR / "eval_results" / "base_model"
SFT_EVAL_DIR = (
    YHY_DIR
    / "post_training_framework"
    / "runs"
    / "gsm8k_qwen3_1d7b_len768_lr2e-5_ep1"
    / "eval"
    / "sft"
)
OUT_PATH = (
    YHY_DIR
    / "post_training_framework"
    / "runs"
    / "gsm8k_qwen3_1d7b_len768_lr2e-5_ep1"
    / "compare"
    / "qwen3_1d7b_base_vs_sft_compare_report.md"
)


def load(path: Path) -> pd.DataFrame:
    """读取 JSONL 评估结果。"""
    return pd.read_json(path, lines=True)


def metric_cell(df: pd.DataFrame, column: str) -> str:
    """把布尔指标渲染为 命中/总数(比例)。"""
    hit = int(df[column].sum())
    total = len(df)
    rate = hit / total if total else 0.0
    return f"{hit}/{total} ({rate:.0%})"


def row_for(name: str, path: Path) -> tuple[str, pd.DataFrame, str]:
    """生成汇总表的一行。"""
    df = load(path)
    correct_idx = df.loc[df["exact_match"], "idx"].tolist()
    line = (
        f"| {name} | {metric_cell(df, 'exact_match')} | "
        f"{metric_cell(df, 'format_ok')} | "
        f"{metric_cell(df, 'single_final_answer_ok')} | "
        f"{metric_cell(df, 'repeat_like')} | "
        f"{df['pred_chars'].mean():.1f} | {correct_idx} |"
    )
    return name, df, line


def compare_correct_sets(left_name: str, left: pd.DataFrame, right_name: str, right: pd.DataFrame) -> list[str]:
    """对比两份结果的正确题号集合。"""
    left_ok = set(left.loc[left["exact_match"], "idx"])
    right_ok = set(right.loc[right["exact_match"], "idx"])
    all_idx = set(left["idx"].tolist()) | set(right["idx"].tolist())
    return [
        f"### {left_name} vs {right_name}",
        "",
        f"- {right_name} 新增正确: {sorted(right_ok - left_ok)}",
        f"- {right_name} 退化错误: {sorted(left_ok - right_ok)}",
        f"- 两者都正确: {sorted(left_ok & right_ok)}",
        f"- 两者都错误: {sorted(all_idx - left_ok - right_ok)}",
        "",
    ]


def main() -> None:
    """生成 Markdown 对比报告。"""
    runs = [
        ("0.6B base max512", BASE_EVAL_DIR / "base_eval_20_max512_full.jsonl"),
        ("0.6B SFT max160", YHY_DIR / "eval_results" / "sft_model" / "sft_lora_len768_lr3e-5_ep1_run2_eval_20_max160_full.jsonl"),
        ("1.7B base max160", BASE_EVAL_DIR / "qwen3_1d7b_base_eval_20_max160_full.jsonl"),
        ("1.7B base max512", BASE_EVAL_DIR / "qwen3_1d7b_base_eval_20_max512_full.jsonl"),
        ("1.7B SFT max160", SFT_EVAL_DIR / "qwen3_1d7b_sft_lora_len768_lr2e-5_ep1_eval_20_max160_full.jsonl"),
        ("1.7B SFT max512", SFT_EVAL_DIR / "qwen3_1d7b_sft_lora_len768_lr2e-5_ep1_eval_20_max512_full.jsonl"),
    ]

    frames: dict[str, pd.DataFrame] = {}
    table_lines: list[str] = []
    for name, path in runs:
        if not path.exists():
            continue
        run_name, df, line = row_for(name, path)
        frames[run_name] = df
        table_lines.append(line)

    lines = [
        "# Qwen3-1.7B Base vs SFT 对比报告",
        "",
        "## Summary",
        "",
        "| run | exact | format | single final | repeat-like | avg chars | correct idx |",
        "|---|---:|---:|---:|---:|---:|---|",
        *table_lines,
        "",
        "## Correctness Diff",
        "",
    ]

    pairs = [
        ("1.7B base max160", "1.7B SFT max160"),
        ("1.7B base max512", "1.7B SFT max512"),
        ("0.6B SFT max160", "1.7B SFT max160"),
    ]
    for left_name, right_name in pairs:
        if left_name in frames and right_name in frames:
            lines.extend(compare_correct_sets(left_name, frames[left_name], right_name, frames[right_name]))

    lines.extend(
        [
            "## 结论",
            "",
            "- 1.7B SFT max160 相比 1.7B base max160，exact 从 1/20 提升到 9/20，format 从 0/20 提升到 12/20，说明 SFT 对短输出问答格式非常有效。",
            "- 1.7B SFT max512 相比 1.7B base max512，format 从 0/20 提升到 19/20，但 exact 从 11/20 降到 10/20，说明这轮 SFT 主要提升格式，不是稳定提升数学能力。",
            "- 1.7B SFT max512 的 repeat-like 是 20/20，avg_hash_count 是 6.0，长输出时出现明显重复最终答案的问题。",
            "- 1.7B SFT max160 和旧 0.6B SFT max160 都是 9/20，但正确题集合不同；1.7B 底座没有在这轮短输出 SFT 中完全兑现为更高准确率。",
            "- 当前最值得改的不是继续加长 max_new_tokens，而是解决停止边界和重复输出：例如更严格的 EOS/stop、训练数据中 assistant 末尾结束标记检查、降低生成长度、或对重复答案加入惩罚/过滤。",
        ]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
