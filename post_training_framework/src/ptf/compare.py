"""Base/SFT/GRPO 评估结果对比。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import summarize_rows
from .reports import read_jsonl, write_json, write_markdown


def _status(row: dict[str, Any]) -> str:
    """把单条样本压缩成便于人工查看的状态。"""
    pieces = []
    pieces.append("EM" if row.get("exact_match") else "wrong")
    pieces.append("fmt" if row.get("format_ok") else "no_fmt")
    if row.get("repeat_like"):
        pieces.append("repeat")
    return "+".join(pieces)


def build_compare_markdown(
    base_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
    sft_summary: dict[str, Any],
    title: str,
) -> str:
    """生成 Base/SFT 对比 Markdown。"""
    sft_by_idx = {int(row["idx"]): row for row in sft_rows}
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| model | n | exact_match | format_rate | single_final_answer_rate | repeat_like_rate | avg_chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| base | {base_summary['n']} | {base_summary['exact_match']:.4f} | "
            f"{base_summary['format_rate']:.4f} | {base_summary['single_final_answer_rate']:.4f} | "
            f"{base_summary['repeat_like_rate']:.4f} | {base_summary['avg_chars']:.2f} |"
        ),
        (
            f"| sft | {sft_summary['n']} | {sft_summary['exact_match']:.4f} | "
            f"{sft_summary['format_rate']:.4f} | {sft_summary['single_final_answer_rate']:.4f} | "
            f"{sft_summary['repeat_like_rate']:.4f} | {sft_summary['avg_chars']:.2f} |"
        ),
        "",
        "## Per Item",
        "",
        "| idx | gold | base_pred | sft_pred | base_status | sft_status | change |",
        "|---:|---:|---:|---:|---|---|---|",
    ]

    for base_row in base_rows:
        idx = int(base_row["idx"])
        sft_row = sft_by_idx.get(idx)
        if not sft_row:
            continue
        base_em = bool(base_row.get("exact_match"))
        sft_em = bool(sft_row.get("exact_match"))
        if (not base_em) and sft_em:
            change = "improved"
        elif base_em and (not sft_em):
            change = "regressed"
        else:
            change = "same"
        lines.append(
            "| {idx} | {gold} | {base_pred} | {sft_pred} | {base_status} | {sft_status} | {change} |".format(
                idx=idx,
                gold=base_row.get("gold_answer"),
                base_pred=base_row.get("pred_answer"),
                sft_pred=sft_row.get("pred_answer"),
                base_status=_status(base_row),
                sft_status=_status(sft_row),
                change=change,
            )
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `improved` 只表示 exact match 从错变对，不代表推理链一定真正变好。",
            "- `format_rate` 看是否出现合法 `#### 数字`。",
            "- `single_final_answer_rate` 和 `repeat_like_rate` 用来观察 EOS/停止边界问题。",
            "- 人工分析时优先看 `regressed`、`same` 但格式变化很大的样本。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_jsonl(base_jsonl: str | Path, sft_jsonl: str | Path, output_dir: str | Path, run_name: str) -> Path:
    """读取两个 JSONL，输出 summary JSON 和 Markdown 对比报告。"""
    base_rows = read_jsonl(base_jsonl)
    sft_rows = read_jsonl(sft_jsonl)
    base_summary = summarize_rows(base_rows, tag="base")
    sft_summary = summarize_rows(sft_rows, tag="sft")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_name}_compare_summary.json"
    md_path = output_dir / f"{run_name}_compare_report.md"

    write_json(
        summary_path,
        {
            "base_jsonl": str(base_jsonl),
            "sft_jsonl": str(sft_jsonl),
            "base": base_summary,
            "sft": sft_summary,
        },
    )
    write_markdown(
        md_path,
        build_compare_markdown(base_rows, sft_rows, base_summary, sft_summary, title=f"{run_name} Base vs SFT"),
    )
    print("saved compare summary:", summary_path)
    print("saved compare report:", md_path)
    return md_path


def _change_from_sft(sft_row: dict[str, Any], grpo_row: dict[str, Any]) -> str:
    """判断 GRPO 相对 SFT 的 exact match 变化。"""
    sft_em = bool(sft_row.get("exact_match"))
    grpo_em = bool(grpo_row.get("exact_match"))
    if (not sft_em) and grpo_em:
        return "improved"
    if sft_em and (not grpo_em):
        return "regressed"
    return "same"


def build_three_way_compare_markdown(
    base_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    grpo_rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
    sft_summary: dict[str, Any],
    grpo_summary: dict[str, Any],
    title: str,
) -> str:
    """生成 Base/SFT/GRPO 三方对比 Markdown。"""
    sft_by_idx = {int(row["idx"]): row for row in sft_rows}
    grpo_by_idx = {int(row["idx"]): row for row in grpo_rows}

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| model | n | exact_match | format_rate | single_final_answer_rate | repeat_like_rate | avg_chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in [("base", base_summary), ("sft", sft_summary), ("grpo", grpo_summary)]:
        lines.append(
            f"| {label} | {summary['n']} | {summary['exact_match']:.4f} | "
            f"{summary['format_rate']:.4f} | {summary['single_final_answer_rate']:.4f} | "
            f"{summary['repeat_like_rate']:.4f} | {summary['avg_chars']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per Item",
            "",
            "| idx | gold | base_pred | sft_pred | grpo_pred | base_status | sft_status | grpo_status | grpo_vs_sft |",
            "|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )

    for base_row in base_rows:
        idx = int(base_row["idx"])
        sft_row = sft_by_idx.get(idx)
        grpo_row = grpo_by_idx.get(idx)
        if not sft_row or not grpo_row:
            continue
        lines.append(
            "| {idx} | {gold} | {base_pred} | {sft_pred} | {grpo_pred} | "
            "{base_status} | {sft_status} | {grpo_status} | {change} |".format(
                idx=idx,
                gold=base_row.get("gold_answer"),
                base_pred=base_row.get("pred_answer"),
                sft_pred=sft_row.get("pred_answer"),
                grpo_pred=grpo_row.get("pred_answer"),
                base_status=_status(base_row),
                sft_status=_status(sft_row),
                grpo_status=_status(grpo_row),
                change=_change_from_sft(sft_row, grpo_row),
            )
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `grpo_vs_sft` 只表示 exact match 相对 SFT 的变化，不代表推理链一定真正变好。",
            "- 如果 GRPO reward 上升但 `exact_match` 下降，需要优先人工查看 full report。",
            "- 如果 `avg_chars` 大幅下降且 EM 下降，可能是模型学会过早结束。",
            "- 如果 `format_rate` 高但 EM 不升，说明 rule reward 可能主要强化了格式。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_three_jsonl(
    base_jsonl: str | Path,
    sft_jsonl: str | Path,
    grpo_jsonl: str | Path,
    output_dir: str | Path,
    run_name: str,
) -> Path:
    """读取三份 JSONL，输出 Base/SFT/GRPO 对比报告。"""
    base_rows = read_jsonl(base_jsonl)
    sft_rows = read_jsonl(sft_jsonl)
    grpo_rows = read_jsonl(grpo_jsonl)
    base_summary = summarize_rows(base_rows, tag="base")
    sft_summary = summarize_rows(sft_rows, tag="sft")
    grpo_summary = summarize_rows(grpo_rows, tag="grpo")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_name}_compare_summary.json"
    md_path = output_dir / f"{run_name}_compare_report.md"

    write_json(
        summary_path,
        {
            "base_jsonl": str(base_jsonl),
            "sft_jsonl": str(sft_jsonl),
            "grpo_jsonl": str(grpo_jsonl),
            "base": base_summary,
            "sft": sft_summary,
            "grpo": grpo_summary,
        },
    )
    write_markdown(
        md_path,
        build_three_way_compare_markdown(
            base_rows,
            sft_rows,
            grpo_rows,
            base_summary,
            sft_summary,
            grpo_summary,
            title=f"{run_name} Base vs SFT vs GRPO",
        ),
    )
    print("saved compare summary:", summary_path)
    print("saved compare report:", md_path)
    return md_path
