"""JSONL、summary 和 Markdown 报告生成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any) -> None:
    """写入 UTF-8 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """写入 UTF-8 JSONL 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _yes_no(value: Any) -> str:
    """把布尔值渲染为报告中的简短文本。"""
    return "yes" if bool(value) else "no"


def build_eval_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], title: str) -> str:
    """生成包含原始输出的评估报告。"""
    lines: list[str] = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- tag: `{summary.get('tag')}`",
        f"- n: {summary.get('n')}",
        f"- exact_match: {summary.get('exact_match', 0):.4f}",
        f"- first_hash_exact_match: {summary.get('first_hash_exact_match', 0):.4f}",
        f"- format_rate: {summary.get('format_rate', 0):.4f}",
        f"- single_final_answer_rate: {summary.get('single_final_answer_rate', 0):.4f}",
        f"- repeat_like_rate: {summary.get('repeat_like_rate', 0):.4f}",
        f"- avg_hash_count: {summary.get('avg_hash_count', 0):.4f}",
        f"- max_hash_count: {summary.get('max_hash_count', 0)}",
        f"- avg_chars: {summary.get('avg_chars', 0):.2f}",
        "",
        "## Items",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"### #{row.get('idx')}",
                "",
                f"- exact_match: {_yes_no(row.get('exact_match'))}",
                f"- format_ok: {_yes_no(row.get('format_ok'))}",
                f"- single_final_answer_ok: {_yes_no(row.get('single_final_answer_ok'))}",
                f"- repeat_like: {_yes_no(row.get('repeat_like'))}",
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


def write_markdown(path: str | Path, text: str) -> None:
    """写入 Markdown 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
