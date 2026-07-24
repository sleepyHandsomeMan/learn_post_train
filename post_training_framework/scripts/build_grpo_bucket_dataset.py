"""根据 SFT greedy/oracle 结果给 GRPO 训练集分桶。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent


FORMAT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按 greedy@1/oracle@k 结果给 GRPO 训练样本分桶。")
    parser.add_argument("--source-parquet", type=Path, required=True, help="原始 SFT messages parquet。")
    parser.add_argument("--greedy-jsonl", type=Path, required=True, help="SFT greedy@1 评估结果 JSONL。")
    parser.add_argument("--oracle-jsonl", type=Path, required=True, help="SFT oracle@k 评估结果 JSONL。")
    parser.add_argument("--output-dir", type=Path, required=True, help="分桶结果输出目录。")
    parser.add_argument("--write-parquets", action="store_true", default=True, help="写出每个桶的 parquet。")
    parser.add_argument("--no-write-parquets", dest="write_parquets", action="store_false")
    return parser.parse_args()


def _read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    """读取 JSONL 并按 idx 建索引。"""
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            rows[int(row["idx"])] = row
    return rows


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """把 parquet 读出的 messages 转成普通 dict 列表。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(message) for message in messages]


def _strip_format_instruction(user_content: str) -> str:
    """从 user prompt 末尾去掉格式指令，得到原始问题。"""
    text = str(user_content).strip()
    suffix = f" {FORMAT_INSTRUCTION}"
    if text.endswith(suffix):
        return text[: -len(suffix)].strip()
    return text


def _bucket_name(greedy_exact: bool, exact_count: int, oracle_k: int) -> str:
    """根据 greedy 是否正确和 oracle 命中数量分桶。"""
    greedy_part = "greedy_correct" if greedy_exact else "greedy_wrong"
    if exact_count <= 0:
        oracle_part = "oracle_all_wrong"
    elif exact_count >= oracle_k:
        oracle_part = "oracle_all_correct"
    else:
        oracle_part = "oracle_mixed"
    return f"{greedy_part}__{oracle_part}"


def _write_markdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    """写出分桶统计 Markdown。"""
    lines = [
        "# GRPO 训练集 greedy/oracle 分桶统计",
        "",
        "| bucket | count | percent | avg_oracle_exact_count | avg_format_count |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| `{row['bucket']}` | {int(row['count'])} | {row['percent']:.2f}% | "
            f"{row['avg_oracle_exact_count']:.3f} | {row['avg_oracle_format_count']:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """入口函数。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(args.source_parquet)
    greedy_rows = _read_jsonl(args.greedy_jsonl)
    oracle_rows = _read_jsonl(args.oracle_jsonl)
    common_indices = sorted(set(greedy_rows) & set(oracle_rows))
    if not common_indices:
        raise RuntimeError("greedy/oracle 结果没有共同 idx，无法分桶。")

    assignments: list[dict[str, Any]] = []
    bucket_to_indices: dict[str, list[int]] = {}
    for idx in common_indices:
        greedy = greedy_rows[idx]
        oracle = oracle_rows[idx]
        exact_count = int(oracle.get("exact_count", 0))
        oracle_k = int(oracle.get("oracle_k", 8))
        greedy_exact = bool(greedy.get("exact_match", False))
        bucket = _bucket_name(greedy_exact, exact_count, oracle_k)
        bucket_to_indices.setdefault(bucket, []).append(idx)

        messages = _normalize_messages(source_df.iloc[idx]["messages"])
        user_content = str(messages[0]["content"]).strip()
        assignments.append(
            {
                "idx": int(idx),
                "bucket": bucket,
                "greedy_exact_match": greedy_exact,
                "greedy_pred_answer": greedy.get("pred_answer"),
                "gold_answer": greedy.get("gold_answer"),
                "oracle_exact_count": exact_count,
                "oracle_k": oracle_k,
                "oracle_format_count": int(oracle.get("format_count", 0)),
                "oracle_pred_answer_counts": json.dumps(oracle.get("pred_answer_counts", {}), ensure_ascii=False),
                "question": _strip_format_instruction(user_content),
            }
        )

    assignment_df = pd.DataFrame(assignments)
    assignment_path = args.output_dir / "bucket_assignments.csv"
    assignment_df.to_csv(assignment_path, index=False, encoding="utf-8-sig")

    total = len(assignment_df)
    summary_rows: list[dict[str, Any]] = []
    for bucket, group in assignment_df.groupby("bucket", sort=True):
        summary_rows.append(
            {
                "bucket": bucket,
                "count": int(len(group)),
                "percent": float(len(group) / total * 100),
                "avg_oracle_exact_count": float(group["oracle_exact_count"].mean()),
                "avg_oracle_format_count": float(group["oracle_format_count"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["bucket"]).reset_index(drop=True)
    summary_csv = args.output_dir / "bucket_summary.csv"
    summary_md = args.output_dir / "bucket_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    _write_markdown(summary_df, summary_md)

    if args.write_parquets:
        for bucket, indices in bucket_to_indices.items():
            message_df = source_df.iloc[indices].copy().reset_index(drop=True)
            message_df["source_index"] = [int(idx) for idx in indices]
            message_df["source_bucket"] = bucket
            message_df.to_parquet(args.output_dir / f"{bucket}.messages.parquet", index=False)

    meta = {
        "source_parquet": str(args.source_parquet.resolve()),
        "greedy_jsonl": str(args.greedy_jsonl.resolve()),
        "oracle_jsonl": str(args.oracle_jsonl.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "source_rows": int(len(source_df)),
        "greedy_rows": int(len(greedy_rows)),
        "oracle_rows": int(len(oracle_rows)),
        "bucketed_rows": int(total),
        "bucket_parquet_format": "messages",
    }
    with (args.output_dir / "bucket_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"meta": meta, "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
