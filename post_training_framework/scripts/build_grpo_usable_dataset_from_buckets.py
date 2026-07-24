"""从 messages 分桶结果合并当前项目可用的 GRPO 训练数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FULL_BUCKETS: dict[str, str] = {
    "greedy_wrong__oracle_mixed": "core_all_greedy_wrong_oracle_mixed",
    "greedy_correct__oracle_mixed": "core_all_greedy_correct_oracle_mixed",
    "greedy_wrong__oracle_all_correct": "small_full_greedy_wrong_oracle_all_correct",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="合并 GRPO messages bucket，并输出训练数据预览。")
    parser.add_argument("--bucket-dir", type=Path, required=True, help="messages 分桶 parquet 所在目录。")
    parser.add_argument("--output-parquet", type=Path, required=True, help="输出合并后的 GRPO 训练 parquet。")
    parser.add_argument("--preview-md", type=Path, required=True, help="输出 Markdown 预览文件。")
    parser.add_argument("--meta-json", type=Path, default=None, help="输出元数据 JSON；默认与 parquet 同名。")
    parser.add_argument("--seed", type=int, default=42, help="边界抽样和最终 shuffle 的随机种子。")
    parser.add_argument("--preview-items", type=int, default=6, help="预览前多少条样本。")
    parser.add_argument("--sample-mastered", type=int, default=128, help="从 mastered easy 桶抽样多少条。")
    parser.add_argument("--sample-hard", type=int, default=128, help="从 hard no-signal 桶抽样多少条。")
    parser.add_argument("--sample-fragile", type=int, default=64, help="从 fragile correctness 桶抽样多少条。")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱最终训练集。")
    return parser.parse_args()


def _read_bucket(bucket_dir: Path, bucket: str) -> pd.DataFrame:
    """读取指定 bucket 的 messages parquet，并补齐来源字段。"""
    path = bucket_dir / f"{bucket}.messages.parquet"
    if not path.exists():
        raise FileNotFoundError(f"缺少 bucket 文件: {path}")
    df = pd.read_parquet(path)
    if "messages" not in df.columns:
        raise ValueError(f"{path} 缺少 messages 列，不是当前项目的 GRPO 数据格式。")

    df = df.copy()
    if "source_index" not in df.columns:
        assignments_path = bucket_dir / "bucket_assignments.csv"
        if not assignments_path.exists():
            raise ValueError(f"{path} 缺少 source_index，且找不到 {assignments_path}。")
        assignments = pd.read_csv(assignments_path, usecols=["idx", "bucket"])
        source_indices = assignments.loc[assignments["bucket"] == bucket, "idx"].astype(int).tolist()
        if len(source_indices) != len(df):
            raise ValueError(
                f"{bucket} 的 messages 行数 {len(df)} 与 bucket_assignments 行数 {len(source_indices)} 不一致。"
            )
        df["source_index"] = source_indices
    if "source_bucket" not in df.columns:
        df["source_bucket"] = bucket
    return df


def _annotate(df: pd.DataFrame, bucket: str, role: str) -> pd.DataFrame:
    """增加来源桶和选择策略，便于训练后追溯。"""
    out = df.copy()
    out["source_bucket"] = bucket
    out["selection_role"] = role
    return out


def _sample_bucket(
    bucket_dir: Path,
    bucket: str,
    sample_count: int,
    role: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从边界 bucket 中确定性抽样。"""
    df = _read_bucket(bucket_dir, bucket)
    selected_count = min(sample_count, len(df))
    if selected_count < len(df):
        selected = df.sample(n=selected_count, random_state=random_state).sort_index()
    else:
        selected = df
    selected = _annotate(selected, bucket=bucket, role=role)
    return selected, {
        "bucket": bucket,
        "mode": "sample",
        "role": role,
        "source_rows": int(len(df)),
        "selected_rows": int(len(selected)),
        "requested_rows": int(sample_count),
    }


def _message_prompt_content(messages: Any) -> str:
    """从 messages 列中提取 user prompt。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(messages)


def _message_answer_content(messages: Any) -> str:
    """从 messages 列中提取 assistant 标准答案。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, list) and len(messages) > 1:
        second = messages[1]
        if isinstance(second, dict):
            return str(second.get("content", ""))
    return ""


def _extract_ground_truth(answer: str) -> str:
    """从 GSM8K assistant 答案里抽取 #### 后的 ground truth。"""
    if "####" not in answer:
        return ""
    return answer.rsplit("####", 1)[-1].strip()


def _short(text: Any, limit: int = 520) -> str:
    """截断长文本，避免预览文件过长。"""
    value = str(text).replace("\r\n", "\n").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _write_preview(
    output_path: Path,
    output_parquet: Path,
    combined: pd.DataFrame,
    manifest: list[dict[str, Any]],
    preview_items: int,
) -> None:
    """写出合并后训练文件的 Markdown 预览。"""
    lines: list[str] = [
        "# GRPO 合并训练数据预览",
        "",
        f"- parquet: `{output_parquet}`",
        f"- total_rows: `{len(combined)}`",
        f"- columns: `{', '.join(combined.columns)}`",
        "",
        "## 合并策略",
        "",
        "| source_bucket | mode | selected_rows | source_rows | role |",
        "|---|---:|---:|---:|---|",
    ]
    for item in manifest:
        lines.append(
            f"| `{item['bucket']}` | {item['mode']} | {item['selected_rows']} | "
            f"{item['source_rows']} | `{item['role']}` |"
        )

    lines.extend(
        [
            "",
            "## 前几条样本",
            "",
        ]
    )
    for pos, (_, row) in enumerate(combined.head(preview_items).iterrows(), start=1):
        prompt_text = _message_prompt_content(row["messages"])
        answer = _message_answer_content(row["messages"])
        ground_truth = _extract_ground_truth(answer)
        source_bucket = row.get("source_bucket")
        selection_role = row.get("selection_role")
        sample_index = row.get("source_index", "NA")
        lines.extend(
            [
                f"### {pos}. idx={sample_index} bucket={source_bucket}",
                "",
                "- data_format: `messages`",
                f"- ground_truth: `{ground_truth}`",
                f"- selection_role: `{selection_role}`",
                "",
                "prompt:",
                "",
                "```text",
                _short(prompt_text),
                "```",
                "",
                "assistant_answer 预览:",
                "",
                "```text",
                _short(answer),
                "```",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    """入口函数。"""
    args = parse_args()
    sample_buckets: dict[str, tuple[int, str]] = {
        "greedy_correct__oracle_all_correct": (
            args.sample_mastered,
            "boundary_sample_mastered_easy",
        ),
        "greedy_wrong__oracle_all_wrong": (
            args.sample_hard,
            "boundary_sample_hard_no_signal",
        ),
        "greedy_correct__oracle_all_wrong": (
            args.sample_fragile,
            "boundary_sample_fragile_correctness",
        ),
    }

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for bucket, role in FULL_BUCKETS.items():
        df = _read_bucket(args.bucket_dir, bucket)
        selected = _annotate(df, bucket=bucket, role=role)
        frames.append(selected)
        manifest.append(
            {
                "bucket": bucket,
                "mode": "full",
                "role": role,
                "source_rows": int(len(df)),
                "selected_rows": int(len(selected)),
            }
        )

    for offset, (bucket, (sample_count, role)) in enumerate(sample_buckets.items(), start=1):
        selected, item = _sample_bucket(
            args.bucket_dir,
            bucket=bucket,
            sample_count=sample_count,
            role=role,
            random_state=args.seed + offset,
        )
        frames.append(selected)
        manifest.append(item)

    combined = pd.concat(frames, ignore_index=True)
    if not args.no_shuffle:
        combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    indices = combined["source_index"].tolist()
    buckets = combined["source_bucket"].tolist()
    duplicate_index_count = int(pd.Series(indices).duplicated().sum())
    selected_by_bucket = {str(key): int(value) for key, value in pd.Series(buckets).value_counts().sort_index().items()}

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output_parquet, index=False)
    _write_preview(args.preview_md, args.output_parquet, combined, manifest, args.preview_items)

    meta_path = args.meta_json or args.output_parquet.with_suffix(".meta.json")
    meta = {
        "bucket_dir": str(args.bucket_dir.resolve()),
        "output_parquet": str(args.output_parquet.resolve()),
        "preview_md": str(args.preview_md.resolve()),
        "seed": int(args.seed),
        "shuffled": not bool(args.no_shuffle),
        "data_format": "messages",
        "total_rows": int(len(combined)),
        "duplicate_source_index_count": duplicate_index_count,
        "selected_rows_by_bucket": selected_by_bucket,
        "selection_manifest": manifest,
        "note": "两个 oracle_mixed 桶全量使用；非 mixed 桶只做少量边界补充。",
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
