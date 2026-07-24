"""根据 SFT greedy/oracle 评估结果构造 GRPO focused 训练集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="筛选 SFT greedy 答错但 oracle@k 命中的题，作为 GRPO 诊断训练集。"
    )
    parser.add_argument("--source-parquet", type=Path, required=True, help="原始 SFT/RL messages parquet。")
    parser.add_argument("--greedy-jsonl", type=Path, required=True, help="SFT greedy@1 评估 JSONL。")
    parser.add_argument("--oracle-jsonl", type=Path, required=True, help="SFT oracle@k 评估 JSONL。")
    parser.add_argument("--output-parquet", type=Path, required=True, help="输出 focused parquet。")
    parser.add_argument("--max-items", type=int, default=96, help="最多输出多少条。")
    parser.add_argument("--min-oracle-exact-count", type=int, default=1, help="oracle 候选中至少几条答对。")
    parser.add_argument("--seed", type=int, default=42, help="筛选后抽样随机种子。")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def main() -> None:
    """入口函数。"""
    args = parse_args()
    if args.max_items < 1:
        raise ValueError("--max-items 必须 >= 1。")

    source_df = pd.read_parquet(args.source_parquet)
    greedy_rows = _read_jsonl(args.greedy_jsonl)
    oracle_rows = _read_jsonl(args.oracle_jsonl)

    greedy_by_idx = {int(row["idx"]): row for row in greedy_rows}
    oracle_by_idx = {int(row["idx"]): row for row in oracle_rows}
    common_indices = sorted(set(greedy_by_idx) & set(oracle_by_idx))

    selected: list[int] = []
    for idx in common_indices:
        greedy_row = greedy_by_idx[idx]
        oracle_row = oracle_by_idx[idx]
        greedy_wrong = not bool(greedy_row.get("exact_match", False))
        oracle_hit = bool(oracle_row.get("oracle_exact_match", False))
        exact_count = int(oracle_row.get("exact_count", 0))
        if greedy_wrong and oracle_hit and exact_count >= args.min_oracle_exact_count:
            selected.append(idx)

    if len(selected) > args.max_items:
        selected_df = pd.Series(selected).sample(n=args.max_items, random_state=args.seed)
        selected = [int(x) for x in selected_df.tolist()]

    selected = sorted(selected)
    if not selected:
        raise RuntimeError("没有筛出 greedy 错但 oracle 命中的样本，请扩大 max-items 或检查评估结果。")

    out_df = source_df.iloc[selected].copy()
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output_parquet, index=False)

    meta = {
        "source_parquet": str(args.source_parquet.resolve()),
        "greedy_jsonl": str(args.greedy_jsonl.resolve()),
        "oracle_jsonl": str(args.oracle_jsonl.resolve()),
        "output_parquet": str(args.output_parquet.resolve()),
        "source_rows": int(len(source_df)),
        "greedy_rows": int(len(greedy_rows)),
        "oracle_rows": int(len(oracle_rows)),
        "common_rows": int(len(common_indices)),
        "selected_rows": int(len(selected)),
        "selected_positions": selected,
        "filter": {
            "greedy_exact_match": False,
            "oracle_exact_match": True,
            "min_oracle_exact_count": args.min_oracle_exact_count,
        },
        "seed": args.seed,
    }
    meta_path = args.output_parquet.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
