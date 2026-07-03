"""对比两份评估 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.compare import compare_jsonl
from ptf.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 base 与 sft 的逐题 JSONL。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_0d6b.json",
        help="实验配置文件路径。",
    )
    parser.add_argument("--base-jsonl", type=Path, required=True, help="base 评估 JSONL。")
    parser.add_argument("--sft-jsonl", type=Path, required=True, help="sft 评估 JSONL。")
    parser.add_argument("--run-name", type=str, default="base_vs_sft", help="对比报告名称。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段。",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    output_dir = cfg.ensure_experiment_dir() / "compare"
    compare_jsonl(args.base_jsonl, args.sft_jsonl, output_dir=output_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
