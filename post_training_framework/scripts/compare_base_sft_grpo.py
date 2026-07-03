"""对比 Base、SFT、GRPO 三个阶段的评估结果。

输入三份固定验证集上的 JSONL，输出 summary JSON 和 Markdown 对比报告。
这个脚本不加载模型、不启动训练，只做已有评估结果的离线汇总。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.compare import compare_three_jsonl


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比 Base/SFT/GRPO 三份评估 JSONL。")
    parser.add_argument("--base-jsonl", type=Path, required=True, help="Base model 评估 JSONL。")
    parser.add_argument("--sft-jsonl", type=Path, required=True, help="SFT model 评估 JSONL。")
    parser.add_argument("--grpo-jsonl", type=Path, required=True, help="GRPO model 评估 JSONL。")
    parser.add_argument("--run-name", type=str, required=True, help="对比报告名称前缀。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 eval_results/grpo_model/<run_name>。",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    output_dir = args.output_dir or (WORKSPACE_ROOT / "eval_results" / "grpo_model" / args.run_name)
    compare_three_jsonl(
        base_jsonl=args.base_jsonl,
        sft_jsonl=args.sft_jsonl,
        grpo_jsonl=args.grpo_jsonl,
        output_dir=output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
