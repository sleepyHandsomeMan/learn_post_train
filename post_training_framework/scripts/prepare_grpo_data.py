"""GRPO/PPO 数据预览工具。

从 SFT messages parquet 生成预览报告，用于人工检查数据质量。

新的 GRPO/PPO 训练器 (train_grpo.py / train_ppo.py) 直接加载 SFT messages
parquet，无需预先转换格式。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.grpo_data import preview_prompt_samples
from ptf.reports import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览 SFT parquet 中可用于 RL 训练的样本。")
    parser.add_argument(
        "--sft-file",
        type=Path,
        default=WORKSPACE_ROOT / "datasets" / "gsm8k_sft" / "train.parquet",
        help="SFT parquet 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "datasets" / "gsm8k_grpo",
        help="预览报告输出目录。",
    )
    parser.add_argument("--num-preview", type=int, default=5, help="预览样本数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = preview_prompt_samples(
        input_file=args.sft_file,
        num_preview=args.num_preview,
    )

    preview_path = args.output_dir / "preview_rl_prompts.md"
    write_markdown(preview_path, report)
    print(f"预览报告已保存: {preview_path}")
    print(f"预览 {args.num_preview} 条样本, 源自: {args.sft_file}")


if __name__ == "__main__":
    main()
