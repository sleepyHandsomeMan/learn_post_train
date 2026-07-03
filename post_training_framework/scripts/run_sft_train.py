"""运行 LoRA SFT 训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.config import ExperimentConfig
from ptf.train_sft import train_lora_sft


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 LoRA SFT adapter。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_0d6b.json",
        help="实验配置文件路径。",
    )
    parser.add_argument("--run-name", type=str, default=None, help="本次训练 checkpoint 名称。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段，例如 --set sft.learning_rate=3e-5。",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    train_lora_sft(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
