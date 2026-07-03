"""运行 SFT LoRA adapter 推理评估。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.config import ExperimentConfig
from ptf.generation import evaluate_model


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 SFT adapter 评估。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_0d6b.json",
        help="实验配置文件路径。",
    )
    parser.add_argument("--adapter-dir", type=Path, default=None, help="LoRA adapter 目录。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="生成最大新 token 数。")
    parser.add_argument("--max-items", type=int, default=None, help="最多评估多少条样本。")
    parser.add_argument("--run-name", type=str, default=None, help="本次评估产物名称。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段，例如 --set prompt.sft_prompt_mode=chat。",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    adapter_dir = args.adapter_dir or cfg.path("sft.eval_adapter_dir")
    evaluate_model(
        cfg,
        model_kind="sft",
        adapter_dir=adapter_dir,
        max_new_tokens=args.max_new_tokens,
        max_items=args.max_items,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
