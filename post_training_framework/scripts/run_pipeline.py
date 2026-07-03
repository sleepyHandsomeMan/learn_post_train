"""按顺序运行 base 评估、SFT 训练、SFT 评估和对比。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.compare import compare_jsonl
from ptf.config import ExperimentConfig
from ptf.generation import evaluate_model
from ptf.train_sft import train_lora_sft


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行一轮 Base -> SFT -> Compare 实验闭环。")
    parser.add_argument(
        "--config",
        type=Path,
        default=FRAMEWORK_ROOT / "configs" / "gsm8k_qwen3_0d6b.json",
        help="实验配置文件路径。",
    )
    parser.add_argument("--skip-base", action="store_true", help="跳过 base 评估。")
    parser.add_argument("--skip-train", action="store_true", help="跳过 SFT 训练。")
    parser.add_argument("--skip-sft-eval", action="store_true", help="跳过 SFT 评估。")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="跳过训练时使用的 adapter 目录。")
    parser.add_argument("--base-jsonl", type=Path, default=None, help="跳过 base 评估时使用的 JSONL。")
    parser.add_argument("--sft-jsonl", type=Path, default=None, help="跳过 SFT 评估时使用的 JSONL。")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="生成最大新 token 数。")
    parser.add_argument("--max-items", type=int, default=None, help="最多评估多少条样本。")
    parser.add_argument("--run-name", type=str, default=None, help="本轮实验名称。")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置字段，例如 --set sft.num_train_epochs=1。",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数。"""
    args = parse_args()
    cfg = ExperimentConfig.load(args.config, overrides=args.overrides)
    run_name = args.run_name or str(cfg.get("experiment_name", "experiment"))
    base_jsonl = args.base_jsonl
    sft_jsonl = args.sft_jsonl
    adapter_dir = args.adapter_dir

    if not args.skip_base:
        _, _, base_jsonl = evaluate_model(
            cfg,
            model_kind="base",
            max_new_tokens=args.max_new_tokens,
            max_items=args.max_items,
            run_name=f"{run_name}_base",
        )

    if not args.skip_train:
        adapter_dir = train_lora_sft(cfg, run_name=f"{run_name}_sft")

    if not args.skip_sft_eval:
        if adapter_dir is None:
            adapter_dir = cfg.path("sft.eval_adapter_dir")
        _, _, sft_jsonl = evaluate_model(
            cfg,
            model_kind="sft",
            adapter_dir=adapter_dir,
            max_new_tokens=args.max_new_tokens,
            max_items=args.max_items,
            run_name=f"{run_name}_sft",
        )

    if base_jsonl is None or sft_jsonl is None:
        print("缺少 base_jsonl 或 sft_jsonl，跳过对比。")
        return

    compare_jsonl(
        base_jsonl,
        sft_jsonl,
        output_dir=cfg.ensure_experiment_dir() / "compare",
        run_name=f"{run_name}_base_vs_sft",
    )


if __name__ == "__main__":
    main()
