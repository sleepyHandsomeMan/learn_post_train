"""运行单个冻结GRPO checkpoint的扩大sample评估。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.sample_eval import evaluate_grpo_checkpoint_sample


def _configure_utf8_output() -> None:
    """固定Windows重定向日志编码，避免中文进度乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="冻结GRPO checkpoint扩大sample评估")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--return-sequences", type=int, default=4)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument(
        "--format-instruction",
        default='Let\'s think step by step and output the final answer after "####".',
    )
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> int:
    """执行单checkpoint评估。"""
    _configure_utf8_output()
    args = parse_args()
    summary = evaluate_grpo_checkpoint_sample(
        config_path=args.config.resolve(),
        checkpoint_dir=args.checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        trial_id=args.trial_id,
        variant=args.variant,
        training_seed=args.training_seed,
        max_items=args.max_items,
        max_response_tokens=args.max_response_tokens,
        eval_batch_size=args.eval_batch_size,
        return_sequences=args.return_sequences,
        eval_seeds=args.eval_seeds,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_prompt_length=args.max_prompt_length,
        format_instruction=args.format_instruction,
        enable_thinking=args.enable_thinking,
    )
    print(
        f"评估完成: trial={summary['trial_id']} responses={summary['full']['responses']} "
        f"summary={args.output_dir.resolve() / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
