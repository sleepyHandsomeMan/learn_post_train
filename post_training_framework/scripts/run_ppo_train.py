"""PPO 训练入口脚本。

用法示例：

    python post_training_framework/scripts/run_ppo_train.py \\
      --config configs/gsm8k_qwen3_1d7b.json \\
      --sft-adapter-dir models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2 \\
      --train-file datasets/gsm8k_sft/train.parquet \\
      --eval-file datasets/gsm8k_sft/eval_20.parquet \\
      --total-training-steps 5 \\
      --train-batch-size 4 \\
      --rollout-n 1 \\
      --run-name ppo_smoke_test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.train_ppo import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 训练 (自包含，不依赖 verl)。")

    # 配置
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON 实验配置文件。")
    parser.add_argument("--run-name", type=str, default="ppo_run",
                        help="本次训练的名称。")

    # 模型
    parser.add_argument("--base-model-dir", type=str, default=None)
    parser.add_argument("--sft-adapter-dir", type=str, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--critic-hidden-size", type=int, default=1024)

    # 数据
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=512)

    # rollout
    parser.add_argument("--rollout-n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)

    # 训练
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    parser.add_argument("--kl-loss-type", type=str, default="low_var_kl",
                        choices=["low_var_kl", "kl"])
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--total-training-steps", type=int, default=500,
                        help="最大训练步数上限（兜底）。")
    parser.add_argument("--max-steps-no-improve", type=int, default=50,
                        help="验证 EM 连续不改善的最大步数 (early stop)。")
    parser.add_argument("--kl-threshold", type=float, default=0.1,
                        help="KL 散度阈值，超过则异常终止。")
    parser.add_argument("--reward-hacking-detect", action="store_true", default=True)
    parser.add_argument("--no-reward-hacking-detect", dest="reward_hacking_detect", action="store_false",
                        help="关闭 reward hacking 检测。")
    parser.add_argument("--reward-hacking-window", type=int, default=30,
                        help="reward hacking 检测窗口 (步数)。")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-steps", type=int, default=1)

    # 验证
    parser.add_argument("--val-before-train", action="store_true", default=True)
    parser.add_argument("--no-val-before-train", dest="val_before_train", action="store_false")
    parser.add_argument("--val-max-items", type=int, default=20)

    return parser.parse_args()


def _resolve_path(raw: str | None, workspace_root: Path) -> str:
    if raw is None:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = workspace_root / path
    return str(path.resolve())


def _load_config_defaults(config_path: Path | None, workspace_root: Path) -> dict:
    import json
    defaults: dict = {}
    if config_path is not None and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults["base_model_dir"] = _resolve_path(
            data.get("model", {}).get("base_model_dir"), workspace_root
        )
        defaults["train_file"] = _resolve_path(
            data.get("dataset", {}).get("train_file"), workspace_root
        )
        defaults["eval_file"] = _resolve_path(
            data.get("dataset", {}).get("eval_file"), workspace_root
        )
    return defaults


def main() -> None:
    args = parse_args()
    config_defaults = _load_config_defaults(args.config, WORKSPACE_ROOT)

    cfg = PPOConfig(
        base_model_dir=args.base_model_dir or config_defaults.get("base_model_dir", ""),
        sft_adapter_dir=_resolve_path(args.sft_adapter_dir, WORKSPACE_ROOT),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        critic_hidden_size=args.critic_hidden_size,
        train_file=args.train_file or config_defaults.get("train_file", ""),
        eval_file=args.eval_file or config_defaults.get("eval_file", ""),
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        train_batch_size=args.train_batch_size,
        ppo_epochs=args.ppo_epochs,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        gamma=args.gamma,
        lam=args.lam,
        kl_loss_coef=args.kl_loss_coef,
        kl_loss_type=args.kl_loss_type,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        total_training_steps=args.total_training_steps,
        max_steps_no_improve=args.max_steps_no_improve,
        kl_threshold=args.kl_threshold,
        reward_hacking_detect=args.reward_hacking_detect,
        reward_hacking_window=args.reward_hacking_window,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        seed=args.seed,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=_resolve_path(args.output_dir, WORKSPACE_ROOT) if args.output_dir else "",
        run_name=args.run_name,
        log_steps=args.log_steps,
        val_before_train=args.val_before_train,
        val_max_items=args.val_max_items,
    )

    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
