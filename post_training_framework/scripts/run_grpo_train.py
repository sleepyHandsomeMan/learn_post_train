"""GRPO 训练入口脚本。

用法示例：

    # 冒烟测试 (5 steps, 用 smoke 数据)
    python post_training_framework/scripts/run_grpo_train.py \\
      --config configs/gsm8k_qwen3_1d7b.json \\
      --sft-adapter-dir models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2 \\
      --train-file datasets/gsm8k_sft/train.parquet \\
      --eval-file datasets/gsm8k_sft/eval_20.parquet \\
      --total-training-steps 5 \\
      --train-batch-size 4 \\
      --rollout-n 4 \\
      --output-dir models/grpo/smoke_test \\
      --run-name smoke_test

    # 正式短训练 (50 steps)
    python post_training_framework/scripts/run_grpo_train.py \\
      --config configs/gsm8k_qwen3_1d7b.json \\
      --sft-adapter-dir models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2 \\
      --total-training-steps 50 \\
      --eval-freq 10 \\
      --run-name short_run
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.train_grpo import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO 训练 (自包含，不依赖 verl)。")

    # 配置
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON 实验配置文件。")
    parser.add_argument("--run-name", type=str, default=None,
                        help="本次训练的名称。")

    # 模型
    parser.add_argument("--base-model-dir", type=str, default=None,
                        help="Base model 目录。")
    parser.add_argument("--sft-adapter-dir", type=str, default=None,
                        help="SFT LoRA adapter 目录 (GRPO 起点)。")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="GRPO checkpoint 目录；恢复 GRPO LoRA、optimizer、训练状态和随机数状态。")
    parser.add_argument(
        "--resume-state-mode",
        type=str,
        choices=["full", "weights_only", "weights_and_optimizer"],
        default=None,
        help="checkpoint 加载模式：完整续训、仅权重分支或权重加旧 optimizer 诊断分支。",
    )
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--allow-base-start", action="store_true", default=None,
                        help="显式允许没有 SFT adapter 时从 base 起训；默认禁止。")
    parser.add_argument("--allow-resume-objective-change", action="store_true", default=None,
                        help="显式允许 resume 时修改 reward/KL 目标；必须使用新的输出分支。")

    # 数据
    parser.add_argument("--train-file", type=str, default=None,
                        help="训练数据 parquet (SFT messages 格式)。")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="验证数据 parquet。")
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-response-length", type=int, default=None)

    # rollout
    parser.add_argument("--rollout-n", type=int, default=None,
                        help="每个 prompt 生成的回答数。")
    parser.add_argument("--rollout-batch-size", type=int, default=None,
                        help="rollout 批量生成时一次处理的 prompt 数。")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)

    # 训练
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="每步处理的 prompt 数。")
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="每次 optimizer 更新前累积的 PPO mini-batch 数。")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--kl-loss-coef", type=float, default=None)
    parser.add_argument("--kl-loss-type", type=str, default=None,
                        choices=["low_var_kl", "kl"])
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--norm-adv-by-std", action="store_true", default=None)
    parser.add_argument("--no-norm-adv", dest="norm_adv_by_std", action="store_false", default=None,
                        help="不对 advantage 做标准差归一化。")
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--total-training-steps", type=int, default=None,
                        help="最大训练步数上限（兜底）。")
    parser.add_argument("--max-steps-no-improve", type=int, default=None,
                        help="验证 EM 连续不改善的最大步数 (early stop)。")
    parser.add_argument("--early-stop-trend-window", type=int, default=None,
                        help="达到早停耐心后，用于判断恢复趋势的最近验证点数量。")
    parser.add_argument("--early-stop-min-recovery-slope", type=float, default=None,
                        help="允许趋势延长所需的最小 EM 斜率（每个验证点）。")
    parser.add_argument("--early-stop-max-extension-steps", type=int, default=None,
                        help="检测到恢复趋势时最多额外训练的步数。")
    parser.add_argument("--kl-threshold", type=float, default=None,
                        help="actor 相对 SFT reference 的 KL 硬阈值。")
    parser.add_argument("--kl-warning-threshold", type=float, default=None,
                        help="actor-reference KL 预警阈值。")
    parser.add_argument("--kl-guard-window", type=int, default=None,
                        help="reference KL 滑动窗口长度。")
    parser.add_argument("--kl-guard-patience-checks", type=int, default=None,
                        help="reference KL 连续超限窗口数。")
    parser.add_argument("--approx-kl-threshold", type=float, default=None,
                        help="单次 PPO 更新相对 old policy 的 KL 阈值。")
    parser.add_argument("--adaptive-kl-enabled", action="store_true", default=None)
    parser.add_argument("--no-adaptive-kl", dest="adaptive_kl_enabled",
                        action="store_false", default=None)
    parser.add_argument("--adaptive-kl-target", type=float, default=None)
    parser.add_argument("--adaptive-kl-interval", type=int, default=None)
    parser.add_argument("--adaptive-kl-factor", type=float, default=None)
    parser.add_argument("--adaptive-kl-tolerance", type=float, default=None)
    parser.add_argument("--adaptive-kl-min-coef", type=float, default=None)
    parser.add_argument("--adaptive-kl-max-coef", type=float, default=None)
    parser.add_argument("--reward-hacking-detect", action="store_true", default=None)
    parser.add_argument("--no-reward-hacking-detect", dest="reward_hacking_detect", action="store_false", default=None,
                        help="关闭 reward hacking 检测。")
    parser.add_argument("--reward-hacking-window", type=int, default=None,
                        help="reward hacking 检测窗口 (步数)。")
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic-prompt-sampling", action="store_true", default=None,
                        help="按 seed 和 step 固定 prompt 序列，用于控制变量实验。")
    parser.add_argument("--no-deterministic-prompt-sampling",
                        dest="deterministic_prompt_sampling", action="store_false", default=None)
    parser.add_argument("--prompt-sampling-seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=None)

    # 奖励权重
    parser.add_argument("--reward-exact-with-format-score", type=float, default=None,
                        help="答案正确且带 #### 格式时的基础分。")
    parser.add_argument("--reward-exact-without-format-score", type=float, default=None,
                        help="答案正确但缺少 #### 格式时的基础分。")
    parser.add_argument("--reward-format-bonus", type=float, default=None,
                        help="存在 #### 格式时的加分。")
    parser.add_argument("--reward-single-final-bonus", type=float, default=None,
                        help="只出现一次最终答案标记时的加分。")
    parser.add_argument("--reward-missing-format-penalty", type=float, default=None,
                        help="缺少 #### 格式时的扣分。")
    parser.add_argument("--reward-multi-final-penalty", type=float, default=None,
                        help="出现多个最终答案标记时的扣分。")
    parser.add_argument("--reward-repeat-penalty", type=float, default=None,
                        help="复读或明显重复时的扣分。")
    parser.add_argument("--reward-overlong-penalty", type=float, default=None,
                        help="回答过长时的扣分。")
    parser.add_argument("--reward-overlong-chars", type=int, default=None,
                        help="触发过长扣分的字符数阈值。")
    parser.add_argument("--reward-long-response-token-threshold", type=int, default=None,
                        help="触发 token 长回答扣分的阈值。")
    parser.add_argument("--reward-long-response-penalty", type=float, default=None,
                        help="token 长回答扣分。")
    parser.add_argument("--reward-truncated-response-penalty", type=float, default=None,
                        help="达到生成上限且未 EOS 时的截断扣分。")
    parser.add_argument("--reward-min", type=float, default=None,
                        help="reward 裁剪下界。")
    parser.add_argument("--reward-max", type=float, default=None,
                        help="reward 裁剪上界。")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None,
                        help="模型输出目录。")
    parser.add_argument("--log-steps", type=int, default=None)

    # 验证
    parser.add_argument("--val-before-train", action="store_true", default=None)
    parser.add_argument("--no-val-before-train", dest="val_before_train", action="store_false", default=None)
    parser.add_argument("--val-max-items", type=int, default=None)
    parser.add_argument("--val-eval-batch-size", type=int, default=None,
                        help="验证集贪心推理 batch size。")
    parser.add_argument("--val-stochastic-n", type=int, default=None,
                        help="随机验证时每题采样的回答数。")
    parser.add_argument("--val-stochastic-max-items", type=int, default=None,
                        help="随机验证使用的固定题目数。")

    # 异常 rollout 留档
    parser.add_argument("--rollout-anomaly-dump-enabled", action="store_true", default=None)
    parser.add_argument("--no-rollout-anomaly-dump", dest="rollout_anomaly_dump_enabled",
                        action="store_false", default=None)
    parser.add_argument("--rollout-anomaly-max-samples", type=int, default=None,
                        help="每个异常 step 最多留档的 rollout 数。")

    # 组内训练信号保护
    parser.add_argument("--signal-guard-window", type=int, default=None)
    parser.add_argument("--signal-guard-warmup-steps", type=int, default=None)
    parser.add_argument("--signal-guard-patience-checks", type=int, default=None,
                        help="组内信号连续多少个滑动窗口不达标后才终止。")
    parser.add_argument("--min-effective-group-rate", type=float, default=None)
    parser.add_argument("--min-mixed-group-rate", type=float, default=None)
    parser.add_argument("--max-zero-advantage-rate", type=float, default=None)
    parser.add_argument("--min-rollout-format-rate", type=float, default=None)

    return parser.parse_args()


def _resolve_path(raw: str | None, workspace_root: Path) -> str:
    """解析路径：None → "", 相对路径 → 绝对路径。"""
    if raw is None or not str(raw).strip():
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = workspace_root / path
    return str(path.resolve())


def _load_config_defaults(config_path: Path | None, workspace_root: Path) -> dict:
    """从 JSON 配置文件提取默认参数。"""
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
        defaults["sft_adapter_dir"] = _resolve_path(
            data.get("model", {}).get("sft_adapter_dir"), workspace_root
        )
        defaults.update(data.get("grpo", {}))
    return defaults


def _build_grpo_config(args: argparse.Namespace, config_defaults: dict) -> GRPOConfig:
    """按“代码安全默认值 < JSON < CLI”合并 GRPO 配置。"""
    values = asdict(GRPOConfig())
    values.update({key: value for key, value in config_defaults.items() if value is not None})
    for key, value in vars(args).items():
        if key != "config" and value is not None and key in values:
            values[key] = value

    for key in ("base_model_dir", "sft_adapter_dir", "resume_from_checkpoint",
                "train_file", "eval_file", "output_dir"):
        values[key] = _resolve_path(values.get(key), WORKSPACE_ROOT)
    values["launch_command"] = subprocess.list2cmdline([sys.executable, *sys.argv])
    return GRPOConfig(**values)


def main() -> None:
    args = parse_args()
    config_defaults = _load_config_defaults(args.config, WORKSPACE_ROOT)

    cfg = _build_grpo_config(args, config_defaults)

    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
