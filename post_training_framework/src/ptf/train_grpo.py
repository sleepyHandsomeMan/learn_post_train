"""GRPO (Group Relative Policy Optimization) 训练器。

不依赖 verl 框架，完全自包含的 GRPO 实现。
与项目已有的数据处理（SFT messages parquet）和奖励函数（GSM8K rule reward）对接。

算法概要：
  1. 对每个 prompt 采样 N 个回答 (rollout)
  2. 用规则奖励对每个回答打分
  3. 组内归一化得到 advantage: A_i = (r_i - mean(r_group)) / std(r_group)
  4. PPO clipped surrogate loss + KL 惩罚
  5. 梯度下降更新 actor

参考：
  - DeepSeek GRPO 论文 (arXiv:2402.03300)
  - 本项目的 GSM8K rule reward (reward.py)
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import extract_final_answer
from .prompting import (
    DEFAULT_FORMAT_INSTRUCTION,
    apply_chat_template_text,
    build_user_content,
)
from .reward import GSM8KRewardConfig, compute_gsm8k_rule_reward
from .rl_dataset import RLPrompt, RLPromptDataset, load_rl_dataset
from .stopping import (
    StopCategory,
    StopDecision,
    StopSeverity,
    TrainingStopController,
    build_stop_decision,
)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger("ptf.grpo")


def setup_logging(output_dir: Path, run_name: str) -> None:
    """配置日志：同时输出到控制台和日志文件。"""
    logger.setLevel(logging.INFO)
    # 控制台handler跨训练会话复用；文件handler按输出目录重新绑定。
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

    # 文件 handler（每条日志立即 flush，防止 OOM 崩溃丢日志）
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    resolved_log_file = str(log_file.resolve())
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if str(Path(handler.baseFilename).resolve()) == resolved_log_file:
            return
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    try:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8", mode="a")
    except PermissionError as exc:
        logger.warning(f"日志文件无法写入，将只输出到控制台: {log_file} ({exc})")
        return
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    # 每条日志写完后立即 flush 到磁盘
    file_handler.flush = lambda: (
        file_handler.stream.flush() if file_handler.stream and not file_handler.stream.closed else None
    )
    logger.addHandler(file_handler)

    logger.info(f"日志文件: {log_file}")


# ---------------------------------------------------------------------------
# 绘图数据 CSV
# ---------------------------------------------------------------------------

# 训练指标 CSV 列定义 (每步一行)
TRAIN_CSV_COLUMNS = [
    "step", "reward_mean", "reward_std", "policy_loss", "kl_loss",
    "approx_kl", "clip_frac", "response_len_mean", "lr", "step_time",
    "grad_norm", "kl_loss_coef", "prompt_count", "rollout_count",
    "mini_batch_count", "optimizer_update_count", "best_val_em", "steps_no_improve",
]

# 验证指标 CSV 列定义 (每 eval_freq 步一行)
VAL_CSV_COLUMNS = [
    "step", "val_reward_mean", "val_exact_match", "val_format_rate",
    "val_response_len_mean", "val_max_tokens_reached_without_eos_rate", "val_eos_rate",
    "val_sample_exact_match", "val_sample_format_rate",
    "val_sample_response_len_mean", "val_sample_max_tokens_reached_without_eos_rate",
    "val_sample_eos_rate",
    "best_val_em_so_far",
]

# 组内 rollout 诊断 CSV 列定义 (每步一行)
GROUP_DIAG_CSV_COLUMNS = [
    "step",
    "group_count",
    "rollout_count",
    "rollout_n",
    "effective_group_rate",
    "zero_signal_group_rate",
    "mixed_group_rate",
    "all_wrong_group_rate",
    "all_correct_group_rate",
    "rollout_exact_rate",
    "rollout_format_rate",
    "rollout_eos_rate",
    "rollout_max_tokens_reached_without_eos_rate",
    "rollout_response_len_mean",
    "fallback_exact_rate",
    "group_reward_std_mean",
    "group_reward_std_nonzero_mean",
    "group_reward_std_max",
    "advantage_mean",
    "advantage_std",
    "advantage_abs_mean",
    "zero_advantage_rate",
    "unique_response_mean",
    "duplicate_group_rate",
    "response_empty_rate",
    "reward_answer_mean",
    "reward_format_mean",
    "reward_single_final_mean",
    "reward_repeat_mean",
    "reward_overlong_mean",
    "reward_length_mean",
    "reward_truncated_mean",
    "reward_raw_mean",
    "sample_indices_json",
    "sample_buckets_json",
    "correct_count_hist_json",
]

# 显存指标 CSV 列定义 (在关键节点追加一行，含各组件拆解)
GPU_CSV_COLUMNS = [
    "step", "tag", "allocated_gb", "reserved_gb", "pool_free_gb",
    "segment_count", "peak_allocated_gb",
    # 模型权重拆解
    "actor_base_weights_gb", "actor_lora_weights_gb", "actor_embed_weights_gb", "actor_weights_total_gb",
    "ref_base_weights_gb", "ref_lora_weights_gb", "ref_weights_total_gb",
    "model_weights_total_gb",
    # 优化器拆解
    "optimizer_momentum_gb", "optimizer_variance_gb", "optimizer_total_gb",
    # 梯度缓冲
    "grad_buffer_gb",
    # 训练数据
    "rollout_data_gb",
    # 激活值/logits/临时 (allocated - 识别项)
    "activations_logits_temp_gb",
    # KV cache 估算 (generate 阶段)
    "kv_cache_estimated_gb",
]


class _CSVWriter:
    """CSV writer 包装类，持有文件句柄以便实时 flush。"""
    def __init__(self, fh, writer):
        self.fh = fh
        self.writer = writer

    def writerow(self, row):
        self.writer.writerow(row)
        self.fh.flush()


def _init_csv(path: Path, columns: list[str]) -> Any:
    """初始化 CSV 文件并返回 writer。文件不存在则写表头，存在则追加。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                old_columns = reader.fieldnames or []
                old_rows = list(reader)
            if old_columns != columns:
                with path.open("w", encoding="utf-8", newline="") as f:
                    migrator = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                    migrator.writeheader()
                    migrator.writerows(old_rows)
                logger.info(f"CSV 列结构已迁移: {path}")
        need_header = not path.exists() or path.stat().st_size == 0
        fh = open(str(path), "a", encoding="utf-8", newline="")
    except PermissionError as exc:
        logger.warning(f"CSV 文件无法写入，将跳过该曲线文件: {path} ({exc})")
        return None
    writer = csv.writer(fh)
    if need_header:
        writer.writerow(columns)
        fh.flush()
    return _CSVWriter(fh, writer)


def _truncate_csv_before_resume(path: Path, start_step: int) -> None:
    """续训前删除 checkpoint 之后的脏行，避免同一 step 重复。"""
    if not path.exists() or start_step <= 0:
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            step = int(float(str(row.get("step", ""))))
        except ValueError:
            continue
        if step < start_step:
            kept_rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept_rows)


def _append_csv_row(writer: Any, row: dict[str, Any], columns: list[str]) -> None:
    """按列顺序写入一行 CSV。缺失字段填空字符串。"""
    values = [row.get(col, "") for col in columns]
    writer.writerow(values)  # _CSVWriter.writerow 会自动 flush


# ---------------------------------------------------------------------------
# 显存诊断
# ---------------------------------------------------------------------------

def _format_bytes(b: int) -> str:
    """字节转人类可读大小。"""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    else:
        return f"{b / 1024**3:.2f} GB"


def log_gpu_memory(tag: str, level: int = logging.INFO) -> None:
    """打印 CUDA 内存分配器的概要状态（独立函数，不依赖 trainer 实例）。"""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    pool_free = reserved - allocated

    stats = torch.cuda.memory_stats()
    active_alloc_count = stats.get("allocation.all.current", 0)
    segment_count = stats.get("segment.all.current", 0)

    msg = (
        f"[显存:{tag}] "
        f"allocated={_format_bytes(allocated)} "
        f"reserved={_format_bytes(reserved)} "
        f"预留池={_format_bytes(pool_free)} "
        f"段数={segment_count} "
        f"活跃块数={active_alloc_count}"
    )
    logger.log(level, msg)


def _calc_model_component_bytes(model: Any) -> dict[str, int]:
    """计算模型各组件的参数显存占用，返回逐项拆解。

    区分: 基座权重(frozen)、LoRA可训练权重、embedding层。
    """
    base_bytes = 0  # 基座冻结权重 (fp16)
    lora_bytes = 0  # LoRA 可训练权重 (fp16, 含 A/B 矩阵)
    embed_bytes = 0  # embedding 层权重

    for name, param in model.named_parameters():
        size = param.nelement() * param.element_size()
        if not param.requires_grad:
            base_bytes += size
        else:
            # LoRA 参数名通常含 "lora_A" 或 "lora_B"
            if "lora_" in name.lower():
                lora_bytes += size
            elif "embed" in name.lower():
                embed_bytes += size
            else:
                lora_bytes += size  # 其他可训练参数也归类到可训练部分

    return {"base": base_bytes, "lora": lora_bytes, "embed": embed_bytes,
            "total": base_bytes + lora_bytes + embed_bytes}


def _calc_optimizer_bytes(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """计算优化器状态的显存占用 (momentum + variance, fp32)。

    返回: {"momentum": ..., "variance": ..., "total": ...}
    """
    momentum_bytes = 0
    variance_bytes = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and p in optimizer.state:
                state = optimizer.state[p]
                if "exp_avg" in state:
                    momentum_bytes += state["exp_avg"].nelement() * state["exp_avg"].element_size()
                if "exp_avg_sq" in state:
                    variance_bytes += state["exp_avg_sq"].nelement() * state["exp_avg_sq"].element_size()
    return {"momentum": momentum_bytes, "variance": variance_bytes,
            "total": momentum_bytes + variance_bytes}


def _calc_grad_bytes(model: Any) -> int:
    """计算梯度缓冲的显存占用 (与可训练参数同大小)。"""
    return sum(
        p.grad.nelement() * p.grad.element_size()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )


def _calc_batch_bytes(batch: dict[str, Any]) -> int:
    """计算训练 batch 数据的显存占用 (input_ids, mask, advantages, log_probs 等)。"""
    total = 0
    for name, tensor in batch.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            total += tensor.nelement() * tensor.element_size()
    return total


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """GRPO 训练的超参数。"""

    # 模型
    base_model_dir: str = ""
    sft_adapter_dir: str = ""  # SFT LoRA adapter 路径，作为 GRPO 的起点
    resume_from_checkpoint: str = ""  # 已保存的 GRPO checkpoint，用于断点续训
    resume_state_mode: str = "full"  # full | weights_only | weights_and_optimizer
    allow_base_start: bool = False  # 只有显式允许时才可跳过 SFT
    allow_resume_objective_change: bool = False  # 显式允许从旧 checkpoint 创建新目标分支
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 数据
    train_file: str = ""
    eval_file: str = ""
    max_prompt_length: int = 512
    max_response_length: int = 256
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION
    enable_thinking: bool = False

    # rollout
    rollout_n: int = 8  # 每个 prompt 生成的回答数
    rollout_batch_size: int = 4  # rollout 批量生成时一次处理的 prompt 数
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50

    # 训练
    train_batch_size: int = 4  # RTX 4070 已验证的每步 prompt 数量
    ppo_epochs: int = 2  # 对每批 rollout 数据重复训练的轮数
    ppo_mini_batch_size: int = 16  # mini-batch 大小 (prompt*N 个回答)
    gradient_accumulation_steps: int = 1  # 累积多少个 PPO mini-batch 后更新一次 optimizer
    learning_rate: float = 5e-6
    kl_loss_coef: float = 0.005
    kl_loss_type: str = "low_var_kl"  # "low_var_kl" | "kl"
    clip_ratio: float = 0.2
    norm_adv_by_std: bool = True
    max_grad_norm: float = 1.0
    total_training_steps: int = 500  # 最大训练步数上限（兜底）
    max_steps_no_improve: int = 50  # 验证 EM 连续不改善的最大步数 (early stop)
    early_stop_trend_window: int = 3  # 判断恢复趋势时使用的最近验证点数量
    early_stop_min_recovery_slope: float = 0.005  # 每个验证点至少恢复的 EM 斜率
    early_stop_max_extension_steps: int = 40  # 趋势恢复时最多额外训练的步数
    kl_threshold: float = 0.1  # actor 相对 SFT reference 的 KL 硬阈值
    kl_warning_threshold: float = 0.06  # reference KL 预警阈值
    kl_guard_window: int = 3  # reference KL 滑动窗口
    kl_guard_patience_checks: int = 3  # 连续超限窗口数
    approx_kl_threshold: float = 0.01  # 单次 PPO 更新相对 old policy 的 KL 阈值
    adaptive_kl_enabled: bool = True
    adaptive_kl_target: float = 0.04
    adaptive_kl_interval: int = 10
    adaptive_kl_factor: float = 1.5
    adaptive_kl_tolerance: float = 1.25
    adaptive_kl_min_coef: float = 0.001
    adaptive_kl_max_coef: float = 0.05
    reward_hacking_detect: bool = True  # 是否检测 reward hacking
    reward_hacking_window: int = 30  # reward hacking 检测窗口 (步数)
    save_freq: int = 10
    eval_freq: int = 10
    seed: int = 42
    deterministic_prompt_sampling: bool = False  # 按 seed+step 固定 prompt，避免其他 RNG 消耗改变题目序列
    prompt_sampling_seed: int = -1  # 小于 0 时复用 seed
    fp16: bool = True
    gradient_checkpointing: bool = True

    # 奖励权重
    reward_exact_with_format_score: float = 1.0
    reward_exact_without_format_score: float = 0.1
    reward_format_bonus: float = 0.1
    reward_single_final_bonus: float = 0.05
    reward_missing_format_penalty: float = -0.2
    reward_multi_final_penalty: float = -0.2
    reward_repeat_penalty: float = -0.5
    reward_overlong_penalty: float = -0.2
    reward_overlong_chars: int = 1200
    reward_long_response_token_threshold: int = 192
    reward_long_response_penalty: float = -0.1
    reward_truncated_response_penalty: float = -0.3
    reward_min: float = -1.0
    reward_max: float = 1.1

    # 输出
    output_dir: str = ""
    run_name: str = "grpo_run"
    log_steps: int = 1
    launch_command: str = ""

    # 验证
    val_before_train: bool = True
    val_max_items: int = 20
    val_eval_batch_size: int = 8
    val_stochastic_n: int = 8
    val_stochastic_max_items: int = 10

    # 异常样本留档
    rollout_anomaly_dump_enabled: bool = True
    rollout_anomaly_max_samples: int = 8

    # 组内训练信号保护
    signal_guard_window: int = 10
    signal_guard_warmup_steps: int = 10
    signal_guard_patience_checks: int = 3
    signal_guard_non_overlapping_windows: bool = True  # 只把互不重叠的窗口视为独立证据
    signal_guard_mixed_hard_stop: bool = False  # mixed只预警，实际reward信号由effective判断
    min_effective_group_rate: float = 0.70
    min_mixed_group_rate: float = 0.60
    max_zero_advantage_rate: float = 0.30
    min_rollout_format_rate: float = 0.90


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_eos_token_id(tokenizer: Any) -> int | None:
    """用 <|im_end|> 作为停止标记。"""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _linear_trend_slope(values: list[float]) -> float:
    """计算等间隔验证点的线性趋势斜率。"""
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = _mean(values)
    numerator = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values))
    denominator = sum((idx - x_mean) ** 2 for idx in range(len(values)))
    return numerator / denominator if denominator > 0 else 0.0


def _should_extend_early_stop(
    val_em_history: list[float],
    trend_window: int,
    min_recovery_slope: float,
    extension_steps: int,
    max_extension_steps: int,
) -> tuple[bool, float | None]:
    """判断达到全局最佳耐心后，近期恢复趋势是否足以有限延长训练。"""
    if len(val_em_history) < trend_window:
        return False, None
    recent_values = val_em_history[-trend_window:]
    slope = _linear_trend_slope(recent_values)
    can_extend = slope >= min_recovery_slope and extension_steps < max_extension_steps
    return can_extend, slope


@dataclass(frozen=True)
class EarlyStopEvaluation:
    """一次验证点的早停状态更新和独立停止判定。"""

    best_val_em: float
    best_step: int
    steps_no_improve: int
    extension_steps: int
    improved: bool
    extended: bool
    recovery_slope: float | None
    decision: StopDecision | None


def _evaluate_early_stopping(
    *,
    step: int,
    val_em: float,
    best_val_em: float,
    best_step: int,
    steps_no_improve: int,
    extension_steps: int,
    val_em_history: list[float],
    eval_freq: int,
    max_steps_no_improve: int,
    trend_window: int,
    min_recovery_slope: float,
    max_extension_steps: int,
) -> EarlyStopEvaluation:
    """只负责验证收益早停，不读取训练rollout健康指标。"""
    improved = val_em > best_val_em
    if improved:
        return EarlyStopEvaluation(
            best_val_em=val_em,
            best_step=step,
            steps_no_improve=0,
            extension_steps=0,
            improved=True,
            extended=False,
            recovery_slope=None,
            decision=None,
        )

    updated_no_improve = steps_no_improve + eval_freq
    if updated_no_improve < max_steps_no_improve:
        return EarlyStopEvaluation(
            best_val_em=best_val_em,
            best_step=best_step,
            steps_no_improve=updated_no_improve,
            extension_steps=extension_steps,
            improved=False,
            extended=False,
            recovery_slope=None,
            decision=None,
        )

    should_extend, recovery_slope = _should_extend_early_stop(
        val_em_history=val_em_history,
        trend_window=trend_window,
        min_recovery_slope=min_recovery_slope,
        extension_steps=extension_steps,
        max_extension_steps=max_extension_steps,
    )
    if should_extend:
        return EarlyStopEvaluation(
            best_val_em=best_val_em,
            best_step=best_step,
            steps_no_improve=updated_no_improve,
            extension_steps=extension_steps + eval_freq,
            improved=False,
            extended=True,
            recovery_slope=recovery_slope,
            decision=None,
        )

    slope_text = "数据不足" if recovery_slope is None else f"{recovery_slope:.4f}"
    decision = build_stop_decision(
        StopCategory.EARLY_STOPPING,
        source="validation_early_stopping",
        reason=(
            f"早停: val_em连续{updated_no_improve}步未改善且无可用恢复趋势"
        ),
        step=step,
        details={
            "val_em": val_em,
            "best_val_em": best_val_em,
            "best_step": best_step,
            "steps_no_improve": updated_no_improve,
            "trend_window": trend_window,
            "recovery_slope": recovery_slope,
            "recovery_slope_text": slope_text,
            "min_recovery_slope": min_recovery_slope,
            "extension_steps": extension_steps,
            "max_extension_steps": max_extension_steps,
        },
    )
    return EarlyStopEvaluation(
        best_val_em=best_val_em,
        best_step=best_step,
        steps_no_improve=updated_no_improve,
        extension_steps=extension_steps,
        improved=False,
        extended=False,
        recovery_slope=recovery_slope,
        decision=decision,
    )


def _evaluate_signal_guard_window(
    diagnostics: list[dict[str, Any]],
    min_effective_group_rate: float,
    min_mixed_group_rate: float,
    max_zero_advantage_rate: float,
    min_rollout_format_rate: float,
    mixed_hard_stop: bool,
) -> tuple[dict[str, float], list[str], list[str]]:
    """汇总一个信号窗口，区分硬停止条件与只读预警。"""
    summary = {
        "effective": _mean([float(x.get("effective_group_rate", 0.0)) for x in diagnostics]),
        "mixed": _mean([float(x.get("mixed_group_rate", 0.0)) for x in diagnostics]),
        "zero_adv": _mean([float(x.get("zero_advantage_rate", 1.0)) for x in diagnostics]),
        "format": _mean([float(x.get("rollout_format_rate", 0.0)) for x in diagnostics]),
    }
    failures: list[str] = []
    warnings: list[str] = []
    if summary["effective"] < min_effective_group_rate:
        failures.append(f"effective={summary['effective']:.3f}")
    if summary["mixed"] < min_mixed_group_rate:
        target = failures if mixed_hard_stop else warnings
        target.append(f"mixed={summary['mixed']:.3f}")
    if summary["zero_adv"] > max_zero_advantage_rate:
        failures.append(f"zero_adv={summary['zero_adv']:.3f}")
    if summary["format"] < min_rollout_format_rate:
        failures.append(f"format={summary['format']:.3f}")
    return summary, failures, warnings


def _rolling_threshold_state(
    values: list[float], window: int, threshold: float
) -> tuple[float, int]:
    """返回最新滑动均值以及连续超过阈值的窗口数。"""
    if not values:
        return 0.0, 0
    if len(values) < window:
        return _mean(values), 0
    rolling_means = [
        _mean(values[end - window:end])
        for end in range(window, len(values) + 1)
    ]
    consecutive = 0
    for value in reversed(rolling_means):
        if value <= threshold:
            break
        consecutive += 1
    return rolling_means[-1], consecutive


def _file_sha256(path: str | Path) -> str:
    """计算小型配置或 parquet 文件的 SHA256，固定实验输入。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str | Path, map_location: Any = None) -> Any:
    """兼容不同 PyTorch 版本加载 checkpoint。"""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


def load_actor_and_reference(
    base_model_dir: str | Path,
    sft_adapter_dir: str | Path | None,
    grpo_adapter_dir: str | Path | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    device: torch.device | None = None,
) -> tuple[Any, Any, Any]:
    """加载 GRPO 训练的 actor 和 reference 模型。

    流程：
      1. 加载 base model
      2. 加载 SFT LoRA adapter → 合并为完整模型
      3. 复制一份冻结作为 reference
      4. 新训练时在合并模型上加新 LoRA；续训时加载已有 GRPO LoRA → actor (可训练)

    Returns:
        (actor, reference, tokenizer)
    """
    try:
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model
    except ImportError:
        raise ImportError("GRPO 训练需要安装 peft。")

    base_model_dir = Path(base_model_dir)
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 base model
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    # 3. 加载并合并 SFT LoRA adapter
    if sft_adapter_dir is not None and Path(sft_adapter_dir).exists():
        sft_adapter_dir = Path(sft_adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir))
        print(f"加载 SFT adapter 自: {sft_adapter_dir}")
        merged_model = model.merge_and_unload()
        print("SFT adapter 已合并到 base model。")
    else:
        merged_model = base_model
        print("未找到 SFT adapter，直接从 base model 开始。")

    # 4. 创建冻结的 reference model
    reference_model = copy.deepcopy(merged_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("Reference model 已创建（冻结）。")

    # 5. 在合并模型上创建或恢复 GRPO LoRA → actor
    if device is not None and torch.cuda.is_available():
        merged_model = merged_model.to(device)

    if grpo_adapter_dir is not None and Path(grpo_adapter_dir).exists():
        grpo_adapter_dir = Path(grpo_adapter_dir)
        actor = PeftModel.from_pretrained(merged_model, str(grpo_adapter_dir), is_trainable=True)
        print(f"加载 GRPO adapter 自: {grpo_adapter_dir}")
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        actor = get_peft_model(merged_model, lora_config)
        print("已创建新的 GRPO LoRA adapter。")

    if gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        if hasattr(actor, "enable_input_require_grads"):
            # LoRA + gradient checkpointing 需要让输入 embedding 参与梯度图。
            actor.enable_input_require_grads()
    actor.train()
    actor.print_trainable_parameters()

    return actor, reference_model, tokenizer


# ---------------------------------------------------------------------------
# Log prob 计算
# ---------------------------------------------------------------------------


def compute_sequence_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算序列中 response 部分的 token-level log probs。

    Args:
        model: 模型（actor 或 reference）
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        response_mask: (batch, seq_len), 1=response token, 0=prompt token

    Returns:
        token_log_probs: (batch, seq_len-1)  每个位置对下一个 token 的 log prob
        seq_log_probs: (batch,)  每条序列 response 部分的总 log prob
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab_size)

    # 向右移位：预测位置 t 的输出对应答案位置 t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()  # (batch, seq_len-1)

    # 每 token 的 log prob
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    # token_log_probs: (batch, seq_len-1)

    # 总 log prob (response 部分)
    seq_log_probs = (token_log_probs * shift_mask).sum(dim=-1)

    return token_log_probs, seq_log_probs


def compute_kl_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_type: str = "low_var_kl",
) -> torch.Tensor:
    """计算 KL 散度损失。

    Args:
        log_probs: 当前策略的 token-level log probs, (batch, seq_len-1)
        ref_log_probs: reference 的 token-level log probs, (batch, seq_len-1)
        response_mask: (batch, seq_len-1)
        kl_type: "low_var_kl" (k3 估计) 或 "kl" (标准)

    Returns:
        kl_loss: 标量
    """
    mask = response_mask[:, 1:] if response_mask.shape[1] == log_probs.shape[1] + 1 else response_mask
    # 确保 mask 与 log_probs 形状一致
    if mask.shape != log_probs.shape:
        if mask.shape[1] == log_probs.shape[1] + 1:
            mask = mask[:, 1:]
        elif mask.shape[1] == log_probs.shape[1] - 1:
            mask = F.pad(mask, (0, 1), value=0)

    mask = mask.to(dtype=log_probs.dtype, device=log_probs.device)
    valid_mask = mask > 0
    valid_tokens = mask.sum()
    if valid_tokens == 0:
        return torch.tensor(0.0, device=log_probs.device)

    if kl_type == "low_var_kl":
        # k3 估计: exp(log_ratio) - log_ratio - 1, 方差更低
        log_ratio = ref_log_probs - log_probs
        log_ratio = torch.where(valid_mask, log_ratio, torch.zeros_like(log_ratio))
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        # 标准 KL: log(p/q) = log_p - log_q
        kl = log_probs - ref_log_probs
        kl = torch.where(valid_mask, kl, torch.zeros_like(kl))

    kl = torch.where(valid_mask, kl, torch.zeros_like(kl))
    return kl.sum() / valid_tokens


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """自包含 GRPO 训练器。"""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self._validate_config()
        set_seed(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型
        sft_adapter = cfg.sft_adapter_dir if cfg.sft_adapter_dir else None
        resume_checkpoint = Path(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else None
        if resume_checkpoint is not None and not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_checkpoint}")
        self.actor, self.reference, self.tokenizer = load_actor_and_reference(
            base_model_dir=cfg.base_model_dir,
            sft_adapter_dir=sft_adapter,
            grpo_adapter_dir=resume_checkpoint,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            fp16=cfg.fp16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            device=self.device,
        )

        # 优化器：只优化 actor 的可训练参数
        trainable_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
        self.reward_config = GSM8KRewardConfig(
            exact_with_format_score=cfg.reward_exact_with_format_score,
            exact_without_format_score=cfg.reward_exact_without_format_score,
            format_bonus=cfg.reward_format_bonus,
            single_final_bonus=cfg.reward_single_final_bonus,
            missing_format_penalty=cfg.reward_missing_format_penalty,
            multi_final_penalty=cfg.reward_multi_final_penalty,
            repeat_penalty=cfg.reward_repeat_penalty,
            overlong_penalty=cfg.reward_overlong_penalty,
            overlong_chars=cfg.reward_overlong_chars,
            long_response_token_threshold=cfg.reward_long_response_token_threshold,
            long_response_penalty=cfg.reward_long_response_penalty,
            truncated_response_penalty=cfg.reward_truncated_response_penalty,
            min_reward=cfg.reward_min,
            max_reward=cfg.reward_max,
        )
        self.current_kl_loss_coef = cfg.kl_loss_coef

        # 续训状态：optimizer/trainer/RNG 如果存在会被恢复；旧 checkpoint 缺失时只做权重 warm-start。
        self.resume_checkpoint_dir = resume_checkpoint
        self.has_trainer_state = False
        self.start_step = 0
        self.last_completed_step = -1
        self.best_val_em = -1.0
        self.best_step = -1
        self.steps_no_improve = 0
        self.early_stop_extension_steps = 0
        self.train_reward_history: list[float] = []
        self.training_session_id: str | None = None
        self.training_status = "initialized"
        self.stop_decision: dict[str, Any] | None = None
        self.stop_reason: str | None = None

        # eos token
        self._eos_token_id = _build_eos_token_id(self.tokenizer)
        self._pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # 加载数据
        self.train_dataset = load_rl_dataset(
            parquet_path=cfg.train_file,
            tokenizer=self.tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            format_instruction=cfg.format_instruction,
            enable_thinking=cfg.enable_thinking,
        )
        self.eval_dataset = load_rl_dataset(
            parquet_path=cfg.eval_file,
            tokenizer=self.tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            format_instruction=cfg.format_instruction,
            enable_thinking=cfg.enable_thinking,
        )

        # 日志
        self.metrics_history: list[dict[str, Any]] = []
        self._load_resume_checkpoint_state()

        # 绘图数据: 训练指标和验证指标分开存储，实时写入 CSV
        self._train_csv_path: Path | None = None
        self._val_csv_path: Path | None = None
        self._group_diag_csv_path: Path | None = None
        self._gpu_csv_path: Path | None = None
        self._train_csv_writer: Any = None
        self._val_csv_writer: Any = None
        self._group_diag_csv_writer: Any = None
        self._gpu_csv_writer: Any = None

        # 显存: 模型加载后的初始状态（含各组件拆解）
        log_gpu_memory("模型加载后")
        self._log_gpu_memory_detailed("模型加载后", batch=None)

    def _validate_config(self) -> None:
        """在加载模型前执行正式训练所需的配置预检。"""
        cfg = self.cfg
        required_files = {
            "base model": cfg.base_model_dir,
            "训练数据": cfg.train_file,
            "验证数据": cfg.eval_file,
        }
        for label, raw_path in required_files.items():
            if not raw_path or not Path(raw_path).exists():
                raise FileNotFoundError(f"{label}不存在: {raw_path or '<empty>'}")
        if not cfg.sft_adapter_dir and not cfg.allow_base_start:
            raise ValueError("缺少 SFT adapter；如确需从 base 起训，必须显式传 --allow-base-start。")
        if cfg.sft_adapter_dir and not Path(cfg.sft_adapter_dir).exists():
            raise FileNotFoundError(f"SFT adapter 不存在: {cfg.sft_adapter_dir}")
        positive_int_fields = (
            "rollout_n", "rollout_batch_size", "train_batch_size", "ppo_epochs",
            "ppo_mini_batch_size", "gradient_accumulation_steps",
            "total_training_steps", "save_freq", "eval_freq",
            "max_response_length", "val_eval_batch_size", "signal_guard_window",
            "signal_guard_warmup_steps", "signal_guard_patience_checks",
            "max_steps_no_improve", "early_stop_trend_window",
            "kl_guard_window", "kl_guard_patience_checks", "adaptive_kl_interval",
            "reward_long_response_token_threshold", "val_stochastic_n",
            "val_stochastic_max_items", "rollout_anomaly_max_samples",
        )
        for name in positive_int_fields:
            if int(getattr(cfg, name)) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if cfg.resume_state_mode not in {"full", "weights_only", "weights_and_optimizer"}:
            raise ValueError(
                "resume_state_mode 必须是 full、weights_only 或 weights_and_optimizer"
            )
        if not 0.0 < cfg.temperature:
            raise ValueError("temperature 必须大于 0")
        if not 0.0 < cfg.top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")
        if cfg.early_stop_trend_window < 2:
            raise ValueError("early_stop_trend_window 必须至少为 2")
        if cfg.early_stop_min_recovery_slope < 0:
            raise ValueError("early_stop_min_recovery_slope 不能小于 0")
        if cfg.early_stop_max_extension_steps < 0:
            raise ValueError("early_stop_max_extension_steps 不能小于 0")
        if not 0 < cfg.kl_warning_threshold < cfg.kl_threshold:
            raise ValueError("KL 预警阈值必须位于 (0, KL 硬阈值) 内")
        if cfg.approx_kl_threshold <= 0:
            raise ValueError("approx_kl_threshold 必须大于 0")
        if cfg.adaptive_kl_target <= 0 or cfg.adaptive_kl_factor <= 1 or cfg.adaptive_kl_tolerance <= 1:
            raise ValueError("自适应 KL 目标必须大于 0，调整因子和容忍系数必须大于 1")
        if not 0 < cfg.adaptive_kl_min_coef <= cfg.kl_loss_coef <= cfg.adaptive_kl_max_coef:
            raise ValueError("初始 KL 系数必须位于自适应 KL 系数上下界内")
        if cfg.output_dir and Path(cfg.output_dir).exists() and not cfg.resume_from_checkpoint:
            if any(Path(cfg.output_dir).glob("checkpoint-*")):
                raise FileExistsError(
                    f"输出目录已有 checkpoint，禁止覆盖；请换 run_name 或使用 resume: {cfg.output_dir}"
                )

    # ------------------------------------------------------------------
    # 显存详细拆解
    # ------------------------------------------------------------------

    def _log_gpu_memory_detailed(self, tag: str, batch: dict[str, Any] | None = None) -> None:
        """打印显存的逐项拆解，区分各组件的实际占用。

        拆解维度:
          - actor 权重: 基座(frozen) + LoRA 可训练 + embedding
          - ref 权重: 基座 + LoRA (全 frozen)
          - optimizer 状态: momentum + variance
          - 梯度缓冲: 与可训练参数同大小
          - rollout 数据: batch 中的 tensor
          - 激活值/logits/临时: allocated - 识别项
          - KV cache 估算: generate 阶段的理论值
        """
        if not torch.cuda.is_available():
            return

        total_allocated = torch.cuda.memory_allocated()
        total_reserved = torch.cuda.memory_reserved()

        stats = torch.cuda.memory_stats()
        peak_allocated = stats.get("allocated_bytes.all.peak", 0)
        peak_reserved = stats.get("reserved_bytes.all.peak", 0)
        segment_count = stats.get("segment.all.current", 0)
        segment_size = stats.get("reserved_bytes.all.current", 0)

        # ---- 逐项计算各组件的显存占用 ----
        actor_comp = _calc_model_component_bytes(self.actor)
        ref_comp = _calc_model_component_bytes(self.reference)
        optimizer_comp = _calc_optimizer_bytes(self.optimizer)
        grad_bytes = _calc_grad_bytes(self.actor)
        batch_bytes = _calc_batch_bytes(batch) if batch is not None else 0

        # 已识别组件合计
        identified_bytes = actor_comp["total"] + ref_comp["total"] + optimizer_comp["total"] + grad_bytes + batch_bytes

        # 激活值/logits/ref临时等 = allocated - 识别项
        # 这部分包含: 前向传播中间激活值、logits、KV cache (generate阶段)、
        # reference 的临时计算、attention score 等
        activations_logits_temp = total_allocated - identified_bytes

        # KV cache 估算 (仅在 generate 阶段有, 训练阶段不存在)
        # Qwen3-0.6B: num_kv_heads=8, head_dim=64, num_layers=28
        # 单条序列 KV cache = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size
        # fp16 dtype_size = 2 bytes
        # 对 rollout_n=4, max_prompt=512, max_response=256 → seq_len ≈ 768
        # kv_per_seq = 2 * 28 * 8 * 64 * 768 * 2 = 2 * 28 * 512 * 768 * 2 bytes
        # 但这只是理论值, 实际 KV cache 是否存在取决于当前是否在 generate
        kv_estimated = 0
        try:
            config = self.actor.config if hasattr(self.actor, "config") else None
            if config is not None:
                num_layers = config.num_hidden_layers
                num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
                head_dim = config.hidden_size // config.num_attention_heads
                dtype_size = 2  # fp16
                # 估算当前 seq_len (如果 batch 存在)
                if batch is not None and "input_ids" in batch:
                    seq_len = batch["input_ids"].shape[-1]
                else:
                    seq_len = self.cfg.max_prompt_length + self.cfg.max_response_length
                # 单条序列 KV cache = 2 * layers * kv_heads * head_dim * seq_len * dtype
                kv_per_seq = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size
                kv_estimated = kv_per_seq * self.cfg.rollout_n * self.cfg.train_batch_size
        except Exception:
            pass

        pct = lambda b: f"{b / total_allocated * 100:.1f}" if total_allocated > 0 else "0.0"

        msg_lines = [
            f"[显存详细:{tag}]",
            f"  ═══ 总量 ═══",
            f"  allocated (真实数据): {_format_bytes(total_allocated)} | 峰值: {_format_bytes(peak_allocated)}",
            f"  reserved  (分配器持有): {_format_bytes(total_reserved)} | 峰值: {_format_bytes(peak_reserved)}",
            f"  预留池 (段内空闲): {_format_bytes(total_reserved - total_allocated)}",
            f"  内存段数: {segment_count} | 每段均: {_format_bytes(int(segment_size / max(segment_count, 1)))}",
            f"  VRAM总量(nvidia): {_format_bytes(torch.cuda.get_device_properties(0).total_memory)}",
            f"",
            f"  ═══ 模型权重 ═══",
            f"  actor 基座权重(frozen): {_format_bytes(actor_comp['base'])} ({pct(actor_comp['base'])}%)",
            f"  actor LoRA权重(可训练): {_format_bytes(actor_comp['lora'])} ({pct(actor_comp['lora'])}%)",
            f"  actor embedding权重:   {_format_bytes(actor_comp['embed'])} ({pct(actor_comp['embed'])}%)",
            f"  actor 权重合计:        {_format_bytes(actor_comp['total'])} ({pct(actor_comp['total'])}%)",
            f"  ref   基座权重(frozen): {_format_bytes(ref_comp['base'])} ({pct(ref_comp['base'])}%)",
            f"  ref   LoRA权重:        {_format_bytes(ref_comp['lora'])} ({pct(ref_comp['lora'])}%)",
            f"  ref   权重合计:        {_format_bytes(ref_comp['total'])} ({pct(ref_comp['total'])}%)",
            f"  模型权重总计:          {_format_bytes(actor_comp['total'] + ref_comp['total'])} ({pct(actor_comp['total'] + ref_comp['total'])}%)",
            f"",
            f"  ═══ 优化器 ═══",
            f"  momentum (exp_avg, fp32): {_format_bytes(optimizer_comp['momentum'])} ({pct(optimizer_comp['momentum'])}%)",
            f"  variance (exp_avg_sq, fp32): {_format_bytes(optimizer_comp['variance'])} ({pct(optimizer_comp['variance'])}%)",
            f"  optimizer 合计:          {_format_bytes(optimizer_comp['total'])} ({pct(optimizer_comp['total'])}%)",
            f"",
            f"  ═══ 梯度与数据 ═══",
            f"  梯度缓冲: {_format_bytes(grad_bytes)} ({pct(grad_bytes)}%)",
            f"  rollout/batch 数据: {_format_bytes(batch_bytes)} ({pct(batch_bytes)}%)",
            f"",
            f"  ═══ 激活值与临时 ═══",
            f"  激活值+logits+ref临时: {_format_bytes(activations_logits_temp)} ({pct(activations_logits_temp)}%)",
            f"  (含: 前向传播中间激活值、logits、attention scores等)",
            f"",
            f"  ═══ KV Cache 估算 ═══",
            f"  KV cache 理论值(generate阶段): {_format_bytes(kv_estimated)}",
            f"  (注意: KV cache 仅在 generate 时存在, 训练阶段为 0)",
            f"",
            f"  ═══ 汇总 ═══",
            f"  已识别组件: {_format_bytes(identified_bytes)} ({pct(identified_bytes)}%)",
            f"  激活值+临时: {_format_bytes(activations_logits_temp)} ({pct(activations_logits_temp)}%)",
            f"  合计 = allocated: {_format_bytes(total_allocated)}",
            f"",
            f"  ═══ WDDM 说明 ═══",
            f"  reserved 包含 VRAM段 + Shared段, 可能超过物理VRAM总量",
        ]
        logger.info("\n".join(msg_lines))

    # ------------------------------------------------------------------
    # Rollout 生成
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_responses(
        self,
        prompts: list[RLPrompt],
    ) -> list[dict[str, Any]]:
        """对一组 prompt 各生成 N 个回答。

        Returns:
            rollout_items: 每个元素包含 prompt, response_text, response_ids 等
        """
        rollout_items: list[dict[str, Any]] = []
        self.actor.eval()

        batch_size = max(1, self.cfg.rollout_batch_size)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            try:
                inputs = self.tokenizer(
                    [prompt.prompt_text for prompt in batch_prompts],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.device)
            finally:
                self.tokenizer.padding_side = old_padding_side

            # 一次性生成 rollout_batch_size * rollout_n 条回答，减少逐条 generate 的调度开销。
            torch.manual_seed(self.cfg.seed + len(self.metrics_history) + start)
            output_ids = self.actor.generate(
                **inputs,
                max_new_tokens=self.cfg.max_response_length,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
                num_return_sequences=self.cfg.rollout_n,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

            prompt_width = inputs.input_ids.shape[-1]
            for flat_idx, item_ids in enumerate(output_ids):
                prompt_idx = flat_idx // self.cfg.rollout_n
                gen_index = flat_idx % self.cfg.rollout_n
                prompt = batch_prompts[prompt_idx]
                raw_response_ids = item_ids[prompt_width:].tolist()
                response_ids = self._trim_generated_response_ids(raw_response_ids)
                terminated_by_eos = (
                    self._eos_token_id is not None and self._eos_token_id in response_ids
                )
                reached_max_tokens_without_eos = (
                    len(response_ids) >= self.cfg.max_response_length and not terminated_by_eos
                )
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                rollout_items.append({
                    "prompt": prompt,
                    "response_text": response_text,
                    "response_ids": response_ids,
                    "prompt_len": len(prompt.prompt_ids),
                    "gen_index": gen_index,
                    "response_token_count": len(response_ids),
                    "terminated_by_eos": terminated_by_eos,
                    "reached_max_tokens_without_eos": reached_max_tokens_without_eos,
                })

        self.actor.train()
        return rollout_items

    def _trim_generated_response_ids(self, response_ids: list[int]) -> list[int]:
        """裁掉批量生成后 eos/pad 后面的尾部 token，避免 pad 进入训练样本。"""
        trimmed: list[int] = []
        for token_id in response_ids:
            if self._eos_token_id is not None and token_id == self._eos_token_id:
                trimmed.append(token_id)
                break
            if token_id == self._pad_token_id:
                break
            trimmed.append(token_id)
        return trimmed

    # ------------------------------------------------------------------
    # 奖励计算
    # ------------------------------------------------------------------

    def _compute_rewards(self, rollout_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对每个 rollout item 计算规则奖励。"""
        for item in rollout_items:
            reward_info = compute_gsm8k_rule_reward(
                response=item["response_text"],
                gold_answer=item["prompt"].ground_truth,
                config=self.reward_config,
                response_token_count=item.get("response_token_count"),
                max_response_tokens=self.cfg.max_response_length,
                terminated_by_eos=item.get("terminated_by_eos"),
            )
            item["reward"] = reward_info["score"]
            item["reward_info"] = reward_info
        return rollout_items

    # ------------------------------------------------------------------
    # 优势计算
    # ------------------------------------------------------------------

    def _compute_advantages(self, rollout_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 prompt 分组，计算组内归一化 advantage。"""
        # 按 prompt idx 分组
        groups: dict[int, list[dict[str, Any]]] = {}
        for item in rollout_items:
            idx = item["prompt"].idx
            groups.setdefault(idx, []).append(item)

        all_advantages: list[float] = []

        for group_items in groups.values():
            rewards = [item["reward"] for item in group_items]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)

            for item in group_items:
                if self.cfg.norm_adv_by_std and std_r > 1e-8:
                    adv = (item["reward"] - mean_r) / std_r
                else:
                    adv = item["reward"] - mean_r
                item["advantage"] = adv
                item["group_mean_reward"] = mean_r
                item["group_std_reward"] = std_r
                all_advantages.append(adv)

        return rollout_items

    def _compute_group_diagnostics(self, rollout_items: list[dict[str, Any]]) -> dict[str, Any]:
        """统计 GRPO 组内 rollout 是否提供了有效学习信号。

        重点看同一 prompt 的多条回答是否有 reward 差异、是否同时包含正确和错误轨迹。
        如果组内 reward 全相同，advantage 接近 0，这组样本基本无法推动策略更新。
        """
        groups: dict[int, list[dict[str, Any]]] = {}
        for item in rollout_items:
            groups.setdefault(item["prompt"].idx, []).append(item)

        group_count = len(groups)
        rollout_count = len(rollout_items)
        if group_count == 0 or rollout_count == 0:
            return {
                "group_count": 0,
                "rollout_count": 0,
                "rollout_n": self.cfg.rollout_n,
            }

        effective_groups = 0
        mixed_groups = 0
        all_wrong_groups = 0
        all_correct_groups = 0
        duplicate_groups = 0
        correct_hist: dict[int, int] = {}
        group_reward_stds: list[float] = []
        nonzero_group_reward_stds: list[float] = []
        unique_response_counts: list[int] = []

        exact_values: list[float] = []
        format_values: list[float] = []
        fallback_exact_values: list[float] = []
        response_empty_values: list[float] = []
        eos_values: list[float] = []
        max_tokens_reached_without_eos_values: list[float] = []
        response_token_counts: list[float] = []
        advantages: list[float] = []

        component_sums = {
            "answer": [],
            "format": [],
            "single_final": [],
            "repeat": [],
            "overlong": [],
            "length": [],
            "truncated": [],
            "raw": [],
        }

        for group_items in groups.values():
            rewards = [float(item["reward"]) for item in group_items]
            reward_std = float(np.std(rewards))
            group_reward_stds.append(reward_std)
            if reward_std > 1e-8:
                effective_groups += 1
                nonzero_group_reward_stds.append(reward_std)

            correct_count = 0
            unique_responses = set()
            for item in group_items:
                info = item.get("reward_info", {})
                exact = bool(info.get("exact_match", False))
                fallback_exact = bool(info.get("fallback_exact_match", False))
                format_ok = bool(info.get("format_ok", False))
                correct_count += int(exact)
                exact_values.append(1.0 if exact else 0.0)
                fallback_exact_values.append(1.0 if fallback_exact else 0.0)
                format_values.append(1.0 if format_ok else 0.0)
                response_empty_values.append(1.0 if len(item.get("response_ids", [])) == 0 else 0.0)
                eos_values.append(1.0 if item.get("terminated_by_eos", False) else 0.0)
                max_tokens_reached_without_eos_values.append(
                    1.0 if item.get("reached_max_tokens_without_eos", False) else 0.0
                )
                response_token_counts.append(float(item.get("response_token_count", 0)))
                advantages.append(float(item.get("advantage", 0.0)))
                unique_responses.add(str(item.get("response_text", "")).strip())

                components = info.get("components", {}) or {}
                component_sums["answer"].append(float(components.get("answer", 0.0)))
                component_sums["format"].append(float(components.get("format", 0.0)))
                component_sums["single_final"].append(float(components.get("single_final", 0.0)))
                component_sums["repeat"].append(float(components.get("repeat", 0.0)))
                component_sums["overlong"].append(float(components.get("overlong", 0.0)))
                component_sums["length"].append(float(components.get("length", 0.0)))
                component_sums["truncated"].append(float(components.get("truncated", 0.0)))
                component_sums["raw"].append(float(info.get("raw_score", item.get("reward", 0.0))))

            group_size = len(group_items)
            correct_hist[correct_count] = correct_hist.get(correct_count, 0) + 1
            if correct_count == 0:
                all_wrong_groups += 1
            elif correct_count == group_size:
                all_correct_groups += 1
            else:
                mixed_groups += 1

            unique_count = len(unique_responses)
            unique_response_counts.append(unique_count)
            if unique_count < group_size:
                duplicate_groups += 1

        advantage_abs = [abs(x) for x in advantages]
        zero_advantages = [1.0 if abs(x) <= 1e-8 else 0.0 for x in advantages]

        return {
            "group_count": group_count,
            "rollout_count": rollout_count,
            "rollout_n": self.cfg.rollout_n,
            "effective_group_rate": effective_groups / group_count,
            "zero_signal_group_rate": 1.0 - effective_groups / group_count,
            "mixed_group_rate": mixed_groups / group_count,
            "all_wrong_group_rate": all_wrong_groups / group_count,
            "all_correct_group_rate": all_correct_groups / group_count,
            "rollout_exact_rate": _mean(exact_values),
            "rollout_format_rate": _mean(format_values),
            "rollout_eos_rate": _mean(eos_values),
            "rollout_max_tokens_reached_without_eos_rate": _mean(
                max_tokens_reached_without_eos_values
            ),
            "rollout_response_len_mean": _mean(response_token_counts),
            "fallback_exact_rate": _mean(fallback_exact_values),
            "group_reward_std_mean": _mean(group_reward_stds),
            "group_reward_std_nonzero_mean": _mean(nonzero_group_reward_stds),
            "group_reward_std_max": max(group_reward_stds) if group_reward_stds else 0.0,
            "advantage_mean": _mean(advantages),
            "advantage_std": float(np.std(advantages)) if advantages else 0.0,
            "advantage_abs_mean": _mean(advantage_abs),
            "zero_advantage_rate": _mean(zero_advantages),
            "unique_response_mean": _mean(unique_response_counts),
            "duplicate_group_rate": duplicate_groups / group_count,
            "response_empty_rate": _mean(response_empty_values),
            "reward_answer_mean": _mean(component_sums["answer"]),
            "reward_format_mean": _mean(component_sums["format"]),
            "reward_single_final_mean": _mean(component_sums["single_final"]),
            "reward_repeat_mean": _mean(component_sums["repeat"]),
            "reward_overlong_mean": _mean(component_sums["overlong"]),
            "reward_length_mean": _mean(component_sums["length"]),
            "reward_truncated_mean": _mean(component_sums["truncated"]),
            "reward_raw_mean": _mean(component_sums["raw"]),
            "correct_count_hist_json": json.dumps(correct_hist, ensure_ascii=False, sort_keys=True),
        }

    def _dump_rollout_anomalies(
        self,
        step: int,
        rollout_items: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> None:
        """格式或截断异常时保存少量原始 rollout，补齐可复盘证据。"""
        if not self.cfg.rollout_anomaly_dump_enabled:
            return
        format_rate = float(diagnostics.get("rollout_format_rate", 1.0))
        max_tokens_reached_without_eos_rate = float(
            diagnostics.get("rollout_max_tokens_reached_without_eos_rate", 0.0)
        )
        if (
            format_rate >= self.cfg.min_rollout_format_rate
            and max_tokens_reached_without_eos_rate < 0.1
        ):
            return
        bad_items = [
            item for item in rollout_items
            if not item.get("reward_info", {}).get("format_ok", False)
            or item.get("reached_max_tokens_without_eos", False)
        ]
        bad_items.sort(
            key=lambda item: (
                not bool(item.get("reached_max_tokens_without_eos", False)),
                -int(item.get("response_token_count", 0)),
            )
        )
        output_dir = Path(self.cfg.output_dir) if self.cfg.output_dir else Path("models/grpo") / self.cfg.run_name
        path = output_dir / "diagnostics" / "rollout_anomalies.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for item in bad_items[:self.cfg.rollout_anomaly_max_samples]:
                prompt = item["prompt"]
                info = item.get("reward_info", {})
                payload = {
                    "step": step,
                    "prompt_idx": prompt.idx,
                    "source_index": prompt.source_index,
                    "source_bucket": prompt.source_bucket,
                    "gen_index": item.get("gen_index"),
                    "response_token_count": item.get("response_token_count"),
                    "terminated_by_eos": item.get("terminated_by_eos"),
                    "reached_max_tokens_without_eos": item.get(
                        "reached_max_tokens_without_eos"
                    ),
                    "format_ok": info.get("format_ok"),
                    "exact_match": info.get("exact_match"),
                    "reward": item.get("reward"),
                    "question": prompt.question,
                    "response": item.get("response_text", ""),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.warning(
            f"[rollout 异常留档] step={step} format={format_rate:.3f} "
            "max_tokens_reached_without_eos="
            f"{max_tokens_reached_without_eos_rate:.3f} path={path}"
        )

    # ------------------------------------------------------------------
    # 构建训练 batch
    # ------------------------------------------------------------------

    def _build_training_batch(
        self, rollout_items: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """将 rollout items 构建为可用于训练的 padded batch。

        Returns:
            dict with: input_ids, attention_mask, response_mask, advantages, old_log_probs
        """
        sequences: list[list[int]] = []
        response_masks: list[list[int]] = []
        advantages: list[float] = []

        for item in rollout_items:
            prompt_ids = item["prompt"].prompt_ids
            response_ids = item["response_ids"]
            full_ids = prompt_ids + response_ids

            # response mask: prompt 部分=0, response 部分=1
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            sequences.append(full_ids)
            response_masks.append(mask)
            advantages.append(item["advantage"])

        # Padding
        max_len = max(len(seq) for seq in sequences)
        pad_id = self._pad_token_id

        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        resp_mask = torch.zeros((len(sequences), max_len), dtype=torch.float)

        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :len(seq)] = 1
            resp_mask[i, :len(seq)] = torch.tensor(response_masks[i], dtype=torch.float)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "response_mask": resp_mask.to(self.device),
            "advantages": torch.tensor(advantages, dtype=torch.float, device=self.device),
        }

    @torch.no_grad()
    def _compute_token_log_probs_in_chunks(
        self,
        model: Any,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """分块计算冻结 log-prob，避免一次对全部 rollout 生成巨大 logits。"""
        total_items = batch["input_ids"].shape[0]
        chunk_size = min(self.cfg.ppo_mini_batch_size, total_items)
        chunks: list[torch.Tensor] = []
        had_gradient_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
        if had_gradient_checkpointing:
            model.gradient_checkpointing_disable()
        try:
            for start in range(0, total_items, chunk_size):
                end = min(start + chunk_size, total_items)
                token_log_probs, _ = compute_sequence_log_probs(
                    model,
                    batch["input_ids"][start:end],
                    batch["attention_mask"][start:end],
                    batch["response_mask"][start:end],
                )
                chunks.append(token_log_probs.detach())
        finally:
            if had_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # 训练步骤
    # ------------------------------------------------------------------

    def _sample_prompt_indices(self, step: int, n_prompts: int) -> list[int]:
        """采样本步 prompt；因果实验可让题目序列不受其他 RNG 消耗影响。"""
        population = list(range(len(self.train_dataset)))
        if not self.cfg.deterministic_prompt_sampling:
            return random.sample(population, n_prompts)

        base_seed = (
            self.cfg.prompt_sampling_seed
            if self.cfg.prompt_sampling_seed >= 0
            else self.cfg.seed
        )
        digest = hashlib.sha256(f"{base_seed}:{step}".encode("utf-8")).digest()
        step_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        step_rng = random.Random(step_seed)
        step_rng.shuffle(population)
        return population[:n_prompts]

    def train_step(self, step: int) -> dict[str, float]:
        """执行一步完整的 GRPO 训练。"""
        start_time = time.time()
        log_gpu_memory(f"step{step}_开始")

        # 1. 采样 prompt batch
        n_prompts = min(self.cfg.train_batch_size, len(self.train_dataset))
        indices = self._sample_prompt_indices(step, n_prompts)
        prompts = [self.train_dataset[i] for i in indices]

        # 2. Rollout: 每个 prompt 生成 N 个回答
        rollout_items = self._generate_responses(prompts)
        log_gpu_memory(f"step{step}_rollout后")

        # 3. 计算奖励
        rollout_items = self._compute_rewards(rollout_items)

        # 4. 计算 group-relative advantage
        rollout_items = self._compute_advantages(rollout_items)
        group_diagnostics = self._compute_group_diagnostics(rollout_items)
        group_diagnostics["sample_indices_json"] = json.dumps(indices, ensure_ascii=False)
        group_diagnostics["sample_buckets_json"] = json.dumps(
            [prompt.source_bucket for prompt in prompts], ensure_ascii=False
        )
        self._dump_rollout_anomalies(step, rollout_items, group_diagnostics)

        # 5. 构建训练 batch，并各计算一次 old/reference token log-prob
        batch = self._build_training_batch(rollout_items)
        log_gpu_memory(f"step{step}_batch构建后")
        batch["old_token_log_probs"] = self._compute_token_log_probs_in_chunks(self.actor, batch)
        batch["ref_token_log_probs"] = self._compute_token_log_probs_in_chunks(self.reference, batch)
        log_gpu_memory(f"step{step}_old+ref分块计算后")

        # 7. PPO epoch(s)
        total_items = len(rollout_items)
        mini_batch_size = min(self.cfg.ppo_mini_batch_size, total_items)
        metrics: dict[str, list[float]] = {
            "policy_loss": [], "kl_loss": [], "total_loss": [],
            "approx_kl": [], "clip_frac": [], "grad_norm": [],
        }

        optimizer_update_count = 0
        mini_batch_count = 0
        for _ in range(self.cfg.ppo_epochs):
            # 随机打乱
            perm = torch.randperm(total_items).tolist()
            epoch_mini_batches = [
                perm[start:start + mini_batch_size]
                for start in range(0, total_items, mini_batch_size)
            ]
            accumulation_steps = self.cfg.gradient_accumulation_steps
            for group_start in range(0, len(epoch_mini_batches), accumulation_steps):
                accumulation_group = epoch_mini_batches[
                    group_start:group_start + accumulation_steps
                ]
                self.optimizer.zero_grad(set_to_none=True)
                group_metrics: list[dict[str, float]] = []
                group_ok = True
                loss_scale = 1.0 / len(accumulation_group)

                for mb_indices in accumulation_group:
                    mb_metrics = self._train_mini_batch(
                        batch, mb_indices, loss_scale=loss_scale
                    )
                    mini_batch_count += 1
                    group_metrics.append(mb_metrics)
                    if mb_metrics.get("backward_ok", 0.0) < 0.5:
                        group_ok = False
                        break

                if group_ok:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.actor.parameters() if p.requires_grad],
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    optimizer_update_count += 1
                    metrics["grad_norm"].append(float(grad_norm.item()))
                    log_gpu_memory("mini_batch_optimizer后")
                else:
                    # 组内任一 mini-batch 非有限时，丢弃整组累积梯度。
                    self.optimizer.zero_grad(set_to_none=True)

                for mb_metrics in group_metrics:
                    for key in ("policy_loss", "kl_loss", "total_loss", "approx_kl", "clip_frac"):
                        if key in mb_metrics:
                            metrics[key].append(mb_metrics[key])

        log_gpu_memory(f"step{step}_PPO更新后")

        # 8. 汇总指标
        avg_metrics = {k: _mean(v) for k, v in metrics.items() if v}
        for key in metrics:
            avg_metrics.setdefault(key, 0.0)
        avg_metrics["reward_mean"] = _mean([item["reward"] for item in rollout_items])
        avg_metrics["reward_std"] = float(np.std([item["reward"] for item in rollout_items]))
        avg_metrics["response_len_mean"] = _mean([len(item["response_ids"]) for item in rollout_items])
        avg_metrics["step_time"] = time.time() - start_time
        avg_metrics["step"] = step
        avg_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        avg_metrics["prompt_count"] = n_prompts
        avg_metrics["rollout_count"] = total_items
        avg_metrics["mini_batch_count"] = mini_batch_count
        avg_metrics["optimizer_update_count"] = optimizer_update_count
        # 记录本步真正参与 loss 计算的系数；自适应调整从下一步起生效。
        avg_metrics["kl_loss_coef"] = self.current_kl_loss_coef
        avg_metrics["group_diagnostics"] = group_diagnostics

        self.metrics_history.append(avg_metrics)
        return avg_metrics

    def _train_mini_batch(
        self,
        batch: dict[str, torch.Tensor],
        indices: list[int],
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        """在一个 mini-batch 上前向和反向；optimizer 更新由外层累积组统一执行。"""
        # 取出 mini-batch 数据
        mb_input_ids = batch["input_ids"][indices]
        mb_attention_mask = batch["attention_mask"][indices]
        mb_response_mask = batch["response_mask"][indices]
        mb_advantages = batch["advantages"][indices]
        mb_old_token_log_probs = batch["old_token_log_probs"][indices]
        mb_ref_token_log_probs = batch["ref_token_log_probs"][indices]

        # 当前策略的 log probs
        curr_token_log_probs, _ = compute_sequence_log_probs(
            self.actor, mb_input_ids, mb_attention_mask, mb_response_mask
        )
        log_gpu_memory("mini_batch_actor前向后")

        # 只在 response token 上计算损失
        resp_mask_shifted = mb_response_mask[:, 1:].to(dtype=curr_token_log_probs.dtype)
        valid_tokens = resp_mask_shifted.sum()
        if valid_tokens == 0:
            return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0, "backward_ok": 0.0}

        # PPO ratio 只在有效 response token 上计算。
        # pad/prompt 位置如果先参与 exp/log，可能产生 nan；后续再乘 0 也无法消除。
        valid_mask = resp_mask_shifted > 0
        raw_log_ratio = curr_token_log_probs - mb_old_token_log_probs
        valid_log_ratio = raw_log_ratio[valid_mask]
        if not torch.isfinite(valid_log_ratio).all():
            logger.warning("[非有限 log_ratio] 跳过当前 mini-batch，避免污染参数。")
            return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0, "backward_ok": 0.0}
        log_ratio = torch.where(valid_mask, raw_log_ratio, torch.zeros_like(raw_log_ratio))
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)  # (mb, seq-1)

        # Clipped surrogate
        adv_expanded = mb_advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_expanded
        policy_loss_per_token = -torch.min(surr1, surr2)
        policy_loss = (policy_loss_per_token * resp_mask_shifted).sum() / valid_tokens

        # KL loss
        kl_loss = compute_kl_loss(
            curr_token_log_probs, mb_ref_token_log_probs,
            mb_response_mask, self.cfg.kl_loss_type,
        )

        # 总损失
        total_loss = policy_loss + self.current_kl_loss_coef * kl_loss
        if not torch.isfinite(total_loss):
            logger.warning("[非有限 loss] 跳过当前 mini-batch，避免污染参数。")
            return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0, "backward_ok": 0.0}

        # 后向传播
        (total_loss * loss_scale).backward()
        log_gpu_memory("mini_batch_backward后")

        # 诊断指标
        with torch.no_grad():
            approx_kl_tensor = ((ratio - 1.0) - log_ratio) * resp_mask_shifted
            approx_kl = (approx_kl_tensor.sum() / valid_tokens).item()
            clipped = ((ratio < 1.0 - self.cfg.clip_ratio) | (ratio > 1.0 + self.cfg.clip_ratio)).float()
            clip_frac = ((clipped * resp_mask_shifted).sum() / valid_tokens).item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "backward_ok": 1.0,
        }

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """在 eval 集上计算规则奖励和格式指标。"""
        eval_items = min(self.cfg.val_max_items, len(self.eval_dataset))
        if eval_items == 0:
            return {}

        self.actor.eval()
        prompts = [self.eval_dataset[i] for i in range(eval_items)]

        # 每个 prompt 生成 1 个回答（确定性）
        rewards: list[float] = []
        exact_matches: list[float] = []
        format_oks: list[float] = []
        response_lens: list[int] = []
        max_tokens_reached_without_eos_values: list[float] = []
        eos_values: list[float] = []

        batch_size = max(1, self.cfg.val_eval_batch_size)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            try:
                inputs = self.tokenizer(
                    [prompt.prompt_text for prompt in batch_prompts],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.device)
            finally:
                self.tokenizer.padding_side = old_padding_side

            output_ids = self.actor.generate(
                **inputs,
                max_new_tokens=self.cfg.max_response_length,
                do_sample=False,  # 验证用贪心解码
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

            prompt_width = inputs.input_ids.shape[-1]
            for prompt, item_ids in zip(batch_prompts, output_ids):
                response_ids = self._trim_generated_response_ids(item_ids[prompt_width:].tolist())
                terminated_by_eos = (
                    self._eos_token_id is not None and self._eos_token_id in response_ids
                )
                reached_max_tokens_without_eos = (
                    len(response_ids) >= self.cfg.max_response_length and not terminated_by_eos
                )
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                reward_info = compute_gsm8k_rule_reward(
                    response=response_text,
                    gold_answer=prompt.ground_truth,
                    config=self.reward_config,
                    response_token_count=len(response_ids),
                    max_response_tokens=self.cfg.max_response_length,
                    terminated_by_eos=terminated_by_eos,
                )
                rewards.append(reward_info["score"])
                exact_matches.append(1.0 if reward_info["exact_match"] else 0.0)
                format_oks.append(1.0 if reward_info["format_ok"] else 0.0)
                response_lens.append(len(response_ids))
                max_tokens_reached_without_eos_values.append(
                    1.0 if reached_max_tokens_without_eos else 0.0
                )
                eos_values.append(1.0 if terminated_by_eos else 0.0)

        metrics = {
            "val_reward_mean": _mean(rewards),
            "val_exact_match": _mean(exact_matches),
            "val_format_rate": _mean(format_oks),
            "val_response_len_mean": _mean(response_lens),
            "val_max_tokens_reached_without_eos_rate": _mean(
                max_tokens_reached_without_eos_values
            ),
            "val_eos_rate": _mean(eos_values),
        }
        metrics.update(self._validate_stochastic_format())
        self.actor.train()
        return metrics

    @torch.no_grad()
    def _validate_stochastic_format(self) -> dict[str, float]:
        """用固定随机种子的 sample@n 验证格式鲁棒性，并恢复训练 RNG。"""
        count = min(self.cfg.val_stochastic_max_items, len(self.eval_dataset))
        rollout_n = self.cfg.val_stochastic_n
        if count == 0 or rollout_n <= 0:
            return {}
        prompts = [self.eval_dataset[i] for i in range(count)]
        exact_matches: list[float] = []
        format_oks: list[float] = []
        response_lens: list[int] = []
        max_tokens_reached_without_eos_values: list[float] = []
        eos_values: list[float] = []
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            batch_size = max(1, self.cfg.val_eval_batch_size)
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start:start + batch_size]
                old_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                try:
                    inputs = self.tokenizer(
                        [prompt.prompt_text for prompt in batch_prompts],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to(self.device)
                finally:
                    self.tokenizer.padding_side = old_padding_side
                torch.manual_seed(self.cfg.seed + 100_000 + start)
                output_ids = self.actor.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_response_length,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    num_return_sequences=rollout_n,
                    pad_token_id=self._pad_token_id,
                    eos_token_id=self._eos_token_id,
                )
                prompt_width = inputs.input_ids.shape[-1]
                for flat_idx, item_ids in enumerate(output_ids):
                    prompt = batch_prompts[flat_idx // rollout_n]
                    response_ids = self._trim_generated_response_ids(item_ids[prompt_width:].tolist())
                    terminated_by_eos = (
                        self._eos_token_id is not None and self._eos_token_id in response_ids
                    )
                    reached_max_tokens_without_eos = (
                        len(response_ids) >= self.cfg.max_response_length and not terminated_by_eos
                    )
                    response_text = self.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    ).strip()
                    info = compute_gsm8k_rule_reward(
                        response=response_text,
                        gold_answer=prompt.ground_truth,
                        config=self.reward_config,
                        response_token_count=len(response_ids),
                        max_response_tokens=self.cfg.max_response_length,
                        terminated_by_eos=terminated_by_eos,
                    )
                    exact_matches.append(1.0 if info["exact_match"] else 0.0)
                    format_oks.append(1.0 if info["format_ok"] else 0.0)
                    response_lens.append(len(response_ids))
                    max_tokens_reached_without_eos_values.append(
                        1.0 if reached_max_tokens_without_eos else 0.0
                    )
                    eos_values.append(1.0 if terminated_by_eos else 0.0)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        return {
            "val_sample_exact_match": _mean(exact_matches),
            "val_sample_format_rate": _mean(format_oks),
            "val_sample_response_len_mean": _mean(response_lens),
            "val_sample_max_tokens_reached_without_eos_rate": _mean(
                max_tokens_reached_without_eos_values
            ),
            "val_sample_eos_rate": _mean(eos_values),
        }

    # ------------------------------------------------------------------
    # 断点续训状态
    # ------------------------------------------------------------------

    def _infer_step_from_checkpoint_name(self) -> int:
        """从 checkpoint-29 这类目录名推断下一步。"""
        if self.resume_checkpoint_dir is None:
            return 0
        name = self.resume_checkpoint_dir.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.rsplit("-", 1)[-1]) + 1
            except ValueError:
                return 0
        return 0

    def _capture_rng_state(self) -> dict[str, Any]:
        """保存随机数状态，减少 resume 后采样轨迹漂移。"""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, ckpt_dir: Path) -> None:
        """恢复随机数状态；旧 checkpoint 没有该文件时跳过。"""
        rng_path = ckpt_dir / "rng_state.pt"
        if not rng_path.exists():
            print(f"未找到 RNG 状态: {rng_path}，将从当前 seed 继续。")
            return
        state = _torch_load(rng_path, map_location="cpu")
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
        print(f"RNG 状态已恢复: {rng_path}")

    def _load_resume_checkpoint_state(self) -> None:
        """按显式模式加载 checkpoint 状态，避免权重分支误继承旧优化器。"""
        if self.resume_checkpoint_dir is None:
            return

        ckpt_dir = self.resume_checkpoint_dir
        mode = self.cfg.resume_state_mode
        state_path = ckpt_dir / "trainer_state.json"
        state: dict[str, Any] = {}
        objective_changes: list[str] = []
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            saved_config = state.get("config", {})
            identity_fields = (
                "base_model_dir", "sft_adapter_dir", "train_file", "eval_file",
                "rollout_n", "max_response_length",
            )
            objective_fields = (
                "kl_loss_coef", "kl_loss_type", "kl_threshold", "kl_warning_threshold",
                "kl_guard_window", "kl_guard_patience_checks", "approx_kl_threshold",
                "adaptive_kl_enabled", "adaptive_kl_target", "adaptive_kl_interval",
                "adaptive_kl_factor", "adaptive_kl_tolerance", "adaptive_kl_min_coef",
                "adaptive_kl_max_coef", "reward_exact_with_format_score",
                "reward_exact_without_format_score", "reward_format_bonus",
                "reward_single_final_bonus", "reward_missing_format_penalty",
                "reward_multi_final_penalty", "reward_repeat_penalty",
                "reward_overlong_penalty", "reward_overlong_chars",
                "reward_long_response_token_threshold", "reward_long_response_penalty",
                "reward_truncated_response_penalty", "reward_min", "reward_max",
            )
            for key in identity_fields:
                if key in saved_config and saved_config[key] != getattr(self.cfg, key):
                    raise ValueError(
                        f"resume 配置不兼容: {key} 当前={getattr(self.cfg, key)!r} "
                        f"checkpoint={saved_config[key]!r}"
                    )
            objective_changes = [
                key for key in objective_fields
                if key in saved_config and saved_config[key] != getattr(self.cfg, key)
            ]
            if objective_changes and not self.cfg.allow_resume_objective_change:
                details = ", ".join(
                    f"{key}: {saved_config[key]!r}->{getattr(self.cfg, key)!r}"
                    for key in objective_changes
                )
                raise ValueError(
                    "resume 会改变训练目标，默认禁止。请从旧 checkpoint 创建新的输出分支，"
                    "并显式设置 --allow-resume-objective-change。变更: " + details
                )
            if objective_changes:
                print(
                    "已显式允许 resume 训练目标变化；本次属于修正版实验分支，不是原实验的严格续训。"
                )

            if mode == "full":
                stateful_fields = (
                    "train_batch_size", "ppo_epochs", "ppo_mini_batch_size",
                    "gradient_accumulation_steps", "learning_rate", "seed",
                    "deterministic_prompt_sampling", "prompt_sampling_seed",
                )
                stateful_changes = [
                    key for key in stateful_fields
                    if key in saved_config and saved_config[key] != getattr(self.cfg, key)
                ]
                if stateful_changes:
                    details = ", ".join(
                        f"{key}: {saved_config[key]!r}->{getattr(self.cfg, key)!r}"
                        for key in stateful_changes
                    )
                    raise ValueError(
                        "full resume 必须保持优化与采样状态兼容；如需做新变量实验，"
                        "请使用 weights_only 新建分支。变更: " + details
                    )

        optimizer_path = ckpt_dir / "optimizer.pt"
        if mode in {"full", "weights_and_optimizer"}:
            if optimizer_path.exists():
                optimizer_state = _torch_load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(optimizer_state)
                print(f"Optimizer 状态已恢复({mode}): {optimizer_path}")
            elif mode == "weights_and_optimizer":
                raise FileNotFoundError(
                    f"weights_and_optimizer 模式要求 checkpoint 含 optimizer.pt: {optimizer_path}"
                )
            else:
                print(f"未找到 optimizer 状态: {optimizer_path}，本次只能从模型权重 warm-start。")
        else:
            print("仅加载 checkpoint LoRA 权重；optimizer、trainer state 与 RNG 均已重置。")

        if mode == "full":
            self.start_step = self._infer_step_from_checkpoint_name()
            self.last_completed_step = self.start_step - 1

        if mode == "full" and state:
            self.has_trainer_state = True
            self.last_completed_step = int(state.get("step", self.last_completed_step))
            self.start_step = int(state.get("next_step", self.last_completed_step + 1))
            self.best_val_em = float(state.get("best_val_em", self.best_val_em))
            self.best_step = int(state.get("best_step", self.best_step))
            self.steps_no_improve = int(state.get("steps_no_improve", self.steps_no_improve))
            self.early_stop_extension_steps = int(
                state.get("early_stop_extension_steps", self.early_stop_extension_steps)
            )
            if objective_changes:
                self.current_kl_loss_coef = self.cfg.kl_loss_coef
            else:
                self.current_kl_loss_coef = float(
                    state.get("current_kl_loss_coef", self.current_kl_loss_coef)
                )
            self.train_reward_history = [float(x) for x in state.get("train_reward_history", [])]
            self.metrics_history = list(state.get("metrics_history", []))
            print(f"Trainer 状态已恢复: {state_path}，下一步从 step {self.start_step} 开始。")
        elif mode == "full":
            print(f"未找到 trainer 状态: {state_path}，将从 step {self.start_step} 近似续跑。")
        elif mode == "weights_and_optimizer":
            print("已加载 LoRA + 旧 optimizer；训练计数、最佳指标和 RNG 从 step 0 重置。")

        if mode == "full":
            self._restore_rng_state(ckpt_dir)

    def _sync_training_state(
        self,
        step: int,
        best_val_em: float,
        best_step: int,
        steps_no_improve: int,
        early_stop_extension_steps: int,
        train_reward_history: list[float],
    ) -> None:
        """同步内存中的训练状态，供 checkpoint 保存使用。"""
        self.last_completed_step = step
        self.best_val_em = best_val_em
        self.best_step = best_step
        self.steps_no_improve = steps_no_improve
        self.early_stop_extension_steps = early_stop_extension_steps
        self.train_reward_history = list(train_reward_history)

    def _save_run_config(self, output_dir: Path) -> None:
        """保存完整配置、命令和数据指纹，保证实验可复现。"""
        path = output_dir / "run_config.json"
        launch_history: list[str] = []
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                launch_history = list(previous.get("launch_history", []))
            except (OSError, json.JSONDecodeError):
                launch_history = []
        if self.cfg.launch_command:
            launch_history.append(self.cfg.launch_command)
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "train_file_sha256": _file_sha256(self.cfg.train_file),
            "eval_file_sha256": _file_sha256(self.cfg.eval_file),
            "launch_history": launch_history,
            "config": asdict(self.cfg),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"完整运行配置已保存: {path}")

    def _check_kl_guard(
        self, step: int, step_metrics: dict[str, float]
    ) -> StopDecision | None:
        """独立判断累计reference KL和单次PPO update KL是否需要停止。"""
        cfg = self.cfg
        update_kl = abs(float(step_metrics.get("approx_kl", 0.0)))
        if update_kl > cfg.approx_kl_threshold:
            return build_stop_decision(
                StopCategory.KL_GUARD,
                source="ppo_update_kl_guard",
                reason=(
                    f"单步 PPO update KL={update_kl:.4e} "
                    f"超过阈值 {cfg.approx_kl_threshold:.4e}"
                ),
                step=step,
                details={
                    "update_kl": update_kl,
                    "threshold": cfg.approx_kl_threshold,
                },
            )
        reference_values = [
            float(item["kl_loss"])
            for item in self.metrics_history
            if isinstance(item, dict) and "kl_loss" in item
        ]
        rolling_kl, consecutive = _rolling_threshold_state(
            reference_values, cfg.kl_guard_window, cfg.kl_threshold
        )
        logger.info(
            f"[KL guard step {step}] reference_kl={rolling_kl:.4f} "
            f"warn={cfg.kl_warning_threshold:.4f} hard={cfg.kl_threshold:.4f} "
            f"failures={consecutive}/{cfg.kl_guard_patience_checks} "
            f"update_kl={update_kl:.4e}/{cfg.approx_kl_threshold:.4e} "
            f"coef={self.current_kl_loss_coef:.5f}"
        )
        if rolling_kl > cfg.kl_warning_threshold:
            logger.warning(
                f"[reference KL 预警] rolling={rolling_kl:.4f} "
                f"超过 {cfg.kl_warning_threshold:.4f}"
            )
        if consecutive >= cfg.kl_guard_patience_checks:
            return build_stop_decision(
                StopCategory.KL_GUARD,
                source="reference_kl_guard",
                reason=(
                    f"actor-reference KL 持续超限({consecutive}个窗口): "
                    f"rolling={rolling_kl:.4f}>{cfg.kl_threshold:.4f}"
                ),
                step=step,
                details={
                    "rolling_reference_kl": rolling_kl,
                    "warning_threshold": cfg.kl_warning_threshold,
                    "hard_threshold": cfg.kl_threshold,
                    "consecutive_failures": consecutive,
                    "patience_checks": cfg.kl_guard_patience_checks,
                    "window": cfg.kl_guard_window,
                    "update_kl": update_kl,
                },
            )
        return None

    def _update_adaptive_kl(self, step: int) -> None:
        """按最近窗口的 reference KL 调整下一阶段的 KL 系数。"""
        cfg = self.cfg
        if not cfg.adaptive_kl_enabled or (step + 1) % cfg.adaptive_kl_interval != 0:
            return
        values = [
            float(item["kl_loss"])
            for item in self.metrics_history
            if isinstance(item, dict) and "kl_loss" in item
        ][-cfg.adaptive_kl_interval:]
        if not values:
            return
        reference_kl = _mean(values)
        old_coef = self.current_kl_loss_coef
        upper = cfg.adaptive_kl_target * cfg.adaptive_kl_tolerance
        lower = cfg.adaptive_kl_target / cfg.adaptive_kl_tolerance
        if reference_kl > upper:
            self.current_kl_loss_coef = min(
                cfg.adaptive_kl_max_coef, old_coef * cfg.adaptive_kl_factor
            )
        elif reference_kl < lower:
            self.current_kl_loss_coef = max(
                cfg.adaptive_kl_min_coef, old_coef / cfg.adaptive_kl_factor
            )
        if self.current_kl_loss_coef != old_coef:
            logger.info(
                f"[自适应 KL] step={step} reference_kl={reference_kl:.4f} "
                f"target={cfg.adaptive_kl_target:.4f} "
                f"coef={old_coef:.5f}->{self.current_kl_loss_coef:.5f}"
            )

    def _check_signal_guard(self, step: int) -> StopDecision | None:
        """检查组内 rollout 是否仍有足够学习信号。"""
        cfg = self.cfg
        if step + 1 < cfg.signal_guard_warmup_steps:
            return None
        all_diagnostics = [
            item["group_diagnostics"]
            for item in self.metrics_history
            if isinstance(item, dict) and item.get("group_diagnostics")
        ]
        if len(all_diagnostics) < cfg.signal_guard_window:
            return None
        if (
            cfg.signal_guard_non_overlapping_windows
            and len(all_diagnostics) % cfg.signal_guard_window != 0
        ):
            # 非重叠模式只在完整窗口边界决策，避免把共享90%样本的窗口当作独立证据。
            return None
        diagnostics = all_diagnostics[-cfg.signal_guard_window:]
        summary, failures, warnings = _evaluate_signal_guard_window(
            diagnostics,
            min_effective_group_rate=cfg.min_effective_group_rate,
            min_mixed_group_rate=cfg.min_mixed_group_rate,
            max_zero_advantage_rate=cfg.max_zero_advantage_rate,
            min_rollout_format_rate=cfg.min_rollout_format_rate,
            mixed_hard_stop=cfg.signal_guard_mixed_hard_stop,
        )
        consecutive_failures = 0
        if failures:
            stride = cfg.signal_guard_window if cfg.signal_guard_non_overlapping_windows else 1
            for end in range(
                len(all_diagnostics),
                cfg.signal_guard_window - 1,
                -stride,
            ):
                window = all_diagnostics[end - cfg.signal_guard_window:end]
                _, window_failures, _ = _evaluate_signal_guard_window(
                    window,
                    min_effective_group_rate=cfg.min_effective_group_rate,
                    min_mixed_group_rate=cfg.min_mixed_group_rate,
                    max_zero_advantage_rate=cfg.max_zero_advantage_rate,
                    min_rollout_format_rate=cfg.min_rollout_format_rate,
                    mixed_hard_stop=cfg.signal_guard_mixed_hard_stop,
                )
                if not window_failures:
                    break
                consecutive_failures += 1
        logger.info(
            f"[signal guard step {step}] window={cfg.signal_guard_window} "
            f"mode={'non_overlapping' if cfg.signal_guard_non_overlapping_windows else 'sliding'} "
            f"effective={summary['effective']:.3f}/{cfg.min_effective_group_rate:.3f} "
            f"mixed={summary['mixed']:.3f}/{cfg.min_mixed_group_rate:.3f} "
            f"zero_adv={summary['zero_adv']:.3f}/{cfg.max_zero_advantage_rate:.3f} "
            f"format={summary['format']:.3f}/{cfg.min_rollout_format_rate:.3f} "
            f"hard_failures={consecutive_failures}/{cfg.signal_guard_patience_checks} "
            f"mixed_policy={'hard' if cfg.signal_guard_mixed_hard_stop else 'warning'}"
        )
        if warnings:
            logger.warning(
                f"[信号保护预警] {', '.join(warnings)}，不计入硬停止耐心"
            )
        if not failures:
            return None
        if consecutive_failures < cfg.signal_guard_patience_checks:
            logger.warning(
                f"[信号保护观察] {', '.join(failures)}，连续 "
                f"{consecutive_failures}/{cfg.signal_guard_patience_checks} 个窗口，暂不终止"
            )
            return None
        return build_stop_decision(
            StopCategory.SIGNAL_GUARD,
            source="training_signal_guard",
            reason=(
                f"组内信号持续不足({consecutive_failures}个窗口): "
                + ", ".join(failures)
            ),
            step=step,
            details={
                "summary": summary,
                "failures": failures,
                "warnings": warnings,
                "consecutive_failures": consecutive_failures,
                "patience_checks": cfg.signal_guard_patience_checks,
                "window": cfg.signal_guard_window,
                "non_overlapping_windows": cfg.signal_guard_non_overlapping_windows,
                "mixed_hard_stop": cfg.signal_guard_mixed_hard_stop,
            },
        )

    def _check_validation_format_guard(
        self,
        *,
        step: int,
        val_format_rate: float,
        best_val_em: float,
    ) -> StopDecision | None:
        """独立判断固定验证集格式是否已经发生极端崩溃。"""
        if val_format_rate >= 0.1 or best_val_em <= 0:
            return None
        return build_stop_decision(
            StopCategory.FORMAT_GUARD,
            source="validation_format_guard",
            reason=f"格式退化: val_fmt={val_format_rate:.3f}",
            step=step,
            details={
                "val_format_rate": val_format_rate,
                "threshold": 0.1,
                "best_val_em": best_val_em,
            },
        )

    @staticmethod
    def _build_runtime_stop_decision(
        exc: BaseException,
        *,
        active_step: int,
        last_completed_step: int,
    ) -> StopDecision:
        """把中断、OOM和未捕获异常统一转换为结构化停止决定。"""
        if isinstance(exc, KeyboardInterrupt):
            category = StopCategory.INTERRUPTED
            source = "keyboard_interrupt"
            reason = "训练被用户中断"
        else:
            message = str(exc)
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_oom = (
                (oom_type is not None and isinstance(exc, oom_type))
                or "out of memory" in message.lower()
            )
            if is_oom:
                category = StopCategory.OUT_OF_MEMORY
                source = "runtime_out_of_memory"
                reason = f"训练发生显存不足: {message}"
            else:
                category = StopCategory.RUNTIME_ERROR
                source = "unhandled_runtime_error"
                reason = f"训练发生未捕获异常: {type(exc).__name__}: {message}"
        return build_stop_decision(
            category,
            source=source,
            reason=reason,
            step=active_step,
            severity=StopSeverity.FATAL,
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "active_step": active_step,
                "last_completed_step": last_completed_step,
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )

    @staticmethod
    def _status_for_stop_decision(decision: StopDecision) -> str:
        """把停止类别映射为稳定的训练终态。"""
        if decision.category == StopCategory.MAX_STEPS:
            return "completed"
        if decision.category == StopCategory.INTERRUPTED:
            return "interrupted"
        if decision.severity == StopSeverity.FATAL:
            return "failed"
        return "stopped"

    @staticmethod
    def _log_selected_stop(stage: str, decision: StopDecision) -> None:
        """统一打印停止决定，同时保留历史日志前缀供现有脚本读取。"""
        legacy_prefix = {
            StopCategory.KL_GUARD: "[KL 保护终止]",
            StopCategory.SIGNAL_GUARD: "[信号保护终止]",
            StopCategory.FORMAT_GUARD: "[格式退化]",
            StopCategory.EARLY_STOPPING: "[早停终止]",
            StopCategory.OUT_OF_MEMORY: "[OOM 终止]",
            StopCategory.INTERRUPTED: "[训练中断]",
            StopCategory.RUNTIME_ERROR: "[训练异常终止]",
        }.get(decision.category)
        if legacy_prefix is not None:
            logger.warning(f"{legacy_prefix} {decision.reason}")
        logger.warning(
            f"[统一停止调度] stage={stage} category={decision.category} "
            f"source={decision.source} reason={decision.reason}"
        )

    def _latest_safe_checkpoint(
        self, output_dir: Path, last_completed_step: int
    ) -> Path | None:
        """查找不晚于最后完整step的最近checkpoint。"""
        candidates: list[tuple[int, Path]] = []
        for path in output_dir.glob("checkpoint-*"):
            if not path.is_dir():
                continue
            try:
                checkpoint_step = int(path.name.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if checkpoint_step <= last_completed_step:
                candidates.append((checkpoint_step, path))
        if self.resume_checkpoint_dir is not None:
            try:
                resume_step = int(self.resume_checkpoint_dir.name.rsplit("-", 1)[-1])
            except ValueError:
                resume_step = -1
            if resume_step <= last_completed_step:
                candidates.append((resume_step, self.resume_checkpoint_dir))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _finalize_training(
        self,
        *,
        output_dir: Path,
        stop_controller: TrainingStopController,
        decision: StopDecision,
        last_completed_step: int,
        completed_steps_in_session: int,
        best_val_em: float,
        best_step: int,
        steps_no_improve: int,
        early_stop_extension_steps: int,
        train_reward_history: list[float],
    ) -> list[str]:
        """统一同步状态、保存产物、关闭资源并写停止档案。"""
        finalization_errors: list[str] = []
        status = self._status_for_stop_decision(decision)
        self.training_session_id = stop_controller.session_id
        self.training_status = status
        self.stop_decision = decision.to_dict()
        self.stop_reason = decision.reason

        if last_completed_step >= 0:
            self._sync_training_state(
                last_completed_step,
                best_val_em,
                best_step,
                steps_no_improve,
                early_stop_extension_steps,
                train_reward_history,
            )

        # 保留既有日志文本，避免编排器和看板的停止原因解析失效。
        logger.info(f"训练结束, 原因: {decision.reason}")
        logger.info(
            f"[统一停止摘要] status={status} category={decision.category} "
            f"source={decision.source} session={stop_controller.session_id}"
        )
        logger.info(f"最佳 val_em={best_val_em:.3f} (step={best_step})")

        # OOM后先释放无引用缓存，提高停止档案和小型状态文件写出的成功率。
        if decision.category == StopCategory.OUT_OF_MEMORY and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - 仅GPU异常分支
                finalization_errors.append(f"清理CUDA缓存失败: {type(exc).__name__}: {exc}")

        checkpoint_path = self._latest_safe_checkpoint(output_dir, last_completed_step)
        active_step = int(decision.details.get("active_step", decision.step))
        incomplete_runtime_step = (
            decision.severity == StopSeverity.FATAL
            and active_step > last_completed_step
        )
        should_save_terminal_checkpoint = (
            last_completed_step >= 0
            and completed_steps_in_session > 0
            and not incomplete_runtime_step
        )
        if should_save_terminal_checkpoint:
            try:
                self._save_checkpoint(output_dir, last_completed_step)
                checkpoint_path = output_dir / f"checkpoint-{last_completed_step}"
            except Exception as exc:
                finalization_errors.append(
                    f"保存终态checkpoint失败: {type(exc).__name__}: {exc}"
                )
                logger.exception("保存终态checkpoint失败")
        elif incomplete_runtime_step:
            logger.warning(
                f"active_step={active_step}未完整结束，不保存可能含部分更新的新checkpoint；"
                f"最近安全checkpoint={checkpoint_path}"
            )

        try:
            self._save_metrics(output_dir)
        except Exception as exc:
            finalization_errors.append(f"保存训练指标失败: {type(exc).__name__}: {exc}")
            logger.exception("保存训练指标失败")

        try:
            if last_completed_step >= 0:
                self._append_gpu_csv(last_completed_step, "训练结束")
            self._log_gpu_memory_detailed("训练结束", batch=None)
        except Exception as exc:
            finalization_errors.append(f"记录终态显存失败: {type(exc).__name__}: {exc}")
            logger.exception("记录终态显存失败")

        finalization_errors.extend(self._close_csv_files())
        try:
            stop_controller.finalize(
                status=status,
                decision=decision,
                last_completed_step=last_completed_step,
                checkpoint_path=checkpoint_path,
                finalization_errors=finalization_errors,
            )
        except Exception as exc:
            finalization_errors.append(f"写停止档案失败: {type(exc).__name__}: {exc}")
            logger.exception("写停止档案失败")

        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                finally:
                    logger.removeHandler(handler)
        return finalization_errors

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self) -> list[dict[str, Any]]:
        """执行GRPO训练，并由统一控制器调度全部停止路径。"""
        cfg = self.cfg
        output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("models/grpo") / cfg.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 停止控制器只统一调度、优先级、留档和收尾；各判定器保持独立。
        stop_controller = TrainingStopController(output_dir, cfg.run_name)
        self.training_session_id = stop_controller.session_id
        self.training_status = "running"
        self.stop_decision = None
        self.stop_reason = None

        best_val_em = self.best_val_em
        best_step = self.best_step
        steps_no_improve = self.steps_no_improve
        early_stop_extension_steps = self.early_stop_extension_steps
        train_reward_history: list[float] = list(self.train_reward_history)
        last_completed_step = self.last_completed_step
        active_step = self.start_step - 1
        completed_steps_in_session = 0
        selected_decision: StopDecision | None = None
        caught_exception: BaseException | None = None
        caught_traceback: Any = None
        finalization_errors: list[str] = []

        # 配置日志（控制台 + 文件），保留原有日志路径和文本格式。
        setup_logging(output_dir, cfg.run_name)

        try:
            stop_controller.start(
                {
                    "start_step": self.start_step,
                    "resume_checkpoint": self.resume_checkpoint_dir,
                    "resume_state_mode": cfg.resume_state_mode,
                    "total_training_steps": cfg.total_training_steps,
                }
            )
            self._save_run_config(output_dir)
            logger.info(f"输出目录: {output_dir}")
            logger.info(f"训练 prompt 数: {len(self.train_dataset)}")
            logger.info(f"每步 prompt 数: {cfg.train_batch_size}")
            logger.info(f"每 prompt 回答数: {cfg.rollout_n}")
            logger.info(f"最大训练步数: {cfg.total_training_steps}")
            logger.info(f"早停耐心: {cfg.max_steps_no_improve} 步 (验证 EM 连续不改善)")
            logger.info(
                f"早停趋势保护: window={cfg.early_stop_trend_window} "
                f"min_slope={cfg.early_stop_min_recovery_slope:.4f}/验证点 "
                f"max_extension={cfg.early_stop_max_extension_steps} 步"
            )
            logger.info(
                f"Reference KL 保护: warning={cfg.kl_warning_threshold:.4f} "
                f"hard={cfg.kl_threshold:.4f} window={cfg.kl_guard_window} "
                f"patience={cfg.kl_guard_patience_checks}"
            )
            logger.info(f"PPO update KL 阈值: {cfg.approx_kl_threshold:.4e}")
            logger.info(
                f"自适应 KL: enabled={cfg.adaptive_kl_enabled} "
                f"target={cfg.adaptive_kl_target:.4f} tolerance={cfg.adaptive_kl_tolerance:.3f} "
                f"coef={self.current_kl_loss_coef:.5f} "
                f"range=[{cfg.adaptive_kl_min_coef:.5f}, {cfg.adaptive_kl_max_coef:.5f}] "
                f"interval={cfg.adaptive_kl_interval} factor={cfg.adaptive_kl_factor:.3f}"
            )
            logger.info(
                f"训练参数: batch={cfg.train_batch_size} rollout_n={cfg.rollout_n} "
                f"rollout_batch={cfg.rollout_batch_size} ppo_epochs={cfg.ppo_epochs} "
                f"ppo_mini_batch={cfg.ppo_mini_batch_size} "
                f"grad_accum={cfg.gradient_accumulation_steps} lr={cfg.learning_rate:.2e} "
                f"max_response={cfg.max_response_length}"
            )
            logger.info(f"Reward 配置: {asdict(self.reward_config)}")
            logger.info(
                f"Reward hacking 检测: {cfg.reward_hacking_detect} "
                f"(窗口={cfg.reward_hacking_window})"
            )
            logger.info(
                f"信号保护: window={cfg.signal_guard_window} "
                f"patience={cfg.signal_guard_patience_checks} 个连续窗口 "
                f"mode={'non_overlapping' if cfg.signal_guard_non_overlapping_windows else 'sliding'} "
                f"mixed={'hard' if cfg.signal_guard_mixed_hard_stop else 'warning'}"
            )
            logger.info(
                f"统一停止调度: session={stop_controller.session_id} "
                "priority=runtime>KL>signal>format>early_stop>max_steps"
            )
            if self.resume_checkpoint_dir is not None:
                logger.info(f"续训 checkpoint: {self.resume_checkpoint_dir}")
                logger.info(f"checkpoint 状态模式: {cfg.resume_state_mode}")
                logger.info(f"续训起始 step: {self.start_step}")
                logger.info(f"完整 trainer state: {self.has_trainer_state}")

            # 原CSV文件名、列名和实时追加行为保持不变，确保现有看板继续工作。
            csv_dir = output_dir / "plots"
            csv_dir.mkdir(parents=True, exist_ok=True)
            self._train_csv_path = csv_dir / "train_metrics.csv"
            self._val_csv_path = csv_dir / "val_metrics.csv"
            self._group_diag_csv_path = csv_dir / "group_diagnostics.csv"
            self._gpu_csv_path = csv_dir / "gpu_memory.csv"
            if self.has_trainer_state:
                for path in (
                    self._train_csv_path,
                    self._val_csv_path,
                    self._group_diag_csv_path,
                    self._gpu_csv_path,
                ):
                    _truncate_csv_before_resume(path, self.start_step)
            self._train_csv_writer = _init_csv(self._train_csv_path, TRAIN_CSV_COLUMNS)
            self._val_csv_writer = _init_csv(self._val_csv_path, VAL_CSV_COLUMNS)
            self._group_diag_csv_writer = _init_csv(
                self._group_diag_csv_path, GROUP_DIAG_CSV_COLUMNS
            )
            self._gpu_csv_writer = _init_csv(self._gpu_csv_path, GPU_CSV_COLUMNS)

            self._log_gpu_memory_detailed("训练前_初始", batch=None)
            if not self.has_trainer_state:
                self._append_gpu_csv(-1, "训练前_初始")

            # 完整resume沿用checkpoint中的早停状态；新分支先建立零步基线。
            should_run_initial_val = cfg.val_before_train and not self.has_trainer_state
            if should_run_initial_val:
                val_metrics = self._validate()
                val_metrics["step"] = (
                    "initial" if self.start_step == 0 else f"resume_{self.start_step - 1}"
                )
                log_prefix = "[初始验证]" if self.start_step == 0 else "[续训前验证]"
                logger.info(f"{log_prefix} {_format_val_metrics(val_metrics)}")
                self.metrics_history.append(val_metrics)
                best_val_em = float(val_metrics.get("val_exact_match", 0.0))
                logger.info(f"初始 baseline: val_em={best_val_em:.3f}")
                self._append_val_csv(self.start_step - 1, val_metrics, best_val_em)
                log_gpu_memory("初始验证后")

            for step in range(self.start_step, cfg.total_training_steps):
                active_step = step
                step_metrics = self.train_step(step)
                last_completed_step = step
                completed_steps_in_session += 1

                train_reward_history.append(step_metrics["reward_mean"])
                self._append_train_csv(step, step_metrics, best_val_em, steps_no_improve)
                self._append_group_diag_csv(
                    step, step_metrics.get("group_diagnostics", {})
                )
                self._sync_training_state(
                    step,
                    best_val_em,
                    best_step,
                    steps_no_improve,
                    early_stop_extension_steps,
                    train_reward_history,
                )

                # 保留既有逐步日志字段，避免日志监控和看板解析回归。
                if step % cfg.log_steps == 0:
                    logger.info(
                        f"[step {step}/{cfg.total_training_steps}] "
                        f"reward={step_metrics['reward_mean']:.3f}±{step_metrics['reward_std']:.3f} "
                        f"policy_loss={step_metrics['policy_loss']:.4f} "
                        f"reference_kl={step_metrics['kl_loss']:.4f} "
                        f"update_kl={step_metrics['approx_kl']:.3e} "
                        f"kl_coef={step_metrics['kl_loss_coef']:.5f} "
                        f"clip_frac={step_metrics['clip_frac']:.3f} "
                        f"grad_norm={step_metrics['grad_norm']:.3f} "
                        f"opt_updates={int(step_metrics['optimizer_update_count'])} "
                        f"resp_len={step_metrics['response_len_mean']:.0f} "
                        f"lr={step_metrics['lr']:.2e} "
                        f"best_em={best_val_em:.3f}(step={best_step}) "
                        f"no_improve={steps_no_improve}/{cfg.max_steps_no_improve} "
                        f"time={step_metrics['step_time']:.1f}s"
                    )
                    group_diag = step_metrics.get("group_diagnostics", {})
                    if group_diag:
                        logger.info(
                            f"[group diag step {step}] "
                            f"effective={group_diag.get('effective_group_rate', 0):.3f} "
                            f"mixed={group_diag.get('mixed_group_rate', 0):.3f} "
                            f"all_wrong={group_diag.get('all_wrong_group_rate', 0):.3f} "
                            f"all_correct={group_diag.get('all_correct_group_rate', 0):.3f} "
                            f"rollout_em={group_diag.get('rollout_exact_rate', 0):.3f} "
                            f"adv_std={group_diag.get('advantage_std', 0):.3f} "
                            f"zero_adv={group_diag.get('zero_advantage_rate', 0):.3f} "
                            f"hist={group_diag.get('correct_count_hist_json', '{}')}"
                        )

                # 更新后阶段统一收集全部安全判定，显式优先级代替if顺序隐式覆盖。
                kl_decision = self._check_kl_guard(step, step_metrics)
                signal_decision = self._check_signal_guard(step)
                selected_decision = stop_controller.select(
                    "post_update",
                    [kl_decision, signal_decision],
                )
                if selected_decision is not None:
                    self._log_selected_stop("post_update", selected_decision)
                    break

                # 当前步通过保护检查后再调整系数，新系数从下一步更新开始生效。
                self._update_adaptive_kl(step)

                scheduled_save = (step + 1) % cfg.save_freq == 0
                should_eval = (step + 1) % cfg.eval_freq == 0
                is_last_budget_step = step == cfg.total_training_steps - 1

                # 终态由统一收尾保存；普通非评估保存点仍保持原行为。
                if scheduled_save and not should_eval and not is_last_budget_step:
                    self._save_checkpoint(output_dir, step)

                if should_eval:
                    val_metrics = self._validate()
                    val_metrics["step"] = step
                    logger.info(f"[eval step {step}] {_format_val_metrics(val_metrics)}")
                    self.metrics_history.append(val_metrics)

                    val_em = float(val_metrics.get("val_exact_match", 0.0))
                    val_fmt = float(val_metrics.get("val_format_rate", 0.0))
                    previous_best_val_em = best_val_em
                    val_em_history = [
                        float(item["val_exact_match"])
                        for item in self.metrics_history
                        if isinstance(item, dict) and "val_exact_match" in item
                    ]
                    early_evaluation = _evaluate_early_stopping(
                        step=step,
                        val_em=val_em,
                        best_val_em=best_val_em,
                        best_step=best_step,
                        steps_no_improve=steps_no_improve,
                        extension_steps=early_stop_extension_steps,
                        val_em_history=val_em_history,
                        eval_freq=cfg.eval_freq,
                        max_steps_no_improve=cfg.max_steps_no_improve,
                        trend_window=cfg.early_stop_trend_window,
                        min_recovery_slope=cfg.early_stop_min_recovery_slope,
                        max_extension_steps=cfg.early_stop_max_extension_steps,
                    )
                    best_val_em = early_evaluation.best_val_em
                    best_step = early_evaluation.best_step
                    steps_no_improve = early_evaluation.steps_no_improve
                    early_stop_extension_steps = early_evaluation.extension_steps

                    if early_evaluation.improved:
                        logger.info(
                            f"[早停追踪] val_em 改善: {val_em:.3f} "
                            f"(最佳 step={best_step})"
                        )
                    else:
                        logger.info(
                            f"[早停追踪] val_em 未改善: {val_em:.3f} "
                            f"(最佳={best_val_em:.3f} at step={best_step}) "
                            f"已连续 {steps_no_improve} 步"
                        )
                    if early_evaluation.extended:
                        slope_text = (
                            "数据不足"
                            if early_evaluation.recovery_slope is None
                            else f"{early_evaluation.recovery_slope:.4f}"
                        )
                        logger.info(
                            f"[早停延长] val_em 连续 {steps_no_improve} 步未刷新全局最佳，"
                            f"但最近 {cfg.early_stop_trend_window} 个验证点斜率={slope_text}，"
                            f"延长 {cfg.eval_freq} 步 "
                            f"({early_stop_extension_steps}/{cfg.early_stop_max_extension_steps})"
                        )
                    elif early_evaluation.decision is not None:
                        slope_text = early_evaluation.decision.details.get(
                            "recovery_slope_text", "数据不足"
                        )
                        logger.info(
                            f"[早停终止候选] val_em 连续 {steps_no_improve} 步未改善，"
                            f"近期斜率={slope_text}，趋势延长="
                            f"{early_stop_extension_steps}/{cfg.early_stop_max_extension_steps} "
                            f"(最佳={best_val_em:.3f} at step={best_step})"
                        )

                    # 保持原CSV列结构，在更新best之后实时写入。
                    self._append_val_csv(step, val_metrics, best_val_em)
                    self._sync_training_state(
                        step,
                        best_val_em,
                        best_step,
                        steps_no_improve,
                        early_stop_extension_steps,
                        train_reward_history,
                    )

                    # Reward hacking仍是预警，不提交停止决定。
                    if (
                        cfg.reward_hacking_detect
                        and len(train_reward_history) >= cfg.reward_hacking_window
                    ):
                        recent_reward = _mean(
                            train_reward_history[-cfg.reward_hacking_window:]
                        )
                        earlier_reward = _mean(
                            train_reward_history[
                                -2 * cfg.reward_hacking_window:-cfg.reward_hacking_window
                            ]
                            if len(train_reward_history) >= 2 * cfg.reward_hacking_window
                            else train_reward_history[:cfg.reward_hacking_window]
                        )
                        if recent_reward > earlier_reward and val_em <= previous_best_val_em:
                            logger.warning(
                                f"[reward hacking 警告] 训练 reward 从 {earlier_reward:.3f} "
                                f"升至 {recent_reward:.3f}, 但 val_em={val_em:.3f} "
                                f"未改善 (最佳={best_val_em:.3f}) → 模型可能在钻 RM/规则漏洞"
                            )

                    format_decision = self._check_validation_format_guard(
                        step=step,
                        val_format_rate=val_fmt,
                        best_val_em=best_val_em,
                    )
                    if format_decision is not None:
                        logger.warning(
                            f"[格式退化候选] val_fmt={val_fmt:.3f} 极低，"
                            "模型丧失格式能力"
                        )

                    selected_decision = stop_controller.select(
                        "post_eval",
                        [format_decision, early_evaluation.decision],
                    )
                    if selected_decision is not None:
                        self._log_selected_stop("post_eval", selected_decision)
                        break

                    if scheduled_save and not is_last_budget_step:
                        self._save_checkpoint(output_dir, step)

            if selected_decision is None:
                if self.start_step >= cfg.total_training_steps:
                    reason = (
                        f"起始 step {self.start_step} 已达到最大训练步数 "
                        f"{cfg.total_training_steps}"
                    )
                else:
                    reason = "达到最大步数"
                budget_decision = build_stop_decision(
                    StopCategory.MAX_STEPS,
                    source="training_budget",
                    reason=reason,
                    step=last_completed_step,
                    severity=StopSeverity.NORMAL,
                    details={
                        "start_step": self.start_step,
                        "last_completed_step": last_completed_step,
                        "total_training_steps": cfg.total_training_steps,
                    },
                )
                selected_decision = stop_controller.select(
                    "budget",
                    [budget_decision],
                )

        except BaseException as exc:
            caught_exception = exc
            caught_traceback = exc.__traceback__
            runtime_decision = self._build_runtime_stop_decision(
                exc,
                active_step=active_step,
                last_completed_step=last_completed_step,
            )
            try:
                selected_decision = stop_controller.select(
                    "runtime_exception",
                    [runtime_decision],
                )
                if selected_decision is not None:
                    self._log_selected_stop("runtime_exception", selected_decision)
            except Exception:
                # 留档本身失败时仍保留内存中的原始异常决定，交由统一收尾再次尝试。
                selected_decision = runtime_decision
                logger.exception("记录运行时停止候选失败")

        finally:
            if selected_decision is None:
                selected_decision = build_stop_decision(
                    StopCategory.RUNTIME_ERROR,
                    source="missing_stop_decision_fallback",
                    reason="训练退出但未形成停止决定",
                    step=active_step,
                    severity=StopSeverity.FATAL,
                    details={"last_completed_step": last_completed_step},
                )
            try:
                finalization_errors = self._finalize_training(
                    output_dir=output_dir,
                    stop_controller=stop_controller,
                    decision=selected_decision,
                    last_completed_step=last_completed_step,
                    completed_steps_in_session=completed_steps_in_session,
                    best_val_em=best_val_em,
                    best_step=best_step,
                    steps_no_improve=steps_no_improve,
                    early_stop_extension_steps=early_stop_extension_steps,
                    train_reward_history=train_reward_history,
                )
            except BaseException as exc:
                finalization_errors = [
                    f"统一收尾发生未捕获异常: {type(exc).__name__}: {exc}"
                ]
                logger.exception("统一收尾发生未捕获异常")

        if caught_exception is not None:
            raise caught_exception.with_traceback(caught_traceback)
        if finalization_errors:
            raise RuntimeError("; ".join(finalization_errors))
        return self.metrics_history

    def _save_checkpoint(self, output_dir: Path, step: int) -> None:
        """保存 actor adapter、optimizer、训练状态和随机数状态。"""
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        torch.save(self.optimizer.state_dict(), str(ckpt_dir / "optimizer.pt"))
        torch.save(self._capture_rng_state(), str(ckpt_dir / "rng_state.pt"))

        trainer_state = {
            "step": step,
            "next_step": step + 1,
            "training_session_id": self.training_session_id,
            "training_status": self.training_status,
            "stop_reason": self.stop_reason,
            "stop_decision": self.stop_decision,
            "best_val_em": self.best_val_em,
            "best_step": self.best_step,
            "steps_no_improve": self.steps_no_improve,
            "early_stop_extension_steps": self.early_stop_extension_steps,
            "current_kl_loss_coef": self.current_kl_loss_coef,
            "train_reward_history": self.train_reward_history,
            "metrics_history": self.metrics_history,
            "run_name": self.cfg.run_name,
            "total_training_steps": self.cfg.total_training_steps,
            "save_freq": self.cfg.save_freq,
            "eval_freq": self.cfg.eval_freq,
            "rollout_n": self.cfg.rollout_n,
            "rollout_batch_size": self.cfg.rollout_batch_size,
            "val_eval_batch_size": self.cfg.val_eval_batch_size,
            "reward_config": asdict(self.reward_config),
            "base_model_dir": self.cfg.base_model_dir,
            "sft_adapter_dir": self.cfg.sft_adapter_dir,
            "train_file": self.cfg.train_file,
            "eval_file": self.cfg.eval_file,
            "config": asdict(self.cfg),
        }
        with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint 已保存: {ckpt_dir}")

    def _save_metrics(self, output_dir: Path) -> None:
        """保存训练指标历史。"""
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
        logger.info(f"训练指标已保存: {metrics_path}")

    # ------------------------------------------------------------------
    # CSV 绘图数据写入
    # ------------------------------------------------------------------

    def _append_train_csv(self, step: int, metrics: dict[str, float],
                          best_val_em: float, steps_no_improve: int) -> None:
        """追加一行训练指标到 CSV。"""
        if self._train_csv_writer is None:
            return
        row = {
            "step": step,
            "reward_mean": round(metrics["reward_mean"], 4),
            "reward_std": round(metrics["reward_std"], 4),
            "policy_loss": round(metrics["policy_loss"], 6),
            "kl_loss": round(metrics["kl_loss"], 6),
            "approx_kl": round(metrics["approx_kl"], 6),
            "clip_frac": round(metrics["clip_frac"], 4),
            "response_len_mean": round(metrics["response_len_mean"], 1),
            "lr": metrics["lr"],
            "step_time": round(metrics["step_time"], 2),
            "grad_norm": round(metrics.get("grad_norm", 0.0), 6),
            "kl_loss_coef": round(metrics.get("kl_loss_coef", self.current_kl_loss_coef), 8),
            "prompt_count": int(metrics.get("prompt_count", 0)),
            "rollout_count": int(metrics.get("rollout_count", 0)),
            "mini_batch_count": int(metrics.get("mini_batch_count", 0)),
            "optimizer_update_count": int(metrics.get("optimizer_update_count", 0)),
            "best_val_em": round(best_val_em, 4),
            "steps_no_improve": steps_no_improve,
        }
        _append_csv_row(self._train_csv_writer, row, TRAIN_CSV_COLUMNS)

    def _append_val_csv(self, step: int, val_metrics: dict[str, float],
                        best_val_em: float) -> None:
        """追加一行验证指标到 CSV。"""
        if self._val_csv_writer is None:
            return
        row = {
            "step": step,
            "val_reward_mean": round(val_metrics.get("val_reward_mean", 0), 4),
            "val_exact_match": round(val_metrics.get("val_exact_match", 0), 4),
            "val_format_rate": round(val_metrics.get("val_format_rate", 0), 4),
            "val_response_len_mean": round(val_metrics.get("val_response_len_mean", 0), 1),
            "val_max_tokens_reached_without_eos_rate": round(
                val_metrics.get("val_max_tokens_reached_without_eos_rate", 0), 4
            ),
            "val_eos_rate": round(val_metrics.get("val_eos_rate", 0), 4),
            "val_sample_exact_match": round(val_metrics.get("val_sample_exact_match", 0), 4),
            "val_sample_format_rate": round(val_metrics.get("val_sample_format_rate", 0), 4),
            "val_sample_response_len_mean": round(
                val_metrics.get("val_sample_response_len_mean", 0), 1
            ),
            "val_sample_max_tokens_reached_without_eos_rate": round(
                val_metrics.get("val_sample_max_tokens_reached_without_eos_rate", 0), 4
            ),
            "val_sample_eos_rate": round(val_metrics.get("val_sample_eos_rate", 0), 4),
            "best_val_em_so_far": round(best_val_em, 4),
        }
        _append_csv_row(self._val_csv_writer, row, VAL_CSV_COLUMNS)

    def _append_group_diag_csv(self, step: int, diag: dict[str, Any]) -> None:
        """追加一行组内 rollout 诊断指标到 CSV。"""
        if self._group_diag_csv_writer is None or not diag:
            return
        row = {
            "step": step,
            "group_count": int(diag.get("group_count", 0)),
            "rollout_count": int(diag.get("rollout_count", 0)),
            "rollout_n": int(diag.get("rollout_n", self.cfg.rollout_n)),
            "effective_group_rate": round(float(diag.get("effective_group_rate", 0.0)), 4),
            "zero_signal_group_rate": round(float(diag.get("zero_signal_group_rate", 0.0)), 4),
            "mixed_group_rate": round(float(diag.get("mixed_group_rate", 0.0)), 4),
            "all_wrong_group_rate": round(float(diag.get("all_wrong_group_rate", 0.0)), 4),
            "all_correct_group_rate": round(float(diag.get("all_correct_group_rate", 0.0)), 4),
            "rollout_exact_rate": round(float(diag.get("rollout_exact_rate", 0.0)), 4),
            "rollout_format_rate": round(float(diag.get("rollout_format_rate", 0.0)), 4),
            "rollout_eos_rate": round(float(diag.get("rollout_eos_rate", 0.0)), 4),
            "rollout_max_tokens_reached_without_eos_rate": round(
                float(diag.get("rollout_max_tokens_reached_without_eos_rate", 0.0)), 4
            ),
            "rollout_response_len_mean": round(
                float(diag.get("rollout_response_len_mean", 0.0)), 1
            ),
            "fallback_exact_rate": round(float(diag.get("fallback_exact_rate", 0.0)), 4),
            "group_reward_std_mean": round(float(diag.get("group_reward_std_mean", 0.0)), 6),
            "group_reward_std_nonzero_mean": round(float(diag.get("group_reward_std_nonzero_mean", 0.0)), 6),
            "group_reward_std_max": round(float(diag.get("group_reward_std_max", 0.0)), 6),
            "advantage_mean": round(float(diag.get("advantage_mean", 0.0)), 6),
            "advantage_std": round(float(diag.get("advantage_std", 0.0)), 6),
            "advantage_abs_mean": round(float(diag.get("advantage_abs_mean", 0.0)), 6),
            "zero_advantage_rate": round(float(diag.get("zero_advantage_rate", 0.0)), 4),
            "unique_response_mean": round(float(diag.get("unique_response_mean", 0.0)), 4),
            "duplicate_group_rate": round(float(diag.get("duplicate_group_rate", 0.0)), 4),
            "response_empty_rate": round(float(diag.get("response_empty_rate", 0.0)), 4),
            "reward_answer_mean": round(float(diag.get("reward_answer_mean", 0.0)), 6),
            "reward_format_mean": round(float(diag.get("reward_format_mean", 0.0)), 6),
            "reward_single_final_mean": round(float(diag.get("reward_single_final_mean", 0.0)), 6),
            "reward_repeat_mean": round(float(diag.get("reward_repeat_mean", 0.0)), 6),
            "reward_overlong_mean": round(float(diag.get("reward_overlong_mean", 0.0)), 6),
            "reward_length_mean": round(float(diag.get("reward_length_mean", 0.0)), 6),
            "reward_truncated_mean": round(float(diag.get("reward_truncated_mean", 0.0)), 6),
            "reward_raw_mean": round(float(diag.get("reward_raw_mean", 0.0)), 6),
            "sample_indices_json": diag.get("sample_indices_json", "[]"),
            "sample_buckets_json": diag.get("sample_buckets_json", "[]"),
            "correct_count_hist_json": diag.get("correct_count_hist_json", "{}"),
        }
        _append_csv_row(self._group_diag_csv_writer, row, GROUP_DIAG_CSV_COLUMNS)

    def _append_gpu_csv(self, step: int, tag: str, batch: dict[str, Any] | None = None) -> None:
        """追加一行显存指标到 CSV，含各组件拆解。"""
        if self._gpu_csv_writer is None or not torch.cuda.is_available():
            return
        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0) / (1024**3)
        reserved = stats.get("reserved_bytes.all.current", 0) / (1024**3)
        peak_allocated = stats.get("allocated_bytes.all.peak", 0) / (1024**3)
        segment_count = stats.get("segment.all.current", 0)

        # 各组件拆解
        actor_comp = _calc_model_component_bytes(self.actor)
        ref_comp = _calc_model_component_bytes(self.reference)
        optimizer_comp = _calc_optimizer_bytes(self.optimizer)
        grad_bytes = _calc_grad_bytes(self.actor)
        batch_bytes = _calc_batch_bytes(batch) if batch is not None else 0

        # KV cache 估算
        kv_estimated = 0
        try:
            config = self.actor.config if hasattr(self.actor, "config") else None
            if config is not None:
                num_layers = config.num_hidden_layers
                num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
                head_dim = config.hidden_size // config.num_attention_heads
                dtype_size = 2
                seq_len = self.cfg.max_prompt_length + self.cfg.max_response_length
                kv_per_seq = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size
                kv_estimated = kv_per_seq * self.cfg.rollout_n * self.cfg.train_batch_size
        except Exception:
            pass

        # 激活值+临时 = allocated - 识别项
        identified = (actor_comp["total"] + ref_comp["total"]
                      + optimizer_comp["total"] + grad_bytes + batch_bytes)
        activations = stats.get("allocated_bytes.all.current", 0) - identified

        gb = lambda b: round(b / (1024**3), 4)

        row = {
            "step": step,
            "tag": tag,
            "allocated_gb": round(allocated, 4),
            "reserved_gb": round(reserved, 4),
            "pool_free_gb": round(reserved - allocated, 4),
            "segment_count": segment_count,
            "peak_allocated_gb": round(peak_allocated, 4),
            # 模型权重拆解
            "actor_base_weights_gb": gb(actor_comp["base"]),
            "actor_lora_weights_gb": gb(actor_comp["lora"]),
            "actor_embed_weights_gb": gb(actor_comp["embed"]),
            "actor_weights_total_gb": gb(actor_comp["total"]),
            "ref_base_weights_gb": gb(ref_comp["base"]),
            "ref_lora_weights_gb": gb(ref_comp["lora"]),
            "ref_weights_total_gb": gb(ref_comp["total"]),
            "model_weights_total_gb": gb(actor_comp["total"] + ref_comp["total"]),
            # 优化器拆解
            "optimizer_momentum_gb": gb(optimizer_comp["momentum"]),
            "optimizer_variance_gb": gb(optimizer_comp["variance"]),
            "optimizer_total_gb": gb(optimizer_comp["total"]),
            # 梯度缓冲
            "grad_buffer_gb": gb(grad_bytes),
            # 训练数据
            "rollout_data_gb": gb(batch_bytes),
            # 激活值/logits/临时
            "activations_logits_temp_gb": gb(activations),
            # KV cache
            "kv_cache_estimated_gb": gb(kv_estimated),
        }
        _append_csv_row(self._gpu_csv_writer, row, GPU_CSV_COLUMNS)

    def _close_csv_files(self) -> list[str]:
        """逐个关闭所有CSV，即使单个文件失败也继续清理其余资源。"""
        errors: list[str] = []
        writer_names = (
            "_train_csv_writer",
            "_val_csv_writer",
            "_group_diag_csv_writer",
            "_gpu_csv_writer",
        )
        for name in writer_names:
            writer = getattr(self, name, None)
            try:
                if writer is not None and hasattr(writer, "fh"):
                    if not writer.fh.closed:
                        writer.fh.flush()
                        writer.fh.close()
            except Exception as exc:
                message = f"关闭{name}失败: {type(exc).__name__}: {exc}"
                errors.append(message)
                logger.exception(message)
            finally:
                setattr(self, name, None)
        return errors


def _format_val_metrics(m: dict[str, Any]) -> str:
    """格式化验证指标为字符串。"""
    return (
        f"reward={m.get('val_reward_mean', 0):.3f} "
        f"em={m.get('val_exact_match', 0):.3f} "
        f"fmt={m.get('val_format_rate', 0):.3f} "
        f"len={m.get('val_response_len_mean', 0):.0f} "
        "max_tokens_reached_without_eos="
        f"{m.get('val_max_tokens_reached_without_eos_rate', 0):.3f} "
        f"eos={m.get('val_eos_rate', 0):.3f} "
        f"sample_em={m.get('val_sample_exact_match', 0):.3f} "
        f"sample_fmt={m.get('val_sample_format_rate', 0):.3f} "
        "sample_max_tokens_reached_without_eos="
        f"{m.get('val_sample_max_tokens_reached_without_eos_rate', 0):.3f}"
    )
