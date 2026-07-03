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
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger("ptf.grpo")


def setup_logging(output_dir: Path, run_name: str) -> None:
    """配置日志：同时输出到控制台和日志文件。"""
    logger.setLevel(logging.INFO)
    # 防止重复添加 handler
    if logger.handlers:
        return

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 文件 handler（每条日志立即 flush，防止 OOM 崩溃丢日志）
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8", mode="a")
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
    "best_val_em", "steps_no_improve",
]

# 验证指标 CSV 列定义 (每 eval_freq 步一行)
VAL_CSV_COLUMNS = [
    "step", "val_reward_mean", "val_exact_match", "val_format_rate",
    "val_response_len_mean", "best_val_em_so_far",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists()
    fh = open(str(path), "a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if need_header:
        writer.writerow(columns)
        fh.flush()
    return _CSVWriter(fh, writer)


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
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 数据
    train_file: str = ""
    eval_file: str = ""
    max_prompt_length: int = 512
    max_response_length: int = 512
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION
    enable_thinking: bool = False

    # rollout
    rollout_n: int = 4  # 每个 prompt 生成的回答数
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50

    # 训练
    train_batch_size: int = 8  # 每步处理的 prompt 数量
    ppo_epochs: int = 1  # 对每批 rollout 数据重复训练的轮数
    ppo_mini_batch_size: int = 16  # mini-batch 大小 (prompt*N 个回答)
    learning_rate: float = 1e-6
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"  # "low_var_kl" | "kl"
    clip_ratio: float = 0.2
    norm_adv_by_std: bool = True
    max_grad_norm: float = 1.0
    total_training_steps: int = 500  # 最大训练步数上限（兜底）
    max_steps_no_improve: int = 50  # 验证 EM 连续不改善的最大步数 (early stop)
    kl_threshold: float = 0.1  # KL 散度阈值，超过则异常终止
    reward_hacking_detect: bool = True  # 是否检测 reward hacking
    reward_hacking_window: int = 30  # reward hacking 检测窗口 (步数)
    save_freq: int = 10
    eval_freq: int = 10
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True

    # 输出
    output_dir: str = ""
    run_name: str = "grpo_run"
    log_steps: int = 1

    # 验证
    val_before_train: bool = True
    val_max_items: int = 20


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

    if kl_type == "low_var_kl":
        # k3 估计: exp(log_ratio) - log_ratio - 1, 方差更低
        log_ratio = ref_log_probs - log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        # 标准 KL: log(p/q) = log_p - log_q
        kl = log_probs - ref_log_probs

    valid_tokens = mask.sum()
    if valid_tokens == 0:
        return torch.tensor(0.0, device=log_probs.device)
    return (kl * mask).sum() / valid_tokens


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """自包含 GRPO 训练器。"""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
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
        self.reward_config = GSM8KRewardConfig()

        # 续训状态：optimizer/trainer/RNG 如果存在会被恢复；旧 checkpoint 缺失时只做权重 warm-start。
        self.resume_checkpoint_dir = resume_checkpoint
        self.has_trainer_state = False
        self.start_step = 0
        self.last_completed_step = -1
        self.best_val_em = -1.0
        self.best_step = -1
        self.steps_no_improve = 0
        self.train_reward_history: list[float] = []

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
        self._train_csv_writer: Any = None
        self._val_csv_writer: Any = None

        # 显存: 模型加载后的初始状态（含各组件拆解）
        log_gpu_memory("模型加载后")
        self._log_gpu_memory_detailed("模型加载后", batch=None)

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

        for prompt in prompts:
            for i in range(self.cfg.rollout_n):
                # 每次生成可能因随机性得到不同结果
                inputs = self.tokenizer(
                    prompt.prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)

                # 用不同的 seed 模拟采样多样性
                gen_seed = self.cfg.seed + prompt.idx * self.cfg.rollout_n + i
                torch.manual_seed(gen_seed)

                output_ids = self.actor.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_response_length,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    top_k=self.cfg.top_k,
                    pad_token_id=self._pad_token_id,
                    eos_token_id=self._eos_token_id,
                )

                prompt_len = inputs.input_ids.shape[-1]
                response_ids = output_ids[0, prompt_len:].tolist()
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                rollout_items.append({
                    "prompt": prompt,
                    "response_text": response_text,
                    "response_ids": response_ids,
                    "prompt_len": prompt_len,
                    "gen_index": i,
                })

        self.actor.train()
        return rollout_items

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

    # ------------------------------------------------------------------
    # 训练步骤
    # ------------------------------------------------------------------

    def train_step(self, step: int) -> dict[str, float]:
        """执行一步完整的 GRPO 训练。"""
        start_time = time.time()
        log_gpu_memory(f"step{step}_开始")

        # 1. 采样 prompt batch
        n_prompts = min(self.cfg.train_batch_size, len(self.train_dataset))
        indices = random.sample(range(len(self.train_dataset)), n_prompts)
        prompts = [self.train_dataset[i] for i in indices]

        # 2. Rollout: 每个 prompt 生成 N 个回答
        rollout_items = self._generate_responses(prompts)
        log_gpu_memory(f"step{step}_rollout后")

        # 3. 计算奖励
        rollout_items = self._compute_rewards(rollout_items)

        # 4. 计算 group-relative advantage
        rollout_items = self._compute_advantages(rollout_items)

        # 5. 构建训练 batch，计算 old_log_probs
        batch = self._build_training_batch(rollout_items)
        log_gpu_memory(f"step{step}_batch构建后")

        with torch.no_grad():
            _, old_seq_log_probs = compute_sequence_log_probs(
                self.actor,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_mask"],
            )
            _, ref_seq_log_probs = compute_sequence_log_probs(
                self.reference,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_mask"],
            )

        batch["old_log_probs"] = old_seq_log_probs.detach()
        batch["ref_log_probs"] = ref_seq_log_probs.detach()
        log_gpu_memory(f"step{step}_old+ref计算后")

        # 6. 计算 old_log_probs 的 per-token 版本 (用于 ratio 计算)
        with torch.no_grad():
            old_token_log_probs, _ = compute_sequence_log_probs(
                self.actor,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_mask"],
            )
        batch["old_token_log_probs"] = old_token_log_probs.detach()

        # 7. PPO epoch(s)
        total_items = len(rollout_items)
        mini_batch_size = min(self.cfg.ppo_mini_batch_size, total_items)
        metrics: dict[str, list[float]] = {
            "policy_loss": [], "kl_loss": [], "total_loss": [],
            "approx_kl": [], "clip_frac": [],
        }

        for _ in range(self.cfg.ppo_epochs):
            # 随机打乱
            perm = torch.randperm(total_items).tolist()
            for start in range(0, total_items, mini_batch_size):
                mb_indices = perm[start:start + mini_batch_size]
                mb_metrics = self._train_mini_batch(batch, mb_indices)
                for k, v in mb_metrics.items():
                    if k in metrics:
                        metrics[k].append(v)

        log_gpu_memory(f"step{step}_PPO更新后")

        # 8. 汇总指标
        avg_metrics = {k: _mean(v) for k, v in metrics.items() if v}
        avg_metrics["reward_mean"] = _mean([item["reward"] for item in rollout_items])
        avg_metrics["reward_std"] = float(np.std([item["reward"] for item in rollout_items]))
        avg_metrics["response_len_mean"] = _mean([len(item["response_ids"]) for item in rollout_items])
        avg_metrics["step_time"] = time.time() - start_time
        avg_metrics["step"] = step
        avg_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        self.metrics_history.append(avg_metrics)
        return avg_metrics

    def _train_mini_batch(
        self, batch: dict[str, torch.Tensor], indices: list[int]
    ) -> dict[str, float]:
        """在一个 mini-batch 上执行前向传播和参数更新。"""
        # 取出 mini-batch 数据
        mb_input_ids = batch["input_ids"][indices]
        mb_attention_mask = batch["attention_mask"][indices]
        mb_response_mask = batch["response_mask"][indices]
        mb_advantages = batch["advantages"][indices]
        mb_old_token_log_probs = batch["old_token_log_probs"][indices]
        mb_ref_log_probs = batch["ref_log_probs"][indices]

        # 当前策略的 log probs
        curr_token_log_probs, curr_seq_log_probs = compute_sequence_log_probs(
            self.actor, mb_input_ids, mb_attention_mask, mb_response_mask
        )
        log_gpu_memory("mini_batch_actor前向后")

        # old 序列 log prob
        old_seq_log_probs = (mb_old_token_log_probs * mb_response_mask[:, 1:]).sum(dim=-1)

        # 从 reference 获取 per-token ref log probs
        # mb_ref_log_probs 是序列级别的标量 (batch,)，不能直接扩展成 per-token 版本，
        # 必须重新计算 per-token 的 ref_token_log_probs
        with torch.no_grad():
            ref_token_log_probs, _ = compute_sequence_log_probs(
                self.reference, mb_input_ids, mb_attention_mask, mb_response_mask
            )
        log_gpu_memory("mini_batch_ref前向后_峰值点")

        # PPO ratio
        # per-token ratio: exp(log_prob_current - log_prob_old)
        ratio = torch.exp(curr_token_log_probs - mb_old_token_log_probs)  # (mb, seq-1)

        # Clipped surrogate
        adv_expanded = mb_advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_expanded
        policy_loss_per_token = -torch.min(surr1, surr2)

        # 只在 response token 上计算损失
        resp_mask_shifted = mb_response_mask[:, 1:]
        valid_tokens = resp_mask_shifted.sum()
        if valid_tokens == 0:
            return {"policy_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0}

        policy_loss = (policy_loss_per_token * resp_mask_shifted).sum() / valid_tokens

        # KL loss
        kl_loss = compute_kl_loss(
            curr_token_log_probs, ref_token_log_probs,
            mb_response_mask, self.cfg.kl_loss_type,
        )

        # 总损失
        total_loss = policy_loss + self.cfg.kl_loss_coef * kl_loss

        # 后向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        log_gpu_memory("mini_batch_backward后")
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()
        log_gpu_memory("mini_batch_optimizer后")

        # 诊断指标
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            clip_frac = ((ratio < 1.0 - self.cfg.clip_ratio) | (ratio > 1.0 + self.cfg.clip_ratio)).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
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

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt.prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            output_ids = self.actor.generate(
                **inputs,
                max_new_tokens=self.cfg.max_response_length,
                do_sample=False,  # 验证用贪心解码
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

            prompt_len = inputs.input_ids.shape[-1]
            response_ids = output_ids[0, prompt_len:].tolist()
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            reward_info = compute_gsm8k_rule_reward(
                response=response_text,
                gold_answer=prompt.ground_truth,
            )
            rewards.append(reward_info["score"])
            exact_matches.append(1.0 if reward_info["exact_match"] else 0.0)
            format_oks.append(1.0 if reward_info["format_ok"] else 0.0)
            response_lens.append(len(response_ids))

        self.actor.train()

        return {
            "val_reward_mean": _mean(rewards),
            "val_exact_match": _mean(exact_matches),
            "val_format_rate": _mean(format_oks),
            "val_response_len_mean": _mean(response_lens),
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
        """加载 optimizer/trainer/RNG 状态；缺失时退化为 GRPO LoRA 权重续跑。"""
        if self.resume_checkpoint_dir is None:
            return

        ckpt_dir = self.resume_checkpoint_dir
        self.start_step = self._infer_step_from_checkpoint_name()
        self.last_completed_step = self.start_step - 1

        optimizer_path = ckpt_dir / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = _torch_load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)
            print(f"Optimizer 状态已恢复: {optimizer_path}")
        else:
            print(f"未找到 optimizer 状态: {optimizer_path}，本次只能从模型权重 warm-start。")

        state_path = ckpt_dir / "trainer_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.has_trainer_state = True
            self.last_completed_step = int(state.get("step", self.last_completed_step))
            self.start_step = int(state.get("next_step", self.last_completed_step + 1))
            self.best_val_em = float(state.get("best_val_em", self.best_val_em))
            self.best_step = int(state.get("best_step", self.best_step))
            self.steps_no_improve = int(state.get("steps_no_improve", self.steps_no_improve))
            self.train_reward_history = [float(x) for x in state.get("train_reward_history", [])]
            self.metrics_history = list(state.get("metrics_history", []))
            print(f"Trainer 状态已恢复: {state_path}，下一步从 step {self.start_step} 开始。")
        else:
            print(f"未找到 trainer 状态: {state_path}，将从 step {self.start_step} 近似续跑。")

        self._restore_rng_state(ckpt_dir)

    def _sync_training_state(
        self,
        step: int,
        best_val_em: float,
        best_step: int,
        steps_no_improve: int,
        train_reward_history: list[float],
    ) -> None:
        """同步内存中的训练状态，供 checkpoint 保存使用。"""
        self.last_completed_step = step
        self.best_val_em = best_val_em
        self.best_step = best_step
        self.steps_no_improve = steps_no_improve
        self.train_reward_history = list(train_reward_history)

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self) -> list[dict[str, Any]]:
        """执行完整的 GRPO 训练循环，含早停和异常检测。"""
        cfg = self.cfg
        output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("models/grpo") / cfg.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 配置日志（控制台 + 文件）
        setup_logging(output_dir, cfg.run_name)
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"训练 prompt 数: {len(self.train_dataset)}")
        logger.info(f"每步 prompt 数: {cfg.train_batch_size}")
        logger.info(f"每 prompt 回答数: {cfg.rollout_n}")
        logger.info(f"最大训练步数: {cfg.total_training_steps}")
        logger.info(f"早停耐心: {cfg.max_steps_no_improve} 步 (验证 EM 连续不改善)")
        logger.info(f"KL 阈值: {cfg.kl_threshold}")
        logger.info(f"Reward hacking 检测: {cfg.reward_hacking_detect} (窗口={cfg.reward_hacking_window})")
        if self.resume_checkpoint_dir is not None:
            logger.info(f"续训 checkpoint: {self.resume_checkpoint_dir}")
            logger.info(f"续训起始 step: {self.start_step}")
            logger.info(f"完整 trainer state: {self.has_trainer_state}")

        # 初始化绘图 CSV 文件（实时追加，每步/每评估立即写入）
        csv_dir = output_dir / "plots"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self._train_csv_path = csv_dir / "train_metrics.csv"
        self._val_csv_path = csv_dir / "val_metrics.csv"
        self._gpu_csv_path = csv_dir / "gpu_memory.csv"
        self._train_csv_writer = _init_csv(self._train_csv_path, TRAIN_CSV_COLUMNS)
        self._val_csv_writer = _init_csv(self._val_csv_path, VAL_CSV_COLUMNS)
        self._gpu_csv_writer = _init_csv(self._gpu_csv_path, GPU_CSV_COLUMNS)

        # 显存: 训练前的初始状态（含模型权重等各组件拆解）
        self._log_gpu_memory_detailed("训练前_初始", batch=None)
        self._append_gpu_csv(-1, "训练前_初始")

        # 训练前验证，记录 baseline。完整 resume 时沿用 checkpoint 里的状态。
        best_val_em = self.best_val_em  # 最佳验证 exact_match
        best_step = self.best_step
        steps_no_improve = self.steps_no_improve  # EM 连续不改善的步数
        train_reward_history: list[float] = list(self.train_reward_history)  # 训练 reward 趋势

        should_run_initial_val = cfg.val_before_train and not self.has_trainer_state
        if should_run_initial_val:
            val_metrics = self._validate()
            val_metrics["step"] = "initial" if self.start_step == 0 else f"resume_{self.start_step - 1}"
            log_prefix = "[初始验证]" if self.start_step == 0 else "[续训前验证]"
            logger.info(f"{log_prefix} {_format_val_metrics(val_metrics)}")
            self.metrics_history.append(val_metrics)
            best_val_em = val_metrics.get("val_exact_match", 0.0)
            logger.info(f"初始 baseline: val_em={best_val_em:.3f}")
            self._append_val_csv(self.start_step - 1, val_metrics, best_val_em)
            log_gpu_memory("初始验证后")

        stop_reason = "达到最大步数"
        last_step = self.start_step - 1
        for step in range(self.start_step, cfg.total_training_steps):
            last_step = step
            step_metrics = self.train_step(step)

            # 记录训练 reward 用于 hacking 检测
            train_reward_history.append(step_metrics["reward_mean"])

            # 写入训练 CSV
            self._append_train_csv(step, step_metrics, best_val_em, steps_no_improve)
            self._sync_training_state(step, best_val_em, best_step, steps_no_improve, train_reward_history)

            # 打印日志（含早停追踪状态）
            if step % cfg.log_steps == 0:
                logger.info(
                    f"[step {step}/{cfg.total_training_steps}] "
                    f"reward={step_metrics['reward_mean']:.3f}±{step_metrics['reward_std']:.3f} "
                    f"policy_loss={step_metrics['policy_loss']:.4f} "
                    f"kl={step_metrics['kl_loss']:.4f} "
                    f"approx_kl={step_metrics['approx_kl']:.4f} "
                    f"clip_frac={step_metrics['clip_frac']:.3f} "
                    f"resp_len={step_metrics['response_len_mean']:.0f} "
                    f"lr={step_metrics['lr']:.2e} "
                    f"best_em={best_val_em:.3f}(step={best_step}) "
                    f"no_improve={steps_no_improve}/{cfg.max_steps_no_improve} "
                    f"time={step_metrics['step_time']:.1f}s"
                )

            # KL 异常检测
            if abs(step_metrics["approx_kl"]) > cfg.kl_threshold:
                logger.warning(
                    f"[异常终止] approx_kl={step_metrics['approx_kl']:.4f} "
                    f"超过阈值 {cfg.kl_threshold} → actor 偏离 reference 太远，停止训练"
                )
                stop_reason = f"KL异常: approx_kl={step_metrics['approx_kl']:.4f}"
                self._save_checkpoint(output_dir, step)
                break

            # 保存 checkpoint
            if (step + 1) % cfg.save_freq == 0 or step == cfg.total_training_steps - 1:
                self._save_checkpoint(output_dir, step)

            # 评估 + 早停 + reward hacking 检测
            if (step + 1) % cfg.eval_freq == 0:
                val_metrics = self._validate()
                val_metrics["step"] = step
                logger.info(f"[eval step {step}] {_format_val_metrics(val_metrics)}")
                self.metrics_history.append(val_metrics)

                val_em = val_metrics.get("val_exact_match", 0.0)
                val_fmt = val_metrics.get("val_format_rate", 0.0)
                val_reward = val_metrics.get("val_reward_mean", 0.0)

                # 写入验证 CSV
                self._append_val_csv(step, val_metrics, best_val_em)

                # 早停: 验证 EM 连续不改善
                if val_em > best_val_em:
                    best_val_em = val_em
                    best_step = step
                    steps_no_improve = 0
                    logger.info(f"[早停追踪] val_em 改善: {val_em:.3f} (最佳 step={best_step})")
                else:
                    steps_no_improve += cfg.eval_freq
                    logger.info(
                        f"[早停追踪] val_em 未改善: {val_em:.3f} "
                        f"(最佳={best_val_em:.3f} at step={best_step}) "
                        f"已连续 {steps_no_improve} 步"
                    )
                self._sync_training_state(step, best_val_em, best_step, steps_no_improve, train_reward_history)

                if steps_no_improve >= cfg.max_steps_no_improve:
                    logger.info(
                        f"[早停终止] val_em 连续 {steps_no_improve} 步未改善 "
                        f"(最佳={best_val_em:.3f} at step={best_step})"
                    )
                    stop_reason = f"早停: val_em连续{steps_no_improve}步未改善"
                    break

                # Reward hacking 检测: 训练 reward 上升但验证 EM 下降
                if cfg.reward_hacking_detect and len(train_reward_history) >= cfg.reward_hacking_window:
                    recent_reward = _mean(train_reward_history[-cfg.reward_hacking_window:])
                    earlier_reward = _mean(
                        train_reward_history[-2 * cfg.reward_hacking_window:-cfg.reward_hacking_window]
                        if len(train_reward_history) >= 2 * cfg.reward_hacking_window
                        else train_reward_history[:cfg.reward_hacking_window]
                    )
                    # reward 上升但 EM 不升（甚至降）
                    if recent_reward > earlier_reward and val_em <= best_val_em:
                        logger.warning(
                            f"[reward hacking 警告] 训练 reward 从 {earlier_reward:.3f} "
                            f"升至 {recent_reward:.3f}, 但 val_em={val_em:.3f} "
                            f"未改善 (最佳={best_val_em:.3f}) → 模型可能在钻 RM/规则漏洞"
                        )

                # 格式退化检测: format_rate 降到 0 → 模型完全不输出正确格式
                if val_fmt < 0.1 and best_val_em > 0:
                    logger.warning(
                        f"[格式退化] val_fmt={val_fmt:.3f} 极低, 模型丧失格式能力, 停止训练"
                    )
                    stop_reason = f"格式退化: val_fmt={val_fmt:.3f}"
                    break

        # 最终保存
        if self.start_step >= cfg.total_training_steps:
            stop_reason = f"起始 step {self.start_step} 已达到最大训练步数 {cfg.total_training_steps}"
        final_step = last_step if last_step >= 0 else 0
        self._sync_training_state(final_step, best_val_em, best_step, steps_no_improve, train_reward_history)
        logger.info(f"训练结束, 原因: {stop_reason}")
        logger.info(f"最佳 val_em={best_val_em:.3f} (step={best_step})")
        self._save_checkpoint(output_dir, final_step)
        self._save_metrics(output_dir)
        self._append_gpu_csv(final_step, "训练结束")
        self._log_gpu_memory_detailed("训练结束", batch=None)

        # 关闭 CSV 文件
        self._close_csv_files()

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
            "best_val_em": self.best_val_em,
            "best_step": self.best_step,
            "steps_no_improve": self.steps_no_improve,
            "train_reward_history": self.train_reward_history,
            "metrics_history": self.metrics_history,
            "run_name": self.cfg.run_name,
            "total_training_steps": self.cfg.total_training_steps,
            "save_freq": self.cfg.save_freq,
            "eval_freq": self.cfg.eval_freq,
            "base_model_dir": self.cfg.base_model_dir,
            "sft_adapter_dir": self.cfg.sft_adapter_dir,
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
            "best_val_em_so_far": round(best_val_em, 4),
        }
        _append_csv_row(self._val_csv_writer, row, VAL_CSV_COLUMNS)

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

    def _close_csv_files(self) -> None:
        """关闭所有 CSV 文件。"""
        for writer in [self._train_csv_writer, self._val_csv_writer, self._gpu_csv_writer]:
            if writer is not None and hasattr(writer, "fh"):
                writer.fh.flush()
                writer.fh.close()


def _format_val_metrics(m: dict[str, Any]) -> str:
    """格式化验证指标为字符串。"""
    return (
        f"reward={m.get('val_reward_mean', 0):.3f} "
        f"em={m.get('val_exact_match', 0):.3f} "
        f"fmt={m.get('val_format_rate', 0):.3f} "
        f"len={m.get('val_response_len_mean', 0):.0f}"
    )
