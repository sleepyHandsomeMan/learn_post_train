"""PPO (Proximal Policy Optimization) 训练器。

不依赖 verl 框架，完全自包含的 PPO 实现。
与项目已有的数据处理和奖励函数对接。

GRPO vs PPO：
  - GRPO: 组内相对优势 (group-relative), 无需 critic, 简单高效
  - PPO: GAE (Generalized Advantage Estimation) + critic, 需要 value head

算法概要：
  1. 对每个 prompt 采样回答 (rollout)
  2. 用 critic (value head) 估计每个 token 的价值
  3. 用 GAE 计算 token-level advantage
  4. PPO clipped surrogate loss + value loss + entropy bonus + KL penalty
  5. 梯度下降更新 actor 和 critic
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import extract_final_answer
from .prompting import DEFAULT_FORMAT_INSTRUCTION
from .reward import GSM8KRewardConfig, compute_gsm8k_rule_reward
from .rl_dataset import RLPrompt, RLPromptDataset, load_rl_dataset

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logger = logging.getLogger("ptf.ppo")


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

# PPO 训练指标 CSV 列定义 (每步一行)
PPO_TRAIN_CSV_COLUMNS = [
    "step", "reward_mean", "policy_loss", "value_loss", "entropy",
    "kl_loss", "approx_kl", "clip_frac", "response_len_mean",
    "actor_lr", "critic_lr", "step_time", "best_val_em", "steps_no_improve",
]

# 验证指标 CSV 列定义 (每 eval_freq 步一行)
PPO_VAL_CSV_COLUMNS = [
    "step", "val_reward_mean", "val_exact_match", "val_format_rate",
    "val_response_len_mean", "best_val_em_so_far",
]

# 显存指标 CSV 列定义
PPO_GPU_CSV_COLUMNS = [
    "step", "tag", "allocated_gb", "reserved_gb", "pool_free_gb",
    "segment_count", "peak_allocated_gb",
]


def _init_csv(path: Path, columns: list[str]) -> Any:
    """初始化 CSV 文件并返回 writer。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists()
    fh = open(str(path), "a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    if need_header:
        writer.writerow(columns)
        fh.flush()
    return writer


def _append_csv_row(writer: Any, row: dict[str, Any], columns: list[str]) -> None:
    """按列顺序写入一行 CSV。"""
    values = [row.get(col, "") for col in columns]
    writer.writerow(values)
    if hasattr(writer, "stream"):
        writer.stream.flush()


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
    """打印 CUDA 内存分配器的详细状态。

    输出 allocated (真实数据)、reserved (分配器持有)、
    分配器内部账本（段数、段内空闲），以及 nvidia-smi 总量。
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    pool_free = reserved - allocated

    # 分配器内部账本
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


def log_gpu_memory_detailed(tag: str) -> None:
    """打印显存的逐项拆解（更详细的版本）。"""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    stats = torch.cuda.memory_stats()
    active_alloc_size = stats.get("allocated_bytes.all.current", 0)
    segment_size = stats.get("reserved_bytes.all.current", 0)
    active_alloc_count = stats.get("allocation.all.current", 0)
    segment_count = stats.get("segment.all.current", 0)

    # 历史峰值
    peak_allocated = stats.get("allocated_bytes.all.peak", 0)
    peak_reserved = stats.get("reserved_bytes.all.peak", 0)

    # 按大小分类的分配数
    small_alloc_count = stats.get("allocation.all.small.current", 0)
    large_alloc_count = stats.get("allocation.all.large.current", 0)

    msg_lines = [
        f"[显存详细:{tag}]",
        f"  allocated (真实数据): {_format_bytes(allocated)} | 峰值: {_format_bytes(peak_allocated)}",
        f"  reserved  (分配器持有): {_format_bytes(reserved)} | 峰值: {_format_bytes(peak_reserved)}",
        f"  预留池 (段内空闲): {_format_bytes(reserved - allocated)}",
        f"  活跃分配块: {active_alloc_count} (小={small_alloc_count}, 大={large_alloc_count})",
        f"  内存段数: {segment_count} | 每段均: {_format_bytes(int(segment_size / max(segment_count, 1)))}",
        f"  VRAM总量(nvidia): {_format_bytes(torch.cuda.get_device_properties(0).total_memory)}",
    ]
    logger.info("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """PPO 训练的超参数。"""

    # 模型
    base_model_dir: str = ""
    sft_adapter_dir: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    critic_hidden_size: int = 1024  # value head 隐藏层大小

    # 数据
    train_file: str = ""
    eval_file: str = ""
    max_prompt_length: int = 512
    max_response_length: int = 512
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION
    enable_thinking: bool = False

    # rollout
    rollout_n: int = 1  # PPO 通常每 prompt 采样 1 个回答
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50

    # 训练
    train_batch_size: int = 8
    ppo_epochs: int = 4  # PPO 通常多轮使用同批数据
    ppo_mini_batch_size: int = 4
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-5  # critic lr 通常比 actor 高
    gamma: float = 1.0  # 折扣因子
    lam: float = 0.95  # GAE lambda
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2  # value function clipping
    entropy_coef: float = 0.01
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
    run_name: str = "ppo_run"
    log_steps: int = 1

    # 验证
    val_before_train: bool = True
    val_max_items: int = 20


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_eos_token_id(tokenizer: Any) -> int | None:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Value Head (critic)
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """从 transformer 最后一层 hidden state 预测标量 value。

    输入: (batch, seq_len, hidden_size) — 模型最后一层 hidden states
    输出: (batch, seq_len) — 每个 token 位置的 value 估计
    """

    def __init__(self, hidden_size: int, intermediate_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (batch, seq_len, hidden_size) → values: (batch, seq_len)"""
        x = F.relu(self.fc1(hidden_states))
        return self.fc2(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


def load_actor_critic_and_reference(
    base_model_dir: str | Path,
    sft_adapter_dir: str | Path | None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    critic_hidden_size: int = 1024,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    device: torch.device | None = None,
) -> tuple[Any, ValueHead, Any, Any]:
    """加载 PPO 训练的 actor、critic 和 reference 模型。

    Returns:
        (actor, critic, reference, tokenizer)
    """
    try:
        from peft import LoraConfig, TaskType, PeftModel, get_peft_model
    except ImportError:
        raise ImportError("PPO 训练需要安装 peft。")

    base_model_dir = Path(base_model_dir)
    dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. base model
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    # 3. 合并 SFT LoRA
    if sft_adapter_dir is not None and Path(sft_adapter_dir).exists():
        sft_adapter_dir = Path(sft_adapter_dir)
        model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir))
        logger.info(f"加载 SFT adapter 自: {sft_adapter_dir}")
        merged_model = model.merge_and_unload()
        logger.info("SFT adapter 已合并。")
    else:
        merged_model = base_model
        logger.info("未找到 SFT adapter，直接从 base model 开始。")

    # 4. reference (冻结)
    reference_model = copy.deepcopy(merged_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # 5. actor: 加新 LoRA
    if device is not None and torch.cuda.is_available():
        merged_model = merged_model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    actor = get_peft_model(merged_model, lora_config)
    if gradient_checkpointing:
        actor.gradient_checkpointing_enable()
    actor.train()
    actor.print_trainable_parameters()

    # 6. critic (value head)
    hidden_size = actor.config.hidden_size
    critic = ValueHead(hidden_size, intermediate_size=critic_hidden_size)
    if device is not None:
        critic = critic.to(device)
    critic.train()

    return actor, critic, reference_model, tokenizer


# ---------------------------------------------------------------------------
# Log prob + value 计算
# ---------------------------------------------------------------------------


def compute_log_probs_and_values(
    model: Any,
    critic: ValueHead | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """同时计算 token-level log probs 和 values。

    Returns:
        token_log_probs: (batch, seq_len-1)
        values: (batch, seq_len) or None (如果 critic 为 None)
    """
    # 需要 output_hidden_states 来给 critic 提供输入
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=(critic is not None),
    )
    logits = outputs.logits

    # log probs
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # values
    if critic is not None and outputs.hidden_states is not None:
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        values = critic(last_hidden)  # (batch, seq_len)
    else:
        values = None

    return token_log_probs, values


def compute_kl_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: str = "low_var_kl",
) -> torch.Tensor:
    """计算 KL 散度。"""
    # 对齐 mask
    if mask.shape[1] == log_probs.shape[1] + 1:
        mask = mask[:, 1:]
    elif mask.shape[1] == log_probs.shape[1] - 1:
        mask = F.pad(mask, (0, 1), value=0)

    if kl_type == "low_var_kl":
        log_ratio = ref_log_probs - log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        kl = log_probs - ref_log_probs

    valid = mask.sum()
    if valid == 0:
        return torch.tensor(0.0, device=log_probs.device)
    return (kl * mask).sum() / valid


# ---------------------------------------------------------------------------
# GAE 计算
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """计算 GAE advantage 和 returns。

    Args:
        rewards: 每个时间步的奖励 (token-level, 中间步为 0, 最后一步为 reward)
        values: critic 对每个时间步的 value 估计
        gamma: 折扣因子
        lam: GAE lambda

    Returns:
        advantages: GAE 优势
        returns: advantage + value (用于 value loss target)
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0

    # 逆向遍历计算 GAE
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """自包含 PPO 训练器。"""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 加载模型 (actor + critic + reference)
        sft_adapter = cfg.sft_adapter_dir if cfg.sft_adapter_dir else None
        self.actor, self.critic, self.reference, self.tokenizer = load_actor_critic_and_reference(
            base_model_dir=cfg.base_model_dir,
            sft_adapter_dir=sft_adapter,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            critic_hidden_size=cfg.critic_hidden_size,
            fp16=cfg.fp16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            device=self.device,
        )

        # 优化器
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = torch.optim.AdamW(actor_params, lr=cfg.actor_learning_rate)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_learning_rate)

        self.reward_config = GSM8KRewardConfig()
        self._eos_token_id = _build_eos_token_id(self.tokenizer)
        self._pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # 数据
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

        self.metrics_history: list[dict[str, Any]] = []

        # 绘图数据 CSV
        self._train_csv_writer: Any = None
        self._val_csv_writer: Any = None
        self._gpu_csv_writer: Any = None

        # 显存: 模型加载后的初始状态
        log_gpu_memory("模型加载后")
        log_gpu_memory_detailed("模型加载后")

    # ------------------------------------------------------------------
    # Rollout 生成
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_rollouts(
        self, prompts: list[RLPrompt]
    ) -> list[dict[str, Any]]:
        """对每个 prompt 生成回答，并记录 old log probs 和 values。"""
        rollout_items: list[dict[str, Any]] = []
        self.actor.eval()
        self.critic.eval()

        for prompt in prompts:
            for i in range(self.cfg.rollout_n):
                inputs = self.tokenizer(
                    prompt.prompt_text, return_tensors="pt", add_special_tokens=False
                ).to(self.device)

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

                # 计算 old log probs 和 values
                full_ids = torch.cat([inputs.input_ids, output_ids[0, prompt_len:].unsqueeze(0)], dim=1)
                attn_mask = torch.ones_like(full_ids)
                resp_mask = torch.zeros_like(full_ids, dtype=torch.float)
                resp_mask[0, prompt_len:] = 1.0

                old_token_log_probs, old_values = compute_log_probs_and_values(
                    self.actor, self.critic, full_ids, attn_mask
                )

                rollout_items.append({
                    "prompt": prompt,
                    "response_text": response_text,
                    "response_ids": response_ids,
                    "prompt_len": prompt_len,
                    "gen_index": i,
                    "full_ids": full_ids,
                    "resp_mask": resp_mask,
                    "old_token_log_probs": old_token_log_probs,
                    "old_values": old_values,
                })

        self.actor.train()
        self.critic.train()
        return rollout_items

    # ------------------------------------------------------------------
    # 奖励 + 优势计算
    # ------------------------------------------------------------------

    def _compute_rewards_and_advantages(
        self, rollout_items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """为每个 rollout item 计算规则奖励和 GAE 优势。"""
        for item in rollout_items:
            # 规则奖励
            reward_info = compute_gsm8k_rule_reward(
                response=item["response_text"],
                gold_answer=item["prompt"].ground_truth,
                config=self.reward_config,
            )
            item["reward"] = reward_info["score"]

            # 构建 token-level rewards: 只在最后一个 response token 给奖励
            response_len = len(item["response_ids"])
            token_rewards = [0.0] * (response_len - 1) + [reward_info["score"]] if response_len > 0 else [reward_info["score"]]

            # 取 response 部分的 values
            prompt_len = item["prompt_len"]
            if item["old_values"] is not None:
                response_values = item["old_values"][0, prompt_len:].tolist()
                if len(response_values) >= response_len:
                    response_values = response_values[:response_len]
                else:
                    response_values = response_values + [0.0] * (response_len - len(response_values))
            else:
                response_values = [0.0] * response_len

            # GAE
            advantages, returns = compute_gae(
                rewards=token_rewards,
                values=response_values,
                gamma=self.cfg.gamma,
                lam=self.cfg.lam,
            )
            item["token_rewards"] = token_rewards
            item["advantages"] = advantages
            item["returns"] = returns

        return rollout_items

    # ------------------------------------------------------------------
    # 训练步骤
    # ------------------------------------------------------------------

    def train_step(self, step: int) -> dict[str, float]:
        start_time = time.time()
        log_gpu_memory(f"step{step}_开始")

        # 1. 采样 prompt
        n_prompts = min(self.cfg.train_batch_size, len(self.train_dataset))
        indices = random.sample(range(len(self.train_dataset)), n_prompts)
        prompts = [self.train_dataset[i] for i in indices]

        # 2. Rollout
        rollout_items = self._generate_rollouts(prompts)
        log_gpu_memory(f"step{step}_rollout后")

        # 3. 奖励 + GAE
        rollout_items = self._compute_rewards_and_advantages(rollout_items)

        # 4. 收集所有训练 token 到统一的 batch
        all_input_ids: list[torch.Tensor] = []
        all_attn_masks: list[torch.Tensor] = []
        all_resp_masks: list[torch.Tensor] = []
        all_old_log_probs: list[torch.Tensor] = []
        all_advantages: list[float] = []
        all_token_advantages: list[list[float]] = []
        all_returns: list[list[float]] = []

        for item in rollout_items:
            all_input_ids.append(item["full_ids"].squeeze(0))
            all_attn_masks.append(torch.ones(item["full_ids"].shape[1], dtype=torch.long))
            all_resp_masks.append(item["resp_mask"].squeeze(0))
            all_old_log_probs.append(item["old_token_log_probs"].squeeze(0))
            all_advantages.append(item["reward"])
            all_token_advantages.append(item["advantages"])
            all_returns.append(item["returns"])

        # Padding
        max_len = max(t.shape[0] for t in all_input_ids)
        pad_id = self._pad_token_id

        batch_input_ids = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        batch_attn_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.long)
        batch_resp_mask = torch.zeros((len(all_input_ids), max_len), dtype=torch.float)

        for i in range(len(all_input_ids)):
            seq_len = all_input_ids[i].shape[0]
            batch_input_ids[i, :seq_len] = all_input_ids[i]
            batch_attn_mask[i, :seq_len] = all_attn_masks[i]
            batch_resp_mask[i, :seq_len] = all_resp_masks[i]

        batch_input_ids = batch_input_ids.to(self.device)
        batch_attn_mask = batch_attn_mask.to(self.device)
        batch_resp_mask = batch_resp_mask.to(self.device)
        log_gpu_memory(f"step{step}_batch构建后")

        # 5. PPO epochs
        total_items = len(rollout_items)
        mini_batch_size = min(self.cfg.ppo_mini_batch_size, total_items)
        metrics_tracker: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "kl_loss": [], "total_loss": [],
        }

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(total_items).tolist()
            for start in range(0, total_items, mini_batch_size):
                mb_indices = perm[start:start + mini_batch_size]
                mb_metrics = self._train_mini_batch(
                    batch_input_ids, batch_attn_mask, batch_resp_mask,
                    all_old_log_probs, all_token_advantages, all_returns,
                    mb_indices,
                )
                for k, v in mb_metrics.items():
                    if k in metrics_tracker:
                        metrics_tracker[k].append(v)

        log_gpu_memory(f"step{step}_PPO更新后")

        avg_metrics = {k: _mean(v) for k, v in metrics_tracker.items() if v}
        avg_metrics["reward_mean"] = _mean([item["reward"] for item in rollout_items])
        avg_metrics["response_len_mean"] = _mean([len(item["response_ids"]) for item in rollout_items])
        avg_metrics["step_time"] = time.time() - start_time
        avg_metrics["step"] = step
        avg_metrics["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
        avg_metrics["critic_lr"] = self.critic_optimizer.param_groups[0]["lr"]

        self.metrics_history.append(avg_metrics)
        return avg_metrics

    def _train_mini_batch(
        self,
        batch_input_ids: torch.Tensor,
        batch_attn_mask: torch.Tensor,
        batch_resp_mask: torch.Tensor,
        all_old_log_probs: list[torch.Tensor],
        all_token_advantages: list[list[float]],
        all_returns: list[list[float]],
        mb_indices: list[int],
    ) -> dict[str, float]:
        """在一个 mini-batch 上训练 actor 和 critic。"""
        mb_ids = batch_input_ids[mb_indices]
        mb_attn = batch_attn_mask[mb_indices]
        mb_resp = batch_resp_mask[mb_indices]

        # 当前策略的 log probs + values
        curr_token_log_probs, curr_values = compute_log_probs_and_values(
            self.actor, self.critic, mb_ids, mb_attn,
        )
        log_gpu_memory("mini_batch_actor+critic前向后")

        # 构建 per-token advantages 和 returns tensor
        max_resp_len = mb_resp.shape[1]
        token_adv = torch.zeros((len(mb_indices), max_resp_len), device=self.device)
        token_ret = torch.zeros((len(mb_indices), max_resp_len), device=self.device)

        for bi, idx in enumerate(mb_indices):
            response_mask = mb_resp[bi].bool()
            resp_positions = response_mask.nonzero(as_tuple=True)[0]
            advantages = all_token_advantages[idx]
            returns = all_returns[idx]
            for pi, pos in enumerate(resp_positions.tolist()):
                if pi < len(advantages):
                    token_adv[bi, pos] = advantages[pi]
                    token_ret[bi, pos] = returns[pi]

        # --- Actor loss (PPO clipped) ---
        old_token_log_probs_mb = torch.stack(
            [all_old_log_probs[idx][:max_resp_len - 1] for idx in mb_indices]
        ).to(self.device)

        # 对齐形状
        min_log_prob_len = min(curr_token_log_probs.shape[1], old_token_log_probs_mb.shape[1])
        curr_lp = curr_token_log_probs[:, :min_log_prob_len]
        old_lp = old_token_log_probs_mb[:, :min_log_prob_len]
        resp_mask_shifted = mb_resp[:, 1:min_log_prob_len + 1]

        ratio = torch.exp(curr_lp - old_lp)
        adv_for_loss = token_adv[:, 1:min_log_prob_len + 1]

        surr1 = ratio * adv_for_loss
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_for_loss
        policy_loss_per_token = -torch.min(surr1, surr2)

        valid_actor_tokens = resp_mask_shifted.sum()
        if valid_actor_tokens > 0:
            policy_loss = (policy_loss_per_token * resp_mask_shifted).sum() / valid_actor_tokens
        else:
            policy_loss = torch.tensor(0.0, device=self.device)

        # --- Value loss (clipped) ---
        if curr_values is not None and valid_actor_tokens > 0:
            v_pred = curr_values[:, :min_log_prob_len + 1]
            v_target = token_ret[:, :min_log_prob_len + 1]

            v_pred_clipped = v_pred  # 简化 (完整实现需要 old_values)
            value_loss_per_token = F.mse_loss(v_pred, v_target, reduction="none")

            valid_value_tokens = mb_resp[:, :min_log_prob_len + 1].sum()
            if valid_value_tokens > 0:
                value_loss = (value_loss_per_token * mb_resp[:, :min_log_prob_len + 1]).sum() / valid_value_tokens
            else:
                value_loss = torch.tensor(0.0, device=self.device)
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        # --- Entropy bonus ---
        if valid_actor_tokens > 0:
            outputs = self.actor(input_ids=mb_ids, attention_mask=mb_attn)
            logits = outputs.logits[:, :-1, :]
            probs = F.softmax(logits[:, :min_log_prob_len], dim=-1)
            log_probs_dist = F.log_softmax(logits[:, :min_log_prob_len], dim=-1)
            entropy_per_token = -(probs * log_probs_dist).sum(dim=-1)
            entropy = (entropy_per_token * resp_mask_shifted).sum() / valid_actor_tokens
        else:
            entropy = torch.tensor(0.0, device=self.device)

        # --- KL loss ---
        with torch.no_grad():
            ref_token_log_probs, _ = compute_log_probs_and_values(
                self.reference, None, mb_ids, mb_attn,
            )
        log_gpu_memory("mini_batch_ref前向后_峰值点")

        ref_lp = ref_token_log_probs[:, :min_log_prob_len]
        kl_loss = compute_kl_loss(curr_lp, ref_lp, mb_resp, self.cfg.kl_loss_type)

        # --- 总损失和优化 ---
        total_loss = policy_loss + value_loss - self.cfg.entropy_coef * entropy + self.cfg.kl_loss_coef * kl_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        log_gpu_memory("mini_batch_backward后")

        # 梯度裁剪
        actor_grad_params = [p for p in self.actor.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(actor_grad_params, self.cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        log_gpu_memory("mini_batch_optimizer后")

        # 诊断指标
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            clip_frac = ((ratio < 1.0 - self.cfg.clip_ratio) | (ratio > 1.0 + self.cfg.clip_ratio)).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
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
        eval_items = min(self.cfg.val_max_items, len(self.eval_dataset))
        if eval_items == 0:
            return {}

        self.actor.eval()
        prompts = [self.eval_dataset[i] for i in range(eval_items)]

        rewards, exact_matches, format_oks, response_lens = [], [], [], []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt.prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            output_ids = self.actor.generate(
                **inputs,
                max_new_tokens=self.cfg.max_response_length,
                do_sample=False,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

            prompt_len = inputs.input_ids.shape[-1]
            response_ids = output_ids[0, prompt_len:].tolist()
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            reward_info = compute_gsm8k_rule_reward(
                response=response_text, gold_answer=prompt.ground_truth,
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
    # 训练主循环
    # ------------------------------------------------------------------

    def train(self) -> list[dict[str, Any]]:
        """执行完整的 PPO 训练循环，含早停和异常检测。"""
        cfg = self.cfg
        output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("models/ppo") / cfg.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 配置日志（控制台 + 文件）
        setup_logging(output_dir, cfg.run_name)
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"训练 prompt 数: {len(self.train_dataset)}")
        logger.info(f"最大训练步数: {cfg.total_training_steps}")
        logger.info(f"早停耐心: {cfg.max_steps_no_improve} 步 (验证 EM 连续不改善)")
        logger.info(f"KL 阈值: {cfg.kl_threshold}")
        logger.info(f"Reward hacking 检测: {cfg.reward_hacking_detect} (窗口={cfg.reward_hacking_window})")

        # 初始化绘图 CSV 文件（实时追加）
        csv_dir = output_dir / "plots"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self._train_csv_writer = _init_csv(csv_dir / "train_metrics.csv", PPO_TRAIN_CSV_COLUMNS)
        self._val_csv_writer = _init_csv(csv_dir / "val_metrics.csv", PPO_VAL_CSV_COLUMNS)
        self._gpu_csv_writer = _init_csv(csv_dir / "gpu_memory.csv", PPO_GPU_CSV_COLUMNS)

        # 显存: 训练前的初始状态（含模型权重）
        log_gpu_memory_detailed("训练前_初始")
        self._append_gpu_csv(-1, "训练前_初始")

        # 训练前验证，记录 baseline
        best_val_em = -1.0
        best_step = -1
        steps_no_improve = 0
        train_reward_history: list[float] = []

        if cfg.val_before_train:
            val_metrics = self._validate()
            val_metrics["step"] = "initial"
            logger.info(f"[初始验证] reward={val_metrics.get('val_reward_mean', 0):.3f} "
                         f"em={val_metrics.get('val_exact_match', 0):.3f} "
                         f"fmt={val_metrics.get('val_format_rate', 0):.3f}")
            self.metrics_history.append(val_metrics)
            best_val_em = val_metrics.get("val_exact_match", 0.0)
            logger.info(f"初始 baseline: val_em={best_val_em:.3f}")
            self._append_val_csv(-1, val_metrics, best_val_em)
            log_gpu_memory("初始验证后")

        stop_reason = "达到最大步数"
        for step in range(cfg.total_training_steps):
            step_metrics = self.train_step(step)

            # 记录训练 reward 用于 hacking 检测
            train_reward_history.append(step_metrics["reward_mean"])

            # 写入训练 CSV
            self._append_train_csv(step, step_metrics, best_val_em, steps_no_improve)

            # 打印日志（含早停追踪状态和 PPO 特有指标）
            if step % cfg.log_steps == 0:
                clip_frac = step_metrics.get("clip_frac", 0.0)
                approx_kl = step_metrics.get("approx_kl", 0.0)
                logger.info(
                    f"[step {step}/{cfg.total_training_steps}] "
                    f"reward={step_metrics['reward_mean']:.3f} "
                    f"policy_loss={step_metrics['policy_loss']:.4f} "
                    f"value_loss={step_metrics['value_loss']:.4f} "
                    f"entropy={step_metrics['entropy']:.4f} "
                    f"kl={step_metrics['kl_loss']:.4f} "
                    f"clip_frac={clip_frac:.3f} "
                    f"approx_kl={approx_kl:.4f} "
                    f"resp_len={step_metrics['response_len_mean']:.0f} "
                    f"actor_lr={step_metrics.get('actor_lr', 0):.2e} "
                    f"critic_lr={step_metrics.get('critic_lr', 0):.2e} "
                    f"best_em={best_val_em:.3f}(step={best_step}) "
                    f"no_improve={steps_no_improve}/{cfg.max_steps_no_improve} "
                    f"time={step_metrics['step_time']:.1f}s"
                )

            # KL 异常检测
            approx_kl = step_metrics.get("approx_kl", 0.0)
            if abs(approx_kl) > cfg.kl_threshold:
                logger.warning(
                    f"[异常终止] approx_kl={approx_kl:.4f} "
                    f"超过阈值 {cfg.kl_threshold} → actor 偏离 reference 太远，停止训练"
                )
                stop_reason = f"KL异常: approx_kl={approx_kl:.4f}"
                self._save_checkpoint(output_dir, step)
                break

            # 保存 checkpoint
            if (step + 1) % cfg.save_freq == 0 or step == cfg.total_training_steps - 1:
                self._save_checkpoint(output_dir, step)

            # 评估 + 早停 + reward hacking 检测
            if (step + 1) % cfg.eval_freq == 0:
                val_metrics = self._validate()
                val_metrics["step"] = step
                logger.info(
                    f"[eval step {step}] "
                    f"reward={val_metrics.get('val_reward_mean', 0):.3f} "
                    f"em={val_metrics.get('val_exact_match', 0):.3f} "
                    f"fmt={val_metrics.get('val_format_rate', 0):.3f} "
                    f"len={val_metrics.get('val_response_len_mean', 0):.0f}"
                )
                self.metrics_history.append(val_metrics)

                val_em = val_metrics.get("val_exact_match", 0.0)
                val_fmt = val_metrics.get("val_format_rate", 0.0)

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

                if steps_no_improve >= cfg.max_steps_no_improve:
                    logger.info(
                        f"[早停终止] val_em 连续 {steps_no_improve} 步未改善 "
                        f"(最佳={best_val_em:.3f} at step={best_step})"
                    )
                    stop_reason = f"早停: val_em连续{steps_no_improve}步未改善"
                    break

                # Reward hacking 检测
                if cfg.reward_hacking_detect and len(train_reward_history) >= cfg.reward_hacking_window:
                    recent_reward = _mean(train_reward_history[-cfg.reward_hacking_window:])
                    earlier_reward = _mean(
                        train_reward_history[-2 * cfg.reward_hacking_window:-cfg.reward_hacking_window]
                        if len(train_reward_history) >= 2 * cfg.reward_hacking_window
                        else train_reward_history[:cfg.reward_hacking_window]
                    )
                    if recent_reward > earlier_reward and val_em <= best_val_em:
                        logger.warning(
                            f"[reward hacking 警告] 训练 reward 从 {earlier_reward:.3f} "
                            f"升至 {recent_reward:.3f}, 但 val_em={val_em:.3f} "
                            f"未改善 (最佳={best_val_em:.3f}) → 模型可能在钻漏洞"
                        )

                # 格式退化检测
                if val_fmt < 0.1 and best_val_em > 0:
                    logger.warning(
                        f"[格式退化] val_fmt={val_fmt:.3f} 极低, 模型丧失格式能力, 停止训练"
                    )
                    stop_reason = f"格式退化: val_fmt={val_fmt:.3f}"
                    break

        # 最终保存
        logger.info(f"训练结束, 原因: {stop_reason}")
        logger.info(f"最佳 val_em={best_val_em:.3f} (step={best_step})")
        final_step = step if step < cfg.total_training_steps else cfg.total_training_steps
        self._save_checkpoint(output_dir, final_step)
        self._save_metrics(output_dir)
        self._append_gpu_csv(final_step, "训练结束")
        log_gpu_memory_detailed("训练结束")
        self._close_csv_files()

        return self.metrics_history

    def _save_checkpoint(self, output_dir: Path, step: int) -> None:
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        # 保存 critic
        torch.save(self.critic.state_dict(), str(ckpt_dir / "critic.pt"))
        logger.info(f"Checkpoint 已保存: {ckpt_dir}")

    def _save_metrics(self, output_dir: Path) -> None:
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
        logger.info(f"训练指标已保存: {metrics_path}")

    # ------------------------------------------------------------------
    # CSV 绘图数据写入
    # ------------------------------------------------------------------

    def _append_train_csv(self, step: int, metrics: dict[str, float],
                          best_val_em: float, steps_no_improve: int) -> None:
        """追加一行 PPO 训练指标到 CSV。"""
        if self._train_csv_writer is None:
            return
        row = {
            "step": step,
            "reward_mean": round(metrics["reward_mean"], 4),
            "policy_loss": round(metrics["policy_loss"], 6),
            "value_loss": round(metrics["value_loss"], 6),
            "entropy": round(metrics["entropy"], 4),
            "kl_loss": round(metrics["kl_loss"], 6),
            "approx_kl": round(metrics.get("approx_kl", 0), 6),
            "clip_frac": round(metrics.get("clip_frac", 0), 4),
            "response_len_mean": round(metrics["response_len_mean"], 1),
            "actor_lr": metrics.get("actor_lr", 0),
            "critic_lr": metrics.get("critic_lr", 0),
            "step_time": round(metrics["step_time"], 2),
            "best_val_em": round(best_val_em, 4),
            "steps_no_improve": steps_no_improve,
        }
        _append_csv_row(self._train_csv_writer, row, PPO_TRAIN_CSV_COLUMNS)

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
        _append_csv_row(self._val_csv_writer, row, PPO_VAL_CSV_COLUMNS)

    def _append_gpu_csv(self, step: int, tag: str) -> None:
        """追加一行显存指标到 CSV。"""
        if self._gpu_csv_writer is None or not torch.cuda.is_available():
            return
        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0) / (1024**3)
        reserved = stats.get("reserved_bytes.all.current", 0) / (1024**3)
        peak_allocated = stats.get("allocated_bytes.all.peak", 0) / (1024**3)
        segment_count = stats.get("segment.all.current", 0)
        row = {
            "step": step,
            "tag": tag,
            "allocated_gb": round(allocated, 4),
            "reserved_gb": round(reserved, 4),
            "pool_free_gb": round(reserved - allocated, 4),
            "segment_count": segment_count,
            "peak_allocated_gb": round(peak_allocated, 4),
        }
        _append_csv_row(self._gpu_csv_writer, row, PPO_GPU_CSV_COLUMNS)

    def _close_csv_files(self) -> None:
        """关闭所有 CSV 文件。"""
        for writer in [self._train_csv_writer, self._val_csv_writer, self._gpu_csv_writer]:
            if writer is not None and hasattr(writer, "stream"):
                writer.stream.flush()
                writer.stream.close()
