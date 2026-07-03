"""LoRA SFT 训练流程。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import ExperimentConfig
from .data import normalize_messages
from .prompting import apply_chat_template_text


class MessageSFTDataset(TorchDataset):
    """把 messages parquet 转成 assistant-only label 的 CausalLM 样本。"""

    def __init__(self, parquet_path: Path, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, torch.Tensor]] = []
        self.skipped_all_prompt = 0

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            messages = normalize_messages(row["messages"])
            item = self._tokenize_one(messages)
            if item is None:
                self.skipped_all_prompt += 1
                continue
            self.samples.append(item)

        print(f"loaded train samples: {len(self.samples)} from {parquet_path}")
        if self.skipped_all_prompt:
            print(f"skipped samples without assistant loss: {self.skipped_all_prompt}")

    def _tokenize_one(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor] | None:
        # full_text 是 user + assistant 的完整监督文本。
        # Qwen3 chat template 会在最后一个 <|im_end|> 后追加换行。
        # 训练时要去掉这个尾部空白，让最后一个有效 label 是 <|im_end|>，
        # 否则模型会学到“结束标记后继续生成换行”，推理时容易复读。
        full_text = apply_chat_template_text(
            self.tokenizer,
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
        ).rstrip()

        # prompt_text 停在 assistant 起点，用于 mask 掉 prompt 部分 loss。
        prompt_text = apply_chat_template_text(
            self.tokenizer,
            [messages[0]],
            add_generation_prompt=True,
            enable_thinking=False,
        )

        full_tokens = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False, truncation=False)

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        # max_length 太短时 assistant 全被截掉，这条样本没有训练信号。
        if all(label == -100 for label in labels):
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


@dataclass
class CausalLMCollator:
    """动态 padding collator。"""

    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            batch["input_ids"].append(
                torch.cat([feature["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([feature["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([feature["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {key: torch.stack(value) for key, value in batch.items()}


def _write_run_config(path: Path, cfg: ExperimentConfig, output_dir: Path, train_rows: int, train_samples: int) -> None:
    """保存训练配置快照，方便后续复现实验。"""
    payload = {
        "experiment_name": cfg.get("experiment_name"),
        "base_model_dir": str(cfg.path("model.base_model_dir")),
        "train_file": str(cfg.path("dataset.train_file")),
        "output_dir": str(output_dir),
        "train_row_count": train_rows,
        "train_sample_count": train_samples,
        "sft": cfg.get("sft", {}),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_lora_sft(cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    """按配置训练 LoRA SFT adapter，并返回 adapter 目录。"""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("训练 LoRA SFT 需要安装 peft。") from exc

    base_model_dir = cfg.path("model.base_model_dir")
    train_file = cfg.path("dataset.train_file")
    sft_cfg = cfg.get("sft", {})
    run_name = run_name or str(sft_cfg.get("run_name", f"sft_{time.strftime('%Y%m%d_%H%M%S')}"))
    output_dir = cfg.ensure_experiment_dir() / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_row_count = len(pd.read_parquet(train_file, columns=["messages"]))
    train_dataset = MessageSFTDataset(train_file, tokenizer, max_length=int(sft_cfg.get("max_length", 768)))
    collator = CausalLMCollator(tokenizer)

    dtype = torch.float16 if torch.cuda.is_available() and bool(sft_cfg.get("fp16", True)) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map=sft_cfg.get("device_map", "auto") if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if bool(sft_cfg.get("gradient_checkpointing", False)):
        # 1.7B 这类更大的 base 在 12GB 显存上训练时，重算激活比保留全部激活更稳。
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(sft_cfg.get("lora_r", 16)),
        lora_alpha=int(sft_cfg.get("lora_alpha", 32)),
        lora_dropout=float(sft_cfg.get("lora_dropout", 0.05)),
        target_modules=list(
            sft_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(sft_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(sft_cfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(sft_cfg.get("num_train_epochs", 1.0)),
        learning_rate=float(sft_cfg.get("learning_rate", 3e-5)),
        warmup_ratio=float(sft_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(sft_cfg.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(sft_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(sft_cfg.get("logging_steps", 20)),
        save_strategy=str(sft_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(sft_cfg.get("save_total_limit", 2)),
        fp16=torch.cuda.is_available() and bool(sft_cfg.get("fp16", True)),
        gradient_checkpointing=bool(sft_cfg.get("gradient_checkpointing", False)),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(sft_cfg.get("dataloader_num_workers", 0)),
        optim=str(sft_cfg.get("optim", "adamw_torch")),
        seed=int(sft_cfg.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    _write_run_config(output_dir / "run_config.json", cfg, output_dir, train_row_count, len(train_dataset))
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_run_config(output_dir / "run_config.json", cfg, output_dir, train_row_count, len(train_dataset))
    print("saved LoRA adapter to:", output_dir)
    return output_dir
