"""可配置的 Qwen3 GSM8K LoRA SFT 训练脚本。

这个脚本从 notebook 的模块 7 训练逻辑抽取而来，目标是方便用不同
max_length / learning_rate / epoch 等参数重复训练多个 SFT adapter。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


YHY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"
DEFAULT_TRAIN_FILE = YHY_DIR / "datasets" / "gsm8k_sft" / "train.parquet"
DEFAULT_OUTPUT_ROOT = YHY_DIR / "models" / "sft"


def build_model_tag(base_model_dir: Path) -> str:
    """根据 base model 目录名生成适合放进实验名称的短标签。"""
    raw = base_model_dir.name.lower()
    chars = [char if char.isalnum() else "_" for char in raw]
    return "_".join("".join(chars).split("_"))


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """把 parquet 读出的 messages 统一成普通 list[dict]。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(message) for message in messages]


def apply_chat_template_text(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """使用 tokenizer 自带 chat template 渲染训练文本。

    Qwen3 tokenizer 支持 enable_thinking=False；如果当前 tokenizer 不支持，
    则自动回退到普通 apply_chat_template。
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class MessageSFTDataset(TorchDataset):
    """把 messages parquet 转成 CausalLM SFT 样本。

    input_ids 是 user + assistant 的完整 chat 文本。
    labels 中 user/prompt 部分设为 -100，只训练 assistant 回答部分。
    """

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
        # full_text 包含 user 和 assistant 标准答案，用作完整 input_ids。
        # Qwen3 的 chat_template 会在 <|im_end|> 后自动追加 \n，
        # 导致 labels 最后一个有效 token 是 \n 而非 <|im_end|>，
        # 模型学到"跨过 <|im_end|> 继续输出" → 推理时复读/乱码。
        # rstrip 移除尾部空白，确保 <|im_end|> 是文本最后一个 token。
        full_text = apply_chat_template_text(
            self.tokenizer,
            messages,
            add_generation_prompt=False,
        ).rstrip()

        # prompt_text 只包含 user，并停在 assistant 起点后。
        # 这个长度用于 mask prompt，使 loss 只落在 assistant tokens 上。
        prompt_text = apply_chat_template_text(
            self.tokenizer,
            [messages[0]],
            add_generation_prompt=True,
        )

        full_tokens = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = list(input_ids)

        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        # assistant 结尾的 <|im_end|> 之后不应有任何有效 label。
        # chat_template 在 <|im_end|> 后追加 \n，rstrip 只去掉了文本末尾的，
        # 但截断后的序列中 <|im_end|> 后仍可能有残余 token。
        # 从后往前找最后一个 <|im_end|>，把它之后的所有 label 设为 -100，
        # 确保 <|im_end|> 是 assistant 部分最后一个有效监督信号。
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for j in range(len(labels) - 1, prompt_len, -1):
            if labels[j] == im_end_id:
                labels[j + 1:] = [-100] * len(labels[j + 1:])
                break

        # 如果 max_length 太短导致 assistant 部分完全被截掉，这条样本没有训练信号。
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
    """动态 padding collator。

    input_ids 用 pad_token_id padding；attention_mask padding 为 0；
    labels padding 为 -100，避免 padding token 参与 loss。
    """

    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = self.tokenizer.pad_token_id

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable Qwen3 GSM8K LoRA SFT adapter.")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true", help="禁用 CUDA fp16 训练。")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        help="LoRA target modules，默认覆盖 attention 和 MLP 线性层。",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    """根据 run-name 或时间戳创建本轮输出目录。"""
    if args.run_name:
        run_name = args.run_name
    else:
        model_tag = build_model_tag(args.base_model_dir)
        run_name = (
            f"{model_tag}_gsm8k_lora"
            f"_len{args.max_length}_lr{args.learning_rate:g}"
            f"_ep{args.num_train_epochs:g}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_run_config(args: argparse.Namespace, output_dir: Path, train_row_count: int, train_sample_count: int) -> None:
    """保存本轮训练配置，方便后续做实验对比。"""
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "output_dir": str(output_dir),
            "train_row_count": train_row_count,
            "train_sample_count": train_sample_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.base_model_dir = args.base_model_dir.resolve()
    args.train_file = args.train_file.resolve()
    args.output_root = args.output_root.resolve()

    if not args.base_model_dir.exists():
        raise FileNotFoundError(args.base_model_dir)
    if not args.train_file.exists():
        raise FileNotFoundError(args.train_file)

    output_dir = build_output_dir(args)
    train_row_count = len(pd.read_parquet(args.train_file, columns=["messages"]))

    print("=" * 80)
    print("base model:", args.base_model_dir)
    print("train file:", args.train_file)
    print("train rows:", train_row_count)
    print("output dir:", output_dir)
    print("max_length:", args.max_length)
    print("learning_rate:", args.learning_rate)
    print("num_train_epochs:", args.num_train_epochs)
    print("gradient_accumulation_steps:", args.gradient_accumulation_steps)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = MessageSFTDataset(args.train_file, tokenizer, max_length=args.max_length)
    collator = CausalLMCollator(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() and not args.no_fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    write_run_config(args, output_dir, train_row_count, len(train_dataset))
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_run_config(args, output_dir, train_row_count, len(train_dataset))

    print("=" * 80)
    print("saved LoRA adapter to:", output_dir)
    print("run_config:", output_dir / "run_config.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
