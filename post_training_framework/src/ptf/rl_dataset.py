"""GRPO/PPO 共用的 prompt 数据集。

直接从 SFT messages parquet 加载数据，提取 user prompt 和 ground_truth，
不依赖 verl 的数据格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from .data import normalize_messages, strip_format_instruction
from .metrics import extract_final_answer
from .prompting import DEFAULT_FORMAT_INSTRUCTION, apply_chat_template_text


@dataclass
class RLPrompt:
    """RL 训练用的 prompt 样本。"""

    idx: int
    user_content: str  # 包含格式指令的完整用户文本
    question: str  # 原始问题（剥离格式指令）
    ground_truth: str  # 从 assistant 回答中抽取的答案数字字符串
    prompt_text: str  # chat template 渲染后的 prompt 文本
    prompt_ids: list[int]  # tokenize 后的 prompt input_ids


class RLPromptDataset(TorchDataset):
    """从 SFT messages parquet 加载 GRPO/PPO 训练用的 prompt 数据集。"""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: Any,
        max_prompt_length: int = 512,
        format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.format_instruction = format_instruction
        self.enable_thinking = enable_thinking
        self.samples: list[RLPrompt] = []

        df = pd.read_parquet(parquet_path)
        for position, (_, row) in enumerate(df.iterrows()):
            sample = self._build_sample(row, position)
            if sample is not None:
                self.samples.append(sample)

    def _build_sample(self, row: Any, idx: int) -> RLPrompt | None:
        """从 parquet 行构建 RLPrompt。"""
        messages = normalize_messages(row["messages"])
        if len(messages) < 2:
            return None

        user_content = str(messages[0]["content"]).strip()
        assistant_answer = str(messages[1]["content"]).strip()
        question = strip_format_instruction(user_content, self.format_instruction)
        ground_truth = extract_final_answer(assistant_answer)
        if ground_truth is None:
            return None

        # 渲染 chat template prompt
        prompt_text = apply_chat_template_text(
            self.tokenizer,
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        if len(prompt_ids) > self.max_prompt_length:
            return None

        return RLPrompt(
            idx=idx,
            user_content=user_content,
            question=question,
            ground_truth=ground_truth,
            prompt_text=prompt_text,
            prompt_ids=prompt_ids,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RLPrompt:
        return self.samples[idx]

    def collate_fn(self, samples: list[RLPrompt]) -> dict[str, Any]:
        """将一个 batch 的 RLPrompt 聚合为 dict，方便 trainer 使用。"""
        return {
            "prompts": samples,
            "prompt_ids": [s.prompt_ids for s in samples],
            "ground_truths": [s.ground_truth for s in samples],
            "prompt_texts": [s.prompt_text for s in samples],
            "user_contents": [s.user_content for s in samples],
        }


def load_rl_dataset(
    parquet_path: str | Path,
    tokenizer: Any,
    max_prompt_length: int = 512,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
    enable_thinking: bool = False,
) -> RLPromptDataset:
    """工厂函数，加载 RL prompt 数据集并打印统计信息。"""
    dataset = RLPromptDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        format_instruction=format_instruction,
        enable_thinking=enable_thinking,
    )
    print(f"加载 RL prompt 数据集: {len(dataset)} 条样本 from {parquet_path}")
    return dataset
