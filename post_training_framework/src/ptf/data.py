"""数据读取与 GSM8K 样本规范化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import extract_final_answer
from .prompting import DEFAULT_FORMAT_INSTRUCTION


@dataclass
class EvalSample:
    """评估样本的统一结构。"""

    idx: int
    question: str
    gold: str
    gold_answer: str | None
    user_content: str


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """把 parquet 读出的 messages 统一转换为 list[dict]。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(message) for message in messages]


def strip_format_instruction(user_content: str, format_instruction: str = DEFAULT_FORMAT_INSTRUCTION) -> str:
    """从 user 文本末尾剥离格式指令，得到原始题目。"""
    text = str(user_content).strip()
    suffix = f" {format_instruction}".strip()
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)].strip()
    return text


def row_to_eval_sample(row: Any, idx: int, format_instruction: str = DEFAULT_FORMAT_INSTRUCTION) -> EvalSample:
    """把一行 parquet 数据转换为 EvalSample。"""
    if "messages" in row:
        messages = normalize_messages(row["messages"])
        if len(messages) < 2:
            raise ValueError(f"第 {idx} 行 messages 少于两轮，无法评估。")
        user_content = str(messages[0]["content"])
        gold = str(messages[1]["content"])
        question = strip_format_instruction(user_content, format_instruction)
        return EvalSample(
            idx=idx,
            question=question,
            gold=gold,
            gold_answer=extract_final_answer(gold),
            user_content=user_content,
        )

    if "question" in row and ("answer" in row or "gold" in row or "gold_answer" in row):
        question = str(row["question"]).strip()
        gold = str(row.get("answer", row.get("gold", row.get("gold_answer", ""))))
        return EvalSample(
            idx=idx,
            question=question,
            gold=gold,
            gold_answer=extract_final_answer(gold),
            user_content=question,
        )

    raise ValueError("评估数据需要 messages 列，或 question + answer/gold/gold_answer 列。")


def load_eval_samples(
    eval_file: str | Path,
    max_items: int | None = None,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
) -> list[EvalSample]:
    """读取评估 parquet，并返回统一样本列表。"""
    df = pd.read_parquet(eval_file)
    if max_items is not None:
        df = df.head(max_items)

    samples: list[EvalSample] = []
    for position, (_, row) in enumerate(df.iterrows()):
        samples.append(row_to_eval_sample(row, idx=position, format_instruction=format_instruction))
    return samples
