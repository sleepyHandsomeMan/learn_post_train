"""GRPO/PPO 训练数据预览工具。

提供从 SFT messages parquet 直接提取 prompt + ground_truth 的辅助函数。
GRPO/PPO 训练器本身使用 rl_dataset.py 加载数据，本模块主要用于数据预览和验证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data import normalize_messages, strip_format_instruction
from .metrics import extract_final_answer
from .prompting import DEFAULT_FORMAT_INSTRUCTION


def preview_prompt_samples(
    input_file: str | Path,
    num_preview: int = 3,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
) -> str:
    """生成 SFT parquet → RL prompt 的预览报告。

    用于人工检查哪些样本会被用于 GRPO/PPO 训练，
    以及 ground_truth 抽取是否正确。
    """
    input_file = Path(input_file)
    df = pd.read_parquet(input_file).head(num_preview)

    lines = [
        f"# RL Prompt 数据预览",
        "",
        f"- source_file: `{input_file}`",
        f"- preview_rows: {len(df)}",
        "",
    ]

    for position, (_, row) in enumerate(df.iterrows()):
        messages = normalize_messages(row["messages"])
        if len(messages) < 2:
            lines.append(f"## Sample {position}: SKIPPED (少于两轮消息)")
            continue

        user_content = str(messages[0]["content"]).strip()
        assistant_answer = str(messages[1]["content"]).strip()
        question = strip_format_instruction(user_content, format_instruction)
        ground_truth = extract_final_answer(assistant_answer)

        lines.extend([
            f"## Sample {position}",
            "",
            f"- **原始问题**: {question[:200]}",
            f"- **user_content**: {user_content[:300]}",
            f"- **assistant 回答**: {assistant_answer[:200]}",
            f"- **抽取的 ground_truth**: `{ground_truth}`",
            f"- **可作为 GRPO/PPO prompt?**: {'是' if ground_truth is not None else '否 (缺少答案)'}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)
