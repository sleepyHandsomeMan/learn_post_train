"""Prompt 构造与 chat template 渲染。"""

from __future__ import annotations

from typing import Any


DEFAULT_FORMAT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def build_user_content(
    question: str,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
    include_format_instruction: bool = True,
) -> str:
    """构造 GSM8K 风格 user 文本。"""
    question = str(question).strip()
    if include_format_instruction and format_instruction:
        return f"{question} {format_instruction}"
    return question


def build_user_messages(
    question: str,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
    include_format_instruction: bool = True,
) -> list[dict[str, str]]:
    """构造单轮问答 messages。"""
    return [
        {
            "role": "user",
            "content": build_user_content(question, format_instruction, include_format_instruction),
        }
    ]


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    enable_thinking: bool | None = False,
) -> str:
    """使用 tokenizer 自带 chat template 渲染文本。"""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_generation_prompt(
    tokenizer: Any,
    question: str,
    prompt_mode: str,
    format_instruction: str = DEFAULT_FORMAT_INSTRUCTION,
    include_format_instruction: bool = True,
    enable_thinking: bool | None = False,
) -> str:
    """根据 prompt_mode 渲染模型实际看到的推理 prompt。"""
    if prompt_mode == "plain":
        return build_user_content(question, format_instruction, include_format_instruction)
    if prompt_mode == "chat":
        messages = build_user_messages(question, format_instruction, include_format_instruction)
        return apply_chat_template_text(tokenizer, messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    raise ValueError(f"未知 prompt_mode: {prompt_mode}")
