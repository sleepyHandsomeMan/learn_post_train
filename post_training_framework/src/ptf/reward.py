"""GSM8K 规则奖励函数。

用于 GRPO/PPO 训练的 reward 计算和离线 eval JSONL 分析。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .metrics import (
    answers_equal,
    extract_final_answer,
    extract_first_hash_answer,
    summarize_repetition,
)


@dataclass
class GSM8KRewardConfig:
    """GSM8K rule reward 的权重配置。"""

    exact_with_format_score: float = 1.0
    exact_without_format_score: float = 0.5
    format_bonus: float = 0.2
    single_final_bonus: float = 0.1
    missing_format_penalty: float = -0.2
    multi_final_penalty: float = -0.1
    repeat_penalty: float = -0.4
    overlong_penalty: float = -0.2
    overlong_chars: int = 1200
    min_reward: float = -1.0
    max_reward: float = 1.3


def _clip(value: float, lower: float, upper: float) -> float:
    """把 reward 限制在固定范围内，避免极端值影响 RL 更新。"""
    return max(lower, min(upper, value))


def compute_gsm8k_rule_reward(
    response: str,
    gold_answer: str | None,
    config: GSM8KRewardConfig | None = None,
) -> dict[str, Any]:
    """计算单条 GSM8K response 的规则奖励。

    设计原则：
    - 满分只给“#### 后答案正确”的输出。
    - 没有 #### 但最后数字正确，只给部分分，避免模型放弃格式。
    - 复读、多个最终答案、过长输出要扣分。
    """
    config = config or GSM8KRewardConfig()
    response = str(response)

    first_hash_answer = extract_first_hash_answer(response)
    fallback_answer = extract_final_answer(response)
    format_ok = first_hash_answer is not None
    pred_answer = first_hash_answer if format_ok else fallback_answer

    exact_match = answers_equal(pred_answer, gold_answer)
    first_hash_exact_match = answers_equal(first_hash_answer, gold_answer)
    fallback_exact_match = answers_equal(fallback_answer, gold_answer)
    repetition = summarize_repetition(response)
    single_final_answer_ok = repetition["hash_count"] == 1 and repetition["final_answer_count"] == 1
    overlong = len(response) > config.overlong_chars

    reward = 0.0
    components: dict[str, float] = {}

    if first_hash_exact_match:
        components["answer"] = config.exact_with_format_score
    elif (not format_ok) and fallback_exact_match:
        components["answer"] = config.exact_without_format_score
    else:
        components["answer"] = 0.0

    if format_ok:
        components["format"] = config.format_bonus
    else:
        components["format"] = config.missing_format_penalty

    if single_final_answer_ok:
        components["single_final"] = config.single_final_bonus
    elif repetition["hash_count"] > 1 or repetition["final_answer_count"] > 1:
        components["single_final"] = config.multi_final_penalty
    else:
        components["single_final"] = 0.0

    components["repeat"] = config.repeat_penalty if repetition["repeat_like"] else 0.0
    components["overlong"] = config.overlong_penalty if overlong else 0.0

    reward = sum(components.values())
    clipped_reward = _clip(reward, config.min_reward, config.max_reward)

    return {
        "score": clipped_reward,
        "raw_score": reward,
        "components": components,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "first_hash_answer": first_hash_answer,
        "fallback_answer": fallback_answer,
        "exact_match": exact_match,
        "first_hash_exact_match": first_hash_exact_match,
        "fallback_exact_match": fallback_exact_match,
        "format_ok": format_ok,
        "single_final_answer_ok": single_final_answer_ok,
        "repeat_like": repetition["repeat_like"],
        "hash_count": repetition["hash_count"],
        "final_answer_count": repetition["final_answer_count"],
        "answer_is_count": repetition["answer_is_count"],
        "repeated_line_count": repetition["repeated_line_count"],
        "overlong": overlong,
        "pred_chars": len(response),
        "config": asdict(config),
    }


