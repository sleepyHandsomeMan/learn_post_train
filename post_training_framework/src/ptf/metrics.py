"""自动评估指标。"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any


FINAL_ANSWER_PATTERN = r"####\s*(-?[0-9][0-9,]*(?:\.\d+)?)"


def normalize_number_text(value: str | None) -> str | None:
    """规范化数字字符串，主要去掉逗号和多余空格。"""
    if value is None:
        return None
    return str(value).replace(",", "").strip()


def extract_final_answer(text: Any) -> str | None:
    """优先抽取 #### 后的数字；没有时退化为最后一个数字。"""
    text = str(text)
    match = re.search(FINAL_ANSWER_PATTERN, text)
    if match:
        return normalize_number_text(match.group(1))
    numbers = re.findall(r"-?[0-9][0-9,]*(?:\.\d+)?", text)
    return normalize_number_text(numbers[-1]) if numbers else None


def extract_first_hash_answer(text: Any) -> str | None:
    """只抽取第一个合法 #### 数字，用于判断格式答案。"""
    match = re.search(FINAL_ANSWER_PATTERN, str(text))
    if not match:
        return None
    return normalize_number_text(match.group(1))


def answers_equal(pred: str | None, gold: str | None) -> bool:
    """比较答案；能转成 Decimal 时按数值比较，否则按字符串比较。"""
    pred = normalize_number_text(pred)
    gold = normalize_number_text(gold)
    if pred is None or gold is None:
        return False
    try:
        return Decimal(pred) == Decimal(gold)
    except InvalidOperation:
        return pred == gold


def count_text_pattern(text: Any, pattern: str) -> int:
    """统计正则模式出现次数。"""
    return len(re.findall(pattern, str(text)))


def summarize_repetition(text: Any) -> dict[str, Any]:
    """统计复读和停止边界相关信号。"""
    text = str(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_count = len(lines) - len(set(lines))
    hash_count = count_text_pattern(text, r"####")
    final_answer_count = count_text_pattern(text, FINAL_ANSWER_PATTERN)
    answer_is_count = count_text_pattern(text, r"The answer is")
    repeat_like = hash_count > 1 or answer_is_count > 1 or repeated_line_count > 0
    return {
        "hash_count": hash_count,
        "final_answer_count": final_answer_count,
        "answer_is_count": answer_is_count,
        "repeated_line_count": repeated_line_count,
        "repeat_like": repeat_like,
    }


def score_prediction(prediction: str, gold_answer: str | None) -> dict[str, Any]:
    """对单条输出计算格式、答案和复读指标。"""
    first_hash_answer = extract_first_hash_answer(prediction)
    pred_answer = first_hash_answer if first_hash_answer is not None else extract_final_answer(prediction)
    repetition = summarize_repetition(prediction)
    format_ok = first_hash_answer is not None
    single_final_answer_ok = repetition["final_answer_count"] == 1 and repetition["hash_count"] == 1
    return {
        "gold_answer": gold_answer,
        "first_hash_answer": first_hash_answer,
        "pred_answer": pred_answer,
        "exact_match": answers_equal(pred_answer, gold_answer),
        "first_hash_exact_match": answers_equal(first_hash_answer, gold_answer),
        "format_ok": format_ok,
        "single_final_answer_ok": single_final_answer_ok,
        "hash_count": repetition["hash_count"],
        "final_answer_count": repetition["final_answer_count"],
        "answer_is_count": repetition["answer_is_count"],
        "repeated_line_count": repetition["repeated_line_count"],
        "repeat_like": repetition["repeat_like"],
        "pred_chars": len(str(prediction)),
    }


def summarize_rows(rows: list[dict[str, Any]], tag: str, max_new_tokens: int | None = None) -> dict[str, Any]:
    """汇总多条样本的平均指标。"""
    n = len(rows)

    def mean_bool(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / n if n else 0.0

    def mean_num(key: str) -> float:
        values = [float(row.get(key, 0) or 0) for row in rows]
        return sum(values) / n if n else 0.0

    return {
        "tag": tag,
        "n": n,
        "max_new_tokens": max_new_tokens,
        "exact_match": mean_bool("exact_match"),
        "first_hash_exact_match": mean_bool("first_hash_exact_match"),
        "format_rate": mean_bool("format_ok"),
        "single_final_answer_rate": mean_bool("single_final_answer_ok"),
        "repeat_like_rate": mean_bool("repeat_like"),
        "avg_hash_count": mean_num("hash_count"),
        "max_hash_count": max((int(row.get("hash_count", 0) or 0) for row in rows), default=0),
        "avg_chars": mean_num("pred_chars"),
    }
