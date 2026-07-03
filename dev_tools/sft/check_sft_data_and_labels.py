from pathlib import Path
import re
import statistics

import pandas as pd
from transformers import AutoTokenizer


YHY_DIR = Path(__file__).resolve().parents[2]
BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"
TRAIN_FILE = YHY_DIR / "datasets" / "gsm8k_sft" / "train.parquet"

FORMAT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.\d+)?)")


def normalize_messages(messages):
    """把 parquet 读出的 messages 统一转换成普通 Python list[dict]。"""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(m) for m in messages]


def apply_chat_template_text(tokenizer, messages, add_generation_prompt):
    """使用训练/评估一致的 chat template 渲染文本。"""
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


def pct(num, den):
    """把计数转成百分比字符串。"""
    return f"{num / den:.2%}" if den else "0.00%"


def preview(text, max_chars=180):
    """压缩展示长文本，方便终端查看。"""
    text = str(text).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text.encode("unicode_escape").decode("ascii")
    return (text[:max_chars] + "...").encode("unicode_escape").decode("ascii")


def check_train_data(df):
    """检查 train.parquet 的 messages 和 assistant gold 是否干净。"""
    print("\n=== A. train.parquet 数据干净度检查 ===")
    total = len(df)
    bad_role_rows = []
    missing_format_rows = []
    multi_final_rows = []
    tail_after_final_rows = []
    non_ascii_rows = []
    suspicious_token_rows = []
    assistant_lengths = []

    suspicious_patterns = [
        "erotique",
        "beurette",
        "externalActionCode",
        "assistantprobe",
        "kInstruction",
        "inertia",
        "dóla",
        "锌",
        "褉",
        "懈",
        "丕",
        "賱",
    ]

    for idx, row in df.iterrows():
        messages = normalize_messages(row["messages"])
        roles = [m.get("role") for m in messages]
        if roles != ["user", "assistant"]:
            bad_role_rows.append((idx, roles))
            continue

        user_content = str(messages[0].get("content", ""))
        assistant = str(messages[1].get("content", ""))
        assistant_lengths.append(len(assistant))

        matches = list(FINAL_ANSWER_PATTERN.finditer(assistant))
        if not matches:
            missing_format_rows.append((idx, assistant))
        if len(matches) > 1:
            multi_final_rows.append((idx, len(matches), assistant))
        if matches:
            tail = assistant[matches[0].end() :].strip()
            if tail:
                tail_after_final_rows.append((idx, tail, assistant))

        if any(ord(ch) > 127 for ch in assistant):
            non_ascii_rows.append((idx, assistant))
        lowered = assistant.lower()
        if any(pattern.lower() in lowered for pattern in suspicious_patterns):
            suspicious_token_rows.append((idx, assistant))

        if FORMAT_INSTRUCTION not in user_content:
            missing_format_rows.append((idx, "user missing format instruction: " + user_content))

    print(f"总样本数: {total}")
    print(f"role 不是 user->assistant 的样本: {len(bad_role_rows)} ({pct(len(bad_role_rows), total)})")
    print(f"缺少合法 #### 数字 或 user 格式指令异常的样本: {len(missing_format_rows)} ({pct(len(missing_format_rows), total)})")
    print(f"assistant 中出现多个合法 #### 数字的样本: {len(multi_final_rows)} ({pct(len(multi_final_rows), total)})")
    print(f"第一个 #### 数字之后仍有尾巴的样本: {len(tail_after_final_rows)} ({pct(len(tail_after_final_rows), total)})")
    print(f"assistant 含非 ASCII 字符的样本: {len(non_ascii_rows)} ({pct(len(non_ascii_rows), total)})")
    print(f"assistant 命中已知异常片段的样本: {len(suspicious_token_rows)} ({pct(len(suspicious_token_rows), total)})")
    print(
        "assistant 字符长度: "
        f"min={min(assistant_lengths)}, "
        f"p50={statistics.median(assistant_lengths):.0f}, "
        f"max={max(assistant_lengths)}"
    )

    def show_examples(title, rows, limit=3):
        print(f"\n{title}:")
        if not rows:
            print("  无")
            return
        for item in rows[:limit]:
            print(" ", item[0], preview(item[-1]))

    show_examples("缺少格式/指令异常示例", missing_format_rows)
    show_examples("多个 #### 数字示例", multi_final_rows)
    show_examples("#### 答案后仍有尾巴示例", tail_after_final_rows)
    show_examples("非 ASCII assistant 示例", non_ascii_rows)
    show_examples("命中异常片段示例", suspicious_token_rows)


def find_labeled_end_token(input_ids, labels, end_token_ids):
    """检查 assistant label 区域里是否保留了结束 token。"""
    return any(label != -100 and token_id in end_token_ids for token_id, label in zip(input_ids, labels))


def token_rows_for_tail(tokenizer, token_ids, labels, end_token_ids, tail_n=24):
    """把 label 尾部 token 解码成可读表格，便于检查结束 token 是否被训练。"""
    labeled_positions = [pos for pos, label in enumerate(labels) if label != -100]
    tail_positions = labeled_positions[-tail_n:]
    rows = []
    for pos in tail_positions:
        token_id = token_ids[pos]
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        rows.append(
            {
                "pos": pos,
                "token_id": token_id,
                "is_end_token": token_id in end_token_ids,
                "token_text": token_text.encode("unicode_escape").decode("ascii"),
            }
        )
    return rows


def check_label_tail_decode(df, tokenizer, max_length=512, sample_indices=None):
    """解码 labels!=-100 的尾部，确认模型实际被训练预测的 assistant 结尾。"""
    print("\n=== C. label 尾部 decode 检查 ===")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    end_token_ids = {token_id for token_id in [im_end_id, eos_id] if isinstance(token_id, int) and token_id >= 0}
    if sample_indices is None:
        sample_indices = [0, 1, 10, 310]

    print(
        "检查方法: 按训练代码构造 full_text 和 prompt_text，"
        "把 prompt_len 之前的 labels 设为 -100，只解码 labels!=-100 的尾部 token。"
    )
    print(f"max_length={max_length}, end_token_ids={sorted(end_token_ids)}")

    for sample_idx in sample_indices:
        if sample_idx not in df.index:
            print(f"\nidx={sample_idx} 不在 dataframe index 中，跳过")
            continue

        row = df.loc[sample_idx]
        messages = normalize_messages(row["messages"])
        assistant = str(messages[1]["content"])
        full_text = apply_chat_template_text(tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_chat_template_text(tokenizer, [messages[0]], add_generation_prompt=True)
        full_ids_untruncated = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_ids = full_ids_untruncated[:max_length]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        labeled_ids = [token_id for token_id, label in zip(input_ids, labels) if label != -100]
        labeled_text_tail = tokenizer.decode(labeled_ids[-80:], skip_special_tokens=False)
        has_labeled_end = find_labeled_end_token(input_ids, labels, end_token_ids)
        truncated = len(full_ids_untruncated) > max_length

        print(f"\n--- idx={sample_idx} ---")
        print(f"full_len={len(full_ids_untruncated)}, prompt_len={len(prompt_ids)}, labeled_len={len(labeled_ids)}, truncated={truncated}")
        print(f"label 区域是否包含结束 token: {has_labeled_end}")
        print("assistant gold tail:")
        print("  " + preview(assistant[-220:]))
        print("labeled assistant tail decode:")
        print("  " + preview(labeled_text_tail, max_chars=320))
        print("tail token table:")
        for item in token_rows_for_tail(tokenizer, input_ids, labels, end_token_ids, tail_n=24):
            print(
                f"  pos={item['pos']:>4} "
                f"id={item['token_id']:>6} "
                f"end={str(item['is_end_token']):<5} "
                f"text={item['token_text']}"
            )


def check_label_mask_and_truncation(df, tokenizer, max_lengths):
    """检查训练 tokenization 是否保留 assistant 结尾和结束 token。"""
    print("\n=== B. label mask / max_length 截断检查 ===")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    end_token_ids = {token_id for token_id in [im_end_id, eos_id] if isinstance(token_id, int) and token_id >= 0}
    print(f"eos_token={tokenizer.eos_token!r}, eos_token_id={eos_id}")
    print(f"<|im_end|> token_id={im_end_id}")
    print(f"用于检查的结束 token ids: {sorted(end_token_ids)}")

    full_cache = []
    for idx, row in df.iterrows():
        messages = normalize_messages(row["messages"])
        full_text = apply_chat_template_text(tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_chat_template_text(tokenizer, [messages[0]], add_generation_prompt=True)
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        full_cache.append((idx, messages, full_text, full_ids, prompt_ids))

    full_lengths = [len(item[3]) for item in full_cache]
    prompt_lengths = [len(item[4]) for item in full_cache]
    print(
        "完整 chat token 长度: "
        f"min={min(full_lengths)}, "
        f"p50={statistics.median(full_lengths):.0f}, "
        f"p90={statistics.quantiles(full_lengths, n=10)[8]:.0f}, "
        f"max={max(full_lengths)}"
    )
    print(
        "prompt token 长度: "
        f"min={min(prompt_lengths)}, "
        f"p50={statistics.median(prompt_lengths):.0f}, "
        f"max={max(prompt_lengths)}"
    )

    for max_length in max_lengths:
        truncated_rows = []
        no_assistant_label_rows = []
        no_labeled_end_rows = []
        assistant_label_lengths = []

        for idx, _messages, full_text, full_ids, prompt_ids in full_cache:
            input_ids = full_ids[:max_length]
            labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            assistant_label_len = sum(label != -100 for label in labels)
            assistant_label_lengths.append(assistant_label_len)

            if len(full_ids) > max_length:
                truncated_rows.append((idx, len(full_ids), len(prompt_ids), assistant_label_len, full_text))
            if assistant_label_len == 0:
                no_assistant_label_rows.append((idx, len(full_ids), len(prompt_ids), assistant_label_len, full_text))
            if not find_labeled_end_token(input_ids, labels, end_token_ids):
                no_labeled_end_rows.append((idx, len(full_ids), len(prompt_ids), assistant_label_len, full_text))

        total = len(full_cache)
        print(f"\nmax_length={max_length}")
        print(f"  被截断样本: {len(truncated_rows)} / {total} ({pct(len(truncated_rows), total)})")
        print(f"  assistant label 为空样本: {len(no_assistant_label_rows)} / {total} ({pct(len(no_assistant_label_rows), total)})")
        print(f"  label 区域没有结束 token 的样本: {len(no_labeled_end_rows)} / {total} ({pct(len(no_labeled_end_rows), total)})")
        print(
            "  assistant label token 数: "
            f"min={min(assistant_label_lengths)}, "
            f"p50={statistics.median(assistant_label_lengths):.0f}, "
            f"max={max(assistant_label_lengths)}"
        )

        if no_labeled_end_rows:
            print("  没有保留结束 token 的示例:")
            for idx, full_len, prompt_len, assistant_label_len, full_text in no_labeled_end_rows[:5]:
                print(
                    f"    idx={idx}, full_len={full_len}, prompt_len={prompt_len}, "
                    f"assistant_label_len={assistant_label_len}, tail={preview(full_text[-220:])}"
                )


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(BASE_MODEL_DIR)

    df = pd.read_parquet(TRAIN_FILE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    check_train_data(df)
    check_label_mask_and_truncation(df, tokenizer, max_lengths=[512, 768, 1024])
    check_label_tail_decode(df, tokenizer, max_length=512, sample_indices=[0, 1, 10, 310])
    check_label_tail_decode(df, tokenizer, max_length=768, sample_indices=[0, 1, 10, 310])


if __name__ == "__main__":
    main()
