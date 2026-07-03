"""SFT 训练数据三阶段校验工具。

检查阶段:
  1. 原始 parquet: assistant content 结尾、EOS 标记、特殊字符
  2. chat template 渲染: 模型实际看到的文本、尾部结构
  3. tokenization + labels: 最后几个有效 label token、截断情况

用法:
  python dev_tools/sft/validate_sft_data.py --parquet datasets/gsm8k_sft/train.parquet
  python dev_tools/sft/validate_sft_data.py --parquet datasets/gsm8k_sft/train.parquet --max-length 768
  python dev_tools/sft/validate_sft_data.py --parquet datasets/gsm8k_sft/train.parquet --show-samples 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


YHY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"
DEFAULT_PARQUET = YHY_DIR / "datasets" / "gsm8k_sft" / "train.parquet"


def normalize_messages(messages):
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(m) for m in messages]


def apply_chat_template_text(tokenizer, messages, add_generation_prompt):
    """按训练脚本同样的参数渲染 chat template。"""
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


def render_training_full_text(tokenizer, messages):
    """渲染训练实际使用的完整文本，并去掉末尾空白。"""
    return apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=False,
    ).rstrip()


def render_training_prompt_text(tokenizer, messages):
    """渲染训练实际使用的 prompt 文本，停在 assistant 起点后。"""
    return apply_chat_template_text(
        tokenizer,
        [messages[0]],
        add_generation_prompt=True,
    )


# ── 阶段 1: 原始 parquet 数据 ──────────────────────────────────


def stage1_raw_parquet(df, show_samples=0):
    """检查原始 parquet 中 assistant content 的格式问题。"""
    print("\n" + "=" * 60)
    print("阶段 1: 原始 parquet 数据检查")
    print("=" * 60)

    # 基本信息
    print(f"总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 每条样本的 message 结构
    role_counts = {}
    for _, row in df.iterrows():
        msgs = normalize_messages(row["messages"])
        roles = tuple(m["role"] for m in msgs)
        role_counts[roles] = role_counts.get(roles, 0) + 1
    print(f"\n消息角色结构分布:")
    for roles, count in role_counts.items():
        print(f"  {roles}: {count} 条")

    # assistant content 结尾检查
    endings = {
        "ends_with_newline": 0,
        "ends_with_spaces": 0,
        "ends_with_hash_number": 0,
        "has_im_end": 0,
        "has_eos_token": 0,
        "other_ending": 0,
    }
    ending_samples = []

    for _, row in df.iterrows():
        msgs = normalize_messages(row["messages"])
        assistant_content = None
        for m in msgs:
            if m["role"] == "assistant":
                assistant_content = m["content"]
                break
        if assistant_content is None:
            continue

        stripped = assistant_content.rstrip()

        if assistant_content.endswith("\n"):
            endings["ends_with_newline"] += 1
        if stripped != assistant_content:
            endings["ends_with_spaces"] += 1
        if "<|im_end|>" in assistant_content:
            endings["has_im_end"] += 1
        if "" in assistant_content or tokenizer_eos_check(assistant_content):
            endings["has_eos_token"] += 1

        # #### 数字 结尾
        if re.match(r".*####\s*-?\d+(\.\d+)?$", stripped):
            endings["ends_with_hash_number"] += 1
        else:
            endings["other_ending"] += 1
            if len(ending_samples) < 5:
                ending_samples.append(stripped[-80:])

    print(f"\nassistant content 结尾统计:")
    for key, count in endings.items():
        pct = count / len(df) * 100
        print(f"  {key}: {count} ({pct:.1f}%)")

    if ending_samples:
        print(f"\n非 #### 数字 结尾的样本 (前5条):")
        for s in ending_samples:
            print(f"  ...{repr(s)}")

    # #### 后面的内容检查
    after_hash_issues = 0
    after_hash_samples = []
    for _, row in df.iterrows():
        msgs = normalize_messages(row["messages"])
        for m in msgs:
            if m["role"] != "assistant":
                continue
            content = m["content"]
            last_hash = content.rfind("####")
            if last_hash >= 0:
                after = content[last_hash + 4:].strip()
                if not re.match(r"^-?\d+(\.\d+)?$", after):
                    after_hash_issues += 1
                    if len(after_hash_samples) < 5:
                        after_hash_samples.append(after[:100])

    if after_hash_issues:
        print(f"\n#### 后有非纯数字内容: {after_hash_issues} 条")
        for s in after_hash_samples:
            print(f"  {repr(s)}")
    else:
        print(f"\n#### 后内容: 全部为纯数字 [OK]")

    # 显示样本详情
    if show_samples > 0:
        print(f"\n--- 前 {show_samples} 条样本详情 ---")
        for i in range(min(show_samples, len(df))):
            msgs = normalize_messages(df.iloc[i]["messages"])
            print(f"\n样本 {i}:")
            for m in msgs:
                role = m["role"]
                content = m["content"]
                print(f"  [{role}] 长度={len(content)}")
                print(f"    开头: {repr(content[:50])}")
                print(f"    结尾: {repr(content[-50:])}")


def tokenizer_eos_check(text):
    """检查文本是否包含 EOS 字符 (U+001F)。"""
    return "" in text


# ── 阶段 2: chat template 渲染 ──────────────────────────────────


def stage2_chat_template(df, tokenizer, show_samples=0):
    """检查 chat template 渲染后模型实际看到的文本。"""
    print("\n" + "=" * 60)
    print("阶段 2: chat template 渲染检查")
    print("=" * 60)

    # 用一条样本展示渲染过程
    msgs = normalize_messages(df.iloc[0]["messages"])

    raw_assistant = None
    for m in msgs:
        if m["role"] == "assistant":
            raw_assistant = m["content"]
            break

    # 原始 chat template 会暴露 tokenizer 自带的尾部结构。
    raw_full_text = apply_chat_template_text(tokenizer, msgs, add_generation_prompt=False)
    # 训练脚本实际使用 rstrip 后的完整文本。
    train_full_text = raw_full_text.rstrip()

    print(f"\n--- 样本 0 渲染对比 ---")
    print(f"原始 assistant content 结尾: {repr(raw_assistant[-40:])}")
    print(f"原始 chat template 结尾: {repr(raw_full_text[-60:])}")
    print(f"训练实际 full_text 结尾: {repr(train_full_text[-60:])}")

    # 逐字符检查渲染后文本末尾
    print(f"\n训练实际 full_text 末尾逐字符 (最后15个):")
    tail = train_full_text[-15:]
    for i, c in enumerate(tail):
        pos = len(train_full_text) - 15 + i
        print(f"  pos={pos}: {repr(c)} (Unicode: U+{ord(c):04X})")

    # 关键检查: <|im_end|> 后面是否有 \n
    raw_im_end_pos = raw_full_text.rfind("<|im_end|>")
    if raw_im_end_pos >= 0:
        raw_after_im_end = raw_full_text[raw_im_end_pos + len("<|im_end|>"):]
        print(f"\n原始 chat template 中 <|im_end|> 后面的内容: {repr(raw_after_im_end)}")

    im_end_pos = train_full_text.rfind("<|im_end|>")
    if im_end_pos >= 0:
        after_im_end = train_full_text[im_end_pos + len("<|im_end|>"):]
        print(f"训练实际 full_text 中 <|im_end|> 位置: {im_end_pos}")
        print(f"训练实际 full_text 中 <|im_end|> 后面的内容: {repr(after_im_end)}")
        if after_im_end == "":
            print(f"  [OK] 训练文本以 <|im_end|> 结束，没有尾部换行")
        elif after_im_end.strip() == "":
            print(f"  [WARN] 训练文本 <|im_end|> 后仍有空白字符 ({len(after_im_end)} 字符)")
        else:
            print(f"  [WARN] 训练文本 <|im_end|> 后有非空白内容!")

    # 批量检查: 所有样本渲染后末尾结构
    tail_patterns = {}
    for i in range(min(100, len(df))):
        msgs_i = normalize_messages(df.iloc[i]["messages"])
        try:
            ft = render_training_full_text(tokenizer, msgs_i)
        except TypeError:
            ft = render_training_full_text(tokenizer, msgs_i)
        # 找最后一个 <|im_end|> 后面的内容
        last_im_end = ft.rfind("<|im_end|>")
        if last_im_end >= 0:
            tail_after = ft[last_im_end + len("<|im_end|>"):]
            # 简化尾部为类型
            if tail_after == "\n":
                tail_key = "<|im_end|>\\n"
            elif tail_after == "":
                tail_key = "<|im_end|> (无后续)"
            elif tail_after.strip() == "":
                tail_key = f"<|im_end|> + 空白({len(tail_after)}字符)"
            else:
                tail_key = f"<|im_end|> + 内容({repr(tail_after[:30])})"
            tail_patterns[tail_key] = tail_patterns.get(tail_key, 0) + 1

    print(f"\n前100条样本渲染后 <|im_end|> 尾部模式:")
    for pattern, count in tail_patterns.items():
        print(f"  {pattern}: {count} 条")

    # 显示样本
    if show_samples > 0:
        print(f"\n--- 前 {show_samples} 条渲染后文本结尾 ---")
        for i in range(min(show_samples, len(df))):
            msgs_i = normalize_messages(df.iloc[i]["messages"])
            ft = render_training_full_text(tokenizer, msgs_i)
            print(f"\n样本 {i}:")
            print(f"  训练实际 full_text 结尾: {repr(ft[-80:])}")


# ── 阶段 3: tokenization + labels ──────────────────────────────


def stage3_tokenization_labels(df, tokenizer, max_length, show_samples=0):
    """检查 tokenization 和 labels 构造，特别是最后几个有效 label token。"""
    print("\n" + "=" * 60)
    print(f"阶段 3: tokenization + labels 检查 (max_length={max_length})")
    print("=" * 60)

    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_id = 198  # 常见 \n token id

    print(f"\n关键 token ID:")
    print(f"  eos_token_id: {eos_id} ({repr(tokenizer.decode([eos_id]))})")
    print(f"  <|im_end|>:   {im_end_id} ({repr(tokenizer.decode([im_end_id]))})")
    print(f"  \\n (常见):    {newline_id}")

    # 批量检查所有样本
    stats = {
        "total": 0,
        "truncated": 0,
        "last_label_is_newline": 0,
        "last_label_is_im_end": 0,
        "last_label_is_eos": 0,
        "last_label_is_other": 0,
        "im_end_in_truncated": 0,
        "im_end_missing_truncated": 0,
        "im_end_trailing_nonneg": 0,
    }
    last_label_samples = []

    for i in range(len(df)):
        msgs = normalize_messages(df.iloc[i]["messages"])

        full_text = render_training_full_text(tokenizer, msgs)
        prompt_text = render_training_prompt_text(tokenizer, msgs)

        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        # 模拟修复后的 labels 构造：mask <|im_end|> 之后的所有 token
        for j in range(len(labels) - 1, prompt_len, -1):
            if labels[j] == im_end_id:
                labels[j + 1:] = [-100] * len(labels[j + 1:])
                break

        stats["total"] += 1

        if len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) > max_length:
            stats["truncated"] += 1

        # 找最后一个非 -100 的 label
        last_label = None
        last_label_pos = None
        for j in range(len(labels) - 1, -1, -1):
            if labels[j] != -100:
                last_label = labels[j]
                last_label_pos = j
                break

        if last_label is None:
            continue

        if last_label == newline_id:
            stats["last_label_is_newline"] += 1
        elif last_label == im_end_id:
            stats["last_label_is_im_end"] += 1
        elif last_label == eos_id:
            stats["last_label_is_eos"] += 1
        else:
            stats["last_label_is_other"] += 1

        if len(last_label_samples) < 5:
            decoded = tokenizer.decode([last_label])
            last_label_samples.append(
                {
                    "idx": i,
                    "last_label_id": last_label,
                    "last_label_decode": repr(decoded),
                    "last_3_labels": [
                        repr(tokenizer.decode([labels[k]]))
                        for k in range(last_label_pos - 2, last_label_pos + 1)
                        if labels[k] != -100
                    ],
                }
            )

        # 截断后 <|im_end|> 是否还在
        if len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) > max_length:
            if im_end_id in full_ids:
                stats["im_end_in_truncated"] += 1
            else:
                stats["im_end_missing_truncated"] += 1

    # 输出统计
    print(f"\nlabels 最后一个有效 token 统计:")
    for key, count in stats.items():
        if key.startswith("last_label_"):
            pct = count / stats["total"] * 100 if stats["total"] else 0
            print(f"  {key}: {count} ({pct:.1f}%)")

    print(f"\n截断统计:")
    print(f"  被截断样本: {stats['truncated']}")
    print(f"  截断后 <|im_end|> 仍在: {stats['im_end_in_truncated']}")
    print(f"  截断后 <|im_end|> 丢失: {stats['im_end_missing_truncated']}")

    # eos_token_id 检查
    print(f"\n推理 eos_token_id 检查:")
    print(f"  tokenizer.eos_token_id: {eos_id} (训练数据中未出现的 <|endoftext|>)")
    print(f"  <|im_end|> id: {im_end_id} (训练数据中 assistant 回答的真实结束标记)")
    print(f"  建议: eos_token_id 只用 <|im_end|> ({im_end_id}), 不包含 eos_token ({eos_id})")

    # 详细样本
    if last_label_samples:
        print(f"\n最后有效 label token 样本 (前5条):")
        for s in last_label_samples:
            print(f"  样本 {s['idx']}: id={s['last_label_id']}, "
                  f"decode={s['last_label_decode']}, "
                  f"最后3个有效token={s['last_3_labels']}")

    # 关键结论
    print(f"\n--- 关键结论 ---")
    if stats["last_label_is_newline"] > stats["total"] * 0.5:
        print(f"  [WARN] 大多数样本的最后一个有效 label 是 \\n (token id={newline_id})")
        print(f"   这意味着模型被训练在 <|im_end|> 之后继续生成 \\n")
        print(f"   推理时模型不会在 <|im_end|> 处自然停止 -> 复读风险")
        print(f"   修复方法: 在 labels 中 mask <|im_end|> 之后的所有 token")
    elif stats["last_label_is_im_end"] > stats["total"] * 0.5:
        print(f"  [OK] 大多数样本的最后一个有效 label 是 <|im_end|>")
        print(f"   模型被训练在 <|im_end|> 处停止 -> 正常行为")
    else:
        print(f"  [WARN] 最后有效 label token 分布复杂，需逐条检查")

    # 显示样本详情
    if show_samples > 0:
        print(f"\n--- 前 {show_samples} 条 tokenization 详情 ---")
        for i in range(min(show_samples, len(df))):
            msgs = normalize_messages(df.iloc[i]["messages"])
            full_text = render_training_full_text(tokenizer, msgs)
            prompt_text = render_training_prompt_text(tokenizer, msgs)

            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_ids), len(full_ids))

            print(f"\n样本 {i}:")
            print(f"  总 token 数: {len(full_ids)}")
            print(f"  prompt token 数: {prompt_len}")
            print(f"  assistant token 数: {len(full_ids) - prompt_len}")

            # assistant 部分最后 5 个 token
            assistant_tokens = full_ids[prompt_len:]
            print(f"  assistant 最后5个token:")
            for t in assistant_tokens[-5:]:
                print(f"    id={t}, decode={repr(tokenizer.decode([t]))}")

            # labels 最后 5 个非 -100
            labels = list(full_ids)
            labels[:prompt_len] = [-100] * prompt_len
            non_100 = [(j, labels[j]) for j in range(len(labels)) if labels[j] != -100]
            if non_100:
                print(f"  labels 最后5个非-100 token:")
                for pos, tid in non_100[-5:]:
                    print(f"    pos={pos}, id={tid}, decode={repr(tokenizer.decode([tid]))}")


# ── 主函数 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SFT 训练数据三阶段校验工具")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--show-samples", type=int, default=0,
                        help="显示前N条样本详情 (默认0, 只看统计)")
    parser.add_argument("--stage", type=int, nargs="+", default=[1, 2, 3],
                        help="只运行指定阶段 (1/2/3)")
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    if not args.base_model_dir.exists():
        raise FileNotFoundError(args.base_model_dir)

    df = pd.read_parquet(args.parquet)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_dir), trust_remote_code=True)

    print(f"校验文件: {args.parquet}")
    print(f"Base model: {args.base_model_dir}")
    print(f"max_length: {args.max_length}")

    if 1 in args.stage:
        stage1_raw_parquet(df, show_samples=args.show_samples)
    if 2 in args.stage:
        stage2_chat_template(df, tokenizer, show_samples=args.show_samples)
    if 3 in args.stage:
        stage3_tokenization_labels(df, tokenizer, max_length=args.max_length,
                                  show_samples=args.show_samples)

    print("\n" + "=" * 60)
    print("校验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
