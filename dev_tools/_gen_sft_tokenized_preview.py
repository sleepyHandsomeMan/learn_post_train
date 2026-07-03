"""生成 SFT tokenize 后的训练样本预览 —— 展示实际喂给模型的 input_ids / labels。"""
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

# ---- 模拟 MessageSFTDataset._tokenize_one 的完整逻辑 ----
BASE = Path('models/base/qwen3_0d6B')
PARQUET = Path('datasets/gsm8k_sft/eval_20.parquet')
NUM_SHOW = 3

tokenizer = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_parquet(PARQUET)


def apply_chat_template_text(tokenizer, messages, add_generation_prompt, enable_thinking=False):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_one(messages, tokenizer, max_length=768):
    """完全复刻 MessageSFTDataset._tokenize_one 的逻辑。"""
    full_text = apply_chat_template_text(
        tokenizer, messages,
        add_generation_prompt=False, enable_thinking=False,
    ).rstrip()

    prompt_text = apply_chat_template_text(
        tokenizer, [messages[0]],
        add_generation_prompt=True, enable_thinking=False,
    )

    full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, truncation=False)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if all(label == -100 for label in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "full_text": full_text,
        "prompt_text": prompt_text,
        "prompt_len": prompt_len,
    }


# ---- 生成预览 ----
lines = []
lines.append('# SFT 训练样本 tokenize 后预览')
lines.append('')
lines.append('以下是 `MessageSFTDataset` tokenize 后实际喂给模型的 `input_ids` 和 `labels`。')
lines.append('')
lines.append('- `labels=-100` 的位置：**不计算 loss**（user / prompt 部分）')
lines.append('- `labels=token_id` 的位置：**计算 loss**（assistant 回答部分）')
lines.append('')
lines.append('每个 token 用 `` `token_text`(token_id) `` 表示，labels 在 `[]` 内标注。')
lines.append('')
lines.append('---')
lines.append('')

for i, (_, row) in enumerate(df.iterrows()):
    if i >= NUM_SHOW:
        break

    msgs = row['messages']
    if hasattr(msgs, 'tolist'):
        msgs = msgs.tolist()
    msgs = [dict(m) for m in msgs]

    item = tokenize_one(msgs, tokenizer)
    if item is None:
        lines.append(f'## 样本 {i}: SKIPPED (assistant 全部被截断)')
        lines.append('')
        continue

    input_ids = item['input_ids']
    labels = item['labels']
    prompt_len = item['prompt_len']

    # 逐个 token 输出
    lines.append(f'## 样本 {i}')
    lines.append('')
    lines.append(f'- 总 token 数: {len(input_ids)}')
    lines.append(f'- prompt (user) token 数: {prompt_len} → labels=-100')
    lines.append(f'- assistant token 数: {len(input_ids) - prompt_len} → labels=token_id')
    lines.append('')

    # ---- 第一部分: user/prompt 区域 (labels=-100) ----
    lines.append('### user 部分 (labels=-100, 不计算 loss)')
    lines.append('')
    lines.append('```text')
    for pos in range(0, prompt_len):
        tid = input_ids[pos]
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        tok_text_escaped = tok_text.replace('\n', '\\n').replace('\r', '\\r')
        lines.append(f'{pos:4d}  {tok_text_escaped!r}  token_id={tid}  label=-100')
    lines.append('```')
    lines.append('')

    # ---- 第二部分: assistant 区域 (labels=token_id) ----
    lines.append('### assistant 部分 (labels=token_id, 计算 loss)')
    lines.append('')
    lines.append('```text')
    for pos in range(prompt_len, len(input_ids)):
        tid = input_ids[pos]
        label = labels[pos]
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        tok_text_escaped = tok_text.replace('\n', '\\n').replace('\r', '\\r')
        lines.append(f'{pos:4d}  {tok_text_escaped!r}  token_id={tid}  label={label}')
    lines.append('```')
    lines.append('')

    # ---- 第三部分: 完整解码对照 ----
    lines.append('### 完整 text 解码对照')
    lines.append('')
    lines.append('```text')
    full_decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    lines.append(full_decoded)
    lines.append('```')
    lines.append('')

    # 高亮 prompt 和 assistant 边界
    prompt_decoded = tokenizer.decode(input_ids[:prompt_len], skip_special_tokens=False)
    asst_decoded = tokenizer.decode(input_ids[prompt_len:], skip_special_tokens=False)
    lines.append(f'**prompt 部分解码:**')
    lines.append('')
    lines.append('```text')
    lines.append(prompt_decoded)
    lines.append('```')
    lines.append('')
    lines.append(f'**assistant 部分解码 (这部分计算 loss):**')
    lines.append('')
    lines.append('```text')
    lines.append(asst_decoded)
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')

# ---- 补充：token 统计 ----
lines.append('## 补充：特殊 token 速查')
lines.append('')
lines.append('| token | id | 说明 |')
lines.append('|---|---|---|')
for name in ['<|im_start|>', '<|im_end|>', '<|endoftext|>']:
    tid = tokenizer.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0:
        lines.append(f'| `{name}` | {tid} | |')

out = Path('datasets/gsm8k_sft/preview_eval_20_tokenized.md')
out.write_text('\n'.join(lines), encoding='utf-8')
print(f'OK: {out}')
