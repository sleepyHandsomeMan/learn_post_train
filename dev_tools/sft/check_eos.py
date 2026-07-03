from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/base/qwen3_0d6B", trust_remote_code=True)

messages = [
    {"role": "user", "content": "What is 2+2? Think step by step and output the final answer after \"####\"."},
    {"role": "assistant", "content": "2+2 = <<2+2=4>>4\n#### 4"},
]

# 渲染完整对话
full_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
print("=== full_text ===")
print(full_text)
print()
print("=== 最后 100 字符 ===")
print(repr(full_text[-100:]))
print()

# EOS token 信息
eos_id = tokenizer.eos_token_id
print(f"EOS token id: {eos_id}")
print(f"EOS token str: {repr(tokenizer.decode([eos_id]))}")

# <|im_end|> token
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_end|> token id: {im_end_id}")
print(f"<|im_end|> decode: {repr(tokenizer.decode([im_end_id]))}")
print()

# tokenize 并查看最后的 tokens
tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
print(f"总 token 数: {len(tokens)}")
print(f"最后 10 个 token ids: {tokens[-10:]}")
print(f"最后 10 个 token 解码: {[repr(tokenizer.decode([t])) for t in tokens[-10:]]}")

# 只渲染 prompt（不含 assistant 回复）
prompt_text = tokenizer.apply_chat_template(
    [messages[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print()
print("=== prompt_text (只有 user + generation prompt) ===")
print(repr(prompt_text[-80:]))
