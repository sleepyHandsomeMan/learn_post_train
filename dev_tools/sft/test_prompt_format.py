"""测试修复后的 prompt 格式"""
from pathlib import Path
from transformers import AutoTokenizer

YHY_DIR = Path(__file__).resolve().parents[2]
BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)

# 测试纯文本 prompt（修复后的方式）
question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
format_instruction = 'Let\'s think step by step and output the final answer after "####".'

pure_text_prompt = question + " " + format_instruction

print("=== 纯文本 prompt（修复后）===")
print(pure_text_prompt)
print()

# 测试 chat template prompt（原来的方式）
messages = [{"role": "user", "content": pure_text_prompt}]
chat_template_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("=== Chat template prompt（原来的方式）===")
print(chat_template_prompt)
print()
print(f"纯文本长度: {len(pure_text_prompt)}")
print(f"Chat template 长度: {len(chat_template_prompt)}")
