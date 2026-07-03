"""诊断 base model 生成问题的脚本"""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

YHY_DIR = Path(__file__).resolve().parents[2]
BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"

def diagnose():
    print("=== 加载 tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    print("\n=== 加载模型 ===")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载成功，设备: {model.device}")
    print(f"模型 vocab size: {model.config.vocab_size}")

    print("\n=== 测试简单生成 ===")
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"输入 token IDs: {inputs['input_ids'][0].tolist()}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    print(f"\n生成的 token IDs: {new_tokens.tolist()}")
    print(f"生成的 token 数量: {len(new_tokens)}")

    # 检查 token ID 是否在合法范围内
    max_token_id = new_tokens.max().item() if len(new_tokens) > 0 else -1
    min_token_id = new_tokens.min().item() if len(new_tokens) > 0 else -1
    print(f"Token ID 范围: [{min_token_id}, {max_token_id}]")
    print(f"Vocab size: {tokenizer.vocab_size}")

    if max_token_id >= tokenizer.vocab_size:
        print(f"⚠️ 警告：生成的 token ID {max_token_id} 超出了 vocab size {tokenizer.vocab_size}")

    # 尝试解码
    print("\n=== 解码测试 ===")
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"解码后的文本: {repr(decoded)}")
    print(f"解码后的长度: {len(decoded)}")

    # 逐个 token 解码
    print("\n=== 逐个 token 解码 ===")
    for i, token_id in enumerate(new_tokens[:10].tolist()):
        try:
            single_decoded = tokenizer.decode([token_id])
            print(f"Token {i}: ID={token_id}, decoded={repr(single_decoded)}")
        except Exception as e:
            print(f"Token {i}: ID={token_id}, 解码失败: {e}")

    print("\n=== 测试 GSM8K 问题 ===")
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    format_instruction = 'Let\'s think step by step and output the final answer after "####".'
    user_content = question + " " + format_instruction

    messages = [{"role": "user", "content": user_content}]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    print(f"Prompt 长度: {len(prompt_text)}")
    print(f"Prompt 前 200 字符:\n{prompt_text[:200]}")

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"\n输入 token 数量: {inputs['input_ids'].shape[-1]}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    print(f"\n生成的 token IDs (前 20 个): {new_tokens[:20].tolist()}")
    print(f"生成的 token 数量: {len(new_tokens)}")

    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n解码后的文本:\n{repr(decoded[:200])}")
    print(f"解码后的完整长度: {len(decoded)}")

    # 检查是否全是齿轮符号
    if decoded.startswith('⚙'):
        print("\n❌ 问题确认：生成的文本全是齿轮符号！")
        print(f"前 10 个字符的 Unicode: {[hex(ord(c)) for c in decoded[:10]]}")

if __name__ == "__main__":
    diagnose()
