from pathlib import Path
import argparse
import json
import re
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


YHY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_DIR = YHY_DIR / "models" / "base" / "qwen3_0d6B"
EVAL_FILE = YHY_DIR / "datasets" / "gsm8k_sft" / "test.parquet"
EVAL_OUTPUT_DIR = YHY_DIR / "eval_results" / "base_model"

FORMAT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
FINAL_ANSWER_PATTERN = r"####\s*(-?[0-9][0-9,]*(?:\.\d+)?)"


def build_model_tag(base_model_dir):
    """根据 base model 目录名生成适合放进评估文件名的短标签。"""
    raw = Path(base_model_dir).name.lower()
    chars = [char if char.isalnum() else "_" for char in raw]
    return "_".join("".join(chars).split("_"))


def normalize_messages(messages):
    """Convert parquet-loaded messages into a plain Python list of dicts."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(m) for m in messages]


def build_gsm8k_user_content(question, include_format_instruction=True):
    """Build the same GSM8K user prompt format used by SFT training."""
    question = str(question).strip()
    if include_format_instruction:
        return question + " " + FORMAT_INSTRUCTION
    return question


def build_user_messages(question, include_format_instruction=True):
    """Build inference messages containing only the user turn."""
    return [{"role": "user", "content": build_gsm8k_user_content(question, include_format_instruction)}]


def apply_chat_template_text(tokenizer, messages, add_generation_prompt):
    """Render messages through the tokenizer chat template.

    对于 base model 评估，不应使用 enable_thinking=False，因为这会在 prompt
    末尾插入空的 <think></think> 标签，导致未经指令微调的 base model 陷入循环生成。
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def render_generation_prompt(tokenizer, question, include_format_instruction=True):
    """Render the prompt for base model evaluation.

    Base model 应该直接续写纯文本 prompt，不使用 chat template。
    Chat template 包含 <|im_start|> 等 special tokens，未经 SFT 的 base model
    不理解这些 token，会导致生成循环或乱码。
    """
    # 对于 base model，直接返回纯文本 prompt，不使用 chat template
    return build_gsm8k_user_content(question, include_format_instruction)


def extract_final_answer(text):
    """Prefer a GSM8K #### answer; otherwise fall back to the last number."""
    text = str(text)
    match = re.search(FINAL_ANSWER_PATTERN, text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[0-9][0-9,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_first_hash_answer(text):
    """Extract only the first valid #### number from the full output."""
    match = re.search(FINAL_ANSWER_PATTERN, str(text))
    if not match:
        return None
    return match.group(1).replace(",", "")


def count_text_pattern(text, pattern):
    """Count occurrences of a regex pattern in generated text."""
    return len(re.findall(pattern, str(text)))


def summarize_repetition(text):
    """Compute repetition-like signals using the full generated output."""
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


def build_eos_token_ids(tokenizer):
    """用 <|im_end|> 作为唯一停止标记。

    Qwen3 的 eos_token 亏 (id=151643) 在训练数据中从未出现，
    模型不会在该 token 处自然停止，不应作为 eos_token。
    只保甥 <|im_end|> (id=151645)，这是训练数据中 assistant 回答的真实结束标记。
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return None


def generate_one(model, tokenizer, question, max_new_tokens, include_format_instruction=True):
    """Generate one GSM8K answer with the same prompt path as SFT evaluation."""
    prompt_text = render_generation_prompt(tokenizer, question, include_format_instruction)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_token_id = build_eos_token_ids(tokenizer)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=1.0,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, max_new_tokens, max_items=20, tag_prefix="base"):
    """Evaluate base model on eval_20 using the full SFT metric schema."""
    df = pd.read_parquet(EVAL_FILE)
    if max_items is not None:
        df = df.head(max_items)

    rows = []
    started = time.time()
    for _, row in df.iterrows():
        messages = normalize_messages(row["messages"])
        user_content = messages[0]["content"]
        assistant_gold = messages[1]["content"]
        question = user_content.replace(" " + FORMAT_INSTRUCTION, "").strip()

        prediction = generate_one(model, tokenizer, question, max_new_tokens=max_new_tokens)
        gold_answer = extract_final_answer(assistant_gold)
        first_hash_answer = extract_first_hash_answer(prediction)
        pred_answer = first_hash_answer if first_hash_answer is not None else extract_final_answer(prediction)
        repetition = summarize_repetition(prediction)
        format_ok = first_hash_answer is not None
        single_final_answer_ok = repetition["final_answer_count"] == 1 and repetition["hash_count"] == 1

        row_data = {
            "idx": int(row.name),
            "tag": f"{tag_prefix}_max{max_new_tokens}_full",
            "include_format_instruction": True,
            "max_new_tokens": max_new_tokens,
            "question": question,
            "gold_answer": gold_answer,
            "first_hash_answer": first_hash_answer,
            "pred_answer": pred_answer,
            "exact_match": pred_answer == gold_answer,
            "first_hash_exact_match": first_hash_answer == gold_answer,
            "format_ok": format_ok,
            "single_final_answer_ok": single_final_answer_ok,
            "hash_count": repetition["hash_count"],
            "final_answer_count": repetition["final_answer_count"],
            "answer_is_count": repetition["answer_is_count"],
            "repeated_line_count": repetition["repeated_line_count"],
            "repeat_like": repetition["repeat_like"],
            "pred_chars": len(prediction),
            "prediction": prediction,
            "gold": assistant_gold,
        }
        rows.append(row_data)
        elapsed = time.time() - started
        avg_per_item = elapsed / len(rows)
        remaining = avg_per_item * (len(df) - len(rows))
        em_rate = sum(r["exact_match"] for r in rows) / len(rows)
        fmt_rate = sum(r["format_ok"] for r in rows) / len(rows)
        rep_rate = sum(r["repeat_like"] for r in rows) / len(rows)
        print(
            f"[{len(rows)}/{len(df)} max={max_new_tokens}] "
            f"gold={gold_answer} pred={pred_answer} format={format_ok} "
            f"repeat={repetition['repeat_like']} hashes={repetition['hash_count']} "
            f"chars={len(prediction)}"
        )
        if len(rows) % 50 == 0 or len(rows) == len(df):
            print(
                f"  --- 进度: {len(rows)}/{len(df)} ({len(rows)/len(df)*100:.1f}%) "
                f"EM={em_rate:.1%} format={fmt_rate:.1%} repeat={rep_rate:.1%} "
                f"已用={elapsed:.0f}s 剩余~{remaining:.0f}s ({remaining/60:.1f}min) ---"
            )

    result_df = pd.DataFrame(rows)
    summary = {
        "tag": f"{tag_prefix}_max{max_new_tokens}_full",
        "n": len(result_df),
        "include_format_instruction": True,
        "max_new_tokens": max_new_tokens,
        "exact_match": float(result_df["exact_match"].mean()) if len(result_df) else 0.0,
        "first_hash_exact_match": float(result_df["first_hash_exact_match"].mean()) if len(result_df) else 0.0,
        "format_rate": float(result_df["format_ok"].mean()) if len(result_df) else 0.0,
        "single_final_answer_rate": float(result_df["single_final_answer_ok"].mean()) if len(result_df) else 0.0,
        "repeat_like_rate": float(result_df["repeat_like"].mean()) if len(result_df) else 0.0,
        "avg_hash_count": float(result_df["hash_count"].mean()) if len(result_df) else 0.0,
        "max_hash_count": int(result_df["hash_count"].max()) if len(result_df) else 0,
        "avg_chars": float(result_df["pred_chars"].mean()) if len(result_df) else 0.0,
        "seconds": round(time.time() - started, 2),
    }
    return summary, result_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--tag-prefix", type=str, default=None)
    args = parser.parse_args()

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_model_dir = args.base_model_dir.resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(base_model_dir)
    output_prefix = args.output_prefix or f"{build_model_tag(base_model_dir)}_base"
    tag_prefix = args.tag_prefix or output_prefix

    print("base model:", base_model_dir)
    print("tag_prefix:", tag_prefix)
    print("output_prefix:", output_prefix)

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    all_summaries = []
    for max_new_tokens in args.max_new_tokens:
        summary, rows = evaluate(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_items=args.max_items,
            tag_prefix=tag_prefix,
        )
        output_path = EVAL_OUTPUT_DIR / f"{output_prefix}_eval_{args.max_items}_max{max_new_tokens}_full.jsonl"
        rows.to_json(output_path, orient="records", lines=True, force_ascii=False)
        print("summary:", json.dumps(summary, ensure_ascii=False, indent=2))
        print("saved:", output_path)
        all_summaries.append(summary)

    print("all summaries:", json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
