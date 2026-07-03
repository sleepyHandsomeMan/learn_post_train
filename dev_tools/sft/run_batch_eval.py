"""可断点续跑的批量评估脚本，支持 base 和 sft 模型。

每完成 20 条自动写入 checkpoint 文件，下次运行时自动检测
已有进度并跳过已完成的条目，避免重复生成。

用法:
  # base 模型
  python dev_tools/sft/run_batch_eval.py base --base-model-dir models/base/qwen3_0d6B --max-items 100 --max-new-tokens 512 --output-prefix 0d6b_base_test100

  # sft 模型
  python dev_tools/sft/run_batch_eval.py sft --base-model-dir models/base/qwen3_0d6B --adapter-dir models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2 --max-items 100 --max-new-tokens 512 --output-prefix 0d6b_eosfix2_test100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


YHY_DIR = Path(__file__).resolve().parents[2]
EVAL_FILE = YHY_DIR / "datasets" / "gsm8k_sft" / "test.parquet"
FORMAT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
FINAL_ANSWER_PATTERN = r"####\s*(-?[0-9][0-9,]*(?:\.\d+)?)"
CHECKPOINT_INTERVAL = 20


def normalize_messages(messages):
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(m) for m in messages]


def build_user_messages(question, include_format_instruction=True):
    question = str(question).strip()
    if include_format_instruction:
        content = question + " " + FORMAT_INSTRUCTION
    else:
        content = question
    return [{"role": "user", "content": content}]


def apply_chat_template_text(tokenizer, messages, add_generation_prompt, enable_thinking=False):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def render_prompt(tokenizer, question, model_kind, include_format_instruction=True):
    """渲染推理 prompt。base 模型不用 chat template，sft 模型用 chat template。"""
    messages = build_user_messages(question, include_format_instruction)
    if model_kind == "sft":
        return apply_chat_template_text(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)
    else:
        # base 模型直接用纯文本，不用 chat template
        return messages[0]["content"]


def extract_final_answer(text):
    text = str(text)
    match = re.search(FINAL_ANSWER_PATTERN, text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[0-9][0-9,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_first_hash_answer(text):
    match = re.search(FINAL_ANSWER_PATTERN, str(text))
    if not match:
        return None
    return match.group(1).replace(",", "")


def summarize_repetition(text):
    text = str(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_count = len(lines) - len(set(lines))
    hash_count = len(re.findall(r"####", text))
    final_answer_count = len(re.findall(FINAL_ANSWER_PATTERN, text))
    answer_is_count = len(re.findall(r"The answer is", text))
    repeat_like = hash_count > 1 or answer_is_count > 1 or repeated_line_count > 0
    return {
        "hash_count": hash_count,
        "final_answer_count": final_answer_count,
        "answer_is_count": answer_is_count,
        "repeated_line_count": repeated_line_count,
        "repeat_like": repeat_like,
    }


def build_eos_token_ids(tokenizer, model_kind):
    """根据模型类型选择 eos_token_id。

    base 模型用纯文本 prompt，不会生成 <|im_end|>，需要原生 EOS (151643) 来停止。
    sft 模型用 chat template，会生成 <|im_end|> 作为 assistant 结尾标记。
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id

    if model_kind == "base":
        # base 模型纯文本模式: 同时用原生 EOS 和 <|im_end|>
        ids = []
        if eos_id is not None:
            ids.append(int(eos_id))
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in ids:
            ids.append(im_end_id)
        return ids if ids else None
    else:
        # sft 模型 chat 模式: 只用 <|im_end|>
        if isinstance(im_end_id, int) and im_end_id >= 0:
            return im_end_id
        if eos_id is not None:
            return int(eos_id)
        return None


def score_one(prediction, gold_answer):
    """对单条预测计算全部指标。"""
    first_hash_answer = extract_first_hash_answer(prediction)
    pred_answer = first_hash_answer if first_hash_answer is not None else extract_final_answer(prediction)
    repetition = summarize_repetition(prediction)
    format_ok = first_hash_answer is not None
    single_final_answer_ok = repetition["final_answer_count"] == 1 and repetition["hash_count"] == 1
    return {
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
    }


def load_checkpoint(output_dir, output_prefix, max_items, max_new_tokens):
    """加载已有的 checkpoint 文件，返回已完成的 idx 列表和 rows。"""
    ckpt_path = output_dir / f"{output_prefix}_eval_{max_items}_max{max_new_tokens}_checkpoint.jsonl"
    if not ckpt_path.exists():
        return [], []
    rows = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    done_idxs = {r["idx"] for r in rows}
    print(f"断点续跑: 已有 {len(rows)} 条结果，从第 {len(rows)+1} 条继续")
    return list(done_idxs), rows


def save_checkpoint(output_dir, output_prefix, max_items, max_new_tokens, rows):
    """每 CHECKPOINT_INTERVAL 条写入一次 checkpoint。"""
    ckpt_path = output_dir / f"{output_prefix}_eval_{max_items}_max{max_new_tokens}_checkpoint.jsonl"
    with open(ckpt_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize_output(output_dir, output_prefix, max_items, max_new_tokens, rows):
    """全部完成后写最终 jsonl，删除 checkpoint。"""
    final_path = output_dir / f"{output_prefix}_eval_{max_items}_max{max_new_tokens}_full.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ckpt_path = output_dir / f"{output_prefix}_eval_{max_items}_max{max_new_tokens}_checkpoint.jsonl"
    if ckpt_path.exists():
        ckpt_path.unlink()
    return final_path


def compute_summary(rows, tag, max_new_tokens):
    """从 rows 计算汇总指标。"""
    df = pd.DataFrame(rows)
    return {
        "tag": tag,
        "n": len(df),
        "include_format_instruction": True,
        "max_new_tokens": max_new_tokens,
        "exact_match": float(df["exact_match"].mean()),
        "first_hash_exact_match": float(df["first_hash_exact_match"].mean()),
        "format_rate": float(df["format_ok"].mean()),
        "single_final_answer_rate": float(df["single_final_answer_ok"].mean()),
        "repeat_like_rate": float(df["repeat_like"].mean()),
        "avg_hash_count": float(df["hash_count"].mean()),
        "max_hash_count": int(df["hash_count"].max()),
        "avg_chars": float(df["pred_chars"].mean()),
    }


def run_eval(model, tokenizer, model_kind, max_new_tokens, max_items, output_dir, output_prefix):
    """运行评估，支持断点续跑和 checkpoint 写入。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(EVAL_FILE).head(max_items)

    # 断点续跑: 加载已有结果
    done_idxs, rows = load_checkpoint(output_dir, output_prefix, max_items, max_new_tokens)
    done_set = set(done_idxs)

    tag = f"{output_prefix}_max{max_new_tokens}_full"
    eos_token_id = build_eos_token_ids(tokenizer, model_kind)
    started = time.time()

    for _, row in df.iterrows():
        idx = int(row.name)
        if idx in done_set:
            continue

        messages = normalize_messages(row["messages"])
        user_content = messages[0]["content"]
        assistant_gold = messages[1]["content"]
        question = user_content.replace(" " + FORMAT_INSTRUCTION, "").strip()

        prompt_text = render_prompt(tokenizer, question, model_kind)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        gold_answer = extract_final_answer(assistant_gold)
        metrics = score_one(prediction, gold_answer)

        row_data = {
            "idx": idx,
            "tag": tag,
            "include_format_instruction": True,
            "max_new_tokens": max_new_tokens,
            "question": question,
            "prediction": prediction,
            "gold": assistant_gold,
            **metrics,
        }
        rows.append(row_data)

        # 进度打印: 只显示条数和累计指标
        n_done = len(rows)
        elapsed = time.time() - started
        em_rate = sum(r["exact_match"] for r in rows) / n_done
        fmt_rate = sum(r["format_ok"] for r in rows) / n_done
        avg_time = elapsed / n_done
        remaining = avg_time * (max_items - n_done)
        print(f"[{n_done}/{max_items}] EM={em_rate:.1%} fmt={fmt_rate:.1%} "
              f"elapsed={elapsed:.0f}s remaining~{remaining:.0f}s")

        # checkpoint 写入
        if n_done % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(output_dir, output_prefix, max_items, max_new_tokens, rows)

    # 最终写入
    final_path = finalize_output(output_dir, output_prefix, max_items, max_new_tokens, rows)
    summary = compute_summary(rows, tag, max_new_tokens)
    summary["seconds"] = round(time.time() - started, 2)

    # 写 summary json
    summary_path = output_dir / f"{output_prefix}_eval_{max_items}_max{max_new_tokens}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n完成! {n_done}条, 耗时{summary['seconds']}s")
    print(f"结果: {final_path}")
    print(f"指标: {json.dumps(summary, ensure_ascii=False, indent=2)}")
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description="可断点续跑的批量评估脚本")
    parser.add_argument("model_kind", choices=["base", "sft"], help="模型类型: base 或 sft")
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None, help="sft 模型需要指定 adapter 目录")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="默认根据 model_kind 自动选择")
    args = parser.parse_args()

    if args.model_kind == "sft" and args.adapter_dir is None:
        parser.error("sft 模型必须指定 --adapter-dir")

    # 自动选择输出目录
    if args.output_dir is None:
        if args.model_kind == "base":
            args.output_dir = YHY_DIR / "eval_results" / "base_model"
        else:
            # 按 output_prefix 创建子目录
            args.output_dir = YHY_DIR / "eval_results" / "sft_model" / args.output_prefix.replace("_test100", "").replace("_test500", "")

    base_model_dir = args.base_model_dir.resolve()
    if not base_model_dir.exists():
        raise FileNotFoundError(base_model_dir)

    print(f"模型类型: {args.model_kind}")
    print(f"base model: {base_model_dir}")
    print(f"测试条数: {args.max_items}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出前缀: {args.output_prefix}")

    # 加载 tokenizer
    tokenizer_dir = args.adapter_dir if args.model_kind == "sft" else base_model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir), dtype=dtype, device_map=device_map, trust_remote_code=True,
    )

    if args.model_kind == "sft":
        from peft import PeftModel
        adapter_dir = args.adapter_dir.resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)
        print(f"adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()

    summary, rows = run_eval(
        model, tokenizer, args.model_kind,
        max_new_tokens=args.max_new_tokens,
        max_items=args.max_items,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
