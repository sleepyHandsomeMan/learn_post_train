"""GRPO rollout 多样性与 reward 信号强度诊断脚本。

排查 GRPO 训练无效果的原因:
  1. rollout 多样性: 同一 prompt 生成 8 次回答, 有多少个不同答案
  2. reward 信号强度: 组内 reward 标准差为 0 的比例
  3. 全零步分析: 分离"全正确"和"全错误"的零信号步
  4. reward 分布: 不同 reward 值的出现频率

运行方式:
  D:/Anaconda/envs/test3/python.exe dev_tools/grpo_rollout_diagnosis.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "post_training_framework" / "src"))

import numpy as np
import torch
from transformers import AutoTokenizer

from ptf.train_grpo import (
    GRPOConfig,
    GRPOTrainer,
    compute_sequence_log_probs,
    compute_gsm8k_rule_reward,
    load_actor_and_reference,
    set_seed,
    _mean,
)
from ptf.rl_dataset import load_rl_dataset
from ptf.reward import GSM8KRewardConfig

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def format_bytes(b: int) -> str:
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def diagnose() -> None:
    # 配置: 使用和训练相同的参数
    cfg = GRPOConfig(
        base_model_dir=str(WORKSPACE_ROOT / "models/base/qwen3_0d6B"),
        sft_adapter_dir=str(WORKSPACE_ROOT / "models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2"),
        train_file=str(WORKSPACE_ROOT / "datasets/gsm8k_sft/train.parquet"),
        eval_file=str(WORKSPACE_ROOT / "datasets/gsm8k_sft/eval_20.parquet"),
        max_prompt_length=512,
        max_response_length=256,
        rollout_n=2,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        seed=42,
        fp16=True,
        gradient_checkpointing=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # 加载模型
    print("=" * 60)
    print("  加载模型...")
    print("=" * 60)
    actor, reference, tokenizer = load_actor_and_reference(
        base_model_dir=cfg.base_model_dir,
        sft_adapter_dir=cfg.sft_adapter_dir,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        device=device,
    )

    # 加载 eval 数据
    eval_dataset = load_rl_dataset(
        parquet_path=cfg.eval_file,
        tokenizer=tokenizer,
        max_prompt_length=cfg.max_prompt_length,
    )

    reward_config = GSM8KRewardConfig()
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # =========================================================================
    # 诊断1: rollout 多样性 (对每个 prompt 生成 8 次)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  诊断1: Rollout 多样性 (8次采样)")
    print("=" * 60)

    actor.eval()
    n_prompts = min(10, len(eval_dataset))  # 诊断用 10 个 prompt
    prompts = [eval_dataset[i] for i in range(n_prompts)]
    n_rollout_diag = 8  # 诊断用 8 次

    diversity_per_prompt: list[int] = []  # 每个 prompt 有多少个不同回答
    reward_per_prompt: list[list[float]] = []  # 每个 prompt 的 reward 分布

    for pi, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt.prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(device)

        responses: list[str] = []
        rewards: list[float] = []
        exact_matches: list[bool] = []

        for i in range(n_rollout_diag):
            gen_seed = cfg.seed + pi * n_rollout_diag + i
            torch.manual_seed(gen_seed)

            output_ids = actor.generate(
                **inputs,
                max_new_tokens=cfg.max_response_length,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            prompt_len = inputs.input_ids.shape[-1]
            response_ids = output_ids[0, prompt_len:].tolist()
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            reward_info = compute_gsm8k_rule_reward(
                response=response_text,
                gold_answer=prompt.ground_truth,
                config=reward_config,
            )

            responses.append(response_text)
            rewards.append(reward_info["score"])
            exact_matches.append(reward_info["exact_match"])

        # 统计多样性
        unique_responses = len(set(responses))
        diversity_per_prompt.append(unique_responses)

        # 统计 reward 分布
        reward_counter = Counter(rewards)
        reward_per_prompt.append(rewards)

        em_rate = sum(exact_matches) / len(exact_matches)
        print(f"\n  Prompt {pi}: ground_truth={prompt.ground_truth}")
        print(f"    不同回答数: {unique_responses}/{n_rollout_diag}")
        print(f"    EM率: {em_rate:.2f} ({sum(exact_matches)}/{n_rollout_diag})")
        print(f"    reward分布: {dict(reward_counter)}")
        if unique_responses < 4:
            # 打印部分回答内容帮助理解
            for ri, r in enumerate(responses[:3]):
                print(f"    回答{ri}: {r[:100]}...")

    avg_diversity = _mean(diversity_per_prompt)
    print(f"\n  平均不同回答数: {avg_diversity:.1f}/{n_rollout_diag}")
    print(f"  多样性评估: {'不足' if avg_diversity < 3 else '还行' if avg_diversity < 5 else '丰富'}")

    # =========================================================================
    # 诊断2: reward 信号强度 (模拟 rollout_n=2 和 rollout_n=4)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  诊断2: Reward 信号强度")
    print("=" * 60)

    # 用已有的 8 次 rollout 模拟不同 rollout_n
    for rollout_n in [2, 4, 8]:
        zero_std_count = 0  # 组内 reward_std=0 的次数
        total_groups = 0
        all_advantages: list[float] = []

        for rewards in reward_per_prompt:
            # 每 rollout_n 个为一组
            for start in range(0, len(rewards), rollout_n):
                group = rewards[start:start + rollout_n]
                if len(group) < rollout_n:
                    continue
                total_groups += 1

                group_std = float(np.std(group))
                if group_std == 0:
                    zero_std_count += 1
                else:
                    # 计算 advantage
                    group_mean = _mean(group)
                    for r in group:
                        adv = (r - group_mean) / group_std
                        all_advantages.append(adv)

        zero_ratio = zero_std_count / total_groups if total_groups > 0 else 0
        print(f"\n  rollout_n={rollout_n}:")
        print(f"    组内 reward_std=0 的比例: {zero_std_count}/{total_groups} ({zero_ratio*100:.1f}%)")
        print(f"    有信号组的 advantage 范围: {min(all_advantages):.2f} ~ {max(all_advantages):.2f}")
        print(f"    {'严重问题: 超过30%的组没有学习信号!' if zero_ratio > 0.3 else '可接受'}")

    # =========================================================================
    # 诊断3: 全零步的构成分析
    # =========================================================================
    print("\n" + "=" * 60)
    print("  诊断3: 全零步的构成 (全正确 vs 全错误)")
    print("=" * 60)

    for rollout_n in [2, 4]:
        zero_std_correct = 0  # 组内全答对的零信号步
        zero_std_wrong = 0    # 组内全答错的零信号步
        total_groups = 0

        # 需要同时看 exact_match 信息
        for pi in range(len(prompts)):
            em_list = []
            rewards = reward_per_prompt[pi]
            for i in range(n_rollout_diag):
                gen_seed = cfg.seed + pi * n_rollout_diag + i
                torch.manual_seed(gen_seed)
                # 重新算 exact_match (已经在上面算过了)
                # 从之前的 exact_matches 需要按 prompt 重新组织
                pass

            # 用上面循环的结果
            # rewards 和 exact_matches 已经收集了

        # 用已有训练数据来分析
        import pandas as pd
        train_csv = WORKSPACE_ROOT / "models/grpo/qwen3_0d6b_grpo_v2/plots/train_metrics.csv"
        if train_csv.exists():
            train_df = pd.read_csv(str(train_csv))
            total_steps = len(train_df)
            zero_std_steps = (train_df["reward_std"] == 0).sum()
            nonzero_std_steps = total_steps - zero_std_steps

            print(f"\n  从训练日志分析 (rollout_n=2, 共 {total_steps} 步):")
            print(f"    reward_std=0 的步数: {zero_std_steps}/{total_steps} ({zero_std_steps/total_steps*100:.1f}%)")
            print(f"    reward_std>0 的步数: {nonzero_std_steps}/{total_steps} ({nonzero_std_steps/total_steps*100:.1f}%)")

            # 进一步: policy_loss=0 的步数
            zero_policy = (train_df["policy_loss"] == 0.0).sum()
            print(f"    policy_loss=0 的步数: {zero_policy}/{total_steps} ({zero_policy/total_steps*100:.1f}%)")

            # reward 取值分布
            reward_counts = train_df["reward_mean"].round(2).value_counts().sort_index()
            print(f"    reward_mean 取值分布:")
            for val, cnt in reward_counts.items():
                print(f"      {val}: {cnt}次")

            # 训练前半 vs 后半的 reward 变化
            half = total_steps // 2
            first_half_reward = train_df["reward_mean"].iloc[:half].mean()
            second_half_reward = train_df["reward_mean"].iloc[half:].mean()
            print(f"    前半 reward均值: {first_half_reward:.3f}")
            print(f"    后半 reward均值: {second_half_reward:.3f}")
            print(f"    变化: {second_half_reward - first_half_reward:+.3f}")

        else:
            print("  训练日志不存在, 无法分析")

    # =========================================================================
    # 诊断4: 不同温度下的多样性对比
    # =========================================================================
    print("\n" + "=" * 60)
    print("  诊断4: 不同温度下的 rollout 多样性")
    print("=" * 60)

    temperatures = [0.7, 1.0, 1.2]
    test_prompt = prompts[0]  # 用第一个 prompt 测试

    inputs = tokenizer(
        test_prompt.prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    for temp in temperatures:
        responses_set: set[str] = set()
        rewards_list: list[float] = []

        for i in range(8):
            torch.manual_seed(42 + i)
            output_ids = actor.generate(
                **inputs,
                max_new_tokens=128,  # 缩短输出加快诊断
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                top_k=50,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            prompt_len = inputs.input_ids.shape[-1]
            response_ids = output_ids[0, prompt_len:].tolist()
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            responses_set.add(response_text)
            reward_info = compute_gsm8k_rule_reward(
                response=response_text,
                gold_answer=test_prompt.ground_truth,
                config=reward_config,
            )
            rewards_list.append(reward_info["score"])

        unique_count = len(responses_set)
        reward_std = float(np.std(rewards_list)) if len(rewards_list) > 1 else 0
        print(f"\n  temperature={temp}:")
        print(f"    不同回答数: {unique_count}/8")
        print(f"    reward 分布: {Counter([round(r, 2) for r in rewards_list])}")
        print(f"    reward_std: {reward_std:.3f}")

    # 显存状态
    print("\n" + "=" * 60)
    print("  显存状态")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"  allocated: {format_bytes(torch.cuda.memory_allocated())}")
        print(f"  reserved:  {format_bytes(torch.cuda.memory_reserved())}")
        print(f"  peak allocated: {format_bytes(torch.cuda.max_memory_allocated())}")
        torch.cuda.reset_peak_memory_stats()

    print("\n诊断完成。")


if __name__ == "__main__":
    diagnose()
