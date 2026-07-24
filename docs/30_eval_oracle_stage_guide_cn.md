# Base/SFT/GRPO 阶段能力评估与 oracle@8 诊断

> 本文记录当前 0.6B GSM8K 后训练实验中，如何判断 Base、SFT、GRPO 每个阶段的模型能力。
> 当前固定验证集为 `datasets/gsm8k_grpo/eval_100.parquet`，评估时统一使用 `max_new_tokens=256`。

## 1. 核心结论

每个阶段都要同时看两类能力：

```text
greedy@1：模型现在单次确定性回答能做到什么。
oracle@8：模型采样分布里是否已经包含正确答案。
```

`greedy@1` 更接近真实单次使用能力，`oracle@8` 更适合判断模型是否还有可训练、可搜索、可强化的潜力。

不能只看 oracle@8，因为 oracle@8 默认“有人能从 8 个答案里挑出正确答案”；真实使用中通常没有这个选择器。
后训练前看 `pass@k/oracle@k - greedy@1` 的差距，是为了判断正确轨迹是否已经存在但概率不够高；差距越大，越说明“激发潜力”的空间可能越大，但最终还取决于 reward/verifier 能否识别正确轨迹、训练是否真的把它推高。

## 2. 名词解释

### 2.1 greedy@1

`greedy@1` 指对每道题只生成 1 个答案，并且每一步都选择当前概率最高的 token。

常见设置：

```text
do_sample = false
num_return_sequences = 1
```

它回答的问题是：

```text
模型现在稳定、确定性地回答一次，正确率是多少？
```

在数学任务里，`greedy@1 EM` 是最重要的阶段对比指标之一。

### 2.2 sample@k

`sample@k` 指对同一道题用随机采样生成 `k` 个候选回答。

常见设置：

```text
do_sample = true
temperature = 0.7
top_p = 1.0
top_k = 50
k = 4 或 8
```

它不是单次能力，而是在观察模型的概率分布：

```text
模型会不会偶尔生成正确解法？
错误解法和正确解法的比例如何？
答案是否多样？
```

### 2.3 pass@k / oracle@k / oracle@8

`pass@k` 指采样 `k` 个候选，只要其中任意一个通过自动验证就算成功；代码任务里通常是“通过单元测试”，数学任务里可以类比为“有一个答案 exact match”。`oracle@k` 指假设有一个知道标准答案的裁判帮你从 `k` 个候选里挑，只要存在正确候选就算成功。

例如 `oracle@8`：

```text
同一道题采样 8 个回答
只要有 1 个答案正确
这道题的 oracle@8 就算对
```

它回答的问题是：

```text
模型的采样空间里是否已经存在正确轨迹？
```

这里的 `oracle` 不是一个真实模型，而是“理想裁判/上帝视角”：评估时知道标准答案，所以能判断 8 个候选里哪一个是对的。它不是线上能力指标，因为真实推理时通常没有标准答案帮你挑；它是后训练诊断指标，用来判断模型是否“会但不稳定”。如果 `pass@k/oracle@k` 明显高于 `greedy@1`，说明后训练、verifier 或 self-consistency 有潜力把正确轨迹推高；如果二者接近且都低，优先改 SFT 数据、模型规模或任务设置。

### 2.4 sample exact rate

`sample exact rate` 是所有采样候选里的平均正确率。

如果有 100 道题，每题采样 8 条，共 800 条候选，其中 300 条 exact match：

```text
sample exact rate = 300 / 800 = 37.5%
```

它比 oracle@8 更严格，因为 oracle@8 只关心“每题有没有至少一个对”，而 sample exact rate 关心“所有候选里正确答案占多少”。

### 2.5 temperature

`temperature` 控制采样随机性。

```text
temperature 低：更接近高概率答案，输出更稳定，但探索少。
temperature 高：探索更多，可能采到新路径，也更容易采到错误路径。
```

对当前 GSM8K 小模型实验，`temperature=0.7` 是合理的诊断起点。

### 2.6 top_p / top_k

`top_p` 和 `top_k` 用于限制采样候选 token 范围。

```text
top_k=50：每一步只从概率最高的 50 个 token 里采样。
top_p=1.0：不按累计概率额外截断。
```

它们影响 oracle@k，因此不同采样配置下的 oracle 结果不能直接混比。

### 2.7 exact match / EM

`exact match` 是自动抽取模型最终数字答案后，与标准答案比较是否相等。

当前脚本优先抽取：

```text
#### 后面的数字
```

如果没有 `####`，会退化为抽取最后一个数字。因此 Base 模型可能 EM 不低，但格式仍然很差。

### 2.8 format rate

`format rate` 表示模型是否输出了符合要求的 `#### final_answer` 格式。

对 GSM8K 后训练，SFT 的一个主要目标就是把 Base 模型拉到稳定格式：

```text
Base：可能会算，但不一定按格式输出。
SFT：学习标准解题格式和最终答案格式。
GRPO：在格式基础上提高高 reward 答案的概率。
```

### 2.9 rollout_n

`rollout_n` 是 GRPO 训练时每个 prompt 采样多少条回答。

它和 oracle@k 思想接近，但用途不同：

```text
oracle@8：评估诊断，事后看 8 条里有没有正确答案。
rollout_n=8：训练过程，同一 prompt 生成 8 条，用 reward 做组内比较并更新策略。
```

如果 SFT oracle@8 明显高于 greedy@1，通常说明增大 GRPO 的 `rollout_n` 有意义。

### 2.10 self-consistency@k

`self-consistency@k` 指采样 `k` 条答案后，用投票选择出现次数最多的最终答案。

它和 oracle@k 的区别：

```text
oracle@k：知道标准答案，选出正确候选，是诊断上限。
self-consistency@k：不知道标准答案，只按多数投票选，是可落地推理方法。
```

如果 oracle@8 很高但 self-consistency@8 不高，说明正确答案存在，但还需要 GRPO、verifier 或 reward model 帮模型把正确答案推高或选出来。

## 3. Base 到 SFT 再到 GRPO 的判断流程

### 3.1 Base model 阶段

Base 模型主要判断两件事：

```text
1. 有没有基础数学推理潜力。
2. 是否能遵循 GSM8K 的最终答案格式。
```

如果 Base greedy@1 低、format rate 低，但 oracle@8 有一定空间，说明模型有一些潜在能力，但需要 SFT 学任务格式和解题风格。

### 3.2 SFT model 阶段

SFT 模型主要判断：

```text
1. 相比 Base，greedy@1 是否提升。
2. format rate 是否明显提升。
3. oracle@8 是否明显高于 greedy@1。
```

如果 SFT oracle@8 明显高于 SFT greedy@1，说明模型“会一部分题，但不稳定”。这时 GRPO 有价值，因为 GRPO 可以尝试把采样中的正确轨迹推成更高概率输出。

### 3.3 GRPO model 阶段

GRPO 阶段不能只看 reward mean，还要看：

```text
greedy EM 是否超过 SFT
format rate 是否保持
avg length 是否异常变长或变短
approx_kl 是否过大或几乎为 0
clip_frac 是否长期过低
entropy 是否过快下降
```

如果 reward 上升但 EM 不升，可能是 reward 设计太粗、更新太弱、验证集太小，或者模型学到了格式而没有学到正确推理。

## 4. 当前固定验证集结果

评估口径：

```text
eval_file = datasets/gsm8k_grpo/eval_100.parquet
max_new_tokens = 256
eval_batch_size = 8
oracle temperature = 0.7
oracle top_p = 1.0
oracle top_k = 50
oracle_k = 8
seed = 42
```

注意：GRPO v4 的训练链路中，SFT adapter 是：

```text
models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2
```

GRPO checkpoint 的正确加载链路是：

```text
base model -> load SFT eosfix2 adapter -> merge_and_unload -> load GRPO LoRA checkpoint
```

### 4.1 Base / SFT / GRPO 对比表

| 阶段 | 模型/检查点 | greedy EM | oracle@8 | sample exact rate | format rate | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Base | qwen3_0d6B | 0.21 | 0.64 | 0.2025 | 0.00 / oracle 0.0013 | 有潜在数学能力，但格式基本不会 |
| SFT | eosfix2 adapter | 0.46 | 0.80 | 0.3838 | 0.92 / oracle 0.9538 | SFT 明显提升格式和单次能力，且采样空间里有更多正确轨迹 |
| GRPO | v4 checkpoint-59 | 0.48 | 0.82 | 0.3963 | 0.93 / oracle 0.9600 | 当前最佳 GRPO checkpoint 只比 SFT 小幅波动 |
| GRPO | v4 checkpoint-69 | 0.46 | 未跑 | 未跑 | 0.93 | 早停点没有超过 SFT |

## 5. 如何根据结果决定下一阶段

当前结果说明：

```text
Base greedy@1 = 0.21
Base oracle@8 = 0.64
SFT greedy@1 = 0.46
SFT oracle@8 = 0.80
GRPO checkpoint-59 greedy@1 = 0.48
GRPO checkpoint-59 oracle@8 = 0.82
```

这不是“0.6B 完全不会”的情况。正确答案已经大量存在于 SFT 采样分布里；GRPO checkpoint-59 的 oracle@8 只从 SFT 的 0.80 到 0.82，sample exact rate 只从 0.3838 到 0.3963，说明当前 GRPO v4 没有显著扩大正确轨迹空间，也没有把这些正确轨迹稳定推成 greedy 输出。

因此下一步不是盲目扩大训练步数，而是先检查和调整：

```text
1. rollout_n 从 4 提到 8，必要时试 16。
2. 学习率从 1e-6 小幅试到 2e-6 或 3e-6。
3. 使用 eval_100 作为早停验证集，不再用 eval_20 做主要判断。
4. 记录 reward_std、approx_kl、clip_frac，确认 GRPO 真的在更新。
5. 补跑 self-consistency@8，看多采样投票是否已经能带来实际推理收益。
6. 如果继续 GRPO，重点观察 oracle@8、sample exact rate、greedy EM 是否同步提升。
```

结果文件分别保存在 `eval_results/base_model/0d6b_base_greedy_eval100_len256_bs8/`、`eval_results/base_model/0d6b_base_oracle8_eval100_len256_temp0d7_bs8/`、`eval_results/sft_model/0d6b_eosfix2_greedy_eval100_len256_bs8/`、`eval_results/sft_model/0d6b_eosfix2_oracle8_eval100_len256_temp0d7_bs8/`。后续不要每次改验证集、`max_new_tokens`、`temperature`、`top_k`，否则结果很难比较；实验稳定后再把固定验证集扩大到 200-1000 条。

## 6. v4 小提升的排查重点

先分清“没信号”和“没更新”：组内 `reward_std` 经常为 0，说明 rollout 没有有效好坏差异；`approx_kl` 长期接近 0 且 `clip_frac` 很低，说明 actor 几乎没有离开 SFT/reference；reward 上升但 EM 下降，才更像 rule reward 被钻空子。当前 v4 更像更新太弱和 `rollout_n=4` 偏小，而不是典型 reward hacking。
