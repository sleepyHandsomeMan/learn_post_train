# GRPO 训练指标与终止条件说明

> 本文档说明 GRPO 训练过程中各项指标的含义、判断标准，
> 以及训练何时终止的规则。

---

## 1. 训练指标含义

### 1.1 每步训练日志

每步输出一行，格式：

```
[step 0/500] reward=0.300±0.173 policy_loss=0.0042 kl=0.0001 approx_kl=0.0001 clip_frac=0.001 resp_len=128 lr=1.00e-06 best_em=0.300(step=initial) no_improve=0/50 time=42.5s
```

逐项含义：

| 指标 | 含义 | 正常范围 | 异常信号 |
|---|---|---|---|
| `reward` | 本步 rollout 回答的平均规则奖励分数 | 0.3~0.8 | 全为负 → 模型输出全错; >1.0 → 可能 hacking |
| `reward±std` | 组内 reward 标准差 | >0 (有差异才有梯度) | std=0 → 组内无差异 → advantage=0 → 无学习信号 |
| `policy_loss` | PPO clipped surrogate loss | 小正值 (0.001~0.01) | 接近 0 且 clip_frac=0 → actor 几乎没移动 |
| `kl` | KL penalty loss | 很小 (0.0001~0.001) | 快速增长 → actor 偏离 reference 太远 |
| `approx_kl` | 近似 KL 散度 (ratio 的估计) | <0.02 | >0.1 → 异常终止 |
| `clip_frac` | 被 PPO clip 限制的 token 比例 | 0.01~0.2 | =0 → ratio 没超出 clip → actor 没在学; >0.5 → 更新太激进 |
| `resp_len` | 平均回答长度 | 80~256 | 固定=max_response_length → 模型不输出 EOS |
| `lr` | 当前学习率 | 1e-6 | 不变 (当前无 scheduler) |
| `best_em` | 历史最佳验证 exact_match | 逐步上升 | 始终不升 → 早停将触发 |
| `no_improve` | EM 连续不改善的步数 / 早停耐心 | 0/50 | 接近 50/50 → 即将早停 |
| `time` | 本步耗时 | 40~60s (RTX 4070) | 突然变长 → 可能 GC/OOM 回收 |

### 1.2 验证日志

每 eval_freq 步输出：

```
[eval step 9] reward=0.250 em=0.300 fmt=0.400 len=107
```

逐项含义：

| 指标 | 含义 | 正常范围 | 异常信号 |
|---|---|---|---|
| `reward` | 验证集平均规则奖励 | 逐步上升 | 上升但 EM 不升 → reward hacking |
| `em` | 验证集 exact_match (答案完全正确) | **最重要的指标** | 下降或不升 → 训练无效或 hacking |
| `fmt` | 验证集 format_rate (含 #### 且格式正确) | >0.5 | <0.1 → 格式退化 → 训练终止 |
| `len` | 验证集平均回答长度 | 稳定 | 突然变长 → 可能重复; 突然变短 → 可能跳过推理 |

### 1.3 显存日志

每步关键节点输出：

```
[显存:mini_batch_actor前向后] allocated=5.11 GB reserved=8.40 GB 预留池=3.29 GB 段数=12 活跃块数=58
```

含义：
- `allocated`: 真正数据占用的显存 (tensor 总大小)
- `reserved`: CUDA 分配器从 runtime 持有的显存总量 (含预留池)
- `预留池`: reserved - allocated，分配器持有但未分配给 tensor 的空间
- `段数`: CUDA runtime 分配的大块内存段数
- `活跃块数`: 当前正在使用的分配块数

训练过程中关注：
- `mini_batch_ref前向后_峰值点` 的 allocated → 总显存峰值
- `mini_batch_backward后` 的 allocated → 释放了多少激活值
- `step{N}_PPO更新后` 的 allocated → 回到基线，激活值已全部释放

---

## 2. 训练终止条件

训练不再固定跑 N 步就结束，而是有**三层保护机制**：

### 2.1 早停 (Early Stopping) — 验证 EM 连续不改善

```
条件: val_exact_match 连续 max_steps_no_improve 步没有超过历史最佳
默认: max_steps_no_improve = 50 步

判定逻辑:
  每次验证时, 如果 val_em > best_val_em → 更新 best, 重置计数器
  否则 → steps_no_improve += eval_freq
  当 steps_no_improve >= 50 → 终止训练

为什么用 EM 而不是 reward:
  reward 是优化信号, 模型可以"钻漏洞"让 reward 上升
  EM 是真实答案正确率, 是最终目标
  reward 上升 + EM 不升 = reward hacking, 不应该继续训练
```

### 2.2 KL 异常终止 — actor 偏离 reference 太远

```
条件: approx_kl > kl_threshold
默认: kl_threshold = 0.1

为什么需要:
  GRPO 的 KL penalty 是防止 actor 跑偏的关键机制
  approx_kl > 0.1 意味着 actor 和 reference 的分布差距已经很大
  继续训练 → actor 开始输出 reference 不认可的回答 → reward hacking 高风险

什么是 approx_kl:
  它是 PPO ratio 的近似 KL: ((ratio - 1) - log(ratio)) 的均值
  ratio = exp(log_prob_current - log_prob_old)
  ratio ≈ 1 → KL ≈ 0 → actor 没怎么移动
  ratio 远离 1 → KL 大 → actor 在快速偏离
```

### 2.3 格式退化检测 — 模型丧失格式能力

```
条件: val_format_rate < 0.1 且之前有过正的 EM
默认: 开启

为什么需要:
  格式退化意味着模型不再输出 #### answer 结构
  这是严重的训练崩溃信号, 模型可能陷入了某种退化模式
  例如: 只输出 EOS, 或输出无格式的长文本
```

### 2.4 Reward Hacking 警告 (不终止, 只警告)

```
条件: 训练 reward 上升但验证 EM 不改善
默认: reward_hacking_detect = True, window = 30 步

检测逻辑:
  比较最近 30 步的平均 reward 与之前 30 步的平均 reward
  如果最近 reward 更高, 但 val_em <= best_val_em
  → 发出警告: 模型可能在钻规则奖励的漏洞

不终止的原因:
  hacking 可能只是暂时的, KL penalty 和后续的 hard negative 可能纠正
  但需要人工检查:
    1. 看验证集输出是否格式正确但答案错
    2. 看 response 是否变长/模板化
    3. 决定是否需要回退到上一个 best_em checkpoint
```

### 2.5 最大步数兜底

```
条件: step >= total_training_steps
默认: total_training_steps = 500

这不是"目标步数", 而是上限兜底
正常训练应该被早停终止, 不是跑到 500 步
```

---

## 3. 各终止条件对应的处理

| 终止原因 | 含义 | 下一步 |
|---|---|---|
| 早停: EM连续N步不改善 | 模型已到当前配置的能力上限 | 分析 best checkpoint 的 EM, 考虑换数据/调参数 |
| KL异常终止 | actor 偏离太远, 有 hacking 风险 | 降低 lr, 增大 kl_coef, 从 best checkpoint 重新训练 |
| 格式退化 | 模型丧失格式输出能力 | 从 best checkpoint 重新训练, 增大 format 相关的 reward |
| 达到最大步数 | 兜底, 理论上不应触发 | 检查为什么早停没触发 (可能是 eval_freq 太大) |

---

## 4. GRPO 达标的具体判断标准

GRPO 训练的目标不是"reward 尽可能高", 而是"模型真正变好":

```
真正的改善 = val_exact_match 上升 + val_format_rate 保持 + response 长度稳定

具体达标判断:
  1. val_em 是否超过 SFT baseline → GRPO 有正面收益
  2. val_em 是否持续改善而非昙花一现 → 学习是稳定的
  3. val_fmt 是否没有退化 → 格式能力没有丧失
  4. reward 和 EM 是否同步上升 → 没有 hacking
  5. approx_kl 是否保持在 <0.02 → actor 没有跑偏
  6. response 长度是否稳定 → 没有变长(重复)或变短(跳过推理)

达标结果对比表:

checkpoint  | exact_match | format_rate | avg_length | rule_reward | human_win_rate
base        | 0.25        | 0.00        | 200+       | -0.57       | —
sft         | 0.50        | 0.85        | 150        | 0.62        | —
grpo_best   | ???         | ???         | ???        | ???         | —
```

---

## 5. 参数配置建议

```bash
# 正式训练 (从 SFT checkpoint 开始)
python post_training_framework/scripts/run_grpo_train.py \
  --base-model-dir models/base/qwen3_0d6B \
  --sft-adapter-dir models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2 \
  --train-file datasets/gsm8k_sft/train.parquet \
  --eval-file datasets/gsm8k_sft/eval_20.parquet \
  --total-training-steps 500 \
  --train-batch-size 2 \
  --rollout-n 2 \
  --ppo-mini-batch-size 4 \
  --learning-rate 1e-6 \
  --kl-loss-coef 0.001 \
  --max-steps-no-improve 50 \
  --kl-threshold 0.1 \
  --reward-hacking-detect \
  --reward-hacking-window 30 \
  --eval-freq 10 \
  --save-freq 10 \
  --max-response-length 256 \
  --output-dir models/grpo/qwen3_0d6b_gsm8k_rule_reward \
  --run-name grpo_0d6b_500steps \
  --val-before-train \
  --fp16 \
  --no-gradient-checkpointing \
  --seed 42
```

关键参数说明:
- `total-training-steps=500`: 兜底上限, 正常训练会被早停终止
- `max-steps-no-improve=50`: EM 连续 50 步不改善就停
- `kl-threshold=0.1`: KL 超过 0.1 就异常终止
- `reward-hacking-detect`: 开启 hacking 警告
- `reward-hacking-window=30`: 用 30 步的 reward 趋势做对比

---

## 6. GRPO 无效果的排查方法

当 GRPO 训练连续多步不改善 val_em 时，需要逐层排查信号链路：

### 6.1 信号链路：从 rollout 到梯度

```
Prompt → rollout_n 个回答 → rule reward → 组内 advantage → policy_loss → 梯度

每一步断裂都会导致"无效果":
  1. rollout 没有多样性 → 不同回答相同 → reward 全相同
  2. reward 粒度太粗 → 不同回答得到相同 reward → advantage=0
  3. advantage=0 → policy_loss=0 → 无梯度 → 模型不学习
```

### 6.2 层1：rollout 多样性不足

```
现象: 对同一个 prompt 多次 rollout, 回答几乎完全相同
原因: 模型太小 (0.6B), 概率分布尖锐, top-1 token 概率 > 80%
      即使 do_sample=True, temperature=0.7, 采样也几乎总是选同一个 token

验证方法: 对同一 prompt 做 8 次 rollout, 统计有多少个不同的回答
  ≥3 个不同回答 → 多样性还行, rollout_n=4 就够
  1~2 个不同回答 → 多样性不足, 需要提高 temperature 或 top_p

调整方向:
  - temperature: 从 0.7 提高到 1.0
  - top_p: 从 1.0 降到 0.9 (nucleus sampling, 截断尾部低概率token)
  - top_k: 从 50 提高到 100
```

### 6.3 层2：reward 全相同 → advantage=0

```
现象: 组内所有 rollout 的 rule reward 值完全一样
原因有两种:
  (a) rollout 没有多样性 → 所有回答相同 → reward 自然相同
  (b) rollout 有多样性, 但 rule reward 的粒度太粗

  (b) 的典型情况:
    4 个 rollout 给了 4 个不同答案, 但都是错的, 格式都正确:
    A1: #### 43 → reward = 0.2 (格式OK但答案错)
    A2: #### 44 → reward = 0.2 (格式OK但答案错)
    A3: #### 45 → reward = 0.2 (格式OK但答案错)
    A4: #### 46 → reward = 0.2 (格式OK但答案错)
    组内 reward 全相同 → advantage=0 → 无学习信号

  正常的情况 (reward 有差异):
    A1: #### 42  (正确) → reward = 1.3
    A2: #### 43  (错误) → reward = 0.2
    A3: 42       (正确但无格式) → reward = 0.55
    A4: 无最终答案 → reward = -0.2
    advantage 有强信号 → 可以学习

统计方法: 在训练中计算每步的组内 reward 标准差
  reward_std=0 的步数占比 → 如果 >30%, 说明超过三分之一的步是白学的
```

### 6.4 层3：rule reward 粒度太粗

```
问题: rule reward 只看最终结果 (对/错) 和格式 (有/无 ####)
      不看中间过程, 不区分"接近正确"和"完全离谱"

  对 GSM8K 数学题:
    答案 42 正确 → 1.3 分
    答案 43 (差1) → 0.2 分 (和答案 9999 同分!)
    答案 42 但无 #### → 0.55 分
    没有答案 → -0.2 分

  只分了 4档, 两档之间差距很大 (0.2→0.55→1.3)
  但"错"这个档里所有错误答案都是 0.2, 不管差多少

解决方案:
  1. 加 reward 粒度: 答案距离正确答案越近给越高分
     例如: abs(pred - gold) < 1 → 0.3分, < 10 → 0.1分
     这是 rule reward 的改进, 不需要训练 RM

  2. 加过程 reward: 中间推理步骤正确给部分分
     例如: 第一步加法正确 → +0.1分
     这需要标注中间步骤, 实现复杂度高

  3. 训练 Reward Model: 用 SFT 模型的输出构造偏好数据
     chosen = 格式正确+答案正确, rejected = 格式正确+答案错误
     RM 给连续分数, 粒度远比 rule reward 细
     但这是下一步 (RM→RLHF 流程), 不是当前 GRPO+rule reward 的范围
```

### 6.5 层4：模型能力上限

```
0.6B 模型在 GSM8K 上 EM≈50% 是合理水平
GRPO 能提升的空间主要在"边界情况":
  - 格式正确但答案错的 → GRPO 通过组内比较抑制错误答案模式
  - 答案正确但格式不对的 → GRPO 通过 reward 差异鼓励输出 ####

但 GRPO 不能把"不会做的题"变会做
  对模型完全不懂的题, 无论怎么调策略, rollout 都不会突然答对
  这类题在训练中是"噪声"——reward 随机, advantage 不稳定

结论: 对 0.6B 模型, GRPO+rule reward 的提升空间可能在 5~10% EM
      如果当前已经从 45% 到了 50%, 再提升可能需要:
      - 更大的模型 (1.7B)
      - 更细粒度的 reward (RM)
      - 更多的训练数据
```

### 6.6 排查流程总结

```
Step 1: 统计 rollout 多样性
  → 对 eval 集的 prompt 做 8 次 rollout
  → 看不同回答的数量分布

Step 2: 统计 reward 信号强度
  → 计算每步组内 reward_std
  → reward_std=0 的占比是多少

Step 3: 分析"全零步"的构成
  → 分离"全正确"步 (所有rollout都答对 → 正常的零信号)
    和"全错误"步 (所有rollout都答错但reward相同 → 真正的问题)

Step 4: 根据诊断结果调整
  → 多样性不足 → 提高 temperature/top_p
  → reward 粒度太粗 → 加距离奖励或换 RM
  → 模型上限 → 接受当前水平, 或换更大模型
```

---

## 6.7 实际排查案例 (0.6B Qwen3 + GSM8K)

训练了 60 步后早停终止, val_em 从 45% 提升到 50% 就不再改善。
排查过程和结果如下:

### 诊断1: Rollout 多样性 → 丰富

```
对 10 个 eval prompt 做 8 次 rollout:
  每个 prompt 都产生 8 个不同回答 (8/8)
  平均不同回答数: 8.0/8

结论: 0.6B 模型的输出多样性足够, temperature=0.7 不需要调高
注意: 温度越高反而降低正确率
  temperature=0.7 → EM=62.5%, reward_std=0.583
  temperature=1.0 → EM=12.5%, reward_std=0.390
  temperature=1.2 → EM=12.5%, reward_std=0.464
```

### 诊断2: Reward 信号强度 → 严重不足 (rollout_n=2)

```
组内 reward_std=0 的比例 (模拟不同 rollout_n):

  rollout_n=2 → 65% 的组没有学习信号 ← 这是核心问题!
  rollout_n=4 → 30% 的组没有信号 ← 刚好踩线
  rollout_n=8 → 10% 的组没有信号 ← 良好

  rollout_n=2 时, 65% 的步 advantage 全为 0 → policy_loss=0 → 白学
  这解释了训练中 43% 的步 policy_loss=0

  对比 rollout_n=2 vs 4:
    rollout_n=2: advantage 范围 [-1.0, +1.0], 信号弱
    rollout_n=4: advantage 范围 [-1.73, +1.73], 信号更强
    rollout_n=8: advantage 范围 [-2.65, +2.65], 信号最强
```

### 诊断3: 训练日志验证

```
60 步训练数据:
  reward_std=0: 14/60 (23.3%) — 注意这是均值, 单组更严重
  policy_loss=0: 26/60 (43.3%) — 近半步无梯度更新

  reward_mean 取值分布:
    0.55: 11次, 0.8: 19次, 1.05: 11次, 1.3: 12次
    集中在 {0.55, 0.8, 1.05, 1.3} 四个档

  前半 vs 后半:
    前半 reward均值: 0.888
    后半 reward均值: 0.833
    变化: -0.054 → 模型没有进步, 甚至略微退步
```

### 诊断4: Rule Reward 粒度分析

```
实际 reward 取值 (从诊断数据):

  答对+格式正确: 1.3 (占多数)
  答对+无格式: 0.55 (少)
  格式正确+答错: 0.3 (很多!)
  无格式+答错: -0.1~-0.2

  问题: 0.3 分这一档包含了所有"格式正确但答案错"的情况
  不管答案差1还是差1000, reward 都是 0.3
  → 组内如果几个 rollout 都是"格式对但答错" → reward 全是 0.3 → advantage=0

  典型案例 (Prompt 2, ground_truth=70000):
    8次 rollout 中 7次 reward=0.3 (格式对但答错)
    只有 1次 reward=1.3 (答对了)
    组内: 7个相同值 + 1个不同 → 如果 rollout_n=2,
    抽到两个 0.3 → reward_std=0 → 白学
```

### 调整决策

```
根据排查结果, 问题不是多样性不足, 而是:
  1. rollout_n=2 太少 → 65% 零信号步
  2. rule reward 粒度太粗 → 错误答案统一给 0.3 分

调整方案:
  1. rollout_n 从 2 提高到 4 (零信号比例从 65%降到 30%)
  2. max_response_length 从 256 降到 128 (减小显存压力, 适配更大 rollout_n)
  3. batch_size 保持 2 (显存约束)
  4. ppo_mini_batch_size 从 4 改到 8 (匹配新的总 rollout 数: 2×4=8)
  5. 考虑给 rule reward 加距离奖励 (后续优化)

显存估算 (rollout_n=4, max_response_length=128):
  actor 激活值 = 4条 × 128token × 28层 × ~166MB/层估计 ≈ 缩短序列后显存降低约一半
  之前的峰值 6.25 GB → 预计 ~3-4 GB → 有余量
```

---

## 7. 日志文件位置

训练日志同时输出到控制台和文件:

```
output_dir/logs/{run_name}.log
```

例如:
```
models/grpo/qwen3_0d6b_gsm8k_rule_reward/logs/grpo_0d6b_500steps.log
```

日志文件特点:
- 实时写入, 每条日志立即 flush 到磁盘
- 即使训练中途 OOM 崩溃, 崩溃前的最后一条日志也能保住
- 包含所有显存详细诊断 (DEBUG 级别)
- 格式: `时间 | 级别 | 内容`

可以用 grep 快速查找关键信息:

```bash
# 查看所有早停追踪
grep "早停追踪" logs/*.log

# 查看所有验证结果
grep "eval step" logs/*.log

# 查看显存峰值
grep "峰值点" logs/*.log

# 查看警告
grep "WARNING" logs/*.log
```
