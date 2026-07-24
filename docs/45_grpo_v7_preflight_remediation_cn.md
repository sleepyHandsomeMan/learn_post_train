# GRPO 有序训练、整改复盘与工程优化总笔记

> 本文档是当前 GSM8K GRPO 训练主线的单一知识入口。原 `42–47` 系列笔记已经按训练逻辑合并；后续同一主线的知识直接更新本文及目录，不再因为篇幅增长拆分。

## 文档定位与阅读路线

本文先回答“训练最终要达到什么目标，以及为什么必须按阶段推进”，再进入本轮 step 211 早停的因果排查。名词、显存、评估优化和历史案例作为可按需下钻的支撑部分。

```text
最终目标与验收标准
→ 数据、reward、checkpoint 与工程预检
→ smoke / diagnostic / formal 分阶段训练
→ 日志、验证、早停与 checkpoint 选择
→ 异常后按证据排查主要矛盾
→ 用控制变量实验确认主因
→ 通过稳定性和能力门禁后再扩模
```

阅读建议：

- 第一次系统学习：先读第一部分；遇到陌生名词跳到第三部分，再返回主线。
- 理解 PPO 与 GRPO 为什么采用不同的 KL 惩罚路径：读第一部分第 2.4 节。
- 从策略梯度定理完整推导 importance ratio、PPO clipping、GRPO advantage 和当前 LoRA optimizer update：读独立理论专题 [`50_grpo_ppo_policy_gradient_derivation_cn.md`](50_grpo_ppo_policy_gradient_derivation_cn.md)。
- 理解 signal guard 与 early stopping 的术语、判据和代码执行顺序：读第一部分第 9.3 节。
- 复盘当前训练：第一部分第12节之后接着读第二部分。
- 调整 batch、rollout 或评估速度：分别查第四、第五部分。
- 理解为什么正式训练前必须有诊断门禁：读第六部分。

## 结构化总目录

| 部分 | 内容 | 解决的问题 |
|---|---|---|
| 1 | [第一部分：GRPO v7 从预检到扩模的有序训练主线](#part-1) | 先明确最终目标、因果链、阶段门禁、正式训练与扩模条件。 |
| 2 | [第二部分：checkpoint-169 整改探针 step 211 早停因果分析](#part-2) | 沿日志证据排查主要矛盾、组合问题和下一轮控制变量实验。 |
| 3 | [第三部分：训练名词附录——从 rollout 到 optimizer update](#part-3) | 统一 epoch、batch、rollout、PPO epoch、step 和 optimizer update 的语义。 |
| 4 | [第四部分：GRPO 训练显存、共享内存与批量预算](#part-4) | 解释显存构成，并为 batch、rollout 和全量微调决策提供资源依据。 |
| 5 | [第五部分：GRPO 评估推理加速](#part-5) | 在不改变评估口径的前提下规划 batch size 和推理优化。 |
| 6 | [第六部分：历史案例——GRPO v5 早停与诊断门禁](#part-6) | 用早期失败案例理解组内诊断和小数据过拟合门禁为何必要。 |
| 维护 | [文档维护规则](#maintenance) | 规定后续何时并入、何时才允许拆分。 |

<a id="part-1"></a>

## 第一部分：GRPO v7 从预检到扩模的有序训练主线

> 本部分作用：先明确最终目标、因果链、阶段门禁、正式训练与扩模条件。

本部分目录：

- [1. 最终目标](#part-1-section-1)
- [2. GRPO 的因果链](#part-1-section-2)
- [3. 为什么必须按阶段推进](#part-1-section-3)
- [4. 阶段 0：定义目标与固定基线](#part-1-section-4)
- [5. 阶段 1：确认起点、数据和 reward](#part-1-section-5)
- [6. 阶段 2：静态预检与工程保障](#part-1-section-6)
- [7. 阶段 3：smoke3 验证最小闭环](#part-1-section-7)
- [8. 阶段 4：diagnostic20 验证链路能学](#part-1-section-8)
- [9. 阶段 5：为什么先做 formal120](#part-1-section-9)
- [10. 阶段 6：formal 后如何决策](#part-1-section-10)
- [11. 固化配置、执行与产物](#part-1-section-11)
- [12. 当前 formal120 证据快照](#part-1-section-12)
- [13. 复盘时的提问顺序](#part-1-section-13)
- [14. 按阶段下钻的专题索引](#part-1-section-14)

<a id="part-1-section-1"></a>

### 1. 最终目标

本项目不是以“训练脚本跑完”为目标，而是建立一条可复现、可诊断、可停止、可扩展的 GRPO 路线：

```text
固定 SFT 基线
→ 证明数据和 reward 能产生组内差异
→ 证明 GRPO LoRA 能收到梯度并改变策略
→ 证明变化能改善固定验证集
→ 证明中周期训练稳定
→ 再决定停止、修复或扩大训练
```

最终成功标准是验证集能力改善且无明显副作用。reward 上升、训练跑满或 GPU 忙碌都不能单独证明成功。

<a id="part-1-section-2"></a>

### 2. GRPO 的因果链

理解和排查必须沿同一条链推进：

```text
prompt
→ 同一 prompt 生成多个 rollout
→ rule reward 对回答排序
→ 组内 reward 归一化得到 advantage
→ GRPO/PPO loss 产生梯度
→ 只更新 GRPO LoRA
→ actor 分布发生变化
→ 固定 eval100 检验变化是否泛化
```

如果前一环没有证据，直接讨论后一环没有意义。例如 reward 全相同会使 advantage 接近 0，此时增加总步数不能解决问题。

#### 2.1 为什么 GRPO 必须看“同题组内差异”

对同一个 prompt 生成 `G` 个回答并得到 reward `r1...rG`。当前实现按组归一化，直观形式为：

```text
Ai = (ri - group_mean) / group_std
```

- 回答奖励高于组均值，advantage 为正，训练提高其概率。
- 回答奖励低于组均值，advantage 为负，训练降低其概率。
- 如果一组 reward 完全相同，`group_std=0`，该组几乎没有相对学习信号。

因此 `effective_group_rate` 回答“多少组有非零 reward 方差”，`mixed_group_rate` 回答“多少题同时采到正确和错误答案”。它们比全局平均 reward 更靠近 GRPO 的学习源头。

#### 2.2 actor、old policy 与 reference 的区别

| 对象 | 当前项目中的含义 | 用途 |
|---|---|---|
| actor | SFT model + 可训练 GRPO LoRA | 生成 rollout，并接受梯度更新 |
| old policy | 本批 rollout 生成时的 actor 概率快照 | 计算 PPO ratio，限制一次批内更新幅度 |
| reference | 冻结的 SFT model | 计算 KL，限制长期偏离 SFT 起点 |

old policy 和 reference 不是同一个概念：old policy 约束“这一批 PPO 更新不要跳太远”，reference 约束“整个 GRPO 过程不要远离 SFT”。

#### 2.3 从 log-prob 到 LoRA 梯度

对每个回答计算：

```text
ratio = exp(actor_logp - old_logp)
policy_loss = -min(ratio * A, clip(ratio) * A)
total_loss = policy_loss + kl_coef * KL(actor || reference)
```

`clip_ratio=0.2` 把 ratio 约束在近似 `[0.8, 1.2]` 范围；修正版以 `kl_loss_coef=0.005` 起步并自适应调节，`reference_kl` 与 `update_kl` 分开保护。反向传播后只有 GRPO LoRA 获得梯度。

<a id="part-1-section-2-4"></a>

#### 2.4 PPO 与 GRPO 为什么采用不同的 KL 惩罚路径

这一问题不能简化成“KL 放在 reward 里还是 loss 里哪个更先进”。两种写法都来自同一个 reference-regularized policy optimization 目标；真正决定实现路径的是 advantage、Value/GAE、组内归一化和 token 信用分配发生在哪一层。

##### 2.4.1 先区分 reward function 与 Reward Model

InstructGPT、DeepSpeed-Chat 和 TRLX 所说的“把 KL 加入 reward”，通常不是修改训练好的 Reward Model 网络，而是由 PPO trainer 在 Reward Model 分数之外构造一个 `non_score_reward`：

```text
Reward Model:
  只输出整条回答的偏好分数 RM_score

PPO trainer:
  每个 token 构造 -β × log(π_actor / π_reference)
  最后一个 token 再加 RM_score
```

因此应区分：

| 名称 | 含义 |
|---|---|
| Reward Model | 根据 prompt/response 输出偏好分数的模型 |
| task/rule reward | 数学正确性、格式等奖励 |
| KL non-score reward | trainer 根据 actor/reference log-prob 计算的约束成本 |
| total reward | task/RM reward 与 KL cost 合成后的训练奖励 |

当前项目没有 Reward Model，task reward 来自 GSM8K 规则函数；KL 也没有写入规则奖励，而是在 policy loss 之后作为独立正则项加入。

##### 2.4.2 两种实现共享同一个数学目标

设 `πθ` 为 actor，`πref` 为冻结 SFT reference，完整回答为 `y`。两种实现都希望最大化：

```text
J(θ) = E_{y~πθ}[
  R_task(y)
  - β × log(πθ(y) / πref(y))
]
```

它等价于任务收益减去 reference KL：

```text
J(θ) = E[R_task] - β × KL(πθ || πref)
```

可以从两个角度实现：

```text
奖励塑形:
  R'(y) = R_task(y) - β × K(y)
  再用 R' 计算 return / advantage

独立 loss 正则:
  advantage 只由 R_task 计算
  total_loss = policy_loss + β × kl_loss
```

在完全 on-policy、无限样本、没有归一化、没有 Value 近似、没有 clipping 且精确计算期望的理想条件下，两者可以对应同一个正则化目标。但真实 PPO/GRPO 会加入不同的估计器和归一化，因此最终梯度不再相同。

##### 2.4.3 PPO 为什么通常把 KL 构造成 token reward

经典 PPO-RLHF 使用 actor-critic 数据流：

```text
token reward
→ discounted return
→ Value 预测
→ GAE advantage
→ PPO actor loss + Value loss
```

Reward Model 通常只在回答结束时给一个稀疏的 sequence score：

```text
r_1 ... r_{T-1} = 0
r_T = RM_score
```

PPO trainer 加入 KL 后，每个 token 都有约束成本：

```text
r'_t = -β × log(π_actor(a_t|s_t) / π_reference(a_t|s_t))
r'_T = r'_T + RM_score
```

这样做与 PPO 结构有四个直接关系：

1. **Value 目标更容易保持一致**：critic 学习的是未来 `RM reward - KL cost`，因此可以直接解释为正则化回报的 Value。若只给 actor loss 添加 KL、critic 仍预测纯 RM reward，数学上并非无效——baseline 不必严格等于完整目标的 Value——但它已经是另一种梯度估计与方差控制设计，不能再把 critic 解释成完整正则化回报的预测器。
2. **token 级信用分配**：哪个 token 开始偏离 reference，就在哪个位置产生 KL cost；GAE 再把这些局部成本传播到相应 token advantage。
3. **复用成熟基础设施**：reward buffer、return、GAE、reward whitening、Value loss 和自适应 KL controller 都可以沿用，不必另建一条 actor 正则分支。
4. **适合稀疏终局奖励**：RM score 只在末尾出现，而 KL 提供稠密约束，使长回答的训练过程不只依赖一个终局分数。

所以 PPO 把 KL 放入 reward 既有数学结构原因，也有成熟工程栈的历史原因。它不是把 KL 训练进 Reward Model，而是在 rollout/experience 阶段构造正则化 reward。

##### 2.4.4 GRPO 为什么通常把 KL 留在组 reward 之外

critic-free GRPO 不训练 Value，也不使用 GAE。对同一道题的 `G` 个回答，它先计算：

```text
A_i = (R_i - μ_R) / σ_R
```

如果先把 KL 并入 sequence reward：

```text
R'_i = R_i - βK_i
```

组内 advantage 就变成：

```text
A'_i = [
  (R_i - μ_R)
  - β(K_i - μ_K)
] / σ_{R-βK}
```

这个展开式揭示了三个底层问题。

第一，**绝对 KL 被变成组内相对 KL**。组均值会把共同偏离抵消：

```text
K = [0.10, 0.10, 0.10, 0.10]
K - mean(K) = [0, 0, 0, 0]
```

即使整组回答都已经远离 SFT，只要组内 KL 相近，KL 信号就可能在有限采样中消失。独立 loss 使用绝对 actor-reference KL，不依赖同组是否存在 KL 差异。

第二，**除以组内标准差会扭曲 β 的尺度**。例如：

```text
R = [1.0, 1.0]
K = [0.020, 0.021]
β = 0.01

R' = [0.99980, 0.99979]
```

原始差异极小，但两条样本做组内标准化后仍可能接近 `[+1, -1]`。此时把 `β` 再缩小十倍，也可能得到近似相同的标准化 advantage；`β` 不再线性表达 KL 约束强度。

第三，**task reward 排序被 KL 混入**。GRPO 原本希望回答“同一道数学题中哪个答案更好”；合并后回答的是“任务正确性减去策略偏离后哪个更高”。错误但像 SFT 的回答可能压过正确但路径更新颖的回答，妨碍策略探索超越 SFT。

还有一个 token 信用问题：当前 GRPO 的 sequence advantage 会扩展到回答的所有 token。如果把整条回答的累计 KL 塞进一个 sequence reward，就无法精确指出是哪个 token 开始偏离；独立 token-level KL loss 可以在实际偏离位置产生回拉梯度。

##### 2.4.5 两种路径的结构对照

| 维度 | PPO 中 KL 作为 reward | GRPO 中 KL 作为独立 loss |
|---|---|---|
| task advantage 是否包含 KL | 是 | 否 |
| 是否训练 Value | 是，Value 学习正则化回报 | 否 |
| KL 是否经过单题组内归一化 | 通常不经过 | 不经过 |
| token 信用分配 | token reward + GAE | token KL 梯度 |
| 绝对 reference 偏离 | 进入 return/advantage | 直接进入 loss |
| `β` 的可解释性 | 受 reward/GAE/whitening 影响 | 保持独立线性权重 |
| 更自然的算法结构 | PPO / actor-critic | critic-free GRPO |

这里的边界不是绝对规定。PPO 也可以直接在 actor loss 中加 KL，但要同步处理 critic 目标；GRPO 也可以构造 KL reward，但不应让它与 task reward 一起经过普通组内标准化。若先分开计算 task advantage，再以绝对尺度加入 token KL，最终又接近独立 loss 正则。

##### 2.4.6 当前项目的实现与决策

当前代码明确采用：

```text
group advantage = GroupNormalize(rule_reward)

policy_loss =
  -mean[min(ratio × A, clip(ratio) × A)]

total_loss =
  policy_loss
  + current_kl_loss_coef × kl_loss
```

对应代码证据：

- `post_training_framework/src/ptf/reward.py`：只计算 answer、format、single-final、repeat、length、truncated 等规则奖励，不包含 KL。
- `post_training_framework/src/ptf/train_grpo.py::_compute_advantages()`：只根据 rule reward 做组内归一化。
- `post_training_framework/src/ptf/train_grpo.py::compute_kl_loss()`：比较 actor 与冻结 reference 的 token log-prob。
- `post_training_framework/src/ptf/train_grpo.py::_train_mini_batch()`：执行 `total_loss = policy_loss + current_kl_loss_coef * kl_loss`。

还必须区分“行业常见 PPO 路径”和“本仓库当前 PPO 实现”。本仓库的 `post_training_framework/src/ptf/train_ppo.py` 暂时也采用独立 KL loss：

- `_compute_rewards_and_advantages()` 构造的 token reward 只有末 token 的 rule reward，没有减去 token KL；因此 critic 的 return/Value 目标也是纯 task reward。
- `_train_mini_batch()` 单独计算 actor-reference `kl_loss`，再执行 `total_loss = policy_loss + value_loss - entropy_coef * entropy + kl_loss_coef * kl_loss`。

所以，本仓库当前 PPO 属于“PPO actor 额外添加 KL 正则”的可行变体，不是 InstructGPT、DeepSpeed-Chat、TRLX 的 token-KL reward 路径。它不影响本轮 GRPO 结论，但以后正式做 PPO 对照实验时必须明确实验标签：如果保留当前写法，应称为 `PPO + direct actor KL regularization`；如果改成工业常见写法，则必须把 token KL 纳入 reward 后重新计算 return、GAE 和 Value target，不能只移动一行 loss 代码并宣称两者等价。

当前项目继续保留独立 KL loss，原因是：

1. task advantage 只表达数学正确性和格式质量。
2. KL 不会被单题组均值抵消。
3. `kl_loss_coef` 仍是可单独调节和解释的约束权重。
4. 即使一组 task reward 全相同、policy loss 接近 0，KL 仍可把 actor 拉回 reference。
5. 实现与 DeepSeekMath 的 GRPO 目标和当前 TRL GRPO 的 `per_token_loss + β × per_token_kl` 路径一致。

因此，本轮 repair probe 中 reference KL 仍然失控，首先说明的是 KL 控制器介入太迟或有效纠偏太弱，不说明 KL 放置层级错误。下一轮应优先消融初始系数、target、调整间隔、增长因子、PPO epoch 和 optimizer 状态，而不是直接把 KL 移入 rule reward。

##### 2.4.7 结论：数学结构决定默认路径，实验决定具体超参数

```text
共同数学目标:
  maximize task reward - β × reference KL

PPO 的默认路径:
  KL → token reward → return / Value / GAE → PPO loss

GRPO 的默认路径:
  task reward → group advantage → policy loss
  KL → 独立 token regularization loss

经验实验负责确认:
  β 多大、多久调整一次、target 设多少、何时停止

算法结构负责解释:
  为什么 KL 不应随意穿过 Value/GAE 或组内标准化
```

所以“PPO 更常把 KL 放进 reward、GRPO 更常把 KL 放在 reward 外”不是纯经验习惯，也不是不可违背的定理；它是同一正则化目标在两种 advantage estimator 下的自然分解，随后由工程实验验证稳定性。

外部事实源：

- [InstructGPT：PPO 阶段加入来自 SFT model 的 per-token KL penalty](https://arxiv.org/abs/2203.02155)
- [DeepSpeed-Chat：`compute_rewards()` 先构造 token KL reward，再在末 token 加 RM score](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L181-L194)
- [TRLX：KL penalty 作为 token reward，最终 token 加 score](https://github.com/CarperAI/trlx/blob/main/trlx/trainer/accelerate_ppo_trainer.py#L473-L485)
- [DeepSeekMath：GRPO 目标中的独立 KL 正则](https://arxiv.org/abs/2402.03300)
- [TRL GRPO：`per_token_loss` 直接加 `beta * per_token_kl`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L2958-L3010)

#### 2.5 为什么指标必须联合解释

- `reward_mean`：回答得分水平，受抽题难度影响，单步波动很大。
- `advantage_std`：组内归一化后是否形成相对信号。
- `grad_norm`：信号是否真正进入可训练参数。
- `approx_kl`：一次 PPO 更新相对 old policy 的变化量。
- `clip_frac`：有多少 token ratio 触及裁剪区；很低不等于没训练。
- `val_exact_match`：策略变化是否转化成固定验证集能力。

可靠证据链应是“组内有差异 → 梯度非零 → adapter 变化 → eval 响应”，不能用其中一个指标替代整条链。

<a id="part-1-section-3"></a>

### 3. 为什么必须按阶段推进

| 阶段 | 要回答的问题 | 通过后为什么进入下一步 |
|---|---|---|
| 0. 目标与基线 | 要改善什么、和谁比较 | 没有固定基线就无法判断 RL 收益 |
| 1. 起点与数据 | 模型起点、样本、reward 是否正确 | 输入契约正确后才值得执行训练代码 |
| 2. 静态预检 | 配置、保存、resume、监控是否完整 | 可复现后才允许消耗 GPU 做 smoke |
| 3. smoke3 | 链路能否完成一次最小闭环 | 能运行后再检查“能不能学” |
| 4. diagnostic20 | 是否有信号、梯度、权重和 eval 响应 | 确认能学后才进入中周期正式实验 |
| 5. formal120 | 中周期是否稳定、最佳点在哪里 | 曲线仍在改善才有扩大训练的理由 |
| 6. 决策 | 停止、修复还是扩模 | 由证据决定，不预设一定扩模 |

<a id="part-1-section-4"></a>

### 4. 阶段 0：定义目标与固定基线

当前任务是 GSM8K，最终答案格式为 `#### <answer>`。比较对象固定为 eosfix2 SFT：

```text
固定验证集: datasets/gsm8k_grpo/eval_100.parquet
SFT greedy EM: 0.47
关注指标: EM、format、response length、reward
```

这一阶段先回答“GRPO 要在什么口径上超过 SFT”。下一阶段才能判断训练输入是否有资格服务这个目标。

<a id="part-1-section-5"></a>

### 5. 阶段 1：确认起点、数据和 reward

#### 5.1 模型起点

```text
Base + SFT adapter merge = SFT model
SFT model + 新建 GRPO LoRA = 可训练 actor
冻结 SFT model = reference model
```

Base、已合并的 SFT 权重和 reference 都不更新，优化器只更新 GRPO LoRA。恢复顺序必须是 `Base → SFT merge → GRPO adapter`。

#### 5.2 分桶数据

- 5759 条训练 messages 唯一、无空值，与固定 eval100 重叠为 0。
- 两个 oracle-mixed 桶全量使用，共 5425 条；其他桶抽取 334 条边界样本。
- oracle-mixed 表示同一题的多个 rollout 有对有错，更可能产生有效组内 advantage。

#### 5.3 reward 顺序

rewardv2 保证正确性优先于格式：

```text
正确且有格式: 1.10
正确但无格式: 0.30
错误但格式正确: 0.10
重复: -0.50
多 final: -0.20
```

只有模型起点正确、验证集无污染、数据存在可学差异且 reward 排序合理，才进入静态预检。

#### 5.4 六类分桶各自解决什么问题

| 分桶 | 说明 | 在 GRPO 中的价值 |
|---|---|---|
| greedy wrong / oracle mixed | 贪心答错，但采样有对有错 | 最直接的能力提升样本，组内信号强 |
| greedy correct / oracle mixed | 贪心答对，但采样不稳定 | 稳固正确策略，减少能力回退 |
| greedy wrong / oracle all correct | 贪心路径差，但随机路径普遍会做 | 小量保留，帮助纠正解码偏好 |
| greedy correct / oracle all correct | 已基本掌握 | 只采边界样本，避免训练预算被简单题占满 |
| greedy wrong / oracle all wrong | 当前能力上限外或题目异常 | 少量保留用于困难边界，不作为主体 |
| greedy correct / oracle all wrong | 贪心偶然正确或采样脆弱 | 少量保留，用于稳定性诊断 |

两个 mixed 桶全量使用，是因为它们天然更容易让同题 rollout 产生 reward 差异；其他桶只做边界补充，避免大量零 advantage 组稀释训练。

#### 5.5 数据与 reward 的最低审计清单

1. 训练 parquet 含当前框架使用的 `messages`，不是 verl 格式字段。
2. 训练与固定验证集无 prompt/message 重叠。
3. `####` 答案提取在正确、错误、缺格式、多 final 和重复输出上符合预期。
4. `run_config.json` 记录训练/验证文件 SHA256，防止同名文件被替换后无法复盘。
5. 人工抽查每个桶的题目和选择角色，确认“可学性标签”不是脚本错位产生的。

<a id="part-1-section-6"></a>

### 6. 阶段 2：静态预检与工程保障

| 保障 | 防止的问题 |
|---|---|
| 缺 SFT adapter 默认拒绝起训 | 静默从 Base 开始 GRPO |
| 保存配置、命令和数据 SHA256 | 实验不可复现 |
| checkpoint 保存 adapter/optimizer/trainer/RNG | 只能恢复权重，不能恢复训练轨迹 |
| resume 截断 checkpoint 之后的 CSV | 重复 step 污染曲线 |
| old/reference log-prob 分块计算并复用 | 重复前向和显存浪费 |
| group diagnostics 与 grad norm | reward 日志正常但实际无梯度 |
| 10-step signal guard | 长时间训练无有效信号 |
| dashboard 读取 train/val/group/gpu | 只凭单一指标误判状态 |

静态检查通过只能证明“配置和程序结构可执行”，不能证明 GPU 路径和训练数学链正确，所以必须进入 smoke。

#### 6.1 一个完整 checkpoint 应包含什么

```text
adapter_model.safetensors   # GRPO LoRA 权重
adapter_config.json         # LoRA 结构
optimizer.pt                # Adam 动量与方差
trainer_state.json          # step、best、早停计数、关键配置
rng_state.pt                # Python、NumPy、Torch、CUDA 随机状态
tokenizer files             # 保持生成与答案提取口径
```

只有 adapter 权重只能做“从该权重重新开始的新实验”，不能称为严格续训。严格 resume 还必须恢复 optimizer、trainer 和 RNG。

#### 6.2 resume 前为什么要做兼容性检查

恢复时先比较 Base、SFT adapter、训练/验证文件、rollout 数、最大回答长度和 reward 关键权重。任一项变化都会改变训练语义；如果确需变化，应创建新实验目录，而不是把不同实验拼进同一条曲线。

#### 6.3 可复现不等于结果完全逐 bit 相同

固定 seed、RNG 和配置可以显著降低漂移，但 Windows/WDDM、CUDA kernel 和浮点归约仍可能产生微小差异。复盘重点是 step、reward、组内指标和验证趋势一致，而不是要求所有浮点数逐 bit 相同。

<a id="part-1-section-7"></a>

### 7. 阶段 3：smoke3 验证最小闭环

smoke 使用 32 条数据跑 3 step，验证：SFT 起点、reward、rollout、反向传播、CSV、checkpoint 和 resume 全部连通。

本轮结果：3 步均 `effective=1.0`、`mixed=1.0`、`zero_adv=0`；峰值 allocated 约 7.21 GB；从 checkpoint-1 恢复后轨迹一致且 CSV 无重复。

smoke 通过只说明“链路能跑”，样本太少、步数太短，不能证明全量数据存在持续学习信号。因此下一步必须用全量分布做 diagnostic20。

#### 7.1 smoke 能证明与不能证明的边界

smoke 能发现路径错误、SFT 未加载、reward 配置错、OOM、NaN、CSV 缺失和 checkpoint 不完整。它不能证明 reward 长期正确、全量分桶有效、验证集会提升或训练不会早停。

如果 smoke 失败，应在最小数据上修复；直接增加 batch、步数或换全量数据只会放大定位成本。

<a id="part-1-section-8"></a>

### 8. 阶段 4：diagnostic20 验证链路能学

最近 10 step 的放行门槛：

```text
effective_group_rate >= 0.70
mixed_group_rate >= 0.60
zero_advantage_rate <= 0.30
rollout_format_rate >= 0.90
0 < approx_kl < 0.02
grad_norm > 0
```

还要联合验证 adapter 权重变化和 eval 响应；`clip_frac` 不设单独下限。

本轮 diagnostic20 最后 10 步为 effective 0.900、mixed 0.875、zero-adv 0.100、format 0.984；step 9 EM 从 0.47 升到 0.50；392/392 个 adapter 张量随后继续变化。

这些证据依次证明 reward 有差异、advantage 有效、梯度进入 LoRA、策略变化且验证集能响应。此时才有资格进入 formal120。

#### 8.1 diagnostic 指标按因果顺序怎么读

| 顺序 | 指标 | 正常含义 | 异常时优先检查 |
|---|---|---|---|
| 1 | unique response、empty rate | rollout 有多样性且非空 | EOS、生成参数、tokenizer |
| 2 | mixed/effective group | reward 能区分同题回答 | 数据难度、rollout_n、reward |
| 3 | advantage std、zero-adv | 归一化后有训练信号 | reward 全同、答案提取失败 |
| 4 | grad norm | 梯度进入 GRPO LoRA | loss、requires_grad、优化器参数组 |
| 5 | approx KL、adapter diff | 参数更新改变策略 | 学习率、old log-prob、optimizer.step |
| 6 | eval EM/format/length | 改变产生可泛化收益 | reward hacking、训练验证口径 |

#### 8.2 常见误判

- 单步 reward 下降：可能只是随机抽到更难的 4 道题，应看滑动均值和 eval。
- `clip_frac=0`：只表示 ratio 未触及 20% 裁剪边界；grad 和权重仍可变化。
- KL 很小：0.6B LoRA 小学习率下可能正常，需结合 grad、权重差和 eval。
- GPU 显存很高：WDDM reserved 可能跨物理/共享显存，优先看 allocated、OOM 和 step time。
- 达到最大步数：只是预算耗尽，不自动代表收敛或成功。

#### 8.3 diagnostic 失败时的回退路径

```text
rollout 空/重复 → 修生成与 EOS
组内全同 → 修分桶、rollout_n、reward
advantage 有效但 grad=0 → 修 loss/LoRA/optimizer
grad 非零但权重不变 → 修 optimizer.step/保存逻辑
权重变化但 eval 不动 → 查更新强度、数据密度与评估口径
reward 升而 EM 降 → 查 reward hacking
```

修复后从失败层之前的最小阶段重跑，不必每次从完整 formal 开始。

<a id="part-1-section-9"></a>

### 9. 阶段 5：为什么先做 formal120

120 不是数据集 epoch，也不是必须跑满的成功条件，而是第一轮中周期正式实验的安全上限。

#### 9.1 计算预算可控

每步 4 个 prompt、每题 8 个 rollout、PPO 2 轮。跑满 120 step 约等于 480 次 prompt 抽样、3840 条 rollout 和 480 次 mini-batch 优化器更新，足以观察趋势且不会直接扩大到高风险长训练。

#### 9.2 有足够验证窗口

每 10 step 做一次 eval100，最多得到“训前基线 + 12 次训练中验证”。diagnostic20 已证明短链路能学，formal120 要回答的是这种改善能否在更长时间保持。

#### 9.3 早停优先于跑满

旧规则是验证 EM 连续 40 step 没有刷新全局最佳就停止，即连续 4 次 eval 无新高。但 step 39 达到 0.56 后，EM 经 0.51、0.49、0.54、0.55，最近趋势已经恢复，旧规则仍在 step 79 停止。它把“未创新高”和“没有恢复趋势”误当成同一件事。

整改后的判断必须同时满足：全局最佳耐心已耗尽，并且最近 3 个验证点没有达到最小正斜率，才真正早停。若趋势正在恢复，只有限延长；延长计数不重置全局最佳耐心，最多增加 40 step，避免噪声造成无限训练。刷新全局最佳时，两类计数都重置。

因此早停可能表示收敛，也可能是规则误判。复盘时必须同时查看全局最佳、近期趋势和最大延长期，结果仍使用最佳 checkpoint。

##### 9.3.1 术语边界：signal guard不是GRPO标准算法名

本文正式使用“训练信号监控与停止规则”（training-signal monitoring and stopping rule），项目代码和日志中简称`signal guard`。

`guard`、`guardrail`、`watchdog`和`abort criterion`都是工程系统中已有的通用表达，但学术界没有一个GRPO标准组件叫`Signal Guard`。本项目的`signal guard`是围绕训练rollout健康度实现的自定义运行期停止规则，更正式的表述包括：

- training-signal monitoring and stopping rule；
- training-health monitoring and abort criterion；
- windowed sequential stopping rule。

`early stopping`则是机器学习中的通用术语，通常表示验证指标长期不改善时提前结束训练，用于避免继续消耗预算、过拟合或在退化区间中继续更新。两者都会停止训练，但观察对象和统计问题不同，不能混称。

##### 9.3.2 当前signal guard的完整逻辑

当前每个GRPO step使用4个prompt、每题8条rollout，共形成4个题组和32条回答。每步先计算：

| 指标 | 当前含义 | 当前判据 |
|---|---|---:|
| `effective_group_rate` | 组内总reward标准差非零、能够形成相对advantage的题组比例 | `< 0.700`为硬失败 |
| `mixed_group_rate` | 同一题的8条回答中同时存在数学正确和错误的题组比例 | `< 0.600`只预警 |
| `zero_advantage_rate` | 回答级advantage接近0的比例 | `> 0.300`为硬失败 |
| `rollout_format_rate` | 训练rollout满足最终答案格式的比例 | `< 0.900`为硬失败 |

当前配置为：

```text
signal_guard_window = 10
signal_guard_warmup_steps = 10
signal_guard_patience_checks = 3
signal_guard_non_overlapping_windows = true
signal_guard_mixed_hard_stop = false
```

执行逻辑是：

```text
step 0–8：不检查，数据不足一个完整窗口

step 9： 汇总step 0–9
step 19：汇总step 10–19
step 29：汇总step 20–29
……

对窗口内四项指标取10步算术平均
→ mixed不足只写warning，不增加硬失败次数
→ effective、zero-adv或format任一失败，则当前窗口为硬失败
→ 以10步为跨度反向统计连续失败的独立窗口
→ 不足3个时只记录观察
→ 连续3个独立窗口硬失败时保存当前checkpoint并终止
```

任一中间窗口通过，就会中断连续失败链。例如step 9失败、step 19通过、step 29再失败时，step 29重新按1/3计算，不能继承step 9的失败。

三个耐心窗口至少覆盖30个独立训练step，即120个prompt题组和960条rollout。mixed只作为诊断，是因为数学答案全对或全错不等于总reward没有差异；格式、长度、重复和截断分量仍可能产生有效advantage。seed=2026案例的完整证据见第12.1.1节。

signal guard只决定“是否继续”，不进入loss、不调整梯度，也不修改学习率。它在当前step的optimizer更新和指标写入之后执行，所以“在step 29被signal guard停止”表示step 0–29共30次GRPO step已经完成，随后保存`checkpoint-29`。

##### 9.3.3 GRPO early stopping的完整实现

当前GRPO训练器有独立的、趋势感知的early stopping。它监控固定验证集的`val_exact_match`，不读取训练rollout的effective、mixed或zero advantage。

主要配置项是：

| 配置 | 含义 | GRPO v7正式长训 | C0/L1因果实验基线 |
|---|---|---:|---:|
| `eval_freq` | 每多少GRPO step验证一次 | 10 | 10 |
| `max_steps_no_improve` | 未刷新全局最佳的step耐心 | 90 | 50 |
| `early_stop_trend_window` | 恢复趋势使用的最近验证点数 | 3 | 3 |
| `early_stop_min_recovery_slope` | 允许延长所需的最小斜率，单位为每个验证点 | 0.005 | 0.005 |
| `early_stop_max_extension_steps` | 达到耐心后最多额外延长的GRPO step | 40 | 20 |

新分支以`weights_only`启动时，训练器先执行零步验证，把该EM作为baseline和初始全局最佳；`best_step=-1`表示最佳值来自训练前。完整`full resume`时不重复建立baseline，而是从`trainer_state.json`恢复`best_val_em`、`best_step`、`steps_no_improve`、趋势延长计数和历史验证指标。

每次达到验证点时，逻辑如下：

```text
执行固定eval100
→ 读取val_exact_match

如果 val_em > 历史全局最佳：
  更新best_val_em和best_step
  steps_no_improve = 0
  early_stop_extension_steps = 0

否则，包括与最佳值相等：
  steps_no_improve += eval_freq

如果 steps_no_improve < max_steps_no_improve：
  继续训练

如果 steps_no_improve >= max_steps_no_improve：
  对最近early_stop_trend_window个验证EM做等间隔线性回归
  → slope >= min_recovery_slope
    且已延长步数 < max_extension_steps：再延长eval_freq步
  → 否则：保存状态并早停
```

这里的斜率单位是“每个验证点”，不是每个GRPO step。`eval_freq=10`时，三个验证点通常相隔10个训练step。延长后不会清零`steps_no_improve`；下一次验证仍会重新检查趋势，直到刷新全局最佳、趋势消失或用完最大延长预算。

例如因果实验使用50步耐心：如果baseline之后一直没有新高，step 9/19/29/39/49会依次累计10/20/30/40/50。到step 49才进入趋势判断；若最近三个验证点正在以至少0.005/验证点恢复，则先延长10步，否则在step 49早停。

##### 9.3.4 signal guard与early stopping的具体区别

| 维度 | signal guard | early stopping |
|---|---|---|
| 正式定位 | 训练健康度监控与异常终止 | 验证表现不再改善时提前结束 |
| 回答的问题 | 当前rollout是否仍有可用GRPO信号和基本格式能力 | actor在固定验证集上是否还产生新的泛化收益 |
| 数据来源 | 训练过程中生成的rollout | 固定`eval_100.parquet` |
| 核心指标 | effective、zero advantage、format；mixed只预警 | greedy `val_exact_match` |
| 比较方式 | 与绝对健康阈值比较 | 与历史全局最佳比较，并检查近期趋势 |
| 当前窗口 | 每10步一个非重叠训练窗口 | 最近3个验证点的趋势窗口 |
| patience含义 | 连续3个独立训练窗口硬失败 | 距离最后一次全局最佳已经过多少训练step |
| 恢复方式 | 任一独立窗口通过就中断连续失败链 | 刷新全局最佳时清零；未刷新但趋势恢复时有限延长 |
| 典型停止含义 | 继续训练可能没有有效优势信号或格式已持续异常 | 继续训练暂时看不到新的验证收益 |
| 是否修改训练目标 | 否，只监控并决定是否终止 | 否，只监控并决定是否终止 |

因此可能出现四种组合：

| signal guard | early stopping | 含义 |
|---|---|---|
| 通过 | 未触发 | 有训练信号且验证仍值得继续观察 |
| 通过 | 触发 | 仍能更新，但更新没有形成新的验证收益 |
| 失败 | 未触发 | 历史验证最好值可能尚可，但当前训练信号正在持续失效 |
| 失败 | 触发 | 训练健康度和验证收益都不支持继续 |

##### 9.3.5 两套规则在训练循环中的先后顺序

当前单个GRPO step的关键顺序是：

```text
rollout与reward
→ PPO/GRPO optimizer update
→ 写入训练指标和trainer state
→ 分别计算KL guard与signal guard
→ post_update统一停止调度（KL优先于signal）
→ 自适应KL系数调整
→ 到达eval_freq时执行固定验证
→ early stopping独立判定
→ reward-hacking预警
→ 极端验证格式退化独立判定
→ post_eval统一停止调度（格式优先于early stopping）
→ checkpoint保存
```

KL或signal guard在同一步触发时，会先于该步原定的验证阶段终止，因此日志复盘要先看`stop_reason`，不能仅因缺少本轮val行就判断训练崩溃。正常预算结束和所有受控停止都会进入统一状态同步、checkpoint、metrics、停止档案和CSV关闭流程；若OOM或异常发生在一个尚未完整结束的step中，则不把可能含部分更新的模型伪装成完整checkpoint，而是在停止档案中指向最近安全checkpoint。

##### 9.3.6 框架中哪些训练器实现了early stopping

| 训练入口 | 是否有early stopping | 当前实现 |
|---|---|---|
| GRPO `train_grpo.py` | 有 | 全局最佳耐心 + 最近验证趋势 + 有限延长；支持full resume恢复计数 |
| PPO `train_ppo.py` | 有 | 简单的全局最佳耐心；达到`max_steps_no_improve`直接停止，没有趋势延长 |
| SFT `train_sft.py` | 当前没有 | 按`num_train_epochs`训练并按保存策略落盘，没有注册`EarlyStoppingCallback` |

所以本文第9.3节所说的“整改后early stopping”专指当前GRPO训练器；不能自动假设PPO和SFT已经继承完全相同的趋势保护逻辑。

##### 9.3.7 GRPO统一停止控制器

当前GRPO不再由各停止分支分别执行`save + break`，而是采用“统一调度、分别判定、统一留档、统一收尾”。核心职责拆分如下：

| 层次 | 实现 | 职责 |
|---|---|---|
| 独立判定器 | KL、signal、验证格式、early stopping | 只计算自己的统计量并返回`StopDecision`，不直接退出训练循环 |
| 统一调度器 | `TrainingStopController` | 分阶段收集候选，按显式优先级选择唯一主停止原因，同时保留全部候选 |
| 统一留档 | `training_stop.json`、`logs/stop_events.jsonl`、`trainer_state.json` | 保存状态、类别、来源、step、阈值证据、checkpoint和收尾错误 |
| 统一收尾 | `GRPOTrainer._finalize_training` | 同步trainer state、保存安全checkpoint和metrics、记录显存、关闭CSV并刷新日志 |

显式优先级为：

```text
OOM/运行时异常
> KL guard
> signal guard
> 验证格式崩溃
> validation early stopping
> 最大训练步数
```

这里的“优先”只决定同一调度阶段多个条件同时触发时哪个成为主停止原因，不允许高优先级指标抵消低优先级失败。`stop_events.jsonl`仍会保存同一步的所有候选，避免过去由代码中第一个`break`掩盖后续条件。

三个停止档案的用途不同：

- `training_stop.json`：当前训练会话的机器可读终态摘要；
- `logs/stop_events.jsonl`：跨resume追加的会话开始、候选选择和结束事件；
- `checkpoint-*/trainer_state.json`：与权重、optimizer和RNG一起保存的可恢复状态，并增加`training_status`、`stop_reason`和`stop_decision`。

原有日志中的`训练结束, 原因:`以及四个`plots/*.csv`文件名、列名和追加方式保持不变；dashboard优先读取结构化停止摘要，对旧实验则继续回退到checkpoint和日志。因此本次重构增加了控制面和审计面，没有改变KL、signal或early stopping的数学阈值，也没有破坏历史看板输入契约。

#### 9.4 120 不代表覆盖 5759 条

训练器每步独立随机抽 4 条，跨 step 可能重复。120 step 有 480 次抽样，相当于数据量的 8.3%；按当前独立采样计算，期望覆盖约 461 条唯一 prompt，约占 8.0%。若目标是无放回遍历一轮，需要改成 epoch sampler，并约需 1440 step；当前证据还不足以直接这样扩大。

#### 9.5 formal120 的时间预算怎么估算

总时长不能只用 `120 × step_time`，还要加训前 eval100、每 10 step 的 eval100、checkpoint 写盘和 WDDM 抖动。估算式为：

```text
总时长 ≈ 120 × 平均训练 step 时间
       + 13 × 一次 eval100 时间
       + 12 × checkpoint 保存时间
```

因此 dashboard 同时记录 step time 和日志更新时间。eval 期间训练 step 不增长，但 GPU/CPU 活动和日志时间仍可证明进程未卡死。

#### 9.6 为什么验证频率是 10 step

频率过高会把大量时间花在 eval，过低则可能错过早期峰值和 reward hacking。diagnostic20 已显示 step 9 能出现变化，因此每 10 step 是当前任务中“观察分辨率”和“评估成本”的折中，不是所有任务通用常数。

#### 9.7 为什么使用 best checkpoint 而不是 last checkpoint

GRPO 的验证曲线可能先升后降。last 只表示训练停止时的位置；best 表示固定验证口径下观察到的最优位置。最终评估、模型导出和下一阶段实验都应从 best 开始，并保留 last 用于分析退化过程。

<a id="part-1-section-10"></a>

### 10. 阶段 6：formal 后如何决策

| 观察结果 | 结论 | 下一步 |
|---|---|---|
| 接近 120 时 EM 仍上升，信号和 KL 稳定 | 训练预算可能不足 | 设计更长实验或无放回 sampler |
| EM 平台且达到趋势感知早停 | 已收敛或当前配置到顶 | 使用最佳 checkpoint，先做最终评估 |
| signal guard、NaN、OOM、KL 或格式异常 | 训练链路/配置异常 | 回退到对应诊断层修复，不扩模 |
| reward 上升但 EM 下降 | reward hacking 或口径错位 | 检查输出和 reward 分量，不扩模 |
| probe 能学但全量长期无提升 | 数据有效密度或更新强度不足 | 重查分桶、reward、学习率和采样 |

因此“跑满 120”本身没有决策价值；只有验证曲线仍上升，才推导出扩大训练这一下一步。

#### 10.1 扩大训练前还缺哪些证据

1. best checkpoint 在固定 eval100 上优于 SFT，且格式和长度没有明显退化。
2. 使用更大或官方 held-out 测试集复核，避免对 100 题验证集做选择性过拟合。
3. 最近多个验证点仍有上升趋势，而不是单点随机波动。
4. signal guard、KL、grad norm 和显存长期稳定。
5. 明确扩大训练要解决的是“预算不足”，不是 reward、数据或评估错误。

#### 10.2 扩模实验一次只改变一个主变量

可选主变量包括总步数、采样器、学习率、rollout_n、batch、reward 或数据配比。若同时改变多个变量，即使结果提升也无法知道原因。推荐优先保持 v7 其他配置不变，只比较：

```text
formal120 best
vs 更长总步数但相同随机 sampler
vs 无放回 epoch sampler 但相近 rollout 预算
```

#### 10.3 三种停止的语义不同

| 停止类型 | 含义 | 是否可直接使用 best |
|---|---|---|
| 最大步数停止 | 预算到顶，可能仍在改善 | 可以，但要判断是否值得扩模 |
| EM 无改善早停 | 当前配置进入平台期 | 可以，通常先停止扩模 |
| signal/KL/OOM/NaN 异常停止 | 链路或稳定性失败 | 先审计，不能直接把结果当成功 |

<a id="part-1-section-11"></a>

### 11. 固化配置、执行与产物

配置：`post_training_framework/configs/gsm8k_qwen3_0d6b_grpo_v7.json`

```powershell
# 冒烟、全量诊断、正式中周期实验
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Smoke
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Diagnostic
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Formal
```

每轮至少保留 `run_config.json`、四类 CSV、训练日志以及包含 optimizer/trainer/RNG 的 checkpoint。恢复前先核对 Base、SFT、数据、reward 和关键 rollout 配置兼容。

#### 11.1 四类 CSV 分别回答什么

| 文件 | 核心问题 |
|---|---|
| `train_metrics.csv` | loss、KL、梯度、reward 和 step time 是否正常 |
| `val_metrics.csv` | 固定验证集能力是否改善、最佳点在哪里 |
| `group_diagnostics.csv` | 同题 rollout 是否持续产生有效 advantage |
| `gpu_memory.csv` | allocated、reserved、优化器和激活值是否稳定 |

#### 11.2 看 dashboard 的固定顺序

1. 先确认日志仍更新、进程和最新 step 一致。
2. 再看 group effective/mixed/zero-adv，确认学习信号源头。
3. 再看 grad norm、KL、policy loss，确认更新链路。
4. 再看 eval EM、format、length，判断泛化方向。
5. 最后看显存与 step time，判断工程稳定性。

不要先盯 reward 曲线再为波动寻找解释；应先确认上游信号和下游验证是否一致。

#### 11.3 正式训练结束后的评估顺序

```text
读取 val_metrics 找 best step
→ 验证 best checkpoint 状态完整
→ Base + SFT merge + best GRPO adapter 加载
→ 固定 eval100 复现训练内结果
→ 更大 held-out 集与 SFT 同口径比较
→ 检查错误类型、格式、长度和代表性样例
→ 再决定导出、继续训练或回退
```

#### 11.4 最小实验记录模板

```text
实验问题:
唯一主变量:
Base / SFT / GRPO 起点:
train/eval SHA256:
reward 配置:
rollout/batch/PPO/lr:
停止原因:
best step 与指标:
组内信号摘要:
显存与耗时:
与 SFT/上一轮比较:
结论与下一步依据:
```

<a id="part-1-section-12"></a>

### 12. 当前 formal120 证据快照

截至 2026-07-12，formal120 已完整跑完，验证曲线为：

| step | EM | reward | format | response length |
|---:|---:|---:|---:|---:|
| -1 | 0.47 | 0.529 | 0.92 | 127.0 |
| 9 | 0.49 | 0.559 | 0.92 | 127.5 |
| 19 | 0.50 | 0.564 | 0.92 | 131.0 |
| 29 | 0.52 | 0.588 | 0.94 | 125.4 |
| 39 | 0.56 | 0.635 | 0.95 | 124.4 |
| 49 | 0.51 | 0.585 | 0.95 | 128.8 |
| 59 | 0.49 | 0.572 | 0.96 | 131.3 |
| 69 | 0.54 | 0.629 | 0.97 | 126.8 |
| 79 | 0.55 | 0.641 | 0.98 | 123.9 |
| 89 | 0.55 | 0.635 | 0.95 | 129.2 |
| 99 | 0.64 | 0.7265 | 0.97 | 131.2 |
| 109 | 0.65 | 0.732 | 0.96 | 134.9 |
| 119 | 0.63 | 0.704 | 0.95 | 140.6 |

step 79 的旧规则早停后，从 `checkpoint-79` 恢复 adapter、optimizer、trainer 和 RNG 全状态；step 99/109 随后刷新最佳，证明 step 79 是过早停止而非已经收敛。formal120 最佳点是 `checkpoint-109` 的 EM 0.65，最终 `checkpoint-119` 为 0.63，不能用 last 代替 best。

step 119 最近 10 步的 effective/mixed/zero-adv/format 为 0.700/0.625/0.300/0.956，均仍在放行边界内；approx KL、梯度和格式也未触发异常保护。因此 formal120 给出的下一步不是直接宣称收敛，而是从完整 `checkpoint-119` 扩展总预算到 500，观察更长曲线。

#### 12.1 formal500 延长实验设计

- 总步数是 500，即继续执行 step 120–499，而不是额外再跑 500 步。
- 保持同一 output、run name、主日志、CSV 和 dashboard，确保曲线连续。
- 每 10 step 保存和 eval100；全局最佳耐心使用 90 step，避免短周期波动误停。
- 最近 3 个验证点斜率至少为 0.005 时允许延长，最多延长 40 step。
- signal guard、KL、NaN/OOM、格式和 reward-hacking 监控全部保留。
- 500 只是最大预算；若没有恢复趋势或安全保护触发，应保留现场并停止。

首次从 step 120 延长时，step 121 的当前组仍为 effective/mixed/zero-adv=0.75/0.75/0.25，但 10 步滑动均值为 0.675/0.625/0.325，仅相差一个组便被旧 signal guard 立即终止。该保护随后改为保留原阈值、连续 3 个窗口不达标才停止；单窗口越界只记录观察，避免边界噪声中断长期实验，同时没有关闭学习信号保护。

#### 12.1.1 seed=2026暴露的第二层signal guard过敏与整改

后续C0/L1配对多seed实验说明，“连续3个滑动窗口”仍不等于三份独立证据。seed=2026的C0/L1都在step 11停止；三个失败窗口实际是step 0–9、1–10和2–11，相邻窗口共享9/10的数据。最终C0/L1都只因`mixed_group_rate < 0.600`停止，但对应窗口的`effective_group_rate`仍为0.725/0.700，说明实际reward方差信号并未消失。

`mixed_group_rate`只回答同一题的8条回答是否同时含数学正确与错误；`effective_group_rate`才直接回答组内总reward是否存在非零方差。由于reward还包含格式、单一最终答案、长度、重复和截断分量，全对或全错组也可能仍有可学习的reward差异。因此不能让mixed单项在effective正常时独立硬停。

整改后的默认规则是：

1. 10步窗口只在step 9/19/29等非重叠边界检查；`patience=3`表示连续观察30个独立step，而不是只新增两个step。
2. mixed低于0.600仍写warning，但不计入硬停止耐心。
3. effective低于0.700、zero advantage高于0.300或format低于0.900仍是硬条件；连续三个独立窗口失败才停止。

真实GPU整改复验从同一`checkpoint-169`分别重置optimizer，使用相同seed=2026重新训练C0/L1各30步。C0在step 9出现`mixed=0.525`但effective=0.700，只记录预警；L1在step 9出现effective=0.675、zero advantage=0.325，正确记为1/3硬失败。到step 19两条轨迹均恢复，最终都达到step 29并生成`checkpoint-29`，证明修复既避免了重叠窗口误停，也保留了真实安全保护。

这次复验同时给出结论边界：训练能继续不等于L1一定更优。seed=2026终点C0/L1 greedy EM为0.660/0.620，sample EM为0.4375/0.3875；L1虽然把最后10步reference KL均值从0.045685降到0.039041，但能力保持更差。因此本轮只确认guard整改有效，不能据此自动晋级Confirm50或正式长训。

#### 12.1.2 新统一停止控制器上线后的seed=2026补充协议

统一停止控制器完成后，后续实验按“先验证工程输出，再补长期因果证据”的顺序推进。该补充协议在查看整改版30步结果后登记，属于事后协议修订，不覆盖原seed=2026在旧guard下step 11停止的删失记录。

第一步使用独立短目录`models/grpo/g7smk/`执行3步真实GPU smoke，只验证以下工程链路：

- 逐步训练日志、原四类CSV和终态checkpoint正常写出；
- `training_stop.json`与`logs/stop_events.jsonl`能够记录统一停止结果；
- dashboard能够读取训练step、验证指标、训练状态和停止原因；
- smoke结果不进入C0/L1能力或KL聚合。

第二步将整改版C0/L1的`checkpoint-29`、0–29步CSV和必要状态复制到短目录`models/grpo/g7s26/`，并在`lineage.json`记录原始只读证据路径。复制只缩短后续操作路径，不改变权重、optimizer、RNG或历史指标。续跑规则为：

```text
C0: models/grpo/g7s26/c0_fresh_control/seed-2026/checkpoint-29
L1: models/grpo/g7s26/l1_lr3e6/seed-2026/checkpoint-29
resume_state_mode = full
target = step 49 / 50个累计GRPO step
```

C0和L1必须成对执行；不能根据30步终点EM只选择其中一条。除学习率外，数据、prompt序列、PPO、reward、KL控制器、signal guard和验证设置全部保持一致。实验矩阵使用短配置`post_training_framework/configs/g7s26.json`，编排事件和阶段配置统一写入`models/grpo/g7s26/_orchestration/`。

第三步将新seed=2026的50步配对与已有seed=42、123完整配对汇总。历史seed=42、123的group diagnostics已用当前非重叠窗口、mixed只预警规则离线回放；C0/L1在step 9、19、29、39、49均无硬失败，因此signal guard没有改变这些已完成轨迹的loss、梯度或模型更新，不因guard版本名称不同而机械重跑。

只有三seed汇总同时满足KL、能力、rollout和格式安全条件，才决定继续L2学习率剂量或K3 KL控制器实验。从SFT model重新开始属于最终配方的端到端外部验证，不在本轮checkpoint-169根因实验中提前执行。

#### 12.1.3 真实GPU smoke、seed=2026 Confirm50与三seed最终判定

2026-07-23已按12.1.2的顺序完成工程验证、补充训练和三seed汇总。所有新输出使用短目录，原长路径实验和原seed=2026 step 11删失记录均未覆盖。

第一步的3步真实GPU smoke输出到`models/grpo/g7smk/`。它完成3/3个step并生成`checkpoint-2`；`training_stop.json`、`logs/stop_events.jsonl`、train/group/val CSV、逐步日志和dashboard均能读取最终`latest_step=2`、`status=completed`、`stop_reason=达到最大步数`。因此统一停止控制器上线后，旧指标输出、日志打印、checkpoint和看板链路仍然成立；smoke不进入能力比较。

第二步从短目录中两条整改版`checkpoint-29`使用full resume续跑step 30–49：

```text
models/grpo/g7s26/c0_fresh_control/seed-2026
models/grpo/g7s26/l1_lr3e6/seed-2026
```

两条轨迹均正常完成累计50步、写出`checkpoint-49`，train/group CSV各50行、val CSV各6行，统一停止原因为“达到最大步数”。seed=2026终点证据为：

| 指标 | C0 | L1 | L1-C0或解释 |
|---|---:|---:|---|
| 尾10步reference KL均值 | 0.060442 | 0.047576 | -0.012866，L1更低 |
| 尾10步reference KL斜率 | +0.001408 | +0.000413 | -0.000995，L1漂移较慢但仍为正 |
| step 49 greedy EM | 0.6600 | 0.6200 | -0.0400 |
| step 49 sample EM | 0.3625 | 0.4625 | +0.1000 |
| rollout EM尾10步 | 0.6406 | 0.6312 | -0.0094 |
| sample格式率 | 0.9375 | 0.8750 | L1低于0.90门槛且低于C0 |
| sample截顶率 | 0.1125 | 0.0875 | L1满足0.10门槛 |

KL系数必须区分两个时间点：seed=2026 C0的step 49 loss实际使用0.005，step结束后控制器依据尾窗把checkpoint中的`current_kl_loss_coef`提高到0.0075；若续跑，下一步才会使用0.0075。L1对应为0.003333→0.003333。汇总脚本已分别记录`last_step_kl_coef`和`final_kl_coef`，不再把“本步使用值”和“步后控制器状态”混为一谈。

第三步从原始CSV重新汇总seed 42、123、2026共6条50步轨迹。机器可读入口为：

```text
models/grpo/g7s26/_orchestration/3seed/c50_trials.csv
models/grpo/g7s26/_orchestration/3seed/c50_paired_deltas.csv
models/grpo/g7s26/_orchestration/3seed/c50_aggregate_summary.md
models/grpo/g7s26/_orchestration/3seed/decision.md
```

三seed预注册判定如下：

| 门槛 | 支持seed数 | 是否通过 |
|---|---:|---|
| 尾10步KL均值更低 | 3/3 | 通过 |
| 尾10步KL斜率不更高 | 2/3 | 通过 |
| greedy retention不低于C0 | 2/3 | 通过 |
| sample retention不低于C0 | 3/3 | 通过 |
| rollout EM尾10步不低于C0 | 2/3 | 通过 |
| sample格式率同时`>=0.90`且`>=C0` | 1/3 | 不通过 |
| sample截顶率同时`<=0.10`且`<=C0` | 2/3 | 通过 |
| hard KL越界为0 | 3/3 | 通过 |

因此本轮总体结论是“不晋级”。L1把尾部KL均值平均降低0.012467，说明降低学习率确实是可复现的稳定因素；但它没有形成同时满足KL、能力、rollout和格式安全的完整训练配方。按照预注册规则，本轮不直接启动L2或K3，也不从SFT重新训练，更不能挑选单个最高EM seed继续。

下一步应先在冻结的三组`checkpoint-49`上扩大固定sample评估，区分当前每seed 80条sample带来的抽样波动与稳定格式退化。seed=123的L1格式率0.8875只比门槛少1条合法回答，seed=2026的0.8750少2条；这足以提出新诊断，但不能反向修改本轮1/3通过和“不晋级”的既有结论。扩大评估必须作为新协议单独登记，只有新证据证明格式安全后，才重新讨论L2剂量或K3控制器实验。

#### 12.1.4 g7lk：三seed确认后以L1为新对照的L2/K3剂量与控制器实验

2026-07-23至2026-07-24在短目录`models/grpo/g7lk/`执行三seed汇总后的下一轮单变量实验，检验两个方向：学习率能否继续降低换取更稳定的KL轨迹（`l2`，3e-6→2e-6），以及禁止自适应KL系数降到初始值以下能否同时改善KL和能力（`k3`，`adaptive_kl_min_coef` 0.001→0.005）。所有分支均从`qwen3_0d6b_grpo_v7_full5759_rewardv2_rollout8_len256_lr5e-6_ppo2_eval100/checkpoint-169`同一份LoRA出发，seed=42，`l1`（学习率3e-6）作为本阶段的新对照组，不再直接使用5e-6的C0作对照——因为L1已经过三seed确认为比C0更稳的候选。

按`Gate10→Screen30→Confirm50`三阶段推进，Screen30晋级判据见`models/grpo/g7lk/_orchestration/s30/decision.md`：

| 门槛 | l2 | l2判定 | k3 | k3判定 |
|---|---:|---|---:|---|
| 尾10步KL均值差`<= +0.01`（相对l1） | -0.000335 | 通过 | -0.001820 | 通过 |
| 尾10步KL斜率差`<= +0.0005` | +0.000132 | 通过 | +0.000277 | 通过 |
| greedy retention差`>= -0.02` | +0.0100 | 通过 | -0.0300 | **不通过** |
| rollout EM尾10步差`>= -0.03` | +0.0031 | 通过 | -0.0188 | 通过 |
| sample格式率`>=0.90` | 0.9250 | 通过 | 0.9375 | 通过 |
| sample截顶率`<=0.10` | 0.0875 | 通过 | 0.0750 | 通过 |
| hard KL越界 | 0 | 通过 | 0 | 通过 |

`l2`全部通过，进入Confirm50；`k3`因greedy retention跌破-0.02门槛未晋级——提高KL系数下限确实在30步内让reference KL更低，但代价是能力下降超出预注册边界，不能用更低KL换取能力损失，因此保留`checkpoint-29`不再续训。本轮不运行`L2+K3`组合，避免收益和代价的来源无法区分学习率与KL下限。

`l2`Confirm50（累计50步）于2026-07-24 00:10:49完成，`stop_reason=早停: val_em连续50步未改善且无可用恢复趋势`，`training_stop.json`确认无hard KL越界。与seed=42 `l1`（`grpo_v7_step169_causal_v1/l1_lr3e6/seed-42`）终点直接对照：

| 指标 | l1（3e-6，对照） | l2（2e-6） | l2-l1 |
|---|---:|---:|---:|
| 尾10步reference KL均值 | 0.047127 | 0.038083 | -0.009044 |
| 尾10步reference KL斜率 | -9.4e-05 | -0.000311 | -0.000217 |
| step 49 reference KL | 0.03907 | 0.031658 | -0.007412 |
| final greedy EM | 0.64 | 0.64 | **0.0000** |
| final sample EM | 0.3875 | 0.3875 | **0.0000** |
| rollout EM尾10步 | 0.5594 | 0.5875 | +0.0281 |
| sample格式/截顶 | 0.9125/0.075 | 0.9125/0.0875 | 持平/+0.0125 |

`l2`在KL轨迹上进一步优于`l1`（尾部KL均值再降0.009，斜率转为更负），但**greedy EM与sample EM相对l1完全没有变化**，逐位小数一致。这不是接近，是数值相等，说明继续降低学习率这一方向的能力边际收益已经归零——KL越来越稳，但换不来更高的验证集正确率。

同时必须指出统计置信度问题：固定验证集只有100题，对比例估计而言95%置信区间半宽约为`1.96×√(0.65×0.35/100)≈0.094`，即0.61–0.67这一区间本身可能落在单次评估的抽样噪声内。checkpoint-169起点的0.67峰值和后续所有continuation分支收敛到的0.64，差距不能排除是噪声而非真实退化。

**结论（截至2026-07-24）**：

1. 长度偏置假说未在此轮出现：`l1`、`l2`、`k3`三条轨迹的`val_response_len_mean`全程稳定在140–146 token，没有随训练推进单调上升，说明formal500时代的reward v2整改（token长度惩罚、截断惩罚、严格格式门控）在checkpoint-169之后的所有受控续训中持续有效。
2. 学习率剂量方向已经打平：`l1→l2`只改善KL稳定性，不改善EM，继续下探学习率不是当前僵局的解法。
3. `k3`方向已证伪：更强KL约束以能力下降为代价，不能采用。
4. 当前从checkpoint-169出发的所有continuation分支都收敛在greedy EM 0.61–0.66区间，没有一条分支重新达到或超过起点0.67，且0.67本身可能是100题验证集的单点噪声而非可复现真实水平。
5. 下一步不建议继续在“学习率/KL系数”这条轴上加密网格，应优先验证0.67是否为噪声（扩大held-out评估或做bootstrap置信区间），并考虑此前提出但未执行的R4（独立prompt数4→8）和无放回epoch sampler方向。

证据入口：`models/grpo/g7lk/_orchestration/{g10,s30,c50}/*.md`、`models/grpo/g7lk/{l2,k3}/seed-42/plots/val_metrics.csv`、`models/grpo/g7lk/_orchestration/events.jsonl`。

#### 12.2 formal500 step 218 格式退化案例与整改链

formal500 没有正常跑满 500：最佳点为 step 169、EM 0.67；step 209 已回落到 0.54，最终在 step 218 被 signal guard 正确截停。`checkpoint-218` 状态完整且 `next_step=219`，但“可恢复”不等于“适合继续训练”。

| 证据 | step 169 | step 199 | step 218 |
|---|---:|---:|---:|
| actor-reference `kl_loss` | 0.0386 | 0.0744 | 0.2335 |
| rollout format | - | - | 0.594 |
| response token 长度 | - | - | mean 214 / max 256 |

- step 200–218 的 rollout format 均值仅 0.896；step 218 前连续 3 个窗口低于 0.9，异常停止本身符合保护规则。
- step 169 后，format 与 response length 的相关系数约 -0.77，与 actor-reference KL 的相关系数约 -0.745，指向“偏离 reference → 输出变长/截断 → 格式尾部退化”。
- 5759 条训练数据均含统一 `####` 指令且 gold 可解析；重建 step 209–218 的 40 条实际样本也无格式异常或重复。每步使用 `random.sample`，不存在顺序走入坏数据段；最近样本略难只能解释局部波动，不能解释持续漂移。

旧框架的直接缺陷不是单一 KL 系数：

1. `kl_loss` 表示 actor 相对冻结 SFT reference 的累计偏离，`approx_kl` 表示本次 PPO update 相对 old policy 的局部变化；旧代码却用后者判断“是否远离 reference”，保护指标接错。
2. `kl_loss_coef=0.001` 对长期偏离约束过弱，也没有根据 reference KL 自动增强惩罚。
3. 正确但缺少严格格式的回答仍可获得约 0.3 正奖励，奖励给模型留下了放弃格式的捷径。
4. 旧 overlong 只看 1200 字符，从未触发；没有 token 上限命中率、EOS 率和截断惩罚。
5. greedy eval format 较高会掩盖随机 rollout 的尾部退化，旧验证没有固定 sample@n 格式检查。

整改必须按因果顺序闭环：

```text
分离 reference KL / update KL → reference KL 滑窗预警与硬保护
→ 自适应 KL 系数 → 严格单一 #### 格式门控
→ token 长度/EOS/截断惩罚 → 固定随机 sample@n 验证
→ 异常 rollout 原文、source index 和 bucket 留档
```

signal guard 保留，因为它最终正确阻止了继续退化。下一轮不能从 step 218 强行续跑；应使用 `gsm8k_qwen3_0d6b_grpo_v7_repair_probe_from169.json` 从最佳 `checkpoint-169` 建立新分支，显式允许目标变化，先跑 50 步诊断。只有 reference KL、sample format、hit-max、EOS、EM 和组内信号同时通过，才讨论恢复更长预算。



#### 12.3 checkpoint-169 整改探针实测结论

2026-07-15 从 step 170 启动 50 步上限探针，实际完成 42 步并在 step 211 安全终止；`checkpoint-211` 的 adapter、optimizer、trainer state 与 RNG 完整。
- `update_kl` 最大仅 0.000113；reference KL 连续 3 个窗口超过 0.1，最终 rolling=0.1119，证明新保护使用了正确指标且 patience 生效。
- 自适应系数按 `0.005(step170) → 0.0075(step200) → 0.01125(step210)` 生效，但调节速度不足以在硬线前拉回偏离。
- rollout format 均值 0.9643、hit-max 均值 0.0290；18 条异常在 step 180/198/200/205 留档，长度与截断扣分、sample index/bucket 链路均有效。
- step 179/189/199/209 的 greedy EM 为 `0.61/0.59/0.55/0.61`，sample format 为 `0.925/0.975/0.950/0.9375`；格式没有持续崩塌，但能力未超过 step 169 的 0.67。
- 结论：监控和停止机制整改有效，训练目标尚未达标；继续使用 checkpoint-169，下一轮应更早、更强地约束 reference KL，不能从 checkpoint-211 强行续训。

<a id="part-1-section-13"></a>

### 13. 复盘时的提问顺序

1. 目标和基线是否固定？
2. SFT 起点、训练集、验证集和 reward 是否正确？
3. rollout 是否有差异，reward 是否能排序？
4. advantage、grad norm 和 adapter 权重是否真实变化？
5. 固定验证集是否改善且无格式、长度退化？
6. 停止是收敛早停还是异常终止？
7. 现有证据究竟支持停止、修复，还是扩大训练？

始终沿因果链回答，不能因为后端指标看起来正常而跳过前端证据。

<a id="part-1-section-14"></a>

### 14. 按阶段下钻的专题索引

- 基线与 oracle 可学性：`30_eval_oracle_stage_guide_cn.md`
- 数据格式、rule reward 与 GRPO 实现：`40_grpo_rule_reward_implementation_cn.md`
- 指标因果链、停止条件和故障定位：`41_grpo_metrics_stop_criteria_cn.md`
- 为什么不能跳过 probe/diagnostic：[第六部分：v5 早停与诊断门禁](#part-6)
- batch、rollout、激活值和 WDDM 显存：[第四部分：显存与批量预算](#part-4)；训练名词附录：[第三部分：训练名词附录](#part-3)

<a id="part-2"></a>

## 第二部分：checkpoint-169 整改探针 step 211 早停因果分析

> 本部分作用：沿日志证据排查主要矛盾、组合问题和下一轮控制变量实验。

本部分目录：

- [1. 文档目标](#part-2-section-1)
- [2. 原始问题是什么](#part-2-section-2)
- [3. 先区分“正确停止”与“训练成功”](#part-2-section-3)
- [4. 排查是如何一步步展开的](#part-2-section-4)
- [5. 主要矛盾是什么](#part-2-section-5)
- [6. 多种因素如何组合成失控链](#part-2-section-6)
- [7. 原因分级](#part-2-section-7)
- [8. 为什么现在不能声称找到唯一主因](#part-2-section-8)
- [9. 控制变量实验顺序](#part-2-section-9)
- [10. 下一轮通过标准](#part-2-section-10)
- [11. 证据入口](#part-2-section-11)

<a id="part-2-section-1"></a>

### 1. 文档目标

本文复盘 `qwen3_0d6b_grpo_v7_repair_probe_from169` 为什么没有完成计划中的50步，而是在 step 211 提前终止。

重点不是为早停找一个简单理由，而是区分三个层次：

```text
直接停止条件是什么
→ 什么训练趋势触发了停止条件
→ 哪些训练设计共同造成这一趋势
```

本文把“已由日志确认的事实”“有证据支持但尚未隔离的因素”“已基本排除的猜测”分开记录。最终主因仍需控制变量实验确认。

<a id="part-2-section-2"></a>

### 2. 原始问题是什么

formal500 在 step 218 出现严重格式退化和较高 actor-reference KL。整改后，从最佳 `checkpoint-169` 创建新分支，计划运行 step 170–219，共50个 GRPO step。

整改探针希望同时验证：

1. reference KL 与 update KL 是否按正确语义分开监控。
2. 自适应 KL 系数是否会在累计偏离时增强约束。
3. 严格格式、长度和截断 reward 是否真正生效。
4. sample@n 是否能暴露 greedy 验证看不到的尾部退化。
5. signal/KL guard 是否能避免再次无限制训练。

实际结果是：探针完成 step 170–211，共42步，在 step 211 因 reference KL 连续3个窗口超过0.1而停止。

```text
step 209 rolling reference KL = 0.1073，失败窗口 1/3
step 210 rolling reference KL = 0.1034，失败窗口 2/3
step 211 rolling reference KL = 0.1119，失败窗口 3/3
```

直接停止条件没有错误：新 guard 使用的是 actor 相对冻结 SFT reference 的累计偏离，不再使用局部 `approx_kl` 代替。

真正需要回答的是：为什么 reference KL 没有被拉回，而是持续越过硬阈值。

<a id="part-2-section-3"></a>

### 3. 先区分“正确停止”与“训练成功”

安全机制正确工作，只能证明训练知道何时停车，不能证明训练方向正确。

本次探针可以分成两个结论：

| 层次 | 结论 |
|---|---|
| 工程监控与保护 | 通过：指标语义、预警、自适应系数、异常留档和硬停止均生效 |
| 能力提升目标 | 未通过：EM 没有超过 step 169 的0.67，reference KL 最终失控 |

所以不能因为 guard 成功终止，就把整改理解成“已经能稳定训练”。

<a id="part-2-section-4"></a>

### 4. 排查是如何一步步展开的

#### 4.1 第一问：是不是单次 PPO 更新突然爆炸

先检查 `approx_kl`、clip fraction、grad norm 和非有限值。

证据：

- 42步中 `update_kl` 最大只有0.000113，远低于0.01阈值。
- clip fraction 几乎为0。
- grad norm 均值约0.681，最大约1.023。
- 没有 NaN、Inf、OOM 或 probability tensor 异常。

结论：不是某一个 mini-batch 的局部 PPO 更新突然爆炸，而是许多局部小更新长期累积成 reference 偏离。

这一步把问题从“单次数值异常”转向“长期控制失效”。

#### 4.2 第二问：KL 控制是否真的介入

探针的系数变化为：

```text
step 170–199 使用 0.005
step 200–209 使用 0.0075
step 210–211 使用 0.01125
```

对应 reference KL 均值：

| 窗口 | reference KL 均值 | 本窗口系数 |
|---|---:|---:|
| 170–179 | 0.0330 | 0.005 |
| 180–189 | 0.0398 | 0.005 |
| 190–199 | 0.0566 | 0.005 |
| 200–209 | 0.0913 | 0.0075 |
| 210–211 | 0.1035 | 0.01125 |

自适应机制确实执行了，但第一次增强发生在 step 199，第二次发生在 step 209。它开始增强时，策略已经进入明显偏离区间。

结论：不是“自适应 KL 没有运行”，而是介入频率和控制增益跟不上策略漂移速度。

#### 4.3 第三问：KL 惩罚在总 loss 中是否足够大

当前目标近似为：

```text
total_loss = policy_loss + kl_coef × reference_kl
```

部分实际量级：

| step | abs(policy loss) | KL项 | KL项占policy loss粗略比例 |
|---:|---:|---:|---:|
| 170 | 0.0465 | 0.000235 | 0.50% |
| 189 | 0.1590 | 0.000255 | 0.16% |
| 200 | 0.1332 | 0.000477 | 0.36% |
| 209 | 0.0648 | 0.000965 | 1.49% |
| 210 | 0.0314 | 0.001203 | 3.83% |

在最主要的前40步里，KL 项的记录值大部分时间只占 policy loss 绝对值的百分之零点几到百分之一左右。这里必须注意：loss 标量的比例不等于参数梯度范数的比例，不能只凭这一张表断言 reward 梯度一定是 KL 梯度的多少倍。

结论：日志直接确认的是“记录到的 KL penalty 标量较小，而且系数提高后 reference KL 仍继续上升”；两者结合，强烈支持当前组合下 KL 的有效纠偏不足，但若要精确比较两类信号，还需额外记录 policy 与 KL 各自的梯度范数或做 KL 控制消融。

#### 4.4 第四问：是不是最近抽到了坏数据

检查42步的 prompt 来源、重复率和分桶分布：

- 共168次 prompt 抽取，覆盖166个唯一样本，仅2个样本重复一次。
- 探针分桶比例与5759条训练集总体比例接近。
- step 200–211 反而有更多 `greedy_correct__oracle_mixed` 样本，但 rollout exact 仍下降。
- 每步使用随机抽样，不存在顺序进入某个坏数据尾段。

结论：数据文件损坏、重复训练少数题或顺序走入坏数据段，都不是主要原因。

#### 4.5 第五问：是不是格式和长度 reward 压过数学奖励

检查 group diagnostics：

- reward 与 rollout exact rate 的相关系数约0.968。
- 平均答案 reward 分量约0.521。
- 平均格式 reward 分量约0.089。
- 平均长度和截断分量约为 -0.012 与 -0.009。
- rollout format 总均值约0.964。

数学正确性仍然是 reward 的主要来源。格式与截断规则能改变少量回答排序，但没有证据表明它们压过了答案正确性。

结论：格式 reward 不是 step 211 KL 失控的主因；192-token 长度惩罚是否需要进一步细化，仍应另做 reward A/B，不能用本次混合探针直接定论。

#### 4.6 第六问：训练更新强度是否在重复放大小批次方向

当前每个 GRPO step：

```text
4个独立prompt × 每题8条rollout = 32条rollout
32 ÷ mini-batch 16 = 每个PPO epoch有2个mini-batch
2个mini-batch × PPO epochs 2 = 每步4次optimizer update
```

42个 GRPO step 共执行约168次 optimizer update，但每步只有4道独立题。rollout 32条并不等于32道独立题，同题回答之间高度相关。

PPO 第二个 epoch 会重新使用相同 response、reward、advantage 和 old-policy log-prob。若当前4道题产生的梯度方向有偏差，第二轮会继续强化这一方向。

结论：`batch=4 + ppo_epochs=2 + lr=5e-6` 构成偏强且高方差的更新组合。单独哪一个是主因仍需消融实验。

#### 4.7 第七问：改变目标后是否错误继承了旧优化器历史

checkpoint-169 来自旧 reward/KL 目标。整改分支改变了：

- 无格式正确答案的 reward。
- 严格格式门控。
- token 长度和截断惩罚。
- KL 初始系数和自适应控制。

但 resume 仍无条件加载 checkpoint-169 的 AdamW state。该 optimizer 已执行680次 update，392组参数均有非零 `exp_avg` 动量。

这意味着整改分支第一步使用的是：

```text
旧目标形成的Adam动量和二阶统计
+ 新目标产生的当前梯度
```

同一目标下断点续训应恢复 optimizer；改变 reward/KL 后创建新目标分支，则应把“恢复权重”和“恢复 optimizer”分开。

结论：旧 optimizer state 是确定存在的实验污染，但它在本次失控中占多大比例尚未被单独验证。KL 是后半段才加速上升，因此不能把全部责任都归给旧动量。

#### 4.8 第八问：为什么 EM 没有随着训练稳定上升

固定验证结果：

| step | greedy EM | greedy format | sample EM | sample format |
|---:|---:|---:|---:|---:|
| 169 起点 | 0.67 | - | - | - |
| 179 | 0.61 | 0.96 | 0.3500 | 0.9250 |
| 189 | 0.59 | 0.97 | 0.3875 | 0.9750 |
| 199 | 0.55 | 0.98 | 0.2625 | 0.9500 |
| 209 | 0.61 | 0.99 | 0.2500 | 0.9375 |

格式保持较高，但数学 EM 没有超过起点。训练 rollout exact 从前10步平均0.628下降到后12步约0.471，说明能力退化不只出现在验证集。

结论：当前策略变化没有转化为可泛化的数学能力提升。目标不应要求每个验证点单调上升，但至少需要 EM 均值、最好值或 rollout exact 趋势不低于起点，本次未达到。

<a id="part-2-section-5"></a>

### 5. 主要矛盾是什么

本次训练的主要矛盾不是“guard 太严格”，而是：

```text
reward/advantage 驱动 actor 持续更新的力量
>
reference KL 对 actor 的锚定与纠偏力量
```

具体表现为：

- 局部 update KL 很小，看起来每次更新都安全。
- 但每步有4次 optimizer update，很多小更新持续累积。
- KL 项在总 loss 中长期占比太小。
- 自适应控制每10步才调整，响应滞后。
- 等系数提高时，reference KL 已经接近或越过硬阈值。

因此硬停止只暴露了问题，没有制造问题。若取消硬停止，模型更可能继续偏离，而不是自动恢复 EM。

<a id="part-2-section-6"></a>

### 6. 多种因素如何组合成失控链

目前最值得优先验证的组合机制如下。其中更新次数、KL 趋势和控制频率是日志事实；“跨题梯度方差较大”和“旧动量放大偏移”仍是待消融的机制假设：

```text
每步只有4个独立prompt
→ 跨题梯度方差较大
→ 同一批rollout做2个PPO epoch
→ 小批次偶然方向被重复强化
→ lr=5e-6下累计进行168次optimizer update
→ KL项长期只占policy loss很小比例
→ reference偏离逐步积累
→ 自适应KL每10步才提高一次
→ 控制器来不及在硬线前拉回
→ step 209–211连续超限并停止
```

旧 AdamW 动量可能进一步放大或改变早期更新方向，但需要单独对照才能确定权重。

<a id="part-2-section-7"></a>

### 7. 原因分级

#### 7.1 已确认的直接原因

- step 209–211 的 rolling reference KL 连续3次超过0.1。
- hard guard 按配置正确停止训练。

#### 7.2 已有强证据支持的训练问题

- KL penalty 在总 loss 中长期过弱。
- 自适应 KL 的10步间隔和1.5倍增益反应过慢。
- 局部 update KL 小，但重复 optimizer update 造成长期累计偏离。

#### 7.3 合理但尚未隔离的共同因素

- `ppo_epochs=2` 重复放大小 prompt batch 的梯度方向。
- `lr=5e-6` 对当前 batch/epoch 组合偏高。
- 每步4个独立 prompt 导致跨题梯度方差较大。
- 改变目标后继承旧 optimizer 动量。

#### 7.4 已基本排除的主因猜测

- 最近训练数据文件损坏。
- 顺序进入坏数据区间。
- 少数样本被高频重复抽中。
- 格式 reward 完全压过数学正确性。
- 单次 PPO update 数值爆炸。
- NaN、OOM 或框架异常导致的被动中止。

<a id="part-2-section-8"></a>

### 8. 为什么现在不能声称找到唯一主因

整改探针同时改变了 reward、KL 配置、验证形态和 guard，又继承了旧 optimizer。它是混合修复实验，不是干净的单变量 A/B。

因此当前可以确定“约束相对更新太弱且太慢”，但不能直接断言：

- 只把 PPO epoch 降到1就一定解决。
- 只把学习率降到2e-6就一定解决。
- 只重置 optimizer 就一定解决。
- 只提高 KL 系数就能让 EM 上升。

这些判断必须用控制变量实验验证。

<a id="part-2-section-9"></a>

### 9. 控制变量实验顺序

#### 9.1 所有实验共同固定

- 都从 checkpoint-169 的同一份 LoRA 权重开始。
- 改变目标后统一重置 optimizer。
- 固定训练数据、reward、prompt 抽样种子和生成参数。
- 先测零步 greedy eval100 与 sample@n 基线。
- 保持相同验证频率、guard 和总步数。

重置 optimizer 是实验卫生条件，不应再和其他主变量混在一起。

#### 9.2 第一轮单变量消融

| 实验 | 相对干净基线的唯一变化 | 要回答的问题 |
|---|---|---|
| R0 | 无：fresh optimizer、lr=5e-6、PPO×2 | 建立新目标干净基线 |
| R1 | PPO epoch 2→1 | 同批 rollout 重复更新是否是主因 |
| R2 | lr 5e-6→2e-6或3e-6 | 单次参数步幅是否过大 |
| R3 | KL 在 warning 阶段更快增强 | 控制器滞后是否是主因 |
| R4 | 有效独立 prompt 4→8 | 跨题梯度方差是否是主因 |

R1 与 R2 不能在同一实验中同时修改，否则即使成功也无法归因。

另做一次短程 O1 对照：除 optimizer 恢复方式外完全相同，一组加载 checkpoint-169 的旧 optimizer，一组使用 fresh optimizer。O1 只用于量化旧动量的影响；正式的新目标分支仍应使用 fresh optimizer。

R4 需要额外控制总 rollout 数、有效 token 数和 optimizer update 次数。若 prompt 从4增至8的同时简单把32条轨迹增至64条，计算量和更新次数也会变化，不能把结果只归因于“独立 prompt 更多”；应使用梯度累积或等更新预算设计，并单独报告无法完全消除的组大小差异。

#### 9.3 训练长度

```text
前10步：代码、显存、NaN、格式门禁
前30步：reference KL斜率、rollout exact、EM趋势
通过30步后：再延长到50步
```

上次偏离在约20步后才明显加速，只跑10步不足以判断长期稳定性。

#### 9.4 主要比较指标

- reference KL 的均值、斜率和达到 warning/hard 的步数。
- update KL、clip fraction、grad norm。
- rollout exact rate 与 all-wrong group rate。
- greedy/sample EM 相对 step-169 零步基线的变化。
- format、EOS、hit-max 和异常样本分布。
- 相同 GRPO step 下实际 optimizer update 次数。

#### 9.5 随机性复核

先用固定 seed 筛选配置，再对表现最好的1–2组使用2–3个 seed 重复。单次运行可能受到 prompt 与 rollout 随机性的影响，不能只用一次最高 EM 宣称主因已解决。

<a id="part-2-section-10"></a>

### 10. 下一轮通过标准

下一轮不是以“跑满50步”为唯一成功条件，而应同时满足：

```text
reference KL不持续上升并保持在安全区
+ rollout exact不低于起点窗口
+ greedy EM均值和最好值不低于0.67
+ sample format、EOS和hit-max不恶化
+ 不依赖关闭guard才能完成训练
```

只有满足这些条件，才能说明训练控制不仅会停车，而且能在 reference 约束下产生可泛化的数学能力收益。

<a id="part-2-section-11"></a>

### 11. 证据入口

- 探针训练指标：`models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/plots/train_metrics.csv`
- 组内诊断：`models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/plots/group_diagnostics.csv`
- 验证指标：`models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/plots/val_metrics.csv`
- 异常 rollout：`models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/diagnostics/rollout_anomalies.jsonl`
- 最终状态：`models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/checkpoint-211/trainer_state.json`
- 训练名词附录：[第三部分：训练名词附录](#part-3)

<a id="part-3"></a>

## 第三部分：训练名词附录——从 rollout 到 optimizer update

> 本部分作用：统一 epoch、batch、rollout、PPO epoch、step 和 optimizer update 的语义。

本部分目录：

- [1. 本附录解决什么问题](#part-3-section-1)
- [2. 概念层级总览](#part-3-section-2)
- [3. 数据单位](#part-3-section-3)
- [4. Batch 系列概念](#part-3-section-4)
- [5. Epoch、step 与 update](#part-3-section-5)
- [6. 一次参数更新内部发生什么](#part-3-section-6)
- [7. PPO/GRPO 中的三个策略对象](#part-3-section-7)
- [8. PPO/GRPO 核心指标](#part-3-section-8)
- [9. 当前配置的完整算例](#part-3-section-9)
- [10. 高频误解纠正](#part-3-section-10)
- [11. 看日志时如何映射这些概念](#part-3-section-11)
- [12. 复盘自检问题](#part-3-section-12)

<a id="part-3-section-1"></a>

### 1. 本附录解决什么问题

这份附录专门解释训练日志和配置中容易混淆的名词，并把它们放进同一条执行链：

```text
训练数据集
→ 抽取 prompt batch
→ 每题生成多个 rollout
→ 计算 reward 与 advantage
→ 组成 rollout batch
→ 切成 PPO mini-batch
→ backward 计算梯度
→ optimizer.step 更新参数
→ 重复若干 PPO epoch
→ 完成一个 GRPO step
```

理解任何名词时，先问两个问题：它描述的是“数据单位”，还是“训练循环”；它统计的是“独立题目数”，还是“生成回答数”。

<a id="part-3-section-2"></a>

### 2. 概念层级总览

```text
完整 GRPO 实验
└─ 多个 GRPO training step
   ├─ prompt batch：本步抽到的独立题目
   ├─ rollout group：同一道题的多个回答
   ├─ rollout batch：本步全部回答
   └─ 多个 PPO epoch：重复遍历本步 rollout batch
      └─ 多个 PPO mini-batch
         ├─ forward
         ├─ loss
         ├─ backward
         └─ optimizer.step
```

<a id="part-3-section-3"></a>

### 3. 数据单位

#### 3.1 Sample、problem 与 prompt

- `sample`：数据集中的一条样本；在 GSM8K 中通常是一道题及其参考答案。
- `problem/question`：原始数学题文本。
- `prompt`：真正送入模型的输入，通常包含题目、格式指令和 chat template。
- 一条 sample 经过模板渲染后成为一个 prompt。

prompt 是独立数学信息的主要单位。抽到4个 prompt，表示本步只覆盖4道独立题目。

#### 3.2 Response、generation 与 rollout

- `response`：模型针对一个 prompt 生成的一段回答。
- `generation`：生成 response 的推理过程或操作。
- `rollout`：RL 中用于训练的一条采样轨迹；当前 GSM8K 项目里基本等价于“prompt + 一条生成回答 + token 概率 + reward”。
- `trajectory`：更一般的轨迹概念；多轮 agent 任务可能包含多个动作，在当前单轮 GSM8K 中可近似理解为 rollout。

同一道题生成8次，会得到8条 rollout，但仍然只有1道独立题目。

#### 3.3 Rollout group

`rollout group` 是同一个 prompt 的多条 rollout 集合。GRPO 在组内比较回答：

```text
同一道题
├─ rollout 1 → reward 1
├─ rollout 2 → reward 2
└─ ...
```

组内有高低 reward，才能计算“相对更好”和“相对更差”的 advantage。组内 reward 完全相同，则该题几乎没有相对学习信号。

#### 3.4 Reward 与 advantage

- `reward`：规则或奖励模型给一条 rollout 的分数。
- `advantage`：该回答相对同组平均水平好多少或差多少。

当前 GRPO 可直观写成：

```text
advantage = (当前reward - 组内平均reward) / 组内reward标准差
```

reward 是绝对评分，advantage 是用于决定梯度方向的相对评分。

<a id="part-3-section-4"></a>

### 4. Batch 系列概念

#### 4.1 Batch

`batch` 泛指一次一起处理的一批数据，但必须说明它是哪一种 batch。当前项目至少有 prompt batch、generation batch、rollout batch 和 PPO mini-batch。

#### 4.2 Prompt batch / train batch

`train_batch_size` 表示一个 GRPO step 抽取多少个独立 prompt。

当前配置：

```text
train_batch_size = 4
```

因此每步只抽4道独立数学题。它决定跨题梯度方差，不能用 rollout 数替代。

#### 4.3 rollout_n

`rollout_n` 表示每个 prompt 采样多少条回答。

当前配置：

```text
rollout_n = 8
```

所以本步 rollout 总数是：

```text
4个prompt × 每题8条rollout = 32条rollout
```

增加 `rollout_n` 主要提高同题组内比较质量，不等于增加独立题目覆盖。

#### 4.4 rollout_batch_size

`rollout_batch_size` 是生成阶段一次送入 `generate()` 的 prompt 数，主要控制显存和生成吞吐。

它是工程调度参数，不直接表示 optimizer 一次使用多少训练样本。例如 prompt batch 为8、rollout batch为4时，会分两次生成，但仍属于同一个 GRPO step。

#### 4.5 Rollout batch

本步所有 prompt 的所有回答合在一起，构成 rollout batch：

```text
rollout_batch_count = train_batch_size × rollout_n
```

当前是32条。每条 rollout 带有 response token、reward、advantage、old-policy log-prob 和 reference log-prob。

#### 4.6 Mini-batch

mini-batch 是从当前 rollout batch 中切出、一次送入前向传播和反向传播的小块。

当前：

```text
rollout batch = 32
ppo_mini_batch_size = 16
```

所以完整遍历一次 rollout batch 需要2个 mini-batch。每个 mini-batch 通常对应一次 `optimizer.step()`。

#### 4.7 Effective batch 与梯度累积

`effective batch size` 表示一次参数更新实际汇总了多少独立数据。梯度累积会先对多个 mini-batch 执行 backward，最后只执行一次 optimizer update。

当前 GRPO 实现每个 mini-batch 都立即更新，没有跨 prompt batch 的梯度累积。因此 mini-batch 中虽然有16条 rollout，它们可能只来自2道题，独立数学题数量仍然很少。

<a id="part-3-section-5"></a>

### 5. Epoch、step 与 update

#### 5.1 Epoch 的一般定义

一个 epoch 是对“当前指定数据集合”的一次完整遍历。关键是先确定这个集合指什么。

- SFT dataset epoch：完整遍历一次整个 SFT 数据集。
- PPO epoch：完整遍历一次当前 GRPO step 已生成的 rollout batch。
- epoch 不是“整个训练完成一次”，也不必等于遍历整个5759条 GRPO 数据。

#### 5.2 PPO epoch

`ppo_epochs=2` 表示同一批 rollout 被完整使用两遍。第二遍不会重新生成回答，只会重新打乱并再次训练。

```text
PPO epoch 1：32条rollout → 两个16条mini-batch → 更新2次
PPO epoch 2：同样32条重新打乱 → 两个mini-batch → 再更新2次
```

每条 rollout 在每个 PPO epoch 出现一次，所以每条被训练两遍；整个 step 总计发生4次 optimizer update。

#### 5.3 GRPO training step

一个 GRPO step 是一次完整的“采样—评分—更新”循环：

1. 抽取 prompt。
2. 生成 rollout。
3. 计算 reward 和 advantage。
4. 计算 old/reference log-prob。
5. 完成全部 PPO epoch 和 mini-batch 更新。
6. 写入该 step 的训练指标。

进入下一个 GRPO step 后，才会抽新题并生成新 rollout。

#### 5.4 Optimizer update / optimizer step

一次 `optimizer.step()` 才是一次真正的参数更新。日志中的一个 GRPO step 可以包含多次 optimizer step。

若 rollout 数能整除 mini-batch，单个 GRPO step 的更新次数为：

```text
optimizer_updates_per_step
= (train_batch_size × rollout_n ÷ ppo_mini_batch_size) × ppo_epochs
```

当前为：

```text
(4 × 8 ÷ 16) × 2 = 4次optimizer update
```

整改探针完成42个 GRPO step，因此执行了约 `42 × 4 = 168` 次 optimizer update。

#### 5.5 total_training_steps

`total_training_steps` 限制的是 GRPO step 数，不是 optimizer update 数，也不是 epoch 数。

如果从 `next_step=170` 训练到 `total_training_steps=220`，计划执行的是 step 170–219，共50个 GRPO step。

<a id="part-3-section-6"></a>

### 6. 一次参数更新内部发生什么

#### 6.1 Forward、loss 与 backward

- `forward`：模型根据输入计算 logits 和 token probability。
- `loss`：把策略好坏转成需要最小化的标量。
- `backward`：根据 loss 计算每个可训练参数的梯度。
- `gradient`：参数朝哪个方向、以多大强度变化的局部信号。

#### 6.2 zero_grad、gradient clipping 与 optimizer.step

典型 mini-batch 更新顺序是：

```text
optimizer.zero_grad
→ forward
→ loss
→ backward
→ gradient clipping
→ optimizer.step
```

- `zero_grad`：清除上一次更新残留的梯度。
- `gradient clipping`：限制梯度范数，防止单次更新爆炸。
- `optimizer.step`：AdamW 根据梯度和历史状态真正修改 GRPO LoRA 参数。

#### 6.3 Learning rate

学习率控制每次 optimizer update 的基础步幅。相同 GRPO step 数下，学习率更高、PPO epoch 更多，通常都会加快策略变化，也会提高偏离 reference 的风险。

#### 6.4 Optimizer state

AdamW 不只保存模型权重，还保存：

- `exp_avg`：历史梯度的一阶动量。
- `exp_avg_sq`：历史梯度平方的二阶统计。
- `step`：已经执行过的 optimizer update 次数。

同一目标下断点续训应恢复 optimizer state；改变 reward 或 KL 目标创建新分支时，应优先只加载模型/LoRA 权重并重置 optimizer，避免旧梯度统计污染新目标。

<a id="part-3-section-7"></a>

### 7. PPO/GRPO 中的三个策略对象

#### 7.1 Actor

actor 是当前参与生成和训练的模型。本项目中是：

```text
Base + SFT adapter merge + 可训练 GRPO LoRA
```

反向传播只更新 GRPO LoRA。

#### 7.2 Old policy

old policy 是生成当前 rollout 时 actor 的概率快照。当前实现保存本批回答更新前的 token log-prob，不需要额外复制一个完整模型。

它用于计算 PPO ratio，限制同一批数据上的局部更新幅度。

#### 7.3 Reference policy

reference 是冻结的 SFT model，用于约束 actor 不要在整个 GRPO 过程中长期偏离 SFT 起点。

old policy 约束“这一批更新不要跳太远”，reference 约束“长期训练不要离 SFT 太远”。

<a id="part-3-section-8"></a>

### 8. PPO/GRPO 核心指标

#### 8.1 Policy loss

policy loss 使用 advantage 提高高奖励回答的概率、降低低奖励回答的概率。它是推动模型学习 reward 的主要梯度来源。

#### 8.2 Ratio 与 clip

```text
ratio = exp(current_logp - old_logp)
```

ratio 表示当前 actor 相对生成 rollout 时的 old policy 改变了多少。PPO clipping 限制利用同一批 rollout 时的局部更新幅度。

#### 8.3 update KL / approx_kl

`approx_kl` 衡量当前 PPO 更新相对 old policy 的局部变化。它回答“这一批是否更新过猛”。

#### 8.4 reference KL / kl_loss

`kl_loss` 衡量 actor 相对冻结 SFT reference 的累计偏离。它回答“整个 GRPO 过程是否已经远离 SFT”。

单步 update KL 很小，经过很多 optimizer update 后，reference KL 仍然可能持续累积。

#### 8.5 clip_frac

`clip_frac` 表示有多少有效 token 的 ratio 触及 PPO 裁剪边界。接近0说明 clipping 几乎没有介入，不等于模型没有长期漂移。

#### 8.6 grad_norm

`grad_norm` 表示本次反向传播梯度的整体尺度。非零说明信号进入了 LoRA，但不能单独证明梯度方向有利于验证 EM。

<a id="part-3-section-9"></a>

### 9. 当前配置的完整算例

```text
train_batch_size = 4
rollout_n = 8
rollout batch = 4 × 8 = 32
ppo_mini_batch_size = 16
每个PPO epoch的mini-batch数 = 32 ÷ 16 = 2
ppo_epochs = 2
每个GRPO step的optimizer update数 = 2 × 2 = 4
```

数据复用关系是：

```text
4道独立题目
→ 每题8条相关回答
→ 32条rollout
→ 每条rollout在两个PPO epoch中各使用一次
→ 总共4次optimizer.step
```

如果只把 `ppo_epochs` 改成1，则每个 GRPO step 只执行2次 optimizer update，但生成成本不变。

<a id="part-3-section-10"></a>

### 10. 高频误解纠正

| 常见说法 | 更准确的理解 |
|---|---|
| 一个 epoch 就是一次完整训练 | epoch 只是完整遍历一次当前指定的数据集合 |
| mini-batch 是一个 epoch 的全部数据 | mini-batch 是 epoch 中一次更新使用的小块 |
| 32条 rollout 就是32道独立题 | 当前是4道题，每题8条相关 rollout |
| 一个 training step 只更新一次参数 | 当前一个 GRPO step 更新4次参数 |
| PPO epoch 2 会重新生成两批回答 | 不会；同一批回答被重复训练两遍 |
| rollout batch size 就是训练 batch size | 前者常指生成调度，后者指独立 prompt 数 |
| update KL 小就不会远离 SFT | 局部小更新经过多次累积，reference KL 仍可变大 |
| checkpoint 只保存模型权重 | 完整 checkpoint 还保存 optimizer、trainer 和 RNG 状态 |

<a id="part-3-section-11"></a>

### 11. 看日志时如何映射这些概念

| 指标或字段 | 对应概念 | 回答的问题 |
|---|---|---|
| `step` | GRPO training step | 已完成多少次采样—训练循环 |
| `rollout_n` | 每题回答数 | 同题组内比较有多少候选 |
| `group_count` | 独立 prompt 数 | 本步覆盖多少道独立题 |
| `rollout_count` | 回答总数 | 本步生成多少条轨迹 |
| `mixed_group_rate` | 组内数学差异 | 多少题同时采到正确和错误回答 |
| `policy_loss` | reward 学习目标 | actor 正在多强地追随 advantage |
| `approx_kl` | actor 对 old policy | 当前批次更新是否过猛 |
| `kl_loss` | actor 对 reference | 长期是否偏离 SFT 起点 |
| `clip_frac` | PPO clipping | 局部裁剪是否真正介入 |
| `grad_norm` | 梯度尺度 | LoRA 是否收到训练信号 |

<a id="part-3-section-12"></a>

### 12. 复盘自检问题

1. 当前说的 epoch 遍历的是整个 dataset，还是本步 rollout batch？
2. batch size 统计的是独立 prompt，还是相关 rollout？
3. 一个 GRPO step 内实际执行多少次 optimizer update？
4. 同一批 rollout 被重复使用了多少个 PPO epoch？
5. `approx_kl` 和 `reference_kl` 分别比较哪两个策略？
6. 增加 rollout 数是在降低同题采样噪声，还是增加跨题覆盖？
7. resume 时训练目标是否完全相同，是否应该恢复 optimizer state？

能够回答这七个问题，才算真正理解配置中的 step、epoch、batch、rollout 与 optimizer update 之间的关系。

<a id="part-4"></a>

## 第四部分：GRPO 训练显存、共享内存与批量预算

> 本部分作用：解释显存构成，并为 batch、rollout 和全量微调决策提供资源依据。

本部分目录：

- [1. GRPO 训练的两个阶段](#part-4-section-1)
- [2. 常驻部分：模型权重](#part-4-section-2)
- [3. Training 阶段显存大头详解](#part-4-section-3)
- [4. KV Cache：极小，不是瓶颈](#part-4-section-4)
- [5. Reference 前向传播的显存占用](#part-4-section-5)
- [6. CUDA 内存分配器：为什么 8.7 GB 数据占了 11.6 GB VRAM](#part-4-section-6)
- [7. Gradient Checkpointing：用计算换显存](#part-4-section-7)
- [8. Optimizer 状态和其他](#part-4-section-8)
- [9. 总汇总](#part-4-section-9)
- [10. 关键认知总结](#part-4-section-10)

> 背景：在 RTX 4070 (12 GB VRAM) 上用 Qwen3-0.6B 模型跑 GRPO 训练，
> 任务管理器显示 11.6 GB Dedicated GPU Memory + 3.2 GB Shared GPU Memory = 14.8 GB 总占用。
> 本文逐项拆解每一部分显存占用是什么、为什么需要、有多大。

---

<a id="part-4-section-1"></a>

### 1. GRPO 训练的两个阶段

GRPO 训练的每一步分为两个阶段，它们不会同时占用显存峰值：

```
Rollout 阶段 (generate):
  actor.generate(prompt) → 生成多个回答 → 计算 reward → 计算 advantage
  显存大头: actor generate 的临时 tensor
  KV cache 极小 (~36 MB, 不是大头!)

Training 阶段 (forward + backward + optimizer step):
  用 rollout 数据做 PPO clip + KL penalty 更新 actor 的 LoRA 参数
  显存大头: actor 前向传播的中间激活值 (~5 GB, 供反向传播计算梯度)
```

---

<a id="part-4-section-2"></a>

### 2. 常驻部分：模型权重

无论在哪个阶段，以下模型权重始终驻留在 GPU 上：

#### 2.1 actor 权重 (~1.20 GB)

```
actor = base(0.6B) + SFT(merged) + 新 LoRA(r=16)

参数量: ~600M (Qwen3-0.6B)
存储: 600M × 2 bytes (fp16/bf16) ≈ 1.20 GB

虽然只有 LoRA 参数 (~10M) 可训练，但整个模型的权重都在 GPU 上。
LoRA 是叠加在原模型上的，前向传播需要先算原模型的线性层，
再叠加 LoRA 的结果，所以原模型权重不能卸载。
```

#### 2.2 reference 权重 (~1.20 GB)

```
reference = copy.deepcopy(actor 去掉新 LoRA 后的 merged model)

参数量: ~600M
存储: 600M × 2 bytes ≈ 1.20 GB

reference 是冻结的 SFT 模型，用于 KL 惩罚：
  KL loss = π_current 偏离 π_reference 的程度
  计算 KL 需要用 reference 做前向传播得到 ref_log_probs
  所以 reference 的完整权重也必须在 GPU 上

为什么不能共享 actor 的权重？
  actor 的权重在训练过程中会不断更新（梯度更新 LoRA 参数）
  reference 必须保持不变（冻结的 SFT 起点）
  如果共享，reference 也会跟着变 → KL 惩罚失效
```

**常驻模型权重合计: ~2.40 GB**

---

<a id="part-4-section-3"></a>

### 3. Training 阶段显存大头详解

#### 3.1 为什么 training 阶段是显存瓶颈

训练阶段做 actor 前向传播时，必须保存每一层的**中间激活值**供反向传播使用：

```
前向传播: input → 层1 → 激活1 → 层2 → 激活2 → ... → 层28 → 激活28 → loss

反向传播: loss → ∂loss/∂激活28 → ∂loss/∂层28参数 + ∂loss/∂激活27
                  → ∂loss/∂激活27 → ∂loss/∂层27参数 + ∂loss/∂激活26
                  → ...

要计算 ∂loss/∂层i参数, 需要激活i (前向传播时层i的输出)
所以前向传播时必须保存每一层的激活值, 直到反向传播用完才能释放
```

#### 3.2 激活值的精确计算 (Qwen3-0.6B 正确参数)

Qwen3-0.6B 配置:
- hidden_size = 1024
- intermediate_size = 3072
- num_hidden_layers = 28
- num_attention_heads = 16 (query heads)
- num_key_value_heads = 8 (GQA, KV heads)
- head_dim = 128
- vocab_size = 151936

训练时 batch_size=2, rollout_n=2, 总序列数=4, seq_len≈768:

```
每层激活值的子项:

1. Attention 子激活:
   - Q_proj 输出: B×S×n_q×head_dim = 4×768×16×128 × 2B = 12582912 B ≈ 12.0 MB
   - K_proj 输出: 4×768×8×128 × 2B = 6291456 B ≈ 6.0 MB
   - V_proj 输出: 4×768×8×128 × 2B ≈ 6.0 MB
   - Attention probs 矩阵: B×n_q×S×S × 2B
     = 4×16×768×768 × 2B = 75497472 B ≈ 71.5 MB ← S² 增长!
   - Attention output: B×S×hidden_size × 2B = 4×768×1024×2 ≈ 6.0 MB
   合计: ≈ 95.5 MB/层

2. MLP 子激活 (SwiGLU 结构):
   - gate_proj 输出: B×S×intermediate_size × 2B = 4×768×3072×2 ≈ 18.9 MB
   - up_proj 输出:   B×S×intermediate_size × 2B ≈ 18.9 MB
   - SwiGLU 中间 (gate×up): B×S×intermediate_size × 2B ≈ 18.9 MB
   - down_proj 输出: B×S×hidden_size × 2B ≈ 6.0 MB
   合计: ≈ 62.7 MB/层

3. 其他 (RMSNorm input residual, etc): ≈ 8 MB/层

每层总激活: ≈ 166 MB
28 层总激活: ≈ 28 × 166 MB = 4.65 GB ← 显存大头!
```

加上 logits 输出层:
```
logits: B×S×vocab_size × 2B = 4×768×151936×2 = 933 MB ≈ 0.93 GB
```

**actor 前向阶段总占用: ~4.65 GB (激活) + ~0.93 GB (logits) ≈ 5.58 GB**

#### 3.3 为什么激活值远大于 KV cache

关键对比——同样处理一个 token，激活值为什么远大于 KV cache：

```
KV cache 每层每个 token 只存 K 和 V 两个向量:
  K: 1 个向量, 维度 = num_kv_heads × head_dim = 8 × 128 = 1024
  V: 1 个向量, 维度 = 1024
  合计: 2048 个 fp16 数 ≈ 4 KB

激活值每层每个 token 存的是整层前向传播的所有中间结果:
  - MLP SwiGLU: intermediate_size=3072 >> hidden_size=1024
    每层存 3 个 3072 维的中间向量 → 每个 token 3×3072×2 = 18 KB
    vs KV cache 每个 token 4 KB → 单这一项就差 4.5×

  - Attention probs: 每个 query head 对所有 seq_len 个 token 计算 attention
    这是 S×S 的矩阵 → 二次增长!
    KV cache 不存 attention 权重, 只存 K/V 向量 → 线性增长

差距来源:
  1. MLP intermediate_size (3072) >> hidden_size (1024),
     每层要存 3 个 3072 维的中间向量
  2. Attention 权重矩阵是 S×S (二次增长),
     而 KV cache 只存 K/V 向量 (线性增长)
  3. 激活值必须全部保存供反向传播; KV cache 只存 K/V 向量供推理时复用
```

---

<a id="part-4-section-4"></a>

### 4. KV Cache：极小，不是瓶颈

#### 4.1 KV Cache 是什么

Transformer 生成文本是逐 token 进行的——每次只预测下一个 token。
每预测一个新 token 时，attention 需要所有之前 token 的 Key 和 Value 向量。

```
预测第 100 个 token 时:
  需要: 第 1~99 个 token 的 K 和 V
  这些 K/V 在预测第 1~99 个 token 时已经计算过了

两种选择:
  (a) 丢弃 → 第 100 步重新算 99 次 K/V → 极慢，且每步越来越慢
  (b) 缓存 → 把算过的 K/V 存在显存里 → 直接读取 → 只算当前 token 的 K/V
```

KV cache 就是选择 (b)：缓存已处理 token 的 K/V 向量，供后续 attention 计算。

#### 4.2 KV Cache 的显存计算

```
KV cache (每个 token) = num_layers × 2(K和V) × num_kv_heads × head_dim × dtype_size

Qwen3-0.6B (GQA, 8 个 KV head):
  28 层 × 2 × 8 KV heads × 128 (head_dim) × 2 bytes (fp16)
  = 28 × 2 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB/token

4 条序列 × 328 token (prompt + response):
  总 KV cache ≈ 4 × 328 × 112 KB ≈ 144 MB

即使序列更长 (768 token):
  总 KV cache ≈ 4 × 768 × 112 KB ≈ 344 MB

KV cache 始终在几十到几百 MB 级别，远不到 GB。
它不是显存瓶颈。
```

#### 4.3 为什么训练阶段不需要 KV Cache

```
训练阶段的 compute_sequence_log_probs:
  输入: prompt + response 拼成完整的 input_ids (全部 token 已知)
  处理: 所有 token 并行计算 (矩阵乘法一次性处理整条序列)
  不需要逐 token 生成 → 不需要 KV cache
  模型配置: use_cache=False → 关闭 KV cache
```

---

<a id="part-4-section-5"></a>

### 5. Reference 前向传播的显存占用

代码中 reference 的前向传播全部在 `torch.no_grad()` 下执行：

```python
# train_step() 中:
with torch.no_grad():
    _, ref_seq_log_probs = compute_sequence_log_probs(
        self.reference, batch["input_ids"], ...
    )

# _train_mini_batch() 中:
with torch.no_grad():
    ref_token_log_probs, _ = compute_sequence_log_probs(
        self.reference, mb_input_ids, ...
    )
```

`torch.no_grad()` 的效果:
1. 不构建计算图 → 不保存中间激活值供反向传播
2. 每层的中间结果在算完下一层后立即释放
3. 只保留最终输出 (ref_log_probs), 然后 .detach() 赋值给 batch

**reference 不产生持久中间激活值。**

但有一个时序问题: _train_mini_batch() 中,
reference 的前向传播发生在 actor 的 ~5 GB 激活值还没释放的时候：

```
actor 前向 (有梯度) → ~5 GB 中间激活值全部在显存中
reference 前向 (no_grad) → 在 actor 的 5 GB 之上叠加临时峰值
  峰值 = 单层 hidden_states + logits ≈ ~0.3 GB
total_loss.backward() → actor 激活值逐层释放
```

峰值瞬间: actor 5.6 GB + ref 0.3 GB 临时 + 模型权重 2.4 GB ≈ 8.3 GB

---

<a id="part-4-section-6"></a>

### 6. CUDA 内存分配器：为什么 8.7 GB 数据占了 11.6 GB VRAM

#### 6.1 核心问题

有效数据只有 ~8.7 GB，但 VRAM 实际占用了 11.6 GB。
差额 ~2.9 GB 来自 **CUDA 内存分配器的预留池**，不是"浪费"。

#### 6.2 CUDA 分配器的工作原理

```
PyTorch 的 CUDA 分配器不是"按需精确分配":
  它维护一个内存池 (memory pool), 从 CUDA runtime 批量申请大块 VRAM
  然后从池中切分给各个 tensor

行为示例:
  申请 5 GB 存激活值 → 分配器实际从 CUDA runtime 拿 6-7 GB (多拿 1-2 GB)
  激活值释放后 → 分配器不归还给 CUDA runtime → 留在池里等下次用
  下次申请 → 直接从池里切分 → 快 (不需要再向 CUDA runtime 申请)

为什么这样做?
  1. 防止碎片化: 频繁 malloc/free 导致 VRAM 出现大量小块空洞
  2. 性能: 向 CUDA runtime 申请是慢操作, 从池里切分是快操作
  3. 峰值缓冲: 训练循环中激活值反复申请/释放, 池要预留空间供峰值用
```

#### 6.3 关键指标

```
torch.cuda.memory_allocated() = 实际数据占用的 VRAM ≈ 8.7 GB
torch.cuda.memory_reserved()  = 分配器从 CUDA runtime 持有的 VRAM 总量 ≈ 11.6 GB

差额 = reserved - allocated ≈ 2.9 GB
这 2.9 GB 是分配器持有但未填数据的预留池空间

不是"浪费"——它是分配器正常工作的必要开销:
  - 内存碎片整理空间
  - 下一步训练循环的峰值缓冲
  - cuBLAS/cuDNN kernel 工作空间
```

#### 6.4 Shared GPU Memory：3.2 GB 系统 RAM 溢出

```
Windows WDDM (显示驱动模型) 下的 CUDA:
  当 VRAM 不足时, CUDA 分配器通过 WDDM 向系统请求 Shared GPU Memory
  Shared GPU Memory = 映射到 GPU 地址空间的系统 RAM
  通过 PCIe 访问 → 比显存慢 ~15×

为什么需要 3.2 GB Shared?
  VRAM 已占用 11.6 GB, 只剩 0.4 GB (12 - 11.6)
  分配器需要额外空间做:
    - 激活值释放后重新分配时的缓冲
    - cuBLAS/cuDNN kernel 的临时工作空间
    - 内存碎片整理
  0.4 GB 不够 → WDDM 提供 3.2 GB 系统 RAM 作为备用

  注意: Shared 不意味着有 3.2 GB 数据在系统 RAM 里
  大部分数据仍在 VRAM 内, Shared 只是"可用但未必在用"的备用空间
```

---

<a id="part-4-section-7"></a>

### 7. Gradient Checkpointing：用计算换显存

#### 7.1 原理

不开 checkpointing 时，前向传播保存所有层的激活值：

```
层1 激活 ✓ 保存 → 反向传播时直接读取
层2 激活 ✓ 保存 → 反向传播时直接读取
...
层28 激活 ✓ 保存 → 反向传播时直接读取

显存: 28 层 × ~166 MB/层 ≈ 4.65 GB
反向传播: 直接读取 → 快
```

开 checkpointing 时，只保存部分层的激活（checkpoint 点），其余层丢弃：

```
层1 激活 ✓ 保存 (checkpoint 点)
层2 激活 ✗ 丢弃
层3 激活 ✗ 丢弃
层4 激活 ✓ 保存 (checkpoint 点)
...
层28 激活 ✓ 保存 (checkpoint 点)

显存: ~7-8 层 × 166 MB ≈ 1.2 GB (只存约 1/4)
反向传播:
  到层3时 → 层2、3 的激活已被丢弃
  → 从层1 (checkpoint 点) 重新做前向传播 → 算出层2、3的激活
  → 再用它们计算梯度 → 多了一次前向计算 → 慢 ~30%, 但省显存
```

#### 7.2 效果

| | 不开 checkpointing | 开 checkpointing |
|---|---|---|
| actor 激活值显存 | ~4.65 GB (存所有层) | ~1.2 GB (只存部分层) |
| 训练速度 | 快 | 慢 ~30% (丢弃的层要重新前向) |
| OOM 风险 | 高 (余量 ~0.4 GB) | 低 (多出 ~3.5 GB 余量) |

---

<a id="part-4-section-8"></a>

### 8. Optimizer 状态和其他

#### 8.1 Optimizer 状态 (~0.08 GB)

```
AdamW 为每个可训练参数存 2 份状态:
  - momentum (一阶矩估计): 与参数同大小
  - variance (二阶矩估计): 与参数同大小

可训练参数 = LoRA 参数 ≈ 10M
optimizer 状态 = 10M × 2 × 4 bytes (fp32) ≈ 80 MB ≈ 0.08 GB

很小，不是显存瓶颈。
```

#### 8.2 CUDA 内核与框架开销

```
CUDA runtime 预分配的内存池
PyTorch 的内存分配器碎片
cuBLAS/cuDNN 的工作空间
这些是 CUDA 分配器预留池的一部分 (已包含在第 6 节的 2.9 GB 中)
```

---

<a id="part-4-section-9"></a>

### 9. 总汇总

#### 9.1 有效数据占用 (allocated)

| 类别 | 估算占用 | 说明 |
|---|---|---|
| actor 权重 | 1.20 GB | 600M × 2B, 常驻 |
| reference 权重 | 1.20 GB | 600M × 2B, 常驻 |
| actor 激活值 (28层) | ~4.65 GB | 真正的显存大头 |
| actor logits | ~0.93 GB | vocab_size=151936 导致 |
| ref 前向临时峰值 | ~0.3 GB | no_grad, 不持久 |
| optimizer 状态 | ~0.08 GB | LoRA 参数的 AdamW |
| **有效数据合计** | **~8.4 GB** | torch.cuda.memory_allocated() |

#### 9.2 实际显存占用 (实测)

| 项 | 大小 | 来源 |
|---|---|---|
| 有效数据 (allocated) | ~8.4 GB | 上表 |
| CUDA 分配器预留池 | ~2.9 GB | reserved - allocated |
| **VRAM 实际占用 (reserved)** | **11.6 GB** | 任务管理器 Dedicated |
| Shared GPU Memory | 3.2 GB | WDDM 系统 RAM 溢出 |
| **总占用** | **14.8 GB** | 11.6 + 3.2 |

#### 9.3 为什么 8.4 GB 数据占了 14.8 GB

```
核心原因: CUDA 内存分配器的预留池和 WDDM Shared 溢出

8.4 GB (数据)
  + 2.9 GB (CUDA 预留池——分配器持有但未填数据的 VRAM)
  + 2.5 GB (WDDM Shared 中 CUDA 分配器预留的备用空间)
  = 14.8 GB (总占用)

不是"浪费":
  预留池是 CUDA 分配器正常工作的必要开销
  防止碎片化、提供峰值缓冲、给 cuBLAS/cuDNN 工作空间
  Shared 是 WDDM 模式下 VRAM 不足时的自动溢出机制

如果没有预留池 → 频繁 malloc/free → 碎片化 → 还是会 OOM
```

---

<a id="part-4-section-10"></a>

### 10. 关键认知总结

> **真正的显存大头是 actor 前向传播的中间激活值 (~4.65 GB)，不是 KV cache (~144 MB)。**

> 激活值远大于 KV cache 的原因:
> 1. MLP 的 intermediate_size (3072) >> hidden_size (1024),
>    每层要存 3 个 3072 维的中间向量
> 2. Attention 权重矩阵是 S×S (二次增长),
>    而 KV cache 只存 K/V 向量 (线性增长)
> 3. 激活值必须全部保存供反向传播; KV cache 只存 K/V 向量供推理时复用

> **CUDA 分配器预留池 (~2.9 GB) + WDDM Shared (~3.2 GB) = 6.1 GB 额外占用。**

> 这不是"浪费"——是 CUDA 分配器正常工作的必要开销。
> 有效数据 8.4 GB + 分配器开销 6.4 GB = 总占用 14.8 GB。

> **解决 OOM 的正确方案是开 gradient checkpointing (激活值从 ~4.65 GB 降至 ~1.2 GB),
> 而不是减小 KV cache (它只有 ~144 MB, 优化它几乎没有效果)。**

> **Windows WDDM 模式下，Shared GPU Memory 是 CUDA 的自动溢出机制:
> 当 VRAM 接近满时，系统 RAM 通过 PCIe 映射为 GPU 可访问的地址空间。
> 大部分数据仍在 VRAM 内, Shared 只是备用。**

<a id="part-5"></a>

## 第五部分：GRPO 评估推理加速

> 本部分作用：在不改变评估口径的前提下规划 batch size 和推理优化。

本部分目录：

- [背景](#part-5-section-1)
- [已实现的优化](#part-5-section-2)
- [推荐命令](#part-5-section-3)
- [batch size 怎么选](#part-5-section-4)
- [对准确率的影响](#part-5-section-5)
- [其他加速方案评估](#part-5-section-6)
- [本阶段建议](#part-5-section-7)

本文记录 `post_training_framework` 中 GRPO/SFT/Base 评估脚本的推理加速方案，重点面向当前本机环境：

```text
GPU: NVIDIA GeForce RTX 4070
显存: 12GB
系统: Windows / WDDM
PyTorch: 2.3.1+cu118
当前未安装: flash_attn、vLLM、bitsandbytes
```

<a id="part-5-section-1"></a>

### 背景

评估脚本本质上是在做 LLM 推理：

```text
读取验证集
-> 渲染 prompt
-> model.generate(...)
-> 从输出中提取 #### 后的答案
-> 计算 exact match / format / reward
```

其中最耗时的是 `model.generate(...)`。规则打分和 JSONL/summary 写入耗时很小。

以 `qwen3_0d6b_grpo_v4/checkpoint-69` 的 100 条评估为例：

```text
max_new_tokens: 256
样本数: 100
总耗时: 570.38 秒
平均: 5.7 秒/题
总输出 token: 12254
平均输出速度: 约 21.5 output token/s
```

原始实现是逐条推理：

```text
第 1 条 prompt -> generate
第 2 条 prompt -> generate
...
第 100 条 prompt -> generate
```

对于 RTX 4070 跑 0.6B 模型，这种 batch size = 1 的方式无法充分利用 GPU。

<a id="part-5-section-2"></a>

### 已实现的优化

当前已在 `post_training_framework/src/ptf/generation.py` 中实现批量推理：

```text
一批 prompt -> tokenizer padding -> 一次 model.generate -> 逐条解码和打分
```

新增参数：

```text
--eval-batch-size
```

已支持这些入口：

```text
post_training_framework/scripts/run_grpo_eval.py
post_training_framework/scripts/run_sft_eval.py
post_training_framework/scripts/run_base_eval.py
```

默认值仍为 `1`，因此不传参数时保持旧行为。

<a id="part-5-section-3"></a>

### 推荐命令

#### GRPO checkpoint-69 评估

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_grpo_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --model-dir models\grpo\qwen3_0d6b_grpo_v4\checkpoint-69 `
  --output-dir eval_results\grpo_model\qwen3_0d6b_grpo_v4_checkpoint69_eval100_len256_bs8 `
  --max-new-tokens 256 `
  --max-items 100 `
  --eval-batch-size 8 `
  --run-name qwen3_0d6b_grpo_v4_checkpoint69_eval100_len256_bs8 `
  --set dataset.eval_file=datasets/gsm8k_grpo/eval_100.parquet
```

#### SFT 评估

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --adapter-dir models\sft\qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2 `
  --max-new-tokens 256 `
  --max-items 100 `
  --eval-batch-size 8 `
  --run-name qwen3_0d6b_sft_eval100_len256_bs8 `
  --set dataset.eval_file=datasets\gsm8k_grpo\eval_100.parquet
```

<a id="part-5-section-4"></a>

### batch size 怎么选

当前 0.6B 模型、fp16、RTX 4070 12GB，建议：

| eval_batch_size | 建议 |
|---:|---|
| 1 | 最稳，速度慢，用于复现旧结果 |
| 4 | 保守加速，显存压力低 |
| 8 | 推荐起点，适合当前 0.6B + 256 token 评估 |
| 16 | 可以尝试，若 OOM 就退回 8 |
| 32 | 不建议作为起点，长 prompt 或长输出时容易 OOM |

推荐流程：

```text
先跑 --eval-batch-size 8
如果显存稳定，再试 16
如果出现 CUDA out of memory，退回 8 或 4
```

<a id="part-5-section-5"></a>

### 对准确率的影响

批量推理不会改变 reward、EM、format 的计算逻辑。

但需要注意：decoder-only 模型批量生成必须使用左 padding，否则不同长度 prompt 可能影响生成位置。当前实现已经在批量 tokenization 时临时设置：

```python
tokenizer.padding_side = "left"
```

生成仍然使用贪心解码：

```text
do_sample=False
```

因此同一个模型、同一批数据、同样的 `max_new_tokens` 下，batch 评估应当和单条评估保持一致或只有极小的数值差异。

<a id="part-5-section-6"></a>

### 其他加速方案评估

#### vLLM

vLLM 对批量推理和多请求调度很强，但 Windows 原生支持不如 Linux/WSL2 顺畅。

当前不建议优先投入，因为你的模型只有 0.6B，先做 HuggingFace batched generate 的收益更直接。

适合后续使用的场景：

```text
评估样本数扩大到 1000+
需要 oracle@8 / oracle@16 多采样
迁移到 WSL2 或 Linux 环境
```

#### FlashAttention 2

RTX 4070 支持 FlashAttention 2，但当前环境没有安装 `flash_attn`。Windows 安装可能比较折腾。

对 0.6B、256 token 的评估来说，收益通常不如 batch generate 直接。

#### 4bit/8bit 量化

当前没有安装 `bitsandbytes`。而 0.6B fp16 在 12GB 显存下并不缺显存，量化主要节省显存，不一定明显加速。

如果后续跑 1.7B 或更大模型，再考虑量化。

#### 减小 max_new_tokens

这能直接加速，但会改变评估条件。当前实验要求固定 `max_new_tokens=256`，因此不建议为了速度调整它。

#### 自定义停止条件

理论上可以在模型生成 `#### 数字` 后提前停止，减少无效输出。

但这会改变评估协议，可能掩盖模型是否能自然停止的问题。当前阶段不建议启用。

<a id="part-5-section-7"></a>

### 本阶段建议

当前最合理的加速路线：

```text
保持 max_new_tokens=256
保持同一个 eval_100.parquet
先用 --eval-batch-size 8
需要更快时试 --eval-batch-size 16
所有正式结果在 summary 中记录 eval_batch_size
```

这样既能加速，又不会把“模型效果变化”和“评估协议变化”混在一起。

<a id="part-6"></a>

## 第六部分：历史案例——GRPO v5 早停与诊断门禁

> 本部分作用：用早期失败案例理解组内诊断和小数据过拟合门禁为何必要。

本部分目录：

- [1. 核心结论](#part-6-section-1)
- [2. 本轮训练现象](#part-6-section-2)
- [3. 指标解读](#part-6-section-3)
- [4. oracle@8 高但 greedy 不升的含义](#part-6-section-4)
- [5. 最可能的问题](#part-6-section-5)
- [6. 下一步排查方案](#part-6-section-6)
- [7. 推荐下一轮实验](#part-6-section-7)
- [8. 诊断工具的定位: 组内 rollout + 小数据过拟合](#part-6-section-8)
- [9. 已执行的排查结果](#part-6-section-9)
- [10. 下一步建议](#part-6-section-10)

> 实验对象: `models/grpo/qwen3_0d6b_grpo_v5_rollout8_len256_lr2e-6_eval100`
> 固定验证集: `datasets/gsm8k_grpo/eval_100.parquet`
> 生成长度: `max_response_length=256`
> 训练设置: `rollout_n=8`, `train_batch_size=4`, `rollout_batch_size=4`, `learning_rate=2e-6`

<a id="part-6-section-1"></a>

### 1. 核心结论

这轮 GRPO v5 没有把 SFT 模型的 `oracle@8` 潜力转化为稳定的 `greedy@1` 能力。

早停不是主要问题本身，而是症状: 固定验证集上的 `val_exact_match` 长时间没有趋势性提升，说明当前 GRPO 配置、reward 信号或更新强度没有有效改变模型的单次输出分布。

优先排查顺序:

```text
组内 rollout 是否有有效比较信号
  -> reward 是否真正区分正确/错误轨迹
  -> advantage 是否非零且有足够方差
  -> actor 更新是否太弱
  -> greedy 评估是否和 oracle 诊断口径一致
```

<a id="part-6-section-2"></a>

### 2. 本轮训练现象

最终日志:

```text
最终 step: 119
早停原因: val_em 连续 100 步未改善
最佳 val_em: 0.490 at step 19
最终 val_em: 0.450 at step 119
最终 checkpoint: checkpoint-119
```

去重后的指标统计:

| 指标 | 范围 | 均值 | 解释 |
|---|---:|---:|---|
| `train reward_mean` | 0.3156 - 1.2062 | 0.7712 | 单步训练 reward 起伏大 |
| `train reward_std` | 0.2915 - 0.6009 | 0.4859 | rollout 奖励有波动，但不等于有效学习 |
| `val_reward_mean` | 0.684 - 0.734 | 0.7065 | 验证 reward 基本不变 |
| `val_exact_match` | 0.44 - 0.49 | 0.4623 | 验证正确率没有趋势性提升 |
| `val_format_rate` | 0.91 - 0.93 | 0.9215 | 格式已经由 SFT 基本学好 |
| `approx_kl` | 0.000485 - 0.002720 | 0.001157 | 每步策略变化很小 |
| `clip_frac` | 0.0027 - 0.0185 | 0.00946 | PPO clip 比例很低，不是更新过猛 |

验证集曲线:

| step | val_reward_mean | val_exact_match | val_format_rate |
|---:|---:|---:|---:|
| -1 | 0.710 | 0.47 | 0.92 |
| 9 | 0.725 | 0.48 | 0.93 |
| 19 | 0.734 | 0.49 | 0.92 |
| 29 | 0.734 | 0.49 | 0.92 |
| 59 | 0.684 | 0.44 | 0.92 |
| 89 | 0.689 | 0.44 | 0.93 |
| 109 | 0.699 | 0.46 | 0.91 |
| 119 | 0.694 | 0.45 | 0.92 |

结论: `val_exact_match` 只是在 0.44-0.49 之间摆动，没有持续上升。

<a id="part-6-section-3"></a>

### 3. 指标解读

#### 3.1 reward_mean 起伏大

每步只有 `train_batch_size=4` 个 prompt，每题生成 `rollout_n=8` 条回答，总共 32 条 rollout。GSM8K rule reward 又接近离散奖励，正确/错误会让单步均值明显跳动。

所以 `reward_mean` 单步起伏大不一定代表训练异常。更重要的是:

```text
验证集 reward 是否上升
验证集 exact match 是否上升
组内 reward 是否能产生有效 advantage
```

本轮 `val_reward_mean` 基本不变，因此训练 reward 的波动没有转化为泛化收益。

#### 3.2 val_exact_match 不提升

`val_exact_match` 表示固定验证集上单次输出的答案正确率。

本轮从初始 0.47 到最好 0.49，之后回落到 0.45。这说明 GRPO 没有稳定提升 greedy/single-output 能力。

#### 3.3 val_format_rate 基本不变

`val_format_rate` 长期在 0.91-0.93。说明 SFT 已经基本学会 `#### final_answer` 格式。

因此这轮 GRPO 的主要瓶颈不是格式，而是数学答案正确性。

#### 3.4 approx_kl 很小

`approx_kl` 估计当前 actor 相比产生这些 rollout 时的 old actor 改变了多少。

```text
ratio = exp(curr_log_prob - old_log_prob)
approx_kl = mean((ratio - 1) - log(ratio))
```

如果 `ratio` 接近 1，说明当前模型对这些 token 的概率几乎没有变化，`approx_kl` 接近 0。

本轮 `approx_kl` 平均约 0.00116，说明每步策略更新很小。结合验证 EM 不涨，更像是有效更新不足，而不是策略发散。

#### 3.5 kl_loss 很小

`kl_loss` 约束当前 actor 不要偏离 reference。当前代码里的 reference 是 `base + SFT adapter` 合并后的冻结模型。

`kl_loss` 小表示 GRPO actor 没有明显偏离 SFT reference。这不是坏事，但如果 EM 不涨，说明模型也没有被有效推到更好的分布。

#### 3.6 clip_frac 绝对值很小

`clip_frac` 表示 PPO ratio 超出 `[1 - clip_ratio, 1 + clip_ratio]` 的比例。

本轮范围约 0.27%-1.85%，平均不到 1%。真正需要警惕的通常是 0.2、0.3 这类量级。

因此当前不是 PPO 更新过猛，而更像更新太保守或 advantage 信号不足。

<a id="part-6-section-4"></a>

### 4. oracle@8 高但 greedy 不升的含义

`SFT oracle@8 ≈ 0.80` 的含义是:

```text
同一道题采样 8 条回答
只要其中有 1 条 exact match
这道题就算 oracle 命中
```

它说明 SFT 模型的采样空间里存在不少正确轨迹。

但 `greedy@1 ≈ 0.46-0.49` 表示单次稳定输出仍然只有约一半正确。

GRPO 要完成的是:

```text
把“偶尔能采到的正确轨迹”
推成“更高概率、更稳定的单次输出”
```

本轮没有做到，说明正确轨迹存在，但当前 reward/advantage/update 没有有效提高这些轨迹的概率。

<a id="part-6-section-5"></a>

### 5. 最可能的问题

#### 5.1 组内有效信号不足

GRPO 依赖同一 prompt 的多条 rollout 做相对比较:

```text
advantage = (reward - group_mean_reward) / group_reward_std
```

如果某题 8 条全错、全对，或者 reward 几乎一样，那么 `group_reward_std` 很小，advantage 接近 0，这道题对训练几乎没有贡献。

最有价值的是混合组: 同一题 8 条回答里既有正确答案，也有错误答案。

#### 5.2 reward 粒度可能太粗

当前 rule reward 给格式正确、单一 final answer 等行为也会加分。SFT 的格式率已经很高，因此许多错误答案也能拿到一定正 reward。

如果组内多数错误答案都是“格式正确但答案错”，reward 可能区分度不足，模型学到的主要仍是格式或输出形态，而不是数学正确性。

#### 5.3 更新幅度可能太弱

`learning_rate=2e-6`，LoRA 参数量较小，`approx_kl≈0.001`，`clip_frac≈0.01`。这些都说明 actor 改动很小。

如果 rollout/reward 本身已经信号稀疏，过小的更新很难改变 greedy 输出排序。

#### 5.4 训练口径和验证口径不同

训练阶段使用采样 rollout，验证阶段看固定验证集单次输出。采样能找到正确答案，不代表 greedy 会选中正确答案。

必须同时看 `oracle@8`、`sample exact rate`、`greedy@1`、每题 8 条里的正确数量分布。

<a id="part-6-section-6"></a>

### 6. 下一步排查方案

#### Step 1: 增加组内 rollout 诊断日志

建议在 GRPO 训练中记录:

| 诊断项 | 含义 | 判断 |
|---|---|---|
| `effective_group_rate` | `group_reward_std > 0` 的题占比 | 低则无效组太多 |
| `mixed_group_rate` | 同一题 8 条中既有正确又有错误的占比 | 越高越适合 GRPO |
| `all_wrong_group_rate` | 8 条全错的题占比 | 高则采样空间不足或题太难 |
| `all_correct_group_rate` | 8 条全对的题占比 | 高则这些题已掌握但训练信号少 |
| `correct_count_hist` | 每题 0/8 到 8/8 的分布 | 判断 oracle 潜力来源 |
| `advantage_std` | advantage 标准差 | 太小说明梯度信号弱 |
| `zero_advantage_rate` | advantage 近似 0 的 rollout 占比 | 高则大量 rollout 白学 |
| `reward_component_mean` | answer/format/single_final/repeat/overlong 分项均值 | 判断 reward 是否被格式项主导 |

目标: 判断当前 GRPO 是否真的有可学的组内偏好信号。

#### Step 2: 对 SFT 与 GRPO checkpoint-119 做 oracle 审计

在同一 `eval_100` 上比较:

```text
SFT greedy@1 / oracle@8 / sample exact rate
GRPO checkpoint-119 greedy@1 / oracle@8 / sample exact rate
```

判断:

| 现象 | 解释 |
|---|---|
| oracle@8 不变，sample exact 不变 | GRPO 没提高正确轨迹概率 |
| oracle@8 不变，sample exact 上升 | 采样分布改善，但 greedy 还没转化 |
| oracle@8 下降 | GRPO 损害了采样空间 |
| greedy 上升但 oracle 不变 | 正确轨迹排序变好了 |

#### Step 3: 做小数据过拟合测试

选 32-64 道训练题，只在这些题上跑短程 GRPO。

目标不是泛化，而是验证训练链路能否把固定题集的 greedy EM 推上去。

| 结果 | 解释 |
|---|---|
| 小数据 greedy EM 明显上升 | 训练链路基本有效 |
| 小数据仍不升 | 优先怀疑 reward、advantage、loss 实现或更新强度 |

#### Step 4: 根据诊断结果调参

如果 `mixed_group_rate` 低: 提高 `rollout_n` 到 16，调整 `temperature/top_p`，或选择 SFT oracle 能覆盖但 greedy 错的题做训练子集。

如果 reward 被格式项主导: 降低 `format_bonus` 和 `single_final_bonus`，提高答案正确项相对权重，并记录 reward component。

如果 advantage 正常但 `approx_kl` 仍很小: `learning_rate` 从 `2e-6` 试到 `5e-6`，`ppo_epoch` 从 1 试到 2，同时观察 `approx_kl` 是否仍低于 0.02。

如果小数据也无法过拟合: 检查 old/current log prob、response mask、LoRA requires_grad、optimizer 参数范围、reward 与 exact_match 是否一致。

<a id="part-6-section-7"></a>

### 7. 推荐下一轮实验

```text
1. 给 train_grpo.py 增加组内 rollout 诊断指标
2. 用当前 v5 配置跑 10-20 step 诊断版训练
3. 保存每步 group 诊断 CSV
4. 对 checkpoint-119 跑 greedy@1 + oracle@8 + sample exact rate
5. 基于诊断结果决定改 reward、改 lr，还是改 rollout_n
6. 再做 32-64 题小数据过拟合测试
```

这套排查的目标是证明:

```text
rollout 里有正确轨迹
reward 能识别正确轨迹
advantage 能把正确轨迹变成正梯度
actor 参数确实发生足够变化
greedy 输出排序开始改善
```

只有这条链路通了，继续扩大训练步数才有意义。

<a id="part-6-section-8"></a>

### 8. 诊断工具的定位: 组内 rollout + 小数据过拟合

这两个诊断不是为了证明最终泛化能力，而是为了判断 GRPO 训练闭环有没有有效学习信号。

它们在排查链路中的位置不同:

```text
组内 rollout 诊断:
  看每一步训练数据里有没有可比较的好坏样本
  重点回答: 这一步有没有 reward/advantage 信号?

小数据过拟合诊断:
  看模型能不能在少量固定题上被 GRPO 明显推高
  重点回答: 有信号时, actor/LoRA/optimizer/loss 能不能真的学进去?
```

#### 8.1 组内 rollout 诊断的作用

GRPO 不像 SFT 那样每条样本都有标准监督标签。它依赖同一 prompt 的多条 rollout 做相对比较:

```text
同一道题 -> rollout_n 条回答 -> rule reward 打分
        -> 组内均值/方差 -> advantage
        -> policy loss 更新 actor
```

因此, 只有同一题内部出现“好答案”和“坏答案”的差异时, GRPO 才有清楚的训练方向。

组内 rollout 诊断主要检查:

| 指标 | 作用 |
|---|---|
| `mixed_group_rate` | 判断同一题多条回答里是否既有正确也有错误 |
| `effective_group_rate` | 判断 reward 是否有差异, 能不能产生非零 advantage |
| `all_wrong_group_rate` | 判断模型采样空间里是否根本采不到正确答案 |
| `all_correct_group_rate` | 判断题目是否已经掌握, 但对 GRPO 贡献较少 |
| `zero_advantage_rate` | 判断 rollout 是否大量没有梯度信号 |
| `reward_answer_mean / reward_format_mean` | 判断 reward 是否主要在奖励答案正确, 还是被格式项主导 |

判断原则:

```text
mixed_group_rate 高 + effective_group_rate 高:
  说明 GRPO 有比较信号, 可以继续看更新强度和泛化。

all_wrong_group_rate 高:
  说明模型对这些题采样不到正确轨迹, GRPO 很难凭空学会。

all_correct_group_rate 高:
  说明这些题已经会了, 继续训练提供的相对偏好信号少。

reward_format_mean 占比过高:
  说明 reward 可能过多奖励格式, 对数学正确性的推动不足。
```

#### 8.2 小数据过拟合诊断集的作用

小数据过拟合诊断集不是普通验证集, 更接近训练系统的单元测试。

它通常选 32-64 道固定题, 最有价值的是:

```text
SFT greedy@1 答错
但 SFT oracle@8 能答对
```

这类题说明模型采样空间里已经存在正确轨迹, 只是 greedy 输出没有稳定选中。GRPO 理论上应该能通过 reward 把正确轨迹概率推高。

这个诊断允许:

```text
训练集 = 评估集
训练步数很短, 例如 20-50 step
不用于判断泛化, 只用于判断能否学进去
```

判断原则:

| 小数据结果 | 结论 | 下一步 |
|---|---|---|
| greedy EM 明显上升 | 基础训练链路能学 | 再看大集有效样本密度、reward 权重和超参 |
| reward 上升但 greedy EM 不升 | reward 可能没对齐最终正确率 | 查 reward 组成、答案抽取、长度/格式偏置 |
| reward/EM 都不升 | 训练链路或更新强度有问题 | 查 LoRA requires_grad、optimizer、lr、ppo_epochs、advantage |
| mixed group 很少 | 数据不适合当前 GRPO 设置 | 换题、提高 rollout_n 或调整采样温度 |

#### 8.3 为什么它是正式训练前的门控

正式 GRPO 训练成本更高, 而且早停后很难直接判断问题来自哪里。小数据过拟合诊断能把问题拆开:

```text
如果 32-64 题都无法过拟合:
  先不要跑长训练。
  优先修 reward、answer extraction、LoRA 更新、优化器状态、loss 和超参。

如果 32-64 题能过拟合, 但大集不涨:
  说明基础链路不是完全坏的。
  问题更可能在训练数据选择、有效组比例、reward 权重、更新强度或验证口径。
```

因此, 推荐把它作为 GRPO 正式训练前的实验门控:

```text
1. 先跑 SFT greedy@1 / oracle@8, 找到“会但不稳定”的题。
2. 构造 32-64 题 overfit probe。
3. 跑 20-50 step GRPO。
4. 检查 greedy EM、group_diagnostics、reward component、approx_kl、clip_frac。
5. 小数据能明显提升后, 再扩大到正式训练集。
```

当前实验中, 64 题 overfit probe 已经从 `0.46875` 提升到 `0.625`。这说明基础 GRPO 链路可以工作。因此 v5 大集训练不涨时, 不应优先判断为“GRPO 完全无效”, 而应继续排查有效样本密度、reward 设计和更新强度。

<a id="part-6-section-9"></a>

### 9. 已执行的排查结果

#### 9.1 checkpoint-119 与 SFT 的同口径评估

固定评估集仍是 `datasets/gsm8k_grpo/eval_100.parquet`，生成长度仍是 `256`。

| 模型 | greedy@1 EM | oracle@8 | sample exact rate | format/sample |
|---|---:|---:|---:|---:|
| SFT eosfix2 | 0.46 | 0.80 | 0.38375 | 0.95375 |
| GRPO v5 checkpoint-119 | 0.45 | 0.81 | 0.40750 | 0.95625 |

产物:

```text
eval_results/sft_model/0d6b_eosfix2_greedy_eval100_len256_bs8
eval_results/sft_model/0d6b_eosfix2_oracle8_eval100_len256_temp0d7_bs8
eval_results/grpo_model/qwen3_0d6b_grpo_v5_checkpoint119_greedy_eval100_len256_bs8
eval_results/grpo_model/qwen3_0d6b_grpo_v5_checkpoint119_oracle8_eval100_len256_temp0d7_bs8
```

解释:

```text
GRPO v5 让采样分布略有改善:
  oracle@8: 0.80 -> 0.81
  sample exact rate: 0.38375 -> 0.40750

但 greedy@1 没有改善:
  0.46 -> 0.45
```

这说明 v5 并非完全没有改变模型分布，而是没有把采样概率变化转成单次最高概率输出。

#### 9.2 组内 rollout 诊断指标已加入

新增输出:

```text
models/grpo/<run>/plots/group_diagnostics.csv
```

新增日志示例:

```text
[group diag step 0] effective=... mixed=... all_wrong=... all_correct=...
rollout_em=... adv_std=... zero_adv=... hist=...
```

关键字段:

```text
effective_group_rate: reward 有差异、能产生非零 advantage 的组比例
mixed_group_rate: 同一题 8 条回答里既有正确也有错误的组比例
all_wrong_group_rate: 8 条全错的组比例
all_correct_group_rate: 8 条全对的组比例
rollout_exact_rate: 训练 rollout 样本级正确率
advantage_std: advantage 方差
zero_advantage_rate: advantage 接近 0 的 rollout 比例
reward_answer_mean / reward_format_mean: reward 分项均值
```

smoke 测试已通过:

```text
models/grpo/smoke_group_diag_test/plots/group_diagnostics.csv
```

#### 9.3 小数据过拟合诊断

构造了 64 条诊断集:

```text
datasets/gsm8k_grpo/overfit_probe_64_oracle_hit_greedy_miss.parquet
datasets/gsm8k_grpo/overfit_probe_64_oracle_hit_greedy_miss_meta.json
```

其中 35 题来自 `SFT greedy 错但 SFT oracle@8 能命中`，用于测试 GRPO 能否把已有正确轨迹推成 greedy 输出。

训练设置:

```text
output_dir=models/grpo/overfit_probe64_rollout8_len256_lr5e-6_ppo2_diag
train/eval file=overfit_probe_64_oracle_hit_greedy_miss.parquet
rollout_n=8
max_response_length=256
learning_rate=5e-6
ppo_epochs=2
total_training_steps=20
```

验证结果:

| step | val_reward_mean | val_exact_match | val_format_rate |
|---:|---:|---:|---:|
| -1 | 0.7625 | 0.4844 | 0.9688 |
| 4 | 0.7844 | 0.5000 | 0.9688 |
| 9 | 0.8000 | 0.5156 | 0.9688 |
| 14 | 0.8391 | 0.5469 | 0.9844 |
| 19 | 0.9172 | 0.6250 | 0.9844 |

独立评估确认:

```text
eval_results/grpo_model/overfit_probe64_checkpoint19_greedy_len256_bs8
exact_match=0.625
format_rate=0.984375
repeat_like_rate=0.0
```

组内诊断统计:

| 指标 | 均值 |
|---|---:|
| effective_group_rate | 0.9375 |
| mixed_group_rate | 0.9250 |
| all_wrong_group_rate | 0.0375 |
| all_correct_group_rate | 0.0375 |
| rollout_exact_rate | 0.4250 |
| advantage_std | 0.9665 |
| zero_advantage_rate | 0.0703 |
| reward_answer_mean | 0.4242 |
| reward_format_mean | 0.1900 |

结论:

```text
训练链路本身能学:
  SFT greedy on probe64: 0.46875
  GRPO overfit checkpoint-19: 0.625

因此 v5 大集训练失败，不像是 reward/advantage/loss 完全无效。
更可能是大集训练中的有效样本密度不足、训练配置偏保守、数据选择不聚焦，或 100 条验证集上的 greedy 提升需要更强的分布重排。
```

<a id="part-6-section-10"></a>

### 10. 下一步建议

不要继续直接复用 v5 配置跑完整训练。建议下一轮做一个“诊断增强版 v6”:

```text
1. 保留 group_diagnostics.csv。
2. 训练集优先采样 SFT greedy 错但 oracle@8 能命中的题。
3. 使用 lr=5e-6、ppo_epochs=2 作为短程实验配置。
4. 先跑 50-80 step，而不是直接 500 step。
5. 每 10 step 同时看:
   - val_exact_match
   - oracle@8 / sample exact rate
   - effective_group_rate
   - mixed_group_rate
   - approx_kl
   - clip_frac
```

如果 v6 的 `group_diagnostics` 显示有效组很多，但 `approx_kl` 仍长期接近 0 且 greedy 不涨，再继续提高更新强度或检查 PPO loss 的 token/sequence 聚合方式。

<a id="maintenance"></a>

## 文档维护规则

- 本文是 GRPO 训练流程、工程优化和实验复盘的默认主入口。
- 同一训练链上的新实验先判断应补到哪个既有部分，并同步更新该部分目录。
- 不设置固定行数上限，不以文档变长作为拆分理由。
- 只有受众、事实来源、生命周期或维护责任明显独立时才新建专题文档。
- 原始 CSV、JSONL、模型状态和大型评估报告继续保留在 `models/`、`logs/`、`eval_results/`，主笔记只记录可复用结论和精确证据入口。
