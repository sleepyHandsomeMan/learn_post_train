# 当前 GRPO 实现与标准算法的差异清单

> 创建日期: 2026-07-23
> 适用范围: `post_training_framework/src/ptf/train_grpo.py`
> 目标: 集中记录当前自研 GRPO trainer 与标准/主流实现（DeepSeekMath GRPO、TRL GRPO、verl）之间的已知差异，附代码位置、影响机制和实测证据，供后续排查和整改时追溯。

## 1. 总体结论

核心算法（group-relative advantage、无 critic、KL 独立加入 loss、PPO-clip 更新）忠实复现了 GRPO 论文设计，工程稳定性（KL guard、signal guard、reward hacking 检测、early stopping、显存/RNG/断点续训）超出"最小实现"范畴。以下差异点是在此基础上识别的具体技术缺口，不代表训练器整体不可用。

## 2. 差异点清单

### 2.1 Loss 聚合方式存在长度偏置（token-mean，非 seq-mean-token-mean）

**代码位置**: [train_grpo.py:1771](../post_training_framework/src/ptf/train_grpo.py#L1771)

```python
policy_loss = (policy_loss_per_token * resp_mask_shifted).sum() / valid_tokens
```

**问题**: `valid_tokens` 是 mini-batch 内所有回答的 response token 总数，等价于对全部 token 做全局平均。回答越长，占的 token 越多，对梯度的影响力就越大——与"每条回答等权重"的直觉相悖。

**标准做法（seq-mean-token-mean）**: 先在每条回答内部对 token 求平均，再对回答间求平均，使每条回答无论长短都占相同权重。verl 等框架通过 `loss_agg_mode` 参数支持切换，本项目目前只有 token-mean 一种且不可配置。

**数值示例**（4 条回答，token 数与 advantage 见下表）：

| 回答 | token 数 | advantage | token-mean 权重 | seq-mean 权重 |
|---|---:|---:|---:|---:|
| A | 30 | +1.2 | 6.5% | 25% |
| B | 50 | +0.3 | 10.9% | 25% |
| C | 180 | -0.5 | 39.1% | 25% |
| D | 200 | -0.8 | 43.5% | 25% |

token-mean 下负 advantage 回答的总权重达 82.6%，是正 advantage 回答（17.4%）的近 5 倍；seq-mean 下则是均衡的 50/50。

**缓解因素**（使影响不一定立即显著）：
- `max_response_length=256` 限制了长度倍数上限。
- reward 中已有 `overlong_penalty` / `truncated_response_penalty`，从奖励侧部分抵消了 loss 侧的长度偏置。
- GSM8K 数学题回答长度变化幅度本身有限（不像代码生成、长文写作）。

**排查建议**: 如果后续观察到平均回答长度持续上涨但 EM 不涨、或短且正确的回答比例下降，优先检查此处，可考虑把 `_train_mini_batch` 改为先按序列内部平均再按序列间平均。

### 2.2 无 rollout 动态过滤（dynamic sampling）

标准/后续工作（如 DAPO）会在组内奖励方差为 0（全对或全错）时丢弃该组重新采样，避免浪费无梯度信号的样本。本项目只**监控**（`effective_group_rate`、`signal_guard`），在信号持续不足时**停止训练**，但不会在单步内主动重采/过滤补齐——无效组样本仍进入训练（advantage=0，梯度贡献为 0，但浪费了该次 rollout 的算力）。

**代码位置**: `_compute_group_diagnostics`、`_check_signal_guard`（[train_grpo.py](../post_training_framework/src/ptf/train_grpo.py)）。

### 2.3 全参数训练 vs LoRA

标准/主流 GRPO 实现（verl、DeepSeek 官方）通常做全参数微调；本项目受 RTX 4070 显存限制，只训练 LoRA adapter（`lora_r=16`）。这是资源约束下的合理取舍，但策略容量和更新幅度都受限，收敛速度和效果上限不能直接与全参数实现类比。

### 2.4 Rollout 引擎为 HF `generate()`，非 vLLM/SGLang

标准生产级 GRPO pipeline（verl）使用专门推理引擎做 rollout 以获得数量级吞吐提升。本项目 rollout 直接调用 `actor.generate()`（[train_grpo.py:_generate_responses](../post_training_framework/src/ptf/train_grpo.py)），正确性没有问题，但吞吐明显偏低（文档 `40_grpo_rule_reward_implementation_cn.md` 也提到 `hf` backend "依赖少但慢"）。

### 2.5 单一奖励来源（无 Reward Model / PRM）

目前只有 GSM8K 规则奖励（答案/格式/复读/长度/截断），见 [reward.py](../post_training_framework/src/ptf/reward.py)，没有 reward model 或过程奖励（PRM）。这是当前阶段的主动取舍（`40_grpo_rule_reward_implementation_cn.md` 明确"暂时不要一开始就做 Reward Model"），不算缺陷。

### 2.6 Reference model 全程不更新（无 reference reset）

整个训练过程 reference 恒定为 SFT checkpoint，没有做迭代式 RL 中常见的"定期把 policy 同步给 reference"。对当前单阶段 GRPO 是标准做法，但以后做多轮迭代 RL 时需要注意。

## 3. formal500 实证：长度上涨 + 正确率下降的真实案例

针对 2.1 节的长度偏置假说，在 v7 系列已完成的正式训练 `qwen3_0d6b_grpo_v7_full5759_rewardv2_rollout8_len256_lr5e-6_ppo2_eval100` 上做了核对，用于回答"当前训练是否已经出现长度上涨、EM 不涨甚至下降"的问题。

### 3.1 验证集（greedy，`val_metrics.csv`）

| step | greedy EM | 长度(token) | format |
|---:|---:|---:|---:|
| 169（峰值） | 0.67 | 141.0 | 0.99 |
| 179 | 0.62 | 137.3 | 0.96 |
| 189 | 0.63 | 137.7 | 0.97 |
| 199 | 0.57 | 143.4 | 0.96 |
| 209 | 0.54 | 145.1 | 0.97 |

峰值之后：长度 141→145（+3%），EM 0.67→0.54（相对降 19%）——长度上涨、EM 下跌同时发生。

### 3.2 训练 rollout（采样解码，更敏感，`train_metrics.csv` + `group_diagnostics.csv`）

| 阶段 | 平均长度 | rollout 正确率 | rollout 格式率 |
|---|---:|---:|---:|
| step < 169 | 130.3 | 0.565 | 0.970 |
| step 169–209 | 136.1 | 0.579 | 0.955 |
| step 210–218（KL guard 停止前） | **161.0** | **0.410** | **0.854** |

最后 9 步长度骤增 18%（136→161），同期正确率暴跌 29%（0.58→0.41），格式率同步下滑。这是目前证据中最典型的"变长且变差"窗口，紧接着 signal guard 触发停止。

### 3.3 异常样本佐证

[rollout_anomalies.jsonl](../models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/diagnostics/rollout_anomalies.jsonl) 留档的 18 条异常 rollout 中，多数呈现 `response_token_count=256`（打满上限）、`terminated_by_eos=false`、`format_ok=false`、`exact_match=false` 的特征——模型推导中途开始重复自我修正、数字打架，直到 token 上限也未给出 `####` 答案。这类"又长又错"的回答若混入同一组，按 token-mean 聚合会获得远超短回答的梯度权重。

### 3.4 重要澄清：项目自身根因排查指向 KL 失控，而非单纯 loss 长度偏置

`docs/45_grpo_v7_preflight_remediation_cn.md` 第 12.2、12.3 节及第二部分对同一次崩溃做了控制变量排查，结论是：

1. **主因是 actor-reference KL 失控**：`kl_loss` 从 step169 的 0.0386 升至 step218 的 0.2335，与 format/长度的相关系数约 -0.77，比长度偏置假说更直接地解释了崩溃。
2. 从 checkpoint-169 做的受控复现实验（`repair_probe_from169`）中，长度并未持续上涨：

   | 阶段 | 平均长度 | rollout 正确率 |
   |---|---:|---:|
   | step 170–190 | 128.9 | 0.582 |
   | step 200–211 | 122.6 | 0.467 |

   长度基本持平甚至略降，但正确率仍下降——说明单纯"长度膨胀"不是这条探针里退化的驱动因素，更支持 KL 漂移是主因。
3. 旧版 formal500 缺少 token 上限命中率、EOS 率、截断惩罚和固定 sample@n 格式检查（`45_` 文档 12.2 节"旧框架的直接缺陷"），因此"变长又变错"更可能是 **奖励/监控未约束住 hit-max 长回答** 与 **KL 约束太弱太慢** 共同作用的结果。

**当前结论**：现象层面（长度上涨、正确率下跌、格式退化同时出现）证据充分；但归因层面，现有 CSV 指标（只有按 mini-batch 聚合的 `policy_loss` / `response_len_mean`，没有按回答长度分桶的梯度贡献）**无法把 loss token-mean 聚合的独立贡献从 KL 失控中剥离出来**。两者可能是相互强化的组合因素：KL 失控导致输出分布漂移变长，loss 长度偏置又放大了这些变长回答在同批次内的梯度权重。

## 4. 待办：验证 loss 长度偏置独立贡献的诊断缺口

若要单独验证 2.1 节的假说，需要新增诊断（当前 CSV 体系缺失）：

```text
按 mini-batch 内每条回答的 token 数分桶
→ 分别记录各分桶对 policy_loss 均值和梯度范数的实际贡献占比
→ 而不是只看聚合后的 policy_loss 均值
```

在此诊断补齐前，不应单独把 loss 聚合方式认定为 formal500 崩溃的独立主因，只能作为已识别的算法差异点持续跟踪。

## 5. 证据入口

- 核心训练器: `post_training_framework/src/ptf/train_grpo.py`
- 奖励函数: `post_training_framework/src/ptf/reward.py`
- formal500 训练曲线: `models/grpo/qwen3_0d6b_grpo_v7_full5759_rewardv2_rollout8_len256_lr5e-6_ppo2_eval100/plots/{train_metrics,val_metrics,group_diagnostics}.csv`
- checkpoint-169 整改探针曲线: `models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/plots/{train_metrics,val_metrics,group_diagnostics}.csv`
- 异常 rollout 留档: `models/grpo/qwen3_0d6b_grpo_v7_repair_probe_from169/diagnostics/rollout_anomalies.jsonl`
- KL 失控与格式退化根因排查: `docs/45_grpo_v7_preflight_remediation_cn.md` 第 12.2、12.3 节，第二部分
- GRPO 实现细节: `docs/40_grpo_rule_reward_implementation_cn.md`
- 指标含义与终止条件: `docs/41_grpo_metrics_stop_criteria_cn.md`
