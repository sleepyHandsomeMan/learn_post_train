# GRPO v7 R8 实验方案：先验噪声核验，再决定是否投入干净版对照实验

> 创建日期: 2026-07-24
> 定位: 本文是 `43_grpo_v7_experiment_timeline_cn.md` 第8节"下一轮建议"的具体化方案，回答"R8 具体怎么设计、先做什么、门槛是什么"。执行完成后，结果记录回 `43_` 时间线文档第7节之后（作为R8小节），并按需要更新本文档的"执行状态"。
> 前置阅读: `43_grpo_v7_experiment_timeline_cn.md`（R1-R7历史与R8方向由来）、`48_grpo_causal_control_experiment_plan_cn.md`（控制变量方法论）、`49_grpo_causal_experiment_operations_cn.md`（编排工具用法）。

## 0. 方案定位：为什么要分阶段，不直接开始训练

`43_` 文档第8.2节列出了4个候选方向，但代码调研（见第1节）发现两处会改变优先级的事实：

1. **验证 0.67 是否为噪声这件事，不需要任何新训练**——现有 `sample_eval.py` 已有 `wilson_interval()` 置信区间工具，而 `datasets/gsm8k_sft/test.parquet` 恰好是一个 1319 题、与训练集零重叠、且是 `eval_100` 超集的现成大验证集。这一步成本几乎为零，却直接决定后续所有训练实验是否值得投入。
2. **原计划的"B1干净版"其实建立在一个不准确的前提上**——B1（`train_batch_size=8, gradient_accumulation_steps=2`）经复核**已经**把 optimizer update 次数控制在与基线一致的4次/step；真正没被控制的是**总 rollout 生成量**（64条 vs 基线32条）。因此不是"重跑B1"，而是要设计一个连总rollout量也打平的新对照（本文称B2）。

因此 R8 分三个阶段，**后一阶段是否执行取决于前一阶段的结果**，不预先承诺全部跑完：

```text
Phase 0（零训练成本，必须先做）
  → 验证 checkpoint-169 的 EM=0.67 与 continuation 分支终点 EM=0.64 的差距是否落在评估噪声内

Phase 1（仅当 Phase 0 支持"差距可能真实"或"结论不确定需要更强信号"时执行）
  → B2：连总rollout量也控制住的干净版prompt多样性对照
  → I4：ppo_epochs=1 与 lr=3e-6（L1）的交互实验（此前从未测过的组合）

Phase 2（仅当 Phase 1 有明确正向信号时才开发，当前不承诺）
  → 无放回 epoch sampler（当前代码完全不存在，需要新开发）
```

## 1. 前置调研结论（决定本方案设计的关键事实）

| 调研项 | 结论 |
|---|---|
| bootstrap/置信区间工具 | `post_training_framework/src/ptf/sample_eval.py` 已实现 `wilson_interval()`（51-67行）和 `paired_prompt_bootstrap()`（95-148行），当前仅被 `run_grpo_sample_matrix.py` 使用，可直接复用，不需要新代码 |
| 更大 held-out 集 | `datasets/gsm8k_grpo/eval_100.parquet` 之外没有 eval_200/eval_500，但 `datasets/gsm8k_sft/test.parquet` 有完整 **1319题**，与 `datasets/gsm8k_grpo/train.parquet`（5759条训练集）**零重叠**，且 `eval_100` 的100题**全部是 test.parquet 的子集**。两者 schema 完全一致（都是单列 `messages`），可以直接用 `run_grpo_eval.py --eval-file datasets/gsm8k_sft/test.parquet` 评估，不需要任何格式转换 |
| B1 是否已控制计算量 | `gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json` 中 `b1_prompt8_accum2` 原始配置为 `{train_batch_size: 8, gradient_accumulation_steps: 2}`。实算：64条rollout ÷ mini_batch16=4个mini-batch/epoch ÷ accum2=2次update/epoch × ppo_epochs2 = **4次/step**，与基线（32条rollout÷16=2个mini-batch÷accum1=2次update/epoch×2epoch=4次/step）**完全一致**。B1 从未真正混淆 optimizer update 次数；它没控制的是**总rollout生成量**（64 vs 32），这是本轮要修正的设计缺陷 |
| 无放回 epoch sampler | `train_grpo.py::_sample_prompt_indices`（1597-1612行）是纯有放回随机采样，跨step独立重抽，代码库中不存在任何"一轮内不重复"的索引追踪机制。**当前完全不存在，需要新开发**，不是配置项 |
| ppo_epochs × learning_rate 独立可配置 | 两者都是 `GRPOConfig` 独立字段，`l1_lr3e6`（仅改lr）和 `p1_ppo_epoch1`（仅改ppo_epochs）已有先例，组合无需改代码 |
| 编排脚本新增variant成本 | `run_grpo_causal_experiments.py::build_trial_plan()` 是纯字典覆盖机制，新增variant只需在matrix JSON的`variants`数组里加一条`{id, tier, factor, description, changed_keys, overrides}`，不用改脚本代码 |
| 当前基线一致性 | `g7lk_base.json`（R7用）与 `gsm8k_qwen3_0d6b_grpo_v7_causal_base_from169.json`（R4用）除 `learning_rate`（0.000003 vs 0.000005，符合R6确认L1为新基线的调整）外，`rollout_n/train_batch_size/ppo_epochs/ppo_mini_batch_size/gradient_accumulation_steps/adaptive_kl_min_coef` 全部一致，是同一套基线的延续，R8可以直接衔接 |

## 2. R8-Phase 0：验证 checkpoint-169 峰值是否为评估噪声

### 2.1 待评估对象

| 对象 | 来源 | 理由 |
|---|---|---|
| `checkpoint-169` | `qwen3_0d6b_grpo_v7_full5759_rewardv2_rollout8_len256_lr5e-6_ppo2_eval100/checkpoint-169` | 峰值来源，greedy EM=0.67（100题） |
| L1 (seed42) final `checkpoint-49` | `grpo_v7_step169_causal_v1/l1_lr3e6/seed-42/checkpoint-49` | R5/R6确认的候选基线终点，100题EM=0.64 |
| L2 (seed42) final `checkpoint-49` | `g7lk/l2/seed-42/checkpoint-49` | R7最新终点，100题EM=0.64，与L1完全相等 |

三者都已存在，**不需要任何新训练**。

### 2.2 评估设置

```text
数据源: datasets/gsm8k_sft/test.parquet（1319题，messages格式，与训练集零重叠）
解码: greedy（do_sample=False），与现有 val_metrics.csv 口径一致
max_new_tokens: 256（与训练/历次评估口径一致）
eval-batch-size: 8（沿用45号文档第五部分batch size建议）
```

执行方式：对每个checkpoint跑一次 `run_grpo_eval.py --eval-file datasets/gsm8k_sft/test.parquet --max-items 1319`，输出JSONL后用 `sample_eval.py::wilson_interval()` 计算每个checkpoint在1319题上的95%置信区间。

### 2.3 判定规则（预注册，跑之前先写好，不能跑完再定标准）

设 `EM_169`、`EM_L1`、`EM_L2` 为三者在1319题上的greedy EM点估计，`CI_169`、`CI_L1`、`CI_L2` 为对应Wilson 95%区间：

| 情形 | 判定 | Phase 1 是否执行 |
|---|---|---|
| `CI_169` 与 `CI_L1`、`CI_L2` 有重叠 | 100题差距主要是噪声，checkpoint-169峰值不代表真实优势 | **执行**，但目标改为在扩大验证集上寻找真实提升，不再纠结0.67这个数字本身 |
| `CI_169` 高于 `CI_L1`、`CI_L2` 且不重叠 | continuation分支确实丢失了169拥有的某种能力，差距真实存在 | **执行**，但Phase 1的B2/I4新增一项"完成后必须在1319题上复核，不能只看100题"；同时建议追加一次独立诊断（比较169与L1终点在1319题上的错误类型分布是否系统性偏移，而非直接冲去调prompt/PPO变量） |
| `EM_169` 本身在1319题上也不到0.60（即100题子集本身对1319题整体不具代表性，评估口径偏差） | 100题子集选取有偏，历史所有基于100题的比较都需要打折看待 | 仍执行Phase 1，但**必须把1319题设为R8起新实验的主评估口径**，100题仅作训练中快速监控用 |

### 2.4 预期产出

```text
eval_results/grpo_model/r8_phase0_noise_check/
  checkpoint169_test1319_full.jsonl
  l1_seed42_final_test1319_full.jsonl
  l2_seed42_final_test1319_full.jsonl
  r8_phase0_wilson_summary.md   # 三者EM点估计+Wilson CI对照表，附判定结论
```

## 3. R8-Phase 1：干净版B2 + PPO×LR交互 I4（仅Phase 0支持时执行）

### 3.1 B2：连总rollout量也控制住的prompt多样性对照

**设计动机**：原B1把optimizer update次数控制住了，但总rollout量翻倍（64 vs 32），如果B1表现出差异，无法判断是"更多独立prompt带来更好的跨题梯度方向"还是单纯"这一步用了两倍的训练信号量"。B2改为同时压低`rollout_n`，让总rollout数量维持32不变：

```text
b2_prompt8_rollout4_matched:
  train_batch_size: 4 → 8       （独立prompt翻倍）
  rollout_n: 8 → 4               （每题rollout数减半，总量仍为 8×4=32）
  ppo_mini_batch_size: 16       （不变，2个mini-batch/epoch）
  gradient_accumulation_steps: 1 （不变）
  ppo_epochs: 2                  （不变）
  → optimizer update: 2×2=4次/step，与基线完全一致
  → 总rollout量: 32，与基线完全一致
  → 唯一变量: 独立prompt数(4→8) 与 组内rollout数(8→4) 的分配比例
```

**已知代价，需要监控**：`rollout_n`从8降到4会削弱组内reward均值/标准差估计的稳定性，`effective_group_rate`和`zero_advantage_rate`可能因此变差。这不是bug，是该设计的固有权衡，必须在结果解读时与"prompt多样性收益"分开看——如果B2的KL/EM变差，要先排除是不是`rollout_n=4`本身信号变弱导致，而不是直接归因于"prompt数增加没用"。

**基线**：与R7一致，使用L1（lr=3e-6）而非5e-6的C0作为对照组。

### 3.2 I4：PPO epoch=1 与 学习率3e-6 的交互

**设计动机**：R4中P1（仅降ppo_epochs）在Screen30被"暂停"（rollout能力尚可但KL仍加速，sample截顶率达到0.10边界）；L1（仅降学习率）单独通过。48号文档9.2节已预留"L1+ppo_epochs=1"交互实验位置，但整个v7系列从未实际执行。两个单变量各自只是部分改善，交互项可能同时压住"小样本方向被PPO epoch重复放大"和"单次更新步幅过大"这两条被怀疑共同致因的链路。

```text
i4_ppo1_lr3e6:
  ppo_epochs: 2 → 1
  learning_rate: 沿用L1的3e-6（已是当前基线，不用重复声明覆盖）
  → optimizer update: 2×1=2次/step（低于基线4次，属于本实验的预期变化，不需要额外控制）
```

### 3.3 执行流程

沿用现有 `run_grpo_causal_experiments.py` 的 `Gate10→Screen30→Confirm50` 三阶段和既有晋级判据（见R7 `g7lk/_orchestration/s30/decision.md` 中的门槛结构，直接复用同一套阈值）：

```powershell
# 1. 在 gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json 或新建 r8 专用 matrix 文件中
#    新增 b2_prompt8_rollout4_matched 和 i4_ppo1_lr3e6 两条 variant（仅编辑JSON，不改代码）

# 2. Gate10
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py run `
  --phase gate10 --variants b2_prompt8_rollout4_matched i4_ppo1_lr3e6 --seeds 42 --execute

# 3. Screen30（仅Gate10工程门禁通过的分支）
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py run `
  --phase screen30 --tiers single --seeds 42 --execute

# 4. 汇总，判定是否晋级Confirm50
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py summarize `
  --tiers single --seeds 42
```

### 3.4 晋级门槛（预注册，沿用R7同一套阈值，不因为是新变量就放宽）

| 门槛 | 阈值 | 说明 |
|---|---:|---|
| 尾10步reference KL均值差（相对同seed L1） | `<= +0.01` | 不能比当前基线更不稳定 |
| 尾10步reference KL斜率差 | `<= +0.0005` | 不能持续加速漂移 |
| greedy retention差 | `>= -0.02` | 能力不能明显下降（K3就是在此门槛未通过） |
| rollout EM尾10步差 | `>= -0.03` | 训练分布上的正确率不能明显下降 |
| sample格式率 | `>=0.90` 且 `>=` 对照组 | 沿用三seed复验暴露的关键短板指标 |
| sample截顶率 | `<=0.10` 且 `<=` 对照组 | 避免用截断/退化换取表面EM |
| hard KL越界 | 必须为0 | 硬性安全线 |

### 3.5 Confirm50通过后的额外步骤（R8特有，R7未做）

任何进入Confirm50且通过的分支，**必须**追加一次Phase 0式的1319题评估，而不是止步于100题结果。理由：R7的教训是L1→L2的100题EM完全相等，如果不换更大样本，可能永远分不清"真的打平了"还是"100题分辨率不够看出差异"。

## 4. R8-Phase 2（暂缓，不在本轮承诺范围内）：无放回 epoch sampler

### 4.1 为什么暂缓

- 代码层面**从零开发**：需要在 `GRPOTrainer` 内维护一个跨step的"已用index集合"，在集合覆盖全量训练集（5759条）后重置并开启下一轮epoch，同时要兼容现有的 `deterministic_prompt_sampling`（seed+step派生）和断点续训（RNG/trainer state恢复）语义，改动面比单纯加一个variant大得多。
- 按45号文档9.4节测算，要无放回遍历一轮5759条数据（每步4个prompt）大约需要1440 step，远超过当前所有分支验证过的最大步数（50步），投入产出比在"当前僵局是否与prompt覆盖率有关"这一假设尚未被Phase 1初步验证之前，风险较高。

### 4.2 触发条件

只有以下情况同时满足，才在R9启动epoch sampler开发：

```text
Phase 0确认0.67并非纯噪声（差距真实存在）
且
Phase 1的B2或I4至少有一个分支同时改善KL与能力指标（而不是像L2那样只改善KL）
```

若触发，建议先做小规模验证（覆盖1-2轮、约250-350 step的部分实现），而不是直接承诺1440 step的完整实现。

## 5. 资源与执行顺序总表

| 阶段 | 是否需要新训练 | 预计GPU耗时 | 触发下一阶段的条件 |
|---|---|---|---|
| Phase 0 | 否，复用3个已有checkpoint | 3次1319题greedy推理，约30-60分钟（参考45号文档第五部分batch8速度） | 无条件执行Phase 1（至少需要用扩大验证集复核一次） |
| Phase 1 (B2) | 是，Gate10→Screen30→Confirm50 | 与R7同规模（约3-4小时/分支×50步） | 通过预注册门槛 |
| Phase 1 (I4) | 是，同上 | 同上 | 通过预注册门槛 |
| Phase 2 | 是，且需先开发代码 | 未估算，取决于最终step数设计 | 第4.2节触发条件 |

## 6. 结果记录位置

R8执行完成后：

1. Phase 0结果写入 `eval_results/grpo_model/r8_phase0_noise_check/r8_phase0_wilson_summary.md`。
2. Phase 1完整过程和判定参照R7格式，追加到 `models/grpo/g7lk/_orchestration/` 或新建 `models/grpo/g7r8/_orchestration/`（视是否复用同一短目录体系而定，建议新开 `g7r8` 保持每轮实验目录独立、便于追溯）。
3. 最终结论回填 `43_grpo_v7_experiment_timeline_cn.md`：在第7节后新增"R8"小节，并更新第0节总览表和第8节待改进方向（已验证的方向从"待尝试"移到"已验证"分类）。
