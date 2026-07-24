# GRPO checkpoint-169 根因控制变量实验方案

## 1. 方案目标

本方案用于回答一个严格的因果问题：

```text
checkpoint-169 整改探针为什么在 step 211 出现 reference KL 持续上升，
同时 greedy/sample EM 没有超过起点？
```

目标不是找到一个“能跑满50步”的配置，而是区分以下因素各自造成了多大影响：

1. 新 reward/KL 目标是否被旧 AdamW optimizer 动量污染。
2. 同一批 rollout 做2个 PPO epoch 是否重复放大小样本方向。
3. `5e-6` 学习率是否使每次参数更新的步幅过大。
4. KL penalty 的初始强度是否过弱。
5. 自适应 KL 每10步才响应是否过慢。
6. 每步只有4道独立题是否导致跨题梯度方差过大。
7. 上述因素之间是否存在交互，而不是单一因素独立致因。

实验最终要产出：

- 每个候选原因的支持证据、反证和不确定性。
- 一套经过固定 seed 筛选和多 seed 复验的安全训练配置。
- 从 checkpoint-169 扩展到正式长训前的明确放行条件。

## 2. 当前已经知道什么

已由 step 170–211 日志确认：

- reference KL 从前10步均值约0.033上升到后段约0.10。
- update KL 很小，没有单个 PPO mini-batch 突然爆炸。
- KL 项长期只占 policy loss 的很小比例。
- 自适应 KL 确实从0.005提高到0.0075和0.01125，但介入偏晚。
- 42个 GRPO step 实际约有168次 optimizer update。
- 每个 step 只有4道独立题，32条 rollout 是4组相关回答。
- 最近训练数据损坏、顺序进入坏数据段、格式 reward 压过答案 reward 已基本排除。
- checkpoint-169 的旧 AdamW 状态确实被整改分支加载，旧 `exp_avg` 不为零。

尚不能确认：

- 旧 optimizer 对失败贡献了多少。
- PPO epoch、学习率、KL 强度、KL 响应速度和 prompt 数中哪一个影响最大。
- 单变量改善能否同时降低 KL 漂移并恢复 EM。
- 某一次运行的改善是否只是 seed 运气。

因此下一步必须是控制变量实验，而不是继续凭直觉叠加修改。

## 3. 因果分析的基本单位

本方案把一次实验定义为：

```text
同一 checkpoint-169 LoRA 权重
+ 一套明确的 optimizer 初始化方式
+ 一个固定 seed
+ 一条固定 prompt 调度规则
+ 一组只允许声明字段变化的 GRPO 配置
+ 10→30→50步的连续训练轨迹
```

比较对象不是某一个偶然最高验证点，而是整段轨迹。

主要因变量：

- reference KL 的均值、后10步斜率、首次越过 warning/hard 的 step。
- rollout exact 的前10步均值与后10步均值。
- 初始、最终和最佳 greedy EM。
- sample EM、format、EOS、hit-max。

辅助因变量：

- update KL 最大值。
- clip fraction、grad norm。
- effective/mixed/all-wrong group rate。
- 实际 optimizer update 次数。
- prompt 暴露量与 rollout 数。
- stop reason 和异常 rollout。

## 4. 所有分支必须固定的共同条件

除被声明为实验变量的字段外，以下条件完全相同：

| 类别 | 固定条件 |
|---|---|
| 起点权重 | formal500 最佳 `checkpoint-169` 的同一份 GRPO LoRA |
| reference | 同一 SFT model，训练中冻结 |
| 数据 | `datasets/gsm8k_grpo/train.parquet` 与固定 eval100 |
| 数据格式 | 项目自有 `messages` parquet |
| rollout | 每题8条、temperature 0.7、top-p 1.0、top-k 50 |
| response | 最大256 token |
| reward | 当前整改后的答案、格式、重复、长度、截断规则 |
| PPO | clip ratio 0.2、mini-batch 16，除指定变量外不变 |
| 验证 | 训练前基线、每10步 greedy eval100、固定10题 sample@8 |
| 安全保护 | reference/update KL、signal guard、格式保护全部保留 |
| prompt | 由 `seed + step` 派生的局部 RNG 决定，不受其他 RNG 消耗影响 |
| 日志 | train/val/group/gpu CSV、checkpoint、异常 rollout、完整配置 |

固定 prompt 调度非常重要。

如果继续使用全局 Python RNG，不同分支可能因为验证、PPO 或其他代码消耗随机数的差异，逐渐抽到不同题目。那时指标差异既可能来自超参数，也可能来自题目序列。

当前新增的 deterministic prompt sampling 使用 `prompt_sampling_seed + step` 生成每步排列：

- batch=4 和 batch=8 分支在同一步共享相同排列。
- batch=4 使用排列前4题。
- batch=8 使用同一排列前8题。
- 续跑到30/50步时仍能按 step 精确接上。

## 5. checkpoint 状态必须显式区分

训练器现在支持三种状态模式。

### 5.1 `full`

加载：

- LoRA 权重。
- optimizer 一阶/二阶状态。
- trainer step、best EM、早停计数。
- RNG 状态。

用途：同一实验从10步严格续到30步，再续到50步。

限制：学习率、PPO epoch、batch、梯度累积、seed 和 prompt 调度不能悄悄改变。

### 5.2 `weights_only`

加载：

- 只加载 checkpoint LoRA 权重。

重置：

- AdamW optimizer。
- trainer step 和历史最佳值。
- RNG。
- 自适应 KL 当前状态。

用途：新目标和绝大多数控制变量分支的干净起点。

这类分支从实验 step 0 计数，并执行零步 greedy/sample baseline。

### 5.3 `weights_and_optimizer`

加载：

- LoRA 权重。
- checkpoint-169 的旧 optimizer。

重置：

- trainer step、最佳值和早停计数。
- RNG。

用途：只隔离“旧 optimizer 历史”这一因素。

这是故意保留污染的诊断分支，不是正式训练候选。

## 6. 第一轮单变量矩阵

### 6.1 C0：干净对照

```text
weights_only
lr=5e-6
ppo_epochs=2
prompt=4
KL coef=0.005
KL interval=10
```

作用：建立新目标下的干净基线。

后续所有单变量分支都先与同 seed 的 C0 比较，而不是直接与混合整改 probe 比较。

### 6.2 O1：旧 optimizer

唯一变化：

```text
resume_state_mode: weights_only → weights_and_optimizer
```

问题：旧目标形成的 AdamW 动量是否使早期 KL 斜率更高、EM 更差。

支持该原因的证据：

- O1 在相同 prompt 序列下明显早于 C0 越过 KL warning。
- O1 的 KL 后10步斜率更高。
- O1 rollout exact 和 greedy/sample EM 同时更差。
- 差异不只出现在第一两步，而能持续到30步。

若 O1 与 C0 接近，旧 optimizer 是确定存在的污染，但不是本次失控主因。

### 6.3 P1：PPO epoch 从2降到1

唯一变化：

```text
ppo_epochs: 2 → 1
```

每步 optimizer update 从4次降到2次。

问题：同一批 rollout 的第二轮重用是否放大偶然梯度方向。

必须同时按两种横轴解读：

1. 相同 GRPO step，即相同 prompt/rollout 暴露量。
2. 相同 optimizer update 数，排除仅仅少更新一半的解释。

如果 P1 只让 KL 上升变慢，但 EM 也完全不动，说明它降低了更新强度，不足以证明“重复更新”是唯一主因。

### 6.4 L1/L2：学习率剂量

L1 唯一变化：

```text
learning_rate: 5e-6 → 3e-6
```

L2 作为敏感性复核：

```text
learning_rate: 5e-6 → 2e-6
```

问题：参数步幅是否偏大。

若 `5e-6 → 3e-6 → 2e-6` 呈现稳定剂量关系，即学习率越低、KL 斜率越低，同时 EM 并未冻结，则学习率原因得到较强支持。

若2e-6虽然 KL 最低但 EM 没有学习，说明约束过强或更新不足，不能直接选最低学习率。

### 6.5 K1：提高 KL 初始强度

唯一变化：

```text
kl_loss_coef: 0.005 → 0.02
```

问题：KL penalty 的反向传播量级是否长期过小。

支持证据必须同时满足：

- warning 前的 KL 斜率明显下降，而不是越线后才停车。
- rollout exact、greedy/sample EM 不因过强锚定而整体下降。

只降低 KL、但完全阻止能力学习，说明系数过强或策略起点已经接近该配置的可达上限。

### 6.6 K2：缩短 KL 响应间隔

唯一变化：

```text
adaptive_kl_interval: 10 → 2
```

问题：控制器是否因为10步一次的反馈滞后而来不及纠偏。

K1 与 K2 分开运行，是为了区分：

- 惩罚本身太弱。
- 惩罚调整得太慢。

### 6.7 B1：增加独立 prompt，保持更新次数

结构变化：

```text
train_batch_size: 4 → 8
gradient_accumulation_steps: 1 → 2
```

这不是普通的两个旋钮一起修改，而是一个有约束的结构实验：

```text
C0: 4 prompt × 8 rollout = 32
    32 ÷ mini-batch16 × PPO2 = 4 optimizer update

B1: 8 prompt × 8 rollout = 64
    64 ÷ mini-batch16 × PPO2 = 8 mini-batch backward
    每2个mini-batch累积一次 = 4 optimizer update
```

主变量是每步独立 prompt 数。

梯度累积是为了保持每步 optimizer update 为4次，防止 batch=8 同时把更新次数翻倍。

代价是每步 rollout 和 backward 工作量约增加，因此还要按 prompt 暴露量、optimizer update 和墙钟时间分别比较。

## 7. 第二轮交互实验

只有单变量结果显示两个因素都有效时，才运行交互项：

| ID | 组合 | 回答的问题 |
|---|---|---|
| I1 | PPO epoch1 + KL interval2 | 重复更新与控制器滞后是否共同致因 |
| I2 | lr3e-6 + KL interval2 | 参数步幅与控制器滞后是否共同致因 |
| I3 | PPO epoch1 + prompt8/accum2 | 小批次方差与 rollout 重用是否共同致因 |

不能一开始就只跑组合实验。

组合成功只能说明整套配置有效，无法知道哪一项是必要条件。先做单变量筛选，再做2×2思想下的交互检查，因果链才完整。

## 8. 三阶段连续训练门禁

### 8.1 Gate10

目标：工程正确性，不判断长期收益。

检查：

- checkpoint 状态模式与日志声明一致。
- C0 optimizer 是 fresh，O1 optimizer step 来自旧状态。
- 初始 greedy/sample baseline 已写入。
- deterministic prompt indices 在各分支一致。
- B1 每步8个 prompt、4次 optimizer update。
- C0 每步4个 prompt、4次 optimizer update。
- P1 每步4个 prompt、2次 optimizer update。
- 无 OOM、NaN、Inf、CSV 表头错位。
- format、EOS、hit-max 没有立即异常。

Gate10 通过不等于该变量有效，只允许续到30步。

### 8.2 Screen30

目标：固定 seed=42 的单变量筛选。

重点比较：

- step 0–9、10–19、20–29 的 reference KL 均值与斜率。
- warning crossing 是否被提前或延后。
- rollout exact 前10与后10是否下降。
- greedy/sample EM 相对各自零步 baseline 的变化。
- 实际 optimizer update 与 prompt 暴露量。

进入50步的最低要求：

```text
没有触发 hard KL/signal/格式 guard
+ 后10步 reference KL 斜率不呈持续加速
+ rollout exact 后10步不显著低于前10步
+ greedy 或 sample 能力指标没有系统性低于 C0
```

### 8.3 Confirm50

目标：排除只在前30步暂时稳定。

固定 seed=42 先延长到50步。

要求：

- 不依赖关闭 guard 才能完成。
- reference KL 在 warning 前稳定，或越过 warning 后能回落。
- final/best EM 至少不低于自身零步 baseline 与 C0 的合理波动区间。
- format、EOS、hit-max 不出现方向性退化。

10→30→50 使用同一输出目录和 `full` resume，因此 LoRA、optimizer、trainer state、RNG、CSV 和日志是连续轨迹，不是三次重新起训。

## 9. 多 seed 复验

固定 seed=42 只用于筛选，不足以形成最终结论。

从50步结果中选择1至2个最优候选，加上 C0，使用：

```text
42
123
2026
```

进行复验。

支持因果结论的最低标准：

- 至少多数 seed 的方向一致。
- KL 斜率改善不能只由一个 seed 驱动。
- EM 不要求每个 seed 单调上升，但均值/中位数不能以明显能力损失换 KL。
- 结论同时报告效应方向和波动，不能只报最好一次。

## 10. 判定主因的规则

一个变量被视为“主要原因”，至少需要：

1. 单变量相对 C0 产生明显、持续、方向一致的改善。
2. 改善同时覆盖 KL 稳定性和能力指标，而不是只延后早停。
3. 30→50步仍成立。
4. 多 seed 多数方向一致。
5. 加回该风险因素时问题能够复现，或组合实验显示它是必要成分。

结果分级：

| 级别 | 含义 |
|---|---|
| 已确认主因 | 单变量、多阶段、多 seed 和反向复现均支持 |
| 重要共同因素 | 单变量有改善，且交互实验显示与另一因素共同作用 |
| 次要因素 | 指标有小幅影响，但不足以解释 step 211 失控 |
| 实验污染 | 条件确实不干净，但清除后主要趋势仍存在 |
| 不支持 | 相对 C0 没有稳定改善或方向相反 |

## 11. 后续操作入口

具体命令、产物目录、dashboard、成本顺序和正式长训放行清单见：

`docs/49_grpo_causal_experiment_operations_cn.md`
