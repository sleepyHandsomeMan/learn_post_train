# GSM8K Rule Reward + GRPO 实现指南

> 日期: 2026-06-22  
> 适用范围: `D:\learnAI\verl\yhy` 个人后训练实验工作区  
> 目标: 在已有 Base/SFT 评估基础上，用 rule reward 跑通 GRPO 的最小闭环。

## 1. 当前阶段目标

当前已经完成：

```text
Base model
  -> SFT
  -> SFT 后评估
  -> rule reward 离线验证
```

下一步进入：

```text
SFT checkpoint
  -> GRPO rollout
  -> GSM8K rule reward
  -> GRPO update
  -> GRPO checkpoint
  -> 固定验证集评估
  -> Base / SFT / GRPO 对比
```

本阶段不是追求 SOTA，而是验证：

- verl 的 GRPO 训练链路能否跑通。
- rule reward 能否把答案正确、格式正确、不复读的输出推高。
- GRPO 后是否能在固定验证集上保持或提升 EM，并继续压低复读。

## 2. 新增目录约定

GRPO 阶段使用独立目录，和已有 Base/SFT 风格保持一致：

```text
yhy/
  datasets/
    gsm8k_grpo/             # GRPO/RL 训练数据
  models/
    grpo/                   # GRPO 后 actor checkpoint 或导出的 adapter/model
  eval_results/
    grpo_model/             # GRPO 模型评估结果
  dev_tools/
    grpo/                   # GRPO 阶段脚本和启动命令
  docs/
    grpo_implementation_guide_cn.md
```

建议命名：

```text
qwen3_1d7b_gsm8k_grpo_rule_len512_n4_lr1e-6_smoke
qwen3_1d7b_gsm8k_grpo_rule_len512_n4_lr1e-6_steps100
qwen3_0d6b_gsm8k_grpo_rule_len512_n4_lr1e-6_steps100
```

命名字段含义：

```text
<base_model>_<task>_<method>_<reward>_<response_len>_<rollout_n>_<lr>_<stage>
```

## 3. 选择 GRPO Baseline

推荐优先使用：

```text
models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2/
```

原因：

- `format_rate = 100%`
- `single_final_answer_rate = 100%`
- `repeat_like_rate = 0%`
- 是更干净的 RL 起点

保留 0.6B 作为对照：

```text
models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2/
```

0.6B 的价值是观察 GRPO 是否能进一步压制中途推理复读。

## 4. 数据准备

### 4.1 SFT 数据和 GRPO 数据的区别

SFT 数据是：

```text
messages:
  user: question + format instruction
  assistant: 标准推理过程 + #### answer
```

GRPO 数据不需要 assistant 标准答案作为监督目标，而需要：

```text
data_source
prompt
ability
reward_model: { style: "rule", ground_truth: "..." }
extra_info
```

verl 的 GSM8K 预处理脚本就是这种格式：

```text
examples/data_preprocess/gsm8k.py
```

### 4.2 本项目 GRPO 数据目标格式

建议生成：

```text
datasets/gsm8k_grpo/
  train.parquet
  eval_20.parquet
  smoke_train_32.parquet
  README.md
```

每行字段：

```python
{
    "data_source": "openai/gsm8k",
    "prompt": [
        {
            "role": "user",
            "content": "题目 Let's think step by step and output the final answer after \"####\"."
        }
    ],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": "42"
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "question": "原始题目",
        "answer": "标准推理过程 #### 42"
    }
}
```

### 4.3 数据准备脚本

当前已新增：

```text
post_training_framework/scripts/prepare_grpo_data.py
```

输入：

```text
datasets/gsm8k_sft/train.parquet
datasets/gsm8k_sft/eval_20.parquet
```

输出：

```text
datasets/gsm8k_grpo/train.parquet
datasets/gsm8k_grpo/eval_20.parquet
datasets/gsm8k_grpo/smoke_train_32.parquet
```

同时会生成方便人工检查的预览报告：

```text
datasets/gsm8k_grpo/preview_train.md
datasets/gsm8k_grpo/preview_eval_20.md
datasets/gsm8k_grpo/preview_smoke_train_32.md
```

执行命令：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\prepare_grpo_data.py `
  --sft-train-file datasets\gsm8k_sft\train.parquet `
  --sft-eval-file datasets\gsm8k_sft\eval_20.parquet `
  --output-dir datasets\gsm8k_grpo `
  --smoke-size 32 `
  --preview-items 3
```

只想快速检查转换效果，不重新写训练 parquet 时，可以使用：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\prepare_grpo_data.py `
  --preview-only `
  --preview-items 5
```

注意：

- `train.parquet` 可用于 GRPO 训练。
- `eval_20.parquet` 只用于验证，不参与训练。
- `smoke_train_32.parquet` 只用于链路冒烟测试。
- `preview_*.md` 会并排展示原始 `messages` 和合成后的 GRPO row，用于确认 prompt、ground_truth、extra_info 是否正确。

## 5. Rule Reward 接入

当前已实现：

```text
post_training_framework/src/ptf/reward.py
```

核心入口：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    ...
```

这个签名兼容 verl 的 custom reward function。

当前 reward 规则：

```text
#### 答案正确: +1.0
无 #### 但 fallback 答案正确: +0.5
格式正确: +0.2
只有一个最终答案: +0.1
缺少格式: -0.2
多个最终答案: -0.1
复读: -0.4
过长: -0.2
clip 到 [-1.0, 1.3]
```

离线验证结果：

| run | avg reward | EM | format | single final | repeat |
|---|---:|---:|---:|---:|---:|
| 0.6B base max512 | -0.565 | 25% | 0% | 0% | 90% |
| 0.6B old SFT run2 max160 | 0.240 | 45% | 70% | 10% | 60% |
| 0.6B eosfix2 max512 | 0.620 | 50% | 85% | 85% | 15% |
| 1.7B eosfix2 max512 | 0.800 | 50% | 100% | 100% | 0% |

说明 reward 能区分：

- base 的无格式长输出。
- old SFT 的复读输出。
- eosfix2 的干净格式输出。

## 6. GRPO Smoke Test

第一轮不要直接正式训练。先跑 smoke test。

目标：

```text
rollout -> reward -> advantage -> actor update -> checkpoint
```

推荐配置：

```text
train_file: datasets/gsm8k_grpo/smoke_train_32.parquet
val_file: datasets/gsm8k_grpo/eval_20.parquet
model: models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2
rollout.n: 4
max_prompt_length: 512
max_response_length: 512
train_batch_size: 8 或 16
ppo_mini_batch_size: 16 或 32
ppo_epochs: 1
actor lr: 1e-6
total_training_steps: 1 到 5
save_freq: 1
test_freq: 1
```

如果显存吃紧，优先降低：

```text
train_batch_size
ppo_mini_batch_size
max_response_length
rollout.n
```

不要一开始关闭 KL。建议保留：

```text
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
algorithm.use_kl_in_reward=False
```

当前已提供脚本：

```text
dev_tools/grpo/run_grpo_smoke_qwen3_1d7b.ps1
dev_tools/grpo/run_grpo_smoke_qwen3_0d6b.ps1
```

建议先检查命令，不启动训练：

```powershell
.\scripts\grpo\run_grpo_smoke_qwen3_1d7b.ps1 -DryRun
```

确认命令无误后，再去掉 `-DryRun` 正式启动 smoke test。

## 7. verl 命令结构

verl GRPO 的核心配置是：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=4
reward.custom_reward_function.path=<reward.py 路径>
reward.custom_reward_function.name=compute_score
```

本项目已经把命令封装到公共脚本：

```powershell
.\scripts\grpo\run_grpo_common.ps1 -DryRun
```

更常用的是直接运行 wrapper：

```powershell
.\scripts\grpo\run_grpo_smoke_qwen3_1d7b.ps1 -DryRun
.\scripts\grpo\run_grpo_short_qwen3_1d7b.ps1 -DryRun
.\scripts\grpo\run_grpo_smoke_qwen3_0d6b.ps1 -DryRun
.\scripts\grpo\run_grpo_short_qwen3_0d6b.ps1 -DryRun
```

注意：

- 当前脚本默认使用 `hf` rollout backend，依赖少但慢。
- 如果本地 vLLM/sglang 可用，可以传入 `-RolloutBackend vllm` 或 `-RolloutBackend sglang`。
- SFT 起点是 LoRA adapter，因此脚本使用 `actor_rollout_ref.model.path=<base>` 加 `actor_rollout_ref.model.lora_adapter_path=<sft_adapter>`。
- 第一次建议只跑 smoke，不要直接跑 full train。

## 8. Smoke Test 验收标准

必须看到：

```text
1. 训练正常启动
2. rollout 能生成 response
3. reward 日志出现 score
4. GRPO advantage 能计算
5. actor update 无 NaN
6. checkpoint 能保存
7. validation 能跑完
```

如果失败，按顺序排查：

| 现象 | 优先检查 |
|---|---|
| 数据读取失败 | parquet 字段是否含 `prompt`、`reward_model` |
| reward 全是 0 | `ground_truth` 是否是纯数字字符串，reward path 是否正确 |
| reward 函数 import 失败 | `reward.custom_reward_function.path` 是否绝对路径 |
| 显存不足 | 降 batch、response length、rollout.n |
| 训练不更新 | rollout.n 是否大于等于 2，组内 reward 是否有差异 |
| KL 过大 | 降 lr，增 KL coef，减训练步数 |

## 9. 正式短程 GRPO

smoke test 成功后，进入短程训练。

推荐第一版正式配置：

```text
train_file: datasets/gsm8k_grpo/train.parquet
val_file: datasets/gsm8k_grpo/eval_20.parquet
model: 1.7B eosfix2 SFT
rollout.n: 4
max_response_length: 512
train_batch_size: 32 或 64
actor lr: 1e-6
ppo_epochs: 1
total_training_steps: 50 到 100
test_freq: 10
save_freq: 10
```

如果显存和速度允许，再尝试：

```text
rollout.n: 8
train_batch_size: 128
total_training_steps: 200
```

## 10. GRPO 评估流程

GRPO checkpoint 保存后，先把 actor checkpoint merge/export 成 HuggingFace 模型，再评估。

merge 命令：

```powershell
.\scripts\grpo\merge_grpo_checkpoint.ps1 `
  -RunName qwen3_1d7b_gsm8k_grpo_rule_len512_n4_lr1e-6_smoke `
  -DryRun
```

确认命令无误后去掉 `-DryRun`，默认输出到：

```text
models/grpo/<run_name>/merged_hf/
```

评估仍使用固定验证集：

```text
datasets/gsm8k_sft/eval_20.parquet
```

评估指标与 SFT 保持一致：

```text
exact_match
format_rate
single_final_answer_rate
repeat_like_rate
avg_chars
rule_reward_avg
```

评估命令：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_grpo_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_1d7b.json `
  --model-dir models\grpo\<run_name>\merged_hf `
  --run-name <run_name> `
  --max-new-tokens 512 `
  --max-items 20
```

评估结果归档到：

```text
eval_results/grpo_model/<run_name>/
  <run_name>_eval_20_max512_full.jsonl
  <run_name>_eval_20_max512_summary.json
  <run_name>_eval_20_max512_full_report.md
  <run_name>_reward_summary.json
```

对比报告建议放：

```text
eval_results/grpo_model/<run_name>/<run_name>_base_sft_grpo_compare_report.md
```

对比表至少包含：

```text
checkpoint | exact match | format rate | single final | repeat-like | avg chars | rule reward
base       | ...
sft        | ...
grpo       | ...
```

三方对比命令：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\compare_base_sft_grpo.py `
  --base-jsonl eval_results\base_model\qwen3_1d7b_base_eval_20_max512_full.jsonl `
  --sft-jsonl eval_results\sft_model\1d7b_eosfix2\1d7b_eosfix2_max512_eval_20_max512_full.jsonl `
  --grpo-jsonl eval_results\grpo_model\<run_name>\<run_name>_eval_20_max512_full.jsonl `
  --run-name <run_name>
```

## 11. 重点风险

### 11.1 Reward Hacking

如果模型学会输出：

```text
#### 42
```

但推理过程变差，说明 reward 太偏格式或答案猜测。

检查方法：

- 人工看 `eval_20` 原始输出。
- 看平均长度是否突然变短。
- 看 EM 是否没有同步上升。

### 11.2 复读被压住但正确率下降

这可能说明模型为了避免复读，学会了过早结束。

检查：

```text
avg_chars 是否明显下降
format_rate 是否高但 EM 下降
推理链是否变短且缺步骤
```

### 11.3 KL 过大

表现：

```text
reward 上升很快
输出风格突然变化
EM 下降
格式模板化
```

处理：

```text
降低 actor lr
增大 kl_loss_coef
减少 total_training_steps
降低 ppo_epochs
```

## 12. 第一轮推荐计划

第一轮只做最小闭环：

```text
1. 生成 datasets/gsm8k_grpo/
2. 用 1.7B eosfix2 跑 smoke_train_32
3. 确认 reward 和 update 正常
4. 保存 smoke checkpoint
5. 用 eval_20 跑评估
6. 生成 Base / SFT / GRPO 对比报告
```

如果 smoke test 成功，再进入：

```text
1. 1.7B eosfix2 + train.parquet + 50 steps
2. 0.6B eosfix2 + train.parquet + 50 steps
3. 比较两者是否在 repeat 与 EM 上有不同收益
```

## 13. 当前已补脚本

当前 GRPO 闭环脚本已经包括：

```text
post_training_framework/scripts/prepare_grpo_data.py
dev_tools/grpo/run_grpo_smoke_qwen3_1d7b.ps1
dev_tools/grpo/run_grpo_short_qwen3_1d7b.ps1
dev_tools/grpo/run_grpo_smoke_qwen3_0d6b.ps1
dev_tools/grpo/run_grpo_short_qwen3_0d6b.ps1
dev_tools/grpo/merge_grpo_checkpoint.ps1
post_training_framework/scripts/run_grpo_eval.py
post_training_framework/scripts/compare_base_sft_grpo.py
```

暂时不要一开始就做 Reward Model。先把 rule reward + GRPO 跑通，理解 rollout、reward、advantage、KL 和 actor update 的关系。
