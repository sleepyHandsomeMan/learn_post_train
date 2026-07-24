# Post Training Framework

`post_training_framework/` 是本项目自包含的后训练实验框架，用来把 Base/SFT/GRPO/PPO 的训练、推理、评估和报告流程固定下来。

它不是 verl/TRL/LLaMA-Factory 的替代品，而是学习和排错用的透明实验层。

## 目录结构

```text
post_training_framework/
  configs/   # 实验配置 JSON
  scripts/   # 命令行入口
  src/ptf/   # 核心 Python 模块
```

详细说明:

```text
configs/README.md
scripts/README.md
src/ptf/README.md
```

## 当前默认任务

| 项 | 默认值 |
|---|---|
| 任务 | GSM8K 数学问答 |
| 输出格式 | `#### <number>` |
| 数据格式 | `messages` parquet |
| SFT | LoRA + assistant-only loss |
| GRPO/PPO | 自定义 PyTorch trainer + rule reward |
| 核心评估 | exact match、format rate、single final answer、repeat-like、平均长度 |

## 关键数据格式

SFT、评估、自定义 GRPO/PPO 默认读取:

```text
messages = [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "... #### 42"}
]
```

当前框架只接受 `messages` parquet，不生成也不读取 verl 风格训练数据。

## 常用入口

### Base 评估

```powershell
d:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_base_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --max-new-tokens 256 `
  --max-items 100 `
  --run-name base_eval
```

### SFT 训练

```powershell
d:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_train.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --run-name sft_run
```

### SFT 评估

```powershell
d:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --adapter-dir models\sft\<adapter_dir> `
  --max-new-tokens 256 `
  --run-name sft_eval
```

### GRPO 训练

```powershell
# 正式训练前先依次执行 Smoke 和 Diagnostic
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Smoke
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Diagnostic

# 只有诊断达标后才执行 Formal
post_training_framework\scripts\run_grpo_v7_preflight.ps1 -Mode Formal
```

v7 从基线、预检、诊断、formal120 到扩模决策的有序训练框架见 `docs/45_grpo_v7_preflight_remediation_cn.md`。

### GRPO 数据预览

```powershell
d:\Anaconda\envs\test3\python.exe post_training_framework\scripts\prepare_grpo_data.py `
  --sft-file datasets\gsm8k_grpo\derived\usable_full7473_mixed_plus_boundary\train_full7473_grpo_usable_mixed_plus_boundary.messages.parquet `
  --output-dir datasets\gsm8k_grpo\previews\<run_name> `
  --num-preview 6
```

## 重要脚本

| 脚本 | 作用 |
|---|---|
| `run_base_eval.py` | base 推理评估 |
| `run_sft_train.py` | LoRA SFT 训练 |
| `run_sft_eval.py` | SFT 推理评估 |
| `run_oracle_eval.py` | oracle@k 多采样评估 |
| `build_grpo_bucket_dataset.py` | greedy/oracle 分桶 |
| `build_grpo_usable_dataset_from_buckets.py` | 合并 GRPO 可训练数据 |
| `run_grpo_train.py` | 自定义 GRPO 训练 |
| `run_grpo_causal_experiments.py` | checkpoint-169 根因控制变量实验编排与汇总 |
| `run_ppo_train.py` | 自定义 PPO 训练 |
| `run_grpo_eval.py` | GRPO checkpoint 评估 |
| `compare_base_sft_grpo.py` | 三阶段对比 |
| `run_training_dashboard.py` | 本地训练指标 dashboard |

## GRPO停止控制与输出兼容

GRPO训练采用“统一调度、分别判定、统一留档、统一收尾”：KL、训练信号、验证格式和early stopping分别计算，统一停止控制器按显式优先级选择主原因。原有训练日志以及`plots/train_metrics.csv`、`val_metrics.csv`、`group_diagnostics.csv`、`gpu_memory.csv`的路径和列结构保持不变，现有看板与因果实验汇总脚本可继续读取。

新增两类机器可读停止档案：

- `<output_dir>/training_stop.json`：当前训练会话的最终状态、主停止原因和checkpoint位置；
- `<output_dir>/logs/stop_events.jsonl`：跨resume追加的开始、候选选择和结束事件。

终态checkpoint的`trainer_state.json`同步保存`training_status`、`stop_reason`和完整`stop_decision`。

## 维护规则

- 脚本只做命令行封装，核心逻辑放到 `src/ptf/`。
- 新增配置写入 `configs/`，并在 `configs/README.md` 说明。
- 新增数据格式或训练文件选择时，同步更新 `datasets/README.md`。
- 训练产物不要放入框架源码目录，统一进入 `models/`、`eval_results/`、`logs/`。
