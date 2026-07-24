# post_training_framework/scripts 目录说明

本目录放正式命令行入口。脚本负责参数解析和流程编排，核心逻辑放在 `src/ptf/`。

## 评估入口

| 脚本 | 作用 |
|---|---|
| `run_base_eval.py` | base model 推理评估 |
| `run_sft_eval.py` | SFT adapter 推理评估 |
| `run_grpo_eval.py` | GRPO/HF actor 推理评估 |
| `run_grpo_sample_eval.py` | 单个冻结GRPO checkpoint的固定随机扩大sample评估 |
| `run_oracle_eval.py` | oracle@k 多采样评估 |
| `compare_runs.py` | 两阶段结果对比 |
| `compare_base_sft_grpo.py` | Base/SFT/GRPO 三方对比 |

## 训练入口

| 脚本 | 作用 |
|---|---|
| `run_sft_train.py` | LoRA SFT 训练 |
| `run_grpo_train.py` | 自定义 GRPO 训练 |
| `run_grpo_causal_experiments.py` | 生成、串行运行并汇总 GRPO 控制变量实验 |
| `run_grpo_sample_matrix.py` | 串行执行C0/L1多seed扩大sample评估并生成置信区间与晋级判定 |
| `run_ppo_train.py` | 自定义 PPO 训练 |
| `run_pipeline.py` | 串起 Base/SFT 基础流程 |

## 数据和诊断入口

| 脚本 | 作用 |
|---|---|
| `prepare_grpo_data.py` | 预览 messages parquet 的 RL prompt/ground truth |
| `build_grpo_bucket_dataset.py` | 按 greedy/oracle 结果分桶 |
| `build_grpo_focus_dataset.py` | 构造 greedy 错但 oracle 命中的聚焦集 |
| `build_grpo_usable_dataset_from_buckets.py` | 合并可训练 GRPO 数据 |
| `score_gsm8k_rule_reward.py` | 离线 rule reward 打分 |
| `run_training_dashboard.py` | 本地训练指标 dashboard |
| `run_grpo_v7_preflight.ps1` | 按 Smoke、Diagnostic、Formal 三阶段启动 GRPO v7 |
| `run_full_train_bucket_audit.ps1` | 全量训练集分桶审计编排 |

## 注意

当前 GRPO 数据构建和训练入口只使用 `messages` parquet。

控制变量实验入口默认只做 dry-run；真正启动 GPU 必须显式传 `--execute`，并按 `gate10 → screen30 → confirm50` 连续推进。

GRPO停止原因优先读取`<run_dir>/training_stop.json`；历史实验没有该文件时，控制变量汇总继续回退解析日志中的`训练结束, 原因:`。训练dashboard仍读取原四类CSV，并额外展示结构化`training_status`和`stop_reason`，因此新旧运行目录都可兼容。
