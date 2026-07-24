# GSM8K GRPO/PPO 数据说明

本目录服务 RL 阶段，包括固定 RL prompt、过拟合探针、greedy/oracle 分桶和最终可训练文件。

## 当前推荐训练文件

| 文件 | 格式 | 用途 |
|---|---|---|
| `train.parquet` | messages | 当前稳定 GRPO 训练入口，5759 条 |
| `derived/usable_full7473_mixed_plus_boundary/train_full7473_grpo_usable_mixed_plus_boundary.messages.parquet` | messages | 稳定入口的可追溯来源文件 |
| `derived/usable_full7473_mixed_plus_boundary/train_full7473_grpo_usable_mixed_plus_boundary_messages_preview.md` | md | 上述训练文件人工预览 |
| `derived/usable_full7473_mixed_plus_boundary/train_full7473_grpo_usable_mixed_plus_boundary.messages.meta.json` | json | 合并策略和来源记录 |

## 固定入口文件

| 文件 | 用途 |
|---|---|
| `train.parquet` | 当前 messages 格式稳定训练入口，5759 条 |
| `smoke_train_32.parquet` | 从当前训练入口抽取的 32 条 messages smoke 数据 |
| `eval_20.parquet` | messages 格式固定 20 条评估集 |
| `eval_100.parquet` | 当前 GRPO 主要固定评估集 |

## 派生数据

| 文件或目录 | 说明 |
|---|---|
| `derived/probes/overfit_probe_64_oracle_hit_greedy_miss.*` | 64 题过拟合诊断集 |
| `derived/focus_train256/train_focus_sft_greedy_miss_oracle8_hit_train256.*` | train256 聚焦样本 |
| `derived/legacy_verl_exports_do_not_use/` | 历史 verl 导出，仅归档，不参与当前流程 |
| `audits/buckets_train256_sft_eosfix2_oracle8_len256/` | 256 条分桶审计 |
| `audits/buckets_train_full7473_sft_eosfix2_oracle8_len256_bs64_bs16/` | 全量 7473 条分桶审计 |
| `previews/` | 项目脚本生成的数据预览 |
| `derived/` | 已归类的派生训练集和 probe 数据 |

## 当前训练格式

当前项目自定义 GRPO 训练器读取 `messages` parquet:

```text
messages -> user prompt -> assistant answer -> #### ground_truth
```

当前项目不生成、不读取 `prompt/reward_model/extra_info` 形式的 verl 训练数据。

## 生成脚本

| 脚本 | 作用 |
|---|---|
| `build_grpo_bucket_dataset.py` | 根据 SFT greedy/oracle 结果分桶 |
| `build_grpo_focus_dataset.py` | 筛选 greedy 错但 oracle 命中的聚焦样本 |
| `build_grpo_usable_dataset_from_buckets.py` | 合并可训练 messages 数据 |
| `prepare_grpo_data.py` | 用项目自带逻辑预览 messages 数据 |

## Git 规则

全量 parquet、bucket 目录、派生 JSON/CSV 默认不进 git。固定小样本和说明文档可保留。
