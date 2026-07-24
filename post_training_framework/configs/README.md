# post_training_framework/configs 目录说明

本目录保存实验配置 JSON。

## 当前配置

| 文件 | 说明 |
|---|---|
| `gsm8k_qwen3_0d6b.json` | GSM8K + Qwen3-0.6B 默认配置 |
| `gsm8k_qwen3_1d7b.json` | GSM8K + Qwen3-1.7B 默认配置 |
| `gsm8k_qwen3_0d6b_grpo_v7.json` | 0.6B GRPO v7 正式配置，含 SFT 起点、rewardv2 和信号保护 |
| `gsm8k_qwen3_0d6b_grpo_v7_repair_probe_from169.json` | 从 formal500 最佳 checkpoint-169 建新分支的 50 步整改验证配置 |
| `gsm8k_qwen3_0d6b_grpo_v7_causal_base_from169.json` | checkpoint-169 控制变量实验的干净基础配置，不建议直接运行 |
| `gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json` | 单变量、敏感性和交互实验矩阵，以及10/30/50步阶段定义 |
| `g7s26.json` | seed=2026新guard C0/L1补充Confirm50矩阵；输出使用短目录`models/grpo/g7s26/` |
| `g7fmt.json` | C0/L1三seed冻结checkpoint扩大sample评估；输出使用短目录`eval_results/grpo_model/g7fmt/` |
| `g7lk_base.json` | L2/K3共同L1基线配置，固定`learning_rate=3e-6`；禁止直接运行 |
| `g7lk.json` | L2学习率剂量与K3 KL下限两个独立分支；输出使用短目录`models/grpo/g7lk/` |

## 配置职责

配置文件集中管理:

- base model 路径。
- train/eval 数据路径。
- prompt 模式和格式指令。
- generation 参数。
- SFT/GRPO/PPO 训练参数。

## 维护规则

- 新增配置时命名为 `<task>_<model>.json`。
- 临时覆盖优先用脚本的 `--set`，不要为每个小实验复制配置。
- 如果某个配置成为长期基线，应在 README 或实验文档中说明用途。
