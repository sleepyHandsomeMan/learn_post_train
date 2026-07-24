# dev_tools/sft 目录说明

本目录保存 SFT 阶段的一次性检查、评估和诊断脚本。

## 常用脚本

| 文件 | 作用 |
|---|---|
| `train_lora_sft.py` | 早期 LoRA SFT 训练脚本 |
| `evaluate_base_max_tokens.py` | base model 不同生成长度评估 |
| `evaluate_full_sft_max_tokens.py` | SFT adapter 不同生成长度评估 |
| `validate_sft_data.py` | SFT parquet 数据检查 |
| `check_sft_data_and_labels.py` | label mask 和 assistant-only loss 检查 |
| `check_eos.py` | EOS 与 `<|im_end|>` 检查 |
| `diagnose_generation.py` | 生成异常诊断 |
| `generate_eval_reports.py` | 从评估 JSONL 生成报告 |
| `build_eval_reports.py` | 汇总构建评估报告 |
| `build_qwen3_1d7b_sft_compare.py` | 1.7B SFT 对比辅助 |

## 维护规则

- 新的正式 SFT 流程入口优先放到 `post_training_framework/scripts/`。
- 本目录保留可解释问题的诊断脚本。
- 诊断输出进入 `eval_results/` 或 `logs/`，不要放在本目录。
