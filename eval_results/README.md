# eval_results 目录说明

本目录保存推理、评估、对比和 rule reward 报告。默认不进入 git。

## 子目录

| 目录 | 说明 |
|---|---|
| `base_model/` | base model 评估结果 |
| `sft_model/` | SFT adapter 评估结果 |
| `grpo_model/` | GRPO/PPO checkpoint 评估结果 |
| `rule_reward/` | rule reward 离线打分结果 |
| `manual_test/` | 手工测试输出 |
| `train_logs/` | 训练日志归档 |

## 标准评估产物

每次正式评估尽量保留:

```text
*_full.jsonl
*_summary.json
*_report.md
```

## 查找建议

- 看总体指标: 先找 `*_summary.json`。
- 看逐题输出: 找 `*_full.jsonl`。
- 人工审查: 找 `*_report.md`。
- 比较 Base/SFT/GRPO: 找 `compare` 或 `*_compare_report.md`。

## 指标检查

不要只看 exact match。至少同步检查:

- format rate
- single final answer rate
- repeat-like rate
- average response length
- rule reward 或 RM score
