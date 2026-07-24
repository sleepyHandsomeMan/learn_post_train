# dev_tools/grpo 目录说明

本目录预留给 GRPO 阶段的一次性诊断脚本。

当前常用 GRPO 诊断脚本仍在:

```text
dev_tools/grpo_rollout_diagnosis.py
```

新增脚本放置建议:

| 脚本类型 | 放置位置 |
|---|---|
| 临时 rollout/reward 诊断 | `dev_tools/grpo/` |
| 正式训练入口 | `post_training_framework/scripts/` |
| 正式数据构建入口 | `post_training_framework/scripts/` |

输出文件不要放在本目录。
