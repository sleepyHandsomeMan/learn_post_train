# datasets 目录说明

本目录存放训练、验证、派生训练集和数据预览。

## 子目录

| 目录 | 阶段 | 说明 |
|---|---|---|
| `gsm8k_sft/` | SFT | GSM8K SFT messages 数据和固定验证集 |
| `gsm8k_grpo/` | GRPO/PPO | RL prompt、分桶审计、聚焦训练集、预览 |

## 数据格式

当前项目主要使用 `messages` parquet:

```text
messages = [
  {"role": "user", "content": "... Let's think ... ####"},
  {"role": "assistant", "content": "... #### 42"}
]
```

自定义 GRPO 训练器会从 assistant 回答中抽取 `####` 后的 ground truth。

当前项目不使用 `prompt/reward_model/extra_info` 形式的 verl 数据。历史导出已隔离，不参与训练。

## Git 规则

- 小型固定数据和人工预览可以提交。
- 大型派生 parquet、全量分桶、临时 probe 数据默认不提交。
- 新增数据集必须在对应子目录补 README，说明来源、字段、用途和是否可训练。
