# GSM8K GRPO 派生数据说明

本目录只放由评估、分桶或筛选脚本生成的派生数据。默认不进入 git。

## 子目录

| 目录 | 说明 |
|---|---|
| `usable_full7473_mixed_plus_boundary/` | 当前推荐的 full7473 GRPO 可训练 messages 数据 |
| `focus_train256/` | train256 greedy miss / oracle hit 聚焦样本 |
| `probes/` | 过拟合诊断集和其他 probe 数据 |
| `legacy_verl_exports_do_not_use/` | 历史 verl 导出，仅用于留档 |

## 当前自定义 GRPO 推荐文件

```text
derived/usable_full7473_mixed_plus_boundary/
  train_full7473_grpo_usable_mixed_plus_boundary.messages.parquet
  train_full7473_grpo_usable_mixed_plus_boundary.messages.meta.json
  train_full7473_grpo_usable_mixed_plus_boundary_messages_preview.md
```

## 注意

- `messages.parquet` 用于当前项目自定义 GRPO 训练器。
- `legacy_verl_exports_do_not_use/` 不参与数据构建、训练或评估。
- 派生文件的 meta 可能记录生成时原始输出路径，移动后以本 README 的当前路径为准。
