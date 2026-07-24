# GSM8K GRPO 审计产物说明

本目录保存 greedy/oracle 分桶审计结果。默认不进入 git。

## 子目录

| 目录 | 说明 |
|---|---|
| `buckets_train256_sft_eosfix2_oracle8_len256/` | 256 条样本的分桶审计 |
| `buckets_train_full7473_sft_eosfix2_oracle8_len256_bs64_bs16/` | 全量 7473 条训练集分桶审计 |

## 文件类型

每个 bucket 审计目录通常包含:

```text
bucket_assignments.csv
bucket_summary.csv
bucket_summary.md
bucket_meta.json
*.messages.parquet
```

## 维护规则

- 审计目录用于判断哪些样本适合 GRPO。
- 可训练集应从审计结果合并到 `../derived/`。
- 分桶脚本只生成当前项目使用的 `*.messages.parquet`。
- 审计 parquet 和 CSV 默认不进 git。
