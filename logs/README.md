# logs 目录说明

本目录保存长运行任务和 dashboard 的 stdout/stderr。默认不进入 git。

## 当前结构

| 路径 | 说明 |
|---|---|
| `full_train_bucket_audit/` | 全量训练集 greedy/oracle 分桶审计日志 |

## 归档规则

长任务建议使用:

```text
logs/<run_name>/
  stdout.log
  stderr.log
  command.txt
```

dashboard 日志建议使用:

```text
logs/dashboard/
  stdout.log
  stderr.log
```

## 注意

- 日志是排错证据，不是实验结论。
- 结论性内容应整理到 `docs/` 或评估报告中。
- 大日志不进 git。
