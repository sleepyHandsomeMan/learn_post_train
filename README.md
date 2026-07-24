# yhy 后训练实验工作区

> 工作区: `D:\learnAI\verl\yhy`
> 主线目标: 用 GSM8K 跑通 Base -> SFT -> GRPO/PPO -> 评估对比的最小后训练闭环。

## 当前定位

这是一个个人学习和实验工作区，不是上游 verl 项目的正式贡献目录。当前主要研究:

- 0.6B/1.7B 级别模型的 GSM8K SFT。
- SFT 后固定验证集评估。
- rule reward + GRPO/PPO 小步训练。
- greedy/oracle 分桶，筛选适合 GRPO 的训练样本。
- Base/SFT/GRPO 的指标和人工样例对比。

## 快速查找

| 想找什么 | 查看 |
|---|---|
| 项目架构和归档规则 | `docs/00_workspace_architecture_map_cn.md` |
| Codex/agent 入口提示 | `AGENTS.md` |
| 后训练学习路线 | `docs/10_learning_path_0p5b_post_training_cn.md` |
| 框架用法 | `post_training_framework/README.md` |
| 数据集说明 | `datasets/README.md` |
| 模型目录说明 | `models/README.md` |
| 评估结果说明 | `eval_results/README.md` |
| 日志说明 | `logs/README.md` |
| 诊断工具说明 | `dev_tools/README.md` |
| 文档索引 | `docs/README.md` |

## 顶层目录

```text
yhy/
  datasets/                 # 固定数据、派生训练集、预览
  models/                   # base model、SFT adapter、GRPO checkpoint
  eval_results/             # 评估 JSONL、summary、Markdown 报告
  logs/                     # 长任务 stdout/stderr 和 dashboard 日志
  docs/                     # 架构、路线、复盘、指标说明
  dev_tools/                # 一次性诊断工具
  notebooks/                # 手工实验 notebook
  post_training_framework/  # 自包含训练/评估框架
```

## 当前关键约定

- 默认任务是 GSM8K 数学问答。
- 最终答案格式是 `#### <number>`。
- 固定验证集不能用于训练。
- 当前自定义 GRPO 训练器读取 `messages` parquet。
- 当前项目的 GRPO 数据构建、预览和训练只使用 `messages` parquet。
- 模型、checkpoint、日志、评估大文件和派生数据默认不进入 git。

## 推荐下一步

1. 先阅读 `docs/00_workspace_architecture_map_cn.md`。
2. 查数据格式看 `datasets/README.md` 和 `datasets/gsm8k_grpo/README.md`。
3. 跑训练或评估前看 `post_training_framework/scripts/README.md`。
4. 改源码前看 `post_training_framework/src/ptf/README.md`。
5. 实验结束后把结论写入 `docs/`，不要只留在日志或聊天记录里。
