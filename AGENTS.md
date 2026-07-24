# yhy 后训练学习工作区

> 作用范围: `D:\learnAI\verl\yhy`

本目录是个人 LLM 后训练实验工作区，当前主线是用 GSM8K 跑通:

```text
Base -> SFT -> Eval -> GRPO/PPO -> RLHF Eval
```

默认用中文沟通。代码注释使用中文。长文档不设固定行数上限；优先维护单一主文档并使用结构化目录和章节组织，只有主题、受众或维护周期明显独立时才拆分文档。

## 快速入口

| 需求 | 优先查看 |
|---|---|
| 项目总架构、目录职责、归档规则 | `docs/00_workspace_architecture_map_cn.md` |
| 后训练学习路线和阶段验收 | `docs/10_learning_path_0p5b_post_training_cn.md` |
| 当前框架脚本和源码职责 | `post_training_framework/README.md` |
| 数据集格式和训练文件选择 | `datasets/README.md` |
| 模型与 checkpoint 管理 | `models/README.md` |
| 评估结果和报告查找 | `eval_results/README.md` |
| 临时日志和 dashboard 输出 | `logs/README.md` |
| 一次性诊断工具 | `dev_tools/README.md` |
| verl 上游架构理解 | `docs/90_verl_source_code_map_cn.md` |

## 顶层目录

```text
yhy/
  datasets/                 # 固定数据集、派生训练集、数据预览
  models/                   # 本地 base model、SFT adapter、GRPO checkpoint
  eval_results/             # Base/SFT/GRPO 推理评估结果
  logs/                     # 长运行脚本日志、dashboard 日志
  docs/                     # 架构、学习路线、实验复盘
  dev_tools/                # 单点诊断、数据检查、GPU 检查脚本
  notebooks/                # 手工实验 notebook
  post_training_framework/  # 自包含后训练实验框架
```

## 当前默认实验约定

- 默认任务: GSM8K 数学问答。
- 默认格式: 最终答案写在 `####` 后。
- 固定验证集不能用于训练。
- 讨论实验收益时区分 `base`、`sft`、`rlhf/grpo`。
- 当前项目的 GRPO 数据构建与训练只使用 `messages` parquet，不使用 verl 数据格式。

## 维护规则

- 新增数据、模型、评估、日志时，先查看对应目录 README。
- 新增长期脚本时放入 `post_training_framework/scripts/`；临时诊断脚本放入 `dev_tools/`。
- 模型权重、checkpoint、日志、评估大文件、派生 parquet 默认不进 git，具体规则见 `.gitignore`。
