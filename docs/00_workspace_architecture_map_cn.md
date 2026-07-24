# yhy 项目架构与归档规则

> 最后更新: 2026-07-09

## 1. 项目目标

本项目用于自学 LLM 后训练，围绕 GSM8K 数学问答跑通一个可复盘的最小闭环:

```text
Base -> SFT -> SFT Eval -> GRPO/PPO -> GRPO Eval -> 对比分析
```

当前重点是 0.6B/1.7B 级别模型、固定验证集、rule reward、GRPO 训练信号诊断。

## 2. 目录总览

| 目录 | 类型 | 职责 | 是否适合进 git |
|---|---|---|---|
| `datasets/` | 数据资产 | 固定数据集、派生训练集、预览报告 | 小型固定数据可进，派生产物默认不进 |
| `models/` | 模型资产 | base model、SFT adapter、GRPO checkpoint | 不进 |
| `eval_results/` | 评估资产 | JSONL、summary、Markdown 报告 | 默认不进 |
| `logs/` | 运行日志 | 长任务 stdout/stderr、dashboard 日志 | 不进 |
| `docs/` | 文档 | 学习路线、架构、复盘、指标解释 | 进 |
| `dev_tools/` | 诊断工具 | 一次性检查、GPU/数据/生成诊断 | 脚本进，输出不进 |
| `notebooks/` | 手工实验 | 交互式探索和手工记录 | 可进，但避免内嵌大输出 |
| `post_training_framework/` | 框架源码 | 配置、训练、评估、报告、GRPO/PPO 入口 | 源码和配置进，runs 不进 |

## 2.1 根目录文件

| 文件 | 职责 |
|---|---|
| `AGENTS.md` | agent 快速入口和目录指引 |
| `README.md` | 人类快速入口 |
| `.gitignore` | 过滤模型、日志、评估产物和派生数据 |
| `CLAUDE.md` | 旧协作规则备份，后续以 `AGENTS.md` 为准 |

## 3. 数据层

`datasets/gsm8k_sft/` 是 SFT 阶段固定数据。

`datasets/gsm8k_grpo/` 是 RL/GRPO 阶段数据，分三类:

| 类型 | 命名 | 用途 |
|---|---|---|
| 固定输入 | `train.parquet`, `eval_20.parquet`, `eval_100.parquet` | 训练或验证入口 |
| 派生训练集 | `derived/` | 聚焦训练、过拟合探针、可用桶合并 |
| 审计产物 | `audits/`, `previews/` | greedy/oracle 分桶和人工预览 |

当前项目的 GRPO 数据构建、预览和训练只读取 `messages` parquet。历史 verl 导出仅归档，不参与当前流程。

## 4. 模型层

`models/` 只存本地模型和 checkpoint:

```text
models/
  base/      # 原始 base model
  sft/       # LoRA SFT adapter
  grpo/      # GRPO/PPO checkpoint 或 adapter
```

模型目录默认不进 git。需要复盘时保留 `README.md`、`run_config.json`、关键指标截图或摘要即可。

## 5. 评估层

`eval_results/` 按模型阶段分:

```text
eval_results/
  base_model/
  sft_model/
  grpo_model/
  rule_reward/
  manual_test/
  train_logs/
```

每次正式评估优先保留三类文件:

```text
*_full.jsonl
*_summary.json
*_report.md
```

判断模型是否变好至少看 exact match、format rate、response 长度、reward、人工样例。

## 6. 框架层

`post_training_framework/` 是可复用实验框架:

```text
post_training_framework/
  configs/     # 模型、数据、prompt、训练参数
  scripts/     # 命令行入口
  src/ptf/     # 可复用 Python 模块
```

`scripts/` 只做参数解析和编排，核心逻辑放在 `src/ptf/`。

## 7. 脚本归档规则

| 脚本类型 | 放置位置 |
|---|---|
| 可长期复用的训练/评估入口 | `post_training_framework/scripts/` |
| 框架内部逻辑 | `post_training_framework/src/ptf/` |
| 临时诊断和一次性分析 | `dev_tools/` |
| 手工探索 | `notebooks/` |
| PowerShell 长任务编排 | `post_training_framework/scripts/` 或对应阶段目录 |

## 8. 命名规则

文档使用两位分类号，避免同类文件只靠相似后缀区分:

```text
<分类号>_<明确主题>_cn.md
```

分类号约定:

```text
00 工作区架构    10 学习路线    20 SFT
30 评估          40 GRPO        90 verl 源码
```

同一分类内按阅读或演进顺序递增，例如 `40_` 是实现、`41_` 是指标、`42_` 是案例复盘。

推荐 run 名包含:

```text
<model>_<stage>_<task>_<key_params>_<eval_or_train_scope>
```

例:

```text
qwen3_0d6b_grpo_v5_rollout8_len256_lr2e-6_eval100
0d6b_eosfix2_oracle8_train_full7473_len256_temp0d7_bs16
```

## 9. Git 规则

应提交:

- 源码、配置、README、架构文档、复盘文档。
- 小型固定样例数据和人工预览文档。

不应提交:

- `models/`
- `eval_results/`
- `logs/`
- 大型 parquet/jsonl/checkpoint。
- dashboard stdout/stderr、临时 probe 文件。

## 10. 新增资产检查清单

新增文件前先判断:

1. 它是源码、文档、固定小数据，还是实验产物?
2. 是否有更合适的阶段目录?
3. 文件名是否包含模型、阶段、关键参数和数据范围?
4. 是否需要 README 记录来源、格式、用途和风险?
5. 是否应该加入 `.gitignore`?
