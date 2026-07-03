# yhy 后训练实验工作区 README

> 最后更新: 2026-06-22  
> 工作区路径: `D:\learnAI\verl\yhy`  
> 目标: 记录本项目的目录架构、实验流程、关键资产和修改历史，方便后续复盘与继续迭代。

## 1. 项目定位

本工作区是一个自学 LLM 后训练的个人实验项目，围绕 GSM8K 数学问答任务，逐步跑通以下闭环：

```text
Base model
  -> 固定验证集评估
  -> SFT
  -> SFT 后评估
  -> Base/SFT 对比
  -> 构造偏好数据
  -> Reward Model
  -> PPO/GRPO/RLHF
  -> RLHF 后评估
```

当前已重点完成 Base/SFT 阶段，包括：

- 固定 GSM8K SFT 数据集和 `eval_20` 验证集。
- 评估 Qwen3-0.6B、Qwen3-1.7B base model。
- 训练多版 LoRA SFT adapter。
- 定位并修复 SFT 输出复读、乱码、EOS/停止边界问题。
- 搭建 `post_training_framework/`，用于复用训练、推理、评估和对比流程。

## 2. 顶层目录结构

```text
yhy/
  datasets/                 # 数据集和固定验证集
  models/                   # 本地 base model 与训练产物
  eval_results/             # 推理、评估、对比报告
  dev_tools/               # 开发工具和诊断脚本
  notebooks/                # 实验 notebook 和手工测试记录
  docs/                     # 学习文档、架构说明、经验复盘
  post_training_framework/  # 配置驱动的后训练实验框架
  AGENTS.md                 # 当前工作区的 Codex 协作规则
  CLAUDE.md                 # 同步的协作规则备份
```

核心原则：

- `datasets/` 存数据，不存模型。
- `models/` 存长期保留的模型和 adapter。
- `eval_results/` 存长期保留的评估产物。
- `post_training_framework/` 存框架源码、配置和入口脚本，不作为长期模型仓库。
- 固定验证集不要用于训练，否则无法判断模型是否真的变好。

## 3. 数据目录

```text
datasets/gsm8k_sft/
  train.parquet          # GSM8K SFT 训练数据，约 7473 条
  test.parquet           # GSM8K 测试数据
  train_500.parquet      # 早期 500 条子集实验
  eval_20.parquet        # 固定人工观察验证集，20 条，不参与训练
```

当前默认任务是 GSM8K 数学问答，要求模型在 `####` 后输出最终数字答案。

GRPO 阶段数据将放在：

```text
datasets/gsm8k_grpo/
  train.parquet
  eval_20.parquet
  smoke_train_32.parquet
  README.md
```

后续新增数据集时，建议按以下结构放置：

```text
datasets/<task_name>/
  train.parquet
  test.parquet
  eval_fixed.parquet
  README.md              # 说明数据来源、字段格式、是否可用于训练
```

## 4. 模型目录

```text
models/
  base/                   # 原始 base model，原则上不修改
  sft/                    # LoRA SFT adapter 和训练快照
  grpo/                   # GRPO/RLHF 阶段产出的 actor checkpoint 或 adapter
```

### 4.1 Base Model

```text
models/base/
  qwen3_0d6B/             # Qwen3-0.6B base model
  qwen3_1d7B/             # Qwen3-1.7B base model
  qwen3d5_0d8B/           # Qwen3.5-0.8B，含视觉相关文件
```

### 4.2 SFT Adapter

```text
models/sft/
  qwen3_0d6b_gsm8k_lora_500/
  qwen3_0d6b_gsm8k_lora_full_20260611_000923/
  qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1/
  qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_run2/
  qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix/
  qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2/
  qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1/
  qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix/
  qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2/
```

当前最值得继续观察的 SFT adapter：

```text
models/sft/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2/
```

命名建议：

```text
<model>_<task>_<method>_<key_params>_<version>
```

例子：

```text
qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2
```

## 5. 评估结果目录

```text
eval_results/
  base_model/             # Base model 评估结果
  sft_model/              # SFT model 评估结果和版本对比
  grpo_model/             # GRPO/RLHF 阶段评估结果和对比报告
  manual_test/            # 手工测试输出
  rule_reward/            # rule reward 离线验证结果
  train_logs/             # 训练日志
```

### 5.1 Base 评估

```text
eval_results/base_model/
  base_eval_20_max512_full_report.md
  base_eval_20_max160_full.jsonl
  base_eval_20_max512_full.jsonl
  qwen3_1d7b_base_eval_20_max160_full_report.md
  qwen3_1d7b_base_eval_20_max512_full_report.md
  qwen3_1d7b_base_compare_report.md
```

### 5.2 SFT 评估

```text
eval_results/sft_model/
  eosfix_sft_compare_report.md
  eosfix2_sft_compare_report.md
  0d6b_early/
  0d6b_old_sft/
  0d6b_old_sft_run2/
  0d6b_eosfix/
  0d6b_eosfix2/
  1d7b_old_sft/
  1d7b_eosfix/
  1d7b_eosfix2/
```

评估文件建议同时保留三类产物：

```text
*.jsonl                  # 每道题的完整结构化结果
*_summary.json           # 汇总指标
*_report.md              # 方便人工查看的 Markdown 报告，必须包含原始输出
```

核心指标：

| 指标 | 作用 |
|---|---|
| `exact_match` | 判断最终答案是否正确 |
| `format_rate` | 判断是否输出合法 `#### 数字` |
| `single_final_answer_rate` | 判断是否只有一个最终答案 |
| `repeat_like_rate` | 判断是否有复读风险 |
| `avg_chars` | 观察输出长度是否异常 |

## 6. 脚本目录

```text
dev_tools/
  download_hf_model.py
  grpo/
    README.md
  sft/
    train_lora_sft.py
    evaluate_base_max_tokens.py
    evaluate_full_sft_max_tokens.py
    validate_sft_data.py
    generate_eval_reports.py
    build_eval_reports.py
    diagnose_generation.py
    check_eos.py
    test_prompt_format.py
    check_sft_data_and_labels.py
    build_qwen3_1d7b_sft_compare.py
```

`dev_tools/sft/` 是较直接的阶段脚本，适合单点调试和复现实验。

重要脚本说明：

| 脚本 | 作用 |
|---|---|
| `train_lora_sft.py` | 训练 LoRA SFT adapter，已包含 EOS/label 修复逻辑 |
| `evaluate_base_max_tokens.py` | Base model 推理评估 |
| `evaluate_full_sft_max_tokens.py` | SFT adapter 推理评估 |
| `validate_sft_data.py` | 检查 SFT 数据、chat template、label mask 和 EOS |
| `generate_eval_reports.py` | 从 JSONL 生成 Markdown 报告 |
| `diagnose_generation.py` | 诊断生成异常 |
| `check_eos.py` | 检查 tokenizer EOS 和 `<|im_end|>` 设置 |

## 7. 后训练实验框架

`post_training_framework/` 是当前项目的可复用框架层，目标是把 Base/SFT 实验流程固定下来。

```text
post_training_framework/
  README.md
  configs/
    gsm8k_qwen3_0d6b.json
    gsm8k_qwen3_1d7b.json
  scripts/
    run_base_eval.py
    run_sft_train.py
    run_sft_eval.py
    compare_runs.py
    run_pipeline.py
    score_gsm8k_rule_reward.py
  src/ptf/
    config.py
    data.py
    prompting.py
    metrics.py
    reports.py
    generation.py
    train_sft.py
    reward.py
    compare.py
```

框架职责：

- 用配置文件管理模型路径、数据路径、prompt 策略和训练超参。
- 固定 base 和 sft 的推理方式。
- 自动生成 JSONL、summary、Markdown 报告。
- 支持 Base/SFT 逐题对比。
- 为后续 Reward Model 和 RLHF 扩展预留结构。

当前注意事项：

- `post_training_framework/README.md` 是框架说明。
- `README.md` 是整个 `yhy/` 项目的总 README。
- 长期模型产物仍建议归档到 `models/`。
- 长期评估产物仍建议归档到 `eval_results/`。

## 8. 关键实验结论

### 8.1 SFT 的主要收益

在当前 GSM8K 任务中，SFT 的主要收益包括：

- 学会更稳定地输出 `#### 数字` 格式。
- 在固定 eval_20 上提升 exact match。
- 让模型输出更接近训练数据的问答风格。

但 SFT 不等于真正学会数学推理。判断是否有能力提升，需要同时看：

- exact match 是否提升。
- 格式率是否提升。
- 复读率是否下降。
- 输出长度是否异常增长。
- 人工逐题检查是否理解题意。

### 8.2 EOS/复读问题修复路线

已确认的关键修复点：

1. Qwen3 chat template 会在 `<|im_end|>` 后追加换行，训练时需要处理尾部空白。
2. assistant 结束标记之后的 token 不应继续参与 loss。
3. 评估时优先用 `<|im_end|>` 作为停止 token。
4. 对小模型，必要时需要关注 `lm_head` 是否足以学习停止边界。

详细复盘见：

```text
docs/sft_repeat_garble_fix_experience_cn.md
```

### 8.3 当前代表性结果

根据已有评估记录：

| 模型/版本 | max tokens | exact match | format rate | single final | repeat-like |
|---|---:|---:|---:|---:|---:|
| base 0.6B | 512 | 25% | 0% | 0% | 待补 |
| 0.6B old SFT | 160 | 45% | 70% | 10% | 60% |
| 0.6B eosfix2 | 512 | 50% | 85% | 85% | 15% |
| 1.7B old SFT | 512 | 50% | 95% | 20% | 100% |
| 1.7B eosfix2 | 512 | 50% | 100% | 100% | 0% |

结论：当前最优方向不是单纯追求更高 EM，而是保持 EM 的同时显著降低复读和停止边界错误。

## 9. 文档索引

```text
README.md                                    # 本文件，项目总 README
docs/
  post_training_0_5b_practice_guide_cn.md  # 后训练完整学习路线
  verl_code_architecture_cn.md             # verl 代码结构说明
  verl_architecture_diagram_cn.md          # verl 架构图说明
  sft_repeat_garble_fix_experience_cn.md   # SFT 复读/乱码修复经验
  grpo_implementation_guide_cn.md           # GSM8K rule reward + GRPO 实现指南
  README_sft_rtx4070_cn.md                 # RTX 4070 环境下的 SFT 说明
```

建议阅读顺序：

1. `README.md`
2. `docs/post_training_0_5b_practice_guide_cn.md`
3. `post_training_framework/README.md`
4. `docs/sft_repeat_garble_fix_experience_cn.md`
5. `docs/grpo_implementation_guide_cn.md`
6. `docs/verl_code_architecture_cn.md`

## 10. 实验资产归档规则

训练完成后，建议按以下规则归档：

```text
models/<stage>/<run_name>/                 # 模型或 adapter
eval_results/<stage>/<run_name_or_group>/  # 评估结果
eval_results/train_logs/<run_name>/        # 训练日志
```

每个重要 adapter 目录至少应包含：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
run_config.json
README.md
```

每个重要评估目录至少应包含：

```text
*_full.jsonl
*_summary.json
*_full_report.md
```

## 11. 当前修改记录

| 日期 | 修改 | 说明 |
|---|---|---|
| 2026-06-19 | 新增 `post_training_framework/` | 搭建配置驱动的 Base/SFT 训练、推理、评估、对比框架 |
| 2026-06-19 | 整理长期资产目录 | 将数据、模型、评估结果迁移到 `datasets/`、`models/`、`eval_results/` |
| 2026-06-21 | 修复 SFT EOS/复读问题 | 形成 `eosfix`、`eosfix2` 两轮实验和复盘文档 |
| 2026-06-21 | 增加 1.7B SFT 实验 | 对比 0.6B 与 1.7B 在 GSM8K SFT 上的表现 |
| 2026-06-22 | 重写本 README | 将原目录说明升级为项目级 README，用于追踪架构和修改详情 |
| 2026-06-22 | 新增 GRPO 实现指南和目录骨架 | 增加 `datasets/gsm8k_grpo/`、`models/grpo/`、`eval_results/grpo_model/`、`dev_tools/grpo/` |

后续修改本项目时，请同步更新本表，至少记录：

- 新增或迁移了哪些目录。
- 新增了哪些重要脚本或配置。
- 训练了哪些关键模型版本。
- 评估指标是否有实质变化。
- 是否产生了新的风险或经验结论。

## 12. 下一步路线

推荐下一阶段不要急着扩数据，而是补齐 RM/RLHF 的最小闭环：

1. 固定 200 到 1000 条验证集。
2. 用 SFT 模型多采样生成候选答案。
3. 用 gold answer 和格式规则自动构造 chosen/rejected。
4. 训练一个小 Reward Model。
5. 做 Reward Model sanity check。
6. 先用 rule reward 跑 GRPO/PPO。
7. 再接入 Reward Model 做短程 RLHF。
8. 对比 Base、SFT、RLHF 的 EM、format、repeat、长度和人工样例质量。

最终目标不是单点指标好看，而是形成一个可以持续迭代的后训练实验系统。
