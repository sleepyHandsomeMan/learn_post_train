# docs 目录说明

本目录只放可长期复用的学习、架构、复盘和指标文档。

## 文件名前缀

| 前缀 | 分类 |
|---|---|
| `00_` | 工作区架构和目录规则 |
| `10_` | 后训练学习路线和实验审查 |
| `20_` | SFT 操作指南和问题复盘 |
| `30_` | 模型评估与 oracle 审计 |
| `40_` | GRPO 实现、指标和性能分析 |
| `50_` | GRPO/PPO 算法理论专题 |
| `90_` | verl 上游源码架构 |

## 推荐阅读顺序

1. `00_workspace_architecture_map_cn.md`: 当前工作区目录架构和归档规则。
2. `10_learning_path_0p5b_post_training_cn.md`: 0.5B 后训练学习路线。
3. `11_experiment_review_checklist_cn.md`: 实验审稿和风险信号。
4. `45_grpo_v7_preflight_remediation_cn.md`: GRPO 总笔记，统一覆盖训练主线、名词、显存、评估优化和早停案例。
   - `43_grpo_v7_experiment_timeline_cn.md`: v7 全部已完成实验的单一时间线索引（每轮条件/结果/问题/下一轮调整），先看这份再下钻到 45/48/49 的详细推理。
   - `44_grpo_r8_next_round_experiment_plan_cn.md`: 基于 43 号文档第8节待改进方向设计的下一轮(R8)具体实验方案，含分阶段执行顺序和预注册判据。
5. `50_grpo_ppo_policy_gradient_derivation_cn.md`: 从策略梯度、重要性采样和 PPO clipping 推导到当前 GRPO LoRA optimizer update。
6. `30_eval_oracle_stage_guide_cn.md`: greedy/oracle 评估和数据可学性审计。
7. `40_grpo_rule_reward_implementation_cn.md`: GRPO 数据、reward 和实现细节。
8. `41_grpo_metrics_stop_criteria_cn.md`: GRPO 指标、因果排查和终止条件。
9. `42_grpo_deviation_from_standard_cn.md`: 当前 GRPO 实现与标准算法的差异清单，含 loss 长度偏置等已知技术缺口和 formal500 实证。
10. `48_grpo_causal_control_experiment_plan_cn.md`: 在总笔记 step 211 复盘之后，给出控制变量矩阵、分阶段门禁和因果判定规则。
10. `49_grpo_causal_experiment_operations_cn.md`: 多实验编排、连续 checkpoint、汇总和 dashboard 操作。
11. `90_verl_source_code_map_cn.md`, `91_verl_architecture_diagram_cn.md`: verl 源码和架构。

## 文档分类

| 类别 | 文件 |
|---|---|
| 项目架构 | `00_workspace_architecture_map_cn.md` |
| 学习路线 | `10_learning_path_0p5b_post_training_cn.md`, `11_experiment_review_checklist_cn.md` |
| SFT 指南与复盘 | `20_sft_rtx4070_quickstart_cn.md`, `21_sft_eos_repeat_fix_case_cn.md` |
| 评估与 oracle | `30_eval_oracle_stage_guide_cn.md` |
| GRPO 总笔记 | `45_grpo_v7_preflight_remediation_cn.md` |
| GRPO 实现与指标规范 | `40_grpo_rule_reward_implementation_cn.md`, `41_grpo_metrics_stop_criteria_cn.md`, `42_grpo_deviation_from_standard_cn.md` |
| GRPO 实验时间线 | `43_grpo_v7_experiment_timeline_cn.md` |
| GRPO 下一轮实验方案 | `44_grpo_r8_next_round_experiment_plan_cn.md` |
| GRPO/PPO 公式推导 | `50_grpo_ppo_policy_gradient_derivation_cn.md` |
| GRPO 因果诊断实验 | `45_grpo_v7_preflight_remediation_cn.md` 第二部分, `48_grpo_causal_control_experiment_plan_cn.md`, `49_grpo_causal_experiment_operations_cn.md` |
| verl 架构 | `90_verl_source_code_map_cn.md`, `91_verl_architecture_diagram_cn.md` |

## 维护规则

- 新增阶段性经验写到 `docs/`，不要只留在聊天记录。
- 文件名使用 `<两位分类号>_<明确英文主题>_cn.md`，同类文档继续递增编号。
- 同一训练主线的新知识优先补充到已有主文档，并同步更新目录；不因行数增加而拆分。
- 只有主题、受众、事实来源或维护周期明显独立时才新建文档，并在主文档中保留入口。
- 大型报告优先放在 `eval_results/`，这里只保留可复用结论。
