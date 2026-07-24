# GRPO 根因实验执行与产物指南

## 1. 文件入口

矩阵文件：

```text
post_training_framework/configs/gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json
```

干净基础配置：

```text
post_training_framework/configs/gsm8k_qwen3_0d6b_grpo_v7_causal_base_from169.json
```

编排入口：

```text
post_training_framework/scripts/run_grpo_causal_experiments.py
```

实验原理和判定规则见 `docs/48_grpo_causal_control_experiment_plan_cn.md`。

## 2. 只查看首轮计划

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py plan --phase gate10
```

## 3. 启动 C0 与 O1 的10步实验

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py run `
  --phase gate10 `
  --variants c0_fresh_control o1_old_optimizer `
  --seeds 42 `
  --execute
```

没有 `--execute` 时只打印命令，不启动 GPU。

## 4. 单变量全部通过工程门禁后续到30步

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py run `
  --phase screen30 `
  --tiers single `
  --seeds 42 `
  --execute
```

脚本要求每个 trial 存在 `checkpoint-9`，否则阻止续跑。

如果某分支在 step 7 被 guard 终止，不会自动从 checkpoint-169 覆盖重跑。

## 5. 汇总

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py summarize `
  --tiers single `
  --seeds 42
```

输出：

```text
models/grpo/grpo_v7_step169_causal_v1/_orchestration/summary.csv
models/grpo/grpo_v7_step169_causal_v1/_orchestration/summary.md
```

## 6. 最优组多 seed

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_grpo_causal_experiments.py run `
  --phase gate10 `
  --variants c0_fresh_control <candidate_id> `
  --seeds 123 2026 `
  --execute
```

仍按 gate10→screen30→confirm50 顺序推进。

## 7. 产物和连续性

每个 trial 使用：

```text
models/grpo/grpo_v7_step169_causal_v1/
  <variant>/
    seed-<seed>/
      checkpoint-9/
      checkpoint-19/
      checkpoint-29/
      checkpoint-39/
      checkpoint-49/
      logs/
      plots/
      diagnostics/
      run_config.json
  _orchestration/
    configs/
    events.jsonl
    summary.csv
    summary.md
```

保护规则：

- 单 GPU 串行运行，不并行启动多个模型。
- 已完成目标 checkpoint 的 trial 自动跳过。
- 发现非预期中间 checkpoint 时标记 `partial`，拒绝覆盖。
- 续阶段必须从上一阶段精确 checkpoint 使用 `full` resume。
- 每次启动和结束写入 `events.jsonl`。
- 训练 stdout 同时显示并写入阶段 orchestrator log。

## 8. dashboard 与人工观察

训练 CSV 仍兼容现有 dashboard。

观察单个分支：

```powershell
D:\Anaconda\envs\test3\python.exe -B post_training_framework\scripts\run_training_dashboard.py `
  --run-dir models\grpo\grpo_v7_step169_causal_v1\c0_fresh_control\seed-42 `
  --port 8766
```

看板用于观察趋势，不替代汇总：

- live console 可能有刷新滞后。
- 结论以 CSV、checkpoint state、日志终止原因和配置哈希为准。
- 不把平均 response length 直接等同于真实 truncation。

## 9. 实验成本顺序

首轮不建议一次性把所有分支跑到50步。

推荐顺序：

1. C0 与 O1 先跑 Gate10，验证三种状态模式。
2. P1、L1、K1、K2 跑 Gate10。
3. B1 单独跑 Gate10，先确认显存和每步更新次数。
4. Gate10 全部工程达标后，单变量统一续到30步。
5. 只将有信息增益的分支续到50步。
6. 必要时补 L2 学习率剂量。
7. 根据单变量证据选择 I1/I2/I3，而不是全部盲跑。
8. 最后对 C0 和1至2个候选做3 seed 复验。

相对计算量需要注意：

- P1 rollout 成本接近 C0，但 backward/update 约减半。
- B1 rollout 和 backward 数据量约为 C0 的2倍，optimizer update 次数仍相同。
- sample@8 每个验证点有额外推理成本，但所有分支保持一致。

## 10. 正式长训放行条件

控制变量实验结束后，只有同时满足以下条件才进入新一轮正式 GRPO 长训：

```text
根因结论得到单变量和多 seed 支持
+ 候选配置在50步内能把 reference KL 控制在安全区
+ rollout exact 与 greedy/sample EM 不以能力下降换稳定
+ format、EOS、hit-max 无系统退化
+ optimizer/checkpoint 恢复语义明确
+ 10→30→50日志和 checkpoint 连续
+ dashboard 与 CSV 指标语义一致
```

正式长训应该从最终确认的最佳 LoRA 起点和干净 optimizer 出发。

不能从发生 hard KL guard 的 checkpoint-211 强行续训，也不能因为某一分支跑满50步就直接宣称问题已解决。
