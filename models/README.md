# models 目录说明

本目录保存本地模型权重、adapter 和 RL checkpoint。默认不进入 git。

## 子目录

| 目录 | 说明 |
|---|---|
| `base/` | 原始 base model，本项目不修改其权重 |
| `sft/` | LoRA SFT adapter 和训练 checkpoint |
| `grpo/` | GRPO/PPO 训练输出和 checkpoint |

## 推荐命名

```text
<model>_<task>_<method>_<key_params>_<version>
```

例:

```text
qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2
qwen3_0d6b_grpo_v5_rollout8_len256_lr2e-6_eval100
```

## 保留哪些信息

模型权重不进 git，但每个重要 run 建议保留:

```text
README.md
run_config.json
训练命令
关键评估结果路径
```

## 注意

- `base/` 只放下载的原始模型。
- `sft/` 放监督微调 adapter。
- `grpo/` 放 RL 后训练 checkpoint。
- 临时空文件或 notebook probe 不应放在这里。
