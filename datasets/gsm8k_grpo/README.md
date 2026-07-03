# GSM8K GRPO Data

本目录用于保存 verl GRPO/RLHF 训练所需的 GSM8K prompt 数据。

目标文件：

```text
train.parquet
eval_20.parquet
smoke_train_32.parquet
preview_train.md
preview_eval_20.md
preview_smoke_train_32.md
```

每行建议字段：

```text
data_source
prompt
ability
reward_model
extra_info
```

其中：

```python
reward_model = {
    "style": "rule",
    "ground_truth": "42"
}
```

注意：

- `train.parquet` 用于 GRPO 训练。
- `eval_20.parquet` 只用于验证，不参与训练。
- `smoke_train_32.parquet` 只用于 1 到 5 step 的链路冒烟测试。
- `preview_*.md` 用于人工检查数据合成效果，会展示原始 SFT messages 和转换后的 GRPO row。

重新生成命令：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\prepare_grpo_data.py `
  --sft-train-file datasets\gsm8k_sft\train.parquet `
  --sft-eval-file datasets\gsm8k_sft\eval_20.parquet `
  --output-dir datasets\gsm8k_grpo `
  --smoke-size 32 `
  --preview-items 3
```
