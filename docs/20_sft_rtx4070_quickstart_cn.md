# RTX 4070 单卡 SFT 实操指南

这份文档用于在本项目中准备 SFT 数据集, 并对 `yhy/models/base/qwen3_0d6B` 这样的 base model 做第一轮监督微调。

目标不是追求效果最大化, 而是先在 RTX 4070 上跑通:

```text
base model -> SFT dataset -> LoRA/QLoRA SFT -> 保存 adapter/checkpoint -> 用同一批 prompt 对比 base 和 sft
```

## 1. 推荐路线

RTX 4070 常见显存是 12GB。对 0.6B 模型来说, 全参数 SFT 可能能跑, 但学习阶段更推荐 LoRA 或 QLoRA:

```text
首选: Transformers + TRL + PEFT + LoRA/QLoRA
原因: 单卡最稳, 显存可控, 排错简单。

进阶: verl 的 SFT trainer + LoRA
原因: 更贴近后续 verl/RLHF 流程, 但环境和显存调参更复杂。
```

建议第一轮:

- 模型: `D:/learnAI/verl/yhy/models/base/qwen3_0d6B`
- 数据: GSM8K SFT 格式, 先抽 500-2000 条
- 方法: LoRA 或 QLoRA
- max length: 512 或 1024
- batch size: 1
- gradient accumulation: 8-32
- epoch: 1
- 保存: adapter 和少量日志

## 2. 准备 Python 环境

当前你已有 `test3` 环境, 且已验证可以加载 Qwen3-0.6B:

```powershell
D:\Anaconda\envs\test3\python.exe -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

安装 SFT 需要的包:

```powershell
D:\Anaconda\envs\test3\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  datasets peft trl accelerate
```

如果后续要尝试 4bit QLoRA, Windows 上 `bitsandbytes` 可能不稳定。RTX 4070 + 0.6B 模型第一轮可以先不用 4bit, 直接 LoRA + fp16。

## 3. 下载并转换 GSM8K SFT 数据

verl 已有 GSM8K SFT 预处理脚本:

```powershell
cd D:\learnAI\verl

D:\Anaconda\envs\test3\python.exe examples\data_preprocess\gsm8k_multiturn_sft.py `
  --local_save_dir D:\learnAI\verl\yhy\datasets\gsm8k_sft
```

生成:

```text
datasets/gsm8k_sft/train.parquet
datasets/gsm8k_sft/test.parquet
```

这个数据的核心字段是:

```text
messages:
  - role: user
    content: 问题 + 输出格式要求
  - role: assistant
    content: 标准解题过程 + 最终答案
```

注意: 普通 `examples/data_preprocess/gsm8k.py` 生成的是 RL/PPO 数据格式, 不是当前 SFT trainer 最方便使用的 `messages` 格式。SFT 先用 `gsm8k_multiturn_sft.py`。

## 4. 推荐训练方法 A: Transformers + TRL + LoRA

这是最推荐你第一轮使用的方法。它不依赖分布式训练, 对 RTX 4070 更友好。

建议把可复用训练脚本放在 `dev_tools/sft` 下:

```text
dev_tools/sft/train_qwen3_0d6b_gsm8k_lora.py
```

核心思路:

1. 读取 `train.parquet`
2. 把 `messages` 转成模型 chat template 或简单 prompt-response 文本
3. 用 `trl.SFTTrainer` 训练 LoRA adapter
4. 保存到 `models/sft/qwen3_0d6b_gsm8k_lora`

推荐参数:

```text
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=1e-4
num_train_epochs=1
max_seq_length=512
lora_r=16 或 32
lora_alpha=32 或 64
lora_dropout=0.05
fp16=True
```

如果 OOM:

- `max_seq_length` 从 1024 降到 512
- `gradient_accumulation_steps` 增大, batch 保持 1
- LoRA rank 从 32 降到 16
- 先抽样 500 条跑通

## 5. 推荐训练方法 B: verl SFT Trainer + LoRA

如果你想直接贴近 verl 后续流程, 可以尝试:

```powershell
cd D:\learnAI\verl

D:\Anaconda\envs\test3\python.exe -m torch.distributed.run `
  --standalone `
  --nnodes=1 `
  --nproc_per_node=1 `
  -m verl.trainer.sft_trainer `
  data.train_files=D:\learnAI\verl\yhy\datasets\gsm8k_sft\train.parquet `
  data.val_files=D:\learnAI\verl\yhy\datasets\gsm8k_sft\test.parquet `
  data.messages_key=messages `
  data.train_batch_size=1 `
  data.micro_batch_size_per_gpu=1 `
  data.max_length=512 `
  data.truncation=right `
  model.path=D:\learnAI\verl\yhy\models\base\qwen3_0d6B `
  model.lora_rank=16 `
  model.lora_alpha=32 `
  model.target_modules=all-linear `
  model.use_remove_padding=True `
  optim.lr=1e-4 `
  trainer.default_local_dir=D:\learnAI\verl\models\sft\verl_qwen3_0d6b_gsm8k_lora `
  trainer.project_name=qwen3_0d6b_sft `
  trainer.experiment_name=gsm8k_lora_rtx4070 `
  trainer.total_epochs=1 `
  trainer.logger=console `
  trainer.save_freq=after_each_epoch `
  trainer.test_freq=after_each_epoch
```

如果 verl 路线遇到环境、FSDP、显存问题, 先回到方法 A 跑通。SFT 学习阶段不必一开始强行用 verl trainer。

## 6. 建议先做小样本试跑

第一次不要直接跑完整 GSM8K。建议先抽 500 条:

```powershell
D:\Anaconda\envs\test3\python.exe - <<'PY'
import pandas as pd
from pathlib import Path

src = Path(r"D:\learnAI\verl\yhy\datasets\gsm8k_sft\train.parquet")
dst = Path(r"D:\learnAI\verl\yhy\datasets\gsm8k_sft\train_500.parquet")

df = pd.read_parquet(src).head(500)
df.to_parquet(dst)
print("saved", dst, "rows", len(df))
PY
```

如果没有 `pandas`, 安装:

```powershell
D:\Anaconda\envs\test3\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas pyarrow
```

后续训练时把 `train.parquet` 换成 `train_500.parquet`。

## 7. SFT 前后怎么验证

使用你已有的 notebook:

```text
yhy/notebooks/base_model_manual_test.ipynb
```

先记录 base:

```text
yhy/eval_results/manual_test/base_model_manual_test_outputs/base_model_outputs.txt
```

SFT 后, 修改 notebook 里的 `MODEL_DIR` 或单独写一个加载 LoRA adapter 的 notebook, 用同一批 prompt 再跑:

```text
sft_model_outputs.txt
```

重点比较:

- 是否更像回答问题, 而不是无止境续写
- 是否更稳定输出 `#### final_answer`
- 简单数学题是否更稳
- 是否减少重复和跑题
- 输出长度是否明显变长或变模板化

## 8. 正式 benchmark 对比

SFT 前后还应继续使用 `lm-evaluation-harness` 做横向可比评测。至少保留:

```text
piqa
hellaswag
arc_easy
arc_challenge
winogrande
```

不要只看 SFT loss。SFT loss 下降不一定代表任务能力提升。

## 9. 目录建议

建议保持如下结构:

```text
datasets/
  gsm8k_sft/
    train.parquet
    test.parquet
    train_500.parquet
    eval_20.parquet
models/
  base/
    qwen3_0d6B/
  sft/
    qwen3_0d6b_gsm8k_lora/
eval_results/
  base_model/
  sft_model/
dev_tools/
  sft/
    train_qwen3_0d6b_gsm8k_lora.py
notebooks/
  sft_experiment_notes.ipynb
```

## 10. 下一步行动清单

按顺序执行:

```text
1. 下载/生成 GSM8K SFT 数据到 datasets/gsm8k_sft
2. 抽 500 条小样本
3. 用 Transformers + TRL + LoRA 跑通一轮
4. 用固定 prompt 对比 base 和 SFT
5. 再跑 lm_eval 对比 benchmark
6. 效果和流程都稳定后, 再扩大数据量或接入 verl SFT trainer
```
