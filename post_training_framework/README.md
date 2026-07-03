# Post Training Framework

这是一个面向自学 LLM 后训练的轻量实验框架，目标不是替代 verl、TRL、LLaMA-Factory 这类成熟项目，而是把你当前正在做的 `Base -> SFT -> 推理 -> 评估 -> 对比` 固定成一套可复用流程。

当前默认实验是：

- base model: `models/base/qwen3_0d6B`
- 数据集: `datasets/gsm8k_sft`
- 任务: GSM8K 数学问答
- 输出格式: 最终答案出现在 `####` 后
- SFT 方式: LoRA + assistant-only loss
- 核心评估: exact match、format rate、single final answer、repeat-like、平均输出长度

## 为什么需要这个框架

你之前的实验已经说明一件事：只跑一次训练并看 loss 不够。要判断 SFT 是否真的带来能力提升，至少要固定以下变量：

1. 固定评估集。
2. 固定 base model 的推理方式。
3. 固定 SFT model 的推理方式。
4. 固定答案抽取和格式判断规则。
5. 固定逐题人工观察样本。
6. 每次训练保存参数快照和完整原始输出。

这个框架就是把这些动作固定下来。后续你只需要改配置文件中的数据路径、模型路径、prompt 模式和训练超参，就能快速复跑一轮实验。

## 目录结构

```text
post_training_framework/
  configs/
    gsm8k_qwen3_0d6b.json        # 当前 GSM8K + Qwen3-0.6B 默认配置
  scripts/
    run_base_eval.py             # 运行 base model 推理和评估
    run_sft_train.py             # 训练 LoRA SFT adapter
    run_sft_eval.py              # 运行 SFT adapter 推理和评估
    compare_runs.py              # 对比两份评估 JSONL
    run_pipeline.py              # 串起 base eval -> SFT train -> SFT eval -> compare
    prepare_grpo_data.py         # 合成 verl GRPO/RLHF 所需的 prompt + reward_model 数据
    run_grpo_eval.py             # 评估 merge 后的 GRPO HuggingFace actor
    compare_base_sft_grpo.py     # 对比 Base、SFT、GRPO 三阶段结果
  src/ptf/
    config.py                    # 配置读取、路径解析、命令行覆盖
    data.py                      # parquet/messages 数据读取
    prompting.py                 # prompt 构造和 chat template 渲染
    metrics.py                   # 答案抽取、格式率、复读指标
    reports.py                   # JSONL、summary、Markdown 报告
    generation.py                # 模型加载、推理、评估主流程
    train_sft.py                 # LoRA SFT 训练逻辑
    grpo_data.py                 # SFT messages -> GRPO 数据格式转换
    compare.py                   # Base/SFT/GRPO 对比报告
```

## 一轮实验流程

推荐先分步跑，确认每一步产物都合理，再使用 `run_pipeline.py` 串起来。

### 1. Base model 推理评估

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_base_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --max-new-tokens 160 `
  --max-items 20 `
  --run-name base
```

成功后会生成：

```text
post_training_framework/runs/<experiment_name>/eval/base/
  base_eval_20_max160_full.jsonl
  base_eval_20_max160_summary.json
  base_eval_20_max160_full_report.md
```

其中 `full_report.md` 会保留每道题的原始输出，方便人工检查题意理解、格式和复读。

### 2. 训练 SFT adapter

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_train.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --run-name qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1
```

当前默认训练参数在配置文件中：

```json
{
  "max_length": 768,
  "learning_rate": 0.00003,
  "num_train_epochs": 1.0,
  "gradient_accumulation_steps": 16,
  "lora_r": 16,
  "lora_alpha": 32
}
```

成功后会生成：

```text
post_training_framework/runs/<experiment_name>/checkpoints/<run_name>/
  adapter_config.json
  adapter_model.safetensors
  tokenizer_config.json
  run_config.json
```

`run_config.json` 是本轮训练的参数快照，后续复盘时优先看它。

### 3. SFT model 推理评估

如果刚刚训练完，可以直接指定新 adapter：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_eval.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --adapter-dir post_training_framework\runs\gsm8k_qwen3_0d6b_len768_lr3e-5_ep1\checkpoints\qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1 `
  --max-new-tokens 160 `
  --max-items 20 `
  --run-name sft_lora_len768_lr3e-5_ep1
```

如果只是评估你之前已经训练好的 adapter，可以不传 `--adapter-dir`，配置中默认指向：

```text
models/sft/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_run2
```

### 4. 对比 Base 与 SFT

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\compare_runs.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --base-jsonl post_training_framework\runs\<experiment_name>\eval\base\base_eval_20_max160_full.jsonl `
  --sft-jsonl post_training_framework\runs\<experiment_name>\eval\sft\sft_lora_len768_lr3e-5_ep1_eval_20_max160_full.jsonl `
  --run-name base_vs_sft
```

对比报告会包含：

- 两个模型的 summary 表。
- 每道题的 gold、base_pred、sft_pred。
- 每道题从 base 到 sft 的 exact match 变化：`improved`、`regressed`、`same`。
- 每道题的格式和复读状态。

### 5. 一键流水线

确认分步流程无误后，可以用：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_pipeline.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --run-name gsm8k_qwen3_0d6b_try1 `
  --max-new-tokens 160 `
  --max-items 20
```

如果已经有 base 评估和 adapter，可以跳过部分阶段：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_pipeline.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --skip-base `
  --base-jsonl eval_results\base_model\base_eval_20_max160_full.jsonl `
  --skip-train `
  --adapter-dir models\sft\qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_run2 `
  --run-name existing_len768_lr3e-5_ep1 `
  --max-new-tokens 160 `
  --max-items 20
```

## 配置怎么改

配置文件是 `configs/gsm8k_qwen3_0d6b.json`。

常用字段：

```json
{
  "model": {
    "base_model_dir": "models/base/qwen3_0d6B"
  },
  "dataset": {
    "train_file": "datasets/gsm8k_sft/train.parquet",
    "eval_file": "datasets/gsm8k_sft/eval_20.parquet",
    "max_eval_items": 20
  },
  "prompt": {
    "base_prompt_mode": "plain",
    "sft_prompt_mode": "chat"
  },
  "generation": {
    "max_new_tokens": 160
  },
  "sft": {
    "max_length": 768,
    "learning_rate": 0.00003,
    "num_train_epochs": 1.0
  }
}
```

也可以不改文件，直接用 `--set` 临时覆盖：

```powershell
& D:\Anaconda\envs\test3\python.exe post_training_framework\scripts\run_sft_train.py `
  --config post_training_framework\configs\gsm8k_qwen3_0d6b.json `
  --run-name lr1e-5_len1024_ep1 `
  --set sft.learning_rate=1e-5 `
  --set sft.max_length=1024 `
  --set sft.num_train_epochs=1
```

## 指标解释

| 指标 | 含义 | 用途 |
|---|---|---|
| `exact_match` | 抽取出的预测答案是否等于 gold | 任务正确率 |
| `first_hash_exact_match` | 第一个 `#### 数字` 是否等于 gold | 判断格式答案是否正确 |
| `format_rate` | 是否出现合法 `#### 数字` | 格式遵循能力 |
| `single_final_answer_rate` | 是否只有一个合法最终答案 | 停止边界/EOS 是否干净 |
| `repeat_like_rate` | 是否出现多次 `####`、多次 `The answer is` 或重复行 | 复读风险 |
| `avg_chars` | 平均输出字符数 | 输出长度和啰嗦程度 |

注意：`exact_match` 上升不代表模型真正理解题意；`format_rate` 上升也不代表推理更强。你需要结合 `full_report.md` 做人工逐题判断。

## 当前设计与之前脚本的关系

这个框架复用了你之前已经验证过的关键经验：

- base model 默认用 `plain` prompt，不套 chat template。
- SFT model 默认用 tokenizer 的 chat template，并设置 `enable_thinking=false`。
- SFT 训练只对 assistant tokens 计算 loss。
- 评估时保留完整原始输出，不截断报告文本。
- 答案抽取优先使用第一个合法 `#### 数字`。

因此它不是另起炉灶，而是把已有实验流程模块化。

## 换新数据集时要做什么

最小要求：评估 parquet 支持下面两种格式之一。

第一种，推荐格式：

```text
messages = [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "... #### 42"}
]
```

第二种，简单表格格式：

```text
question | answer
```

如果新任务不是 GSM8K，通常要改三处：

1. `prompt.format_instruction`：改成新任务的输出约束。
2. `metrics.py`：改答案抽取规则。
3. README/实验记录：明确新任务的正确率定义。

## 和热门项目的对比

我查了几个常见项目，它们的方向和这个框架有明显差别：

| 项目 | 主要定位 | 和本框架的区别 |
|---|---|---|
| Hugging Face TRL | 提供 SFT、Reward、DPO、PPO、GRPO 等 trainer；`SFTTrainer` 支持 conversational dataset、chat template、assistant-only loss 和 PEFT | TRL 是训练库，本框架是学习用实验编排层；当前 SFT 训练逻辑可以后续替换成 TRL `SFTTrainer` |
| LLaMA-Factory | 面向大量 LLM/VLM 的统一高效微调框架，支持 SFT、RM、PPO、DPO、KTO、ORPO、LoRA/QLoRA 等 | LLaMA-Factory 更完整、更通用；本框架更透明，适合你逐行理解数据、prompt、loss mask、评估逻辑 |
| Axolotl | 开源后训练/微调工具，强调配置化训练、多模型和多模态支持 | Axolotl 更像生产级训练配置系统；本框架额外强调固定 eval、原始输出报告、Base/SFT 对比 |
| OpenRLHF | 基于 Ray 的高性能 RLHF/Agentic RL 框架，覆盖 PPO、DAPO、REINFORCE++、vLLM、异步 RL 等 | OpenRLHF 偏 RLHF 训练基础设施；本框架当前先服务 Base/SFT/RM/RLHF 学习闭环的前半段 |
| verl | 灵活高效的 RL 后训练框架，强调 PPO/GRPO 等 RLHF/RL 训练的生产级编排 | verl 是你后续做 GRPO/PPO 的主战场；本框架可以作为 verl 前置实验记录和评估层 |

参考：

- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- verl: https://github.com/verl-project/verl

## 为什么不直接用 TRL SFTTrainer

可以用，但我这次没有直接替换，原因是学习阶段更需要透明度：

- 你现在正在排查 EOS、格式、复读和题意稳定性。
- 手写 dataset/collator 可以清楚看到 prompt 部分如何被 mask 成 `-100`。
- 你可以直接验证 `max_length` 是否截掉 assistant 答案。
- 你能把训练输入和评估输入对应起来。

后续如果你想提高工程复用度，可以把 `train_sft.py` 改成 TRL `SFTTrainer`，保留本框架的 eval 和 compare 层。

## 推荐实验节奏

每次只改一个主要变量：

1. 固定 eval_20，先看快速反馈。
2. 对 promising run，再扩大到 200-1000 条固定验证集。
3. 每次只改一个主变量，例如 `learning_rate` 或 `max_length`。
4. 记录：训练参数、EM、format、single answer、repeat、人工观察结论。
5. 如果 EM 上升但复读也上升，要优先看 `full_report.md`，不要只看 summary。

推荐结果表：

```text
checkpoint | exact match | format rate | single answer | repeat rate | avg chars | 人工结论
base       | ...
sft-A      | ...
sft-B      | ...
```

## 后续扩展路线

当前框架已经覆盖 Base/SFT 阶段，并新增了 rule reward、GRPO 数据合成、GRPO 训练入口、checkpoint merge、GRPO 评估和三方对比脚本。下一步可以继续做：

1. 用 `dev_tools/grpo/run_grpo_smoke_qwen3_1d7b.ps1 -DryRun` 检查 GRPO smoke 命令。
2. 实际运行 smoke test，确认 rollout、reward、advantage、actor update、checkpoint 全部正常。
3. 用 `dev_tools/grpo/merge_grpo_checkpoint.ps1` 导出 HuggingFace 模型。
4. 用 `post_training_framework/scripts/run_grpo_eval.py` 评估 GRPO 模型。
5. 用 `post_training_framework/scripts/compare_base_sft_grpo.py` 对比 Base/SFT/GRPO。
6. `reward_data.py`：用 SFT 多采样结果和 gold 自动构造 chosen/rejected。
7. `reward_train.py`：训练 pairwise Reward Model。
8. `reward_eval.py`：做 RM sanity check。

这样就能逐步覆盖完整闭环：

```text
Base -> SFT -> SFT Eval -> Preference Data -> RM -> RM Eval -> GRPO/PPO -> RLHF Eval
```
