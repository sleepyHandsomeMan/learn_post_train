# LoRA SFT 复读/乱码问题诊断与修复经验

> 日期: 2026-06-21
> 模型: Qwen3-0.6B, Qwen3-1.7B
> 任务: GSM8K 数学问答 SFT
> 作者: yhy 自学实验记录

---

## 1. 问题描述

SFT 后模型出现两种典型异常输出:

1. **复读**: `#### 18` 之后重复输出 `#### 18 beurette aida` 等模式数十次
2. **乱码**: 答案后面出现 `呼和浩ط`, `erotique`, `beurette`, `KInstruction`, `保驾护`, `iólaーズ` 等多语言乱码词

而 base model 输出虽然格式不规范、正确率低, 但没有复读和乱码现象。

---

## 2. 根因分析

经过三轮迭代, 发现三个独立但互相叠加的根因:

### 根因 1: chat template 在 `<|im_end|>` 后自动追加 `\n`

**现象**: Qwen3 tokenizer 的 `apply_chat_template` 渲染每条样本后, 文本结尾是 `<|im_end|>\n` (两个 token: 151645 + 198), 而不是单独的 `<|im_end|>`。

**后果**: tokenization 后 labels 的最后一个有效 token 是 `\n` (id=198) 而非 `<|im_end|>` (id=151645)。模型被训练去"在 `<|im_end|>` 之后继续生成 `\n`", 推理时不会在 `<|im_end|>` 处自然停止。

**定位方法**: 用 `validate_sft_data.py` 阶段 3 检查 labels 最后有效 token:

```
last_label_is_newline: 7473 (100.0%)   ← 修复前
last_label_is_im_end:  7473 (100.0%)   ← 修复后
```

**第一次修复 (eosfix)**: 对 `apply_chat_template` 返回的 `full_text` 做 `.rstrip()`。

**效果**: 复读率有所下降 (1.7B max160: 50% → 20%), 但 max512 时复读率仍然 65%。

**为什么不够**: rstrip 只移除最后一个 `<|im_end|>` 之后的 `\n`, 但 chat template 在 **每个** `<|im_end|>` (user 结尾、assistant 结尾) 后面都加了 `\n`。模型仍然从上下文中学到 `<|im_end|>` → `\n` → 继续生成的模式。

### 根因 2: `<|im_end|>` 之后的所有 token 应被 mask

**现象**: chat template 渲染的文本中, `<|im_end|>` 不只在 assistant 结尾出现, 还在 user 结尾出现。每个 `<|im_end|>` 后面都跟着 `\n`, 然后是下一个 `<|im_start|>`。这些 token 都在 labels 中, 模型从所有 `<|im_end|>` 的位置学到"后面还有内容"。

**第二次修复 (eosfix2)**: 在 `_tokenize_one` 中, 找到最后一个 `<|im_end|>` 并将之后的所有 token 设为 `-100`:

```python
im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
for j in range(len(labels) - 1, -1, -1):
    if labels[j] == im_end_id:
        labels[j + 1:] = [-100] * len(labels[j + 1:])
        break
```

**效果**: 1.7B max512 repeat-like 从 65% 降到 **0%**, single final 从 40% 升到 **100%**。

### 根因 3: LoRA 未覆盖 lm_head, `<|im_end|>` 的停止 logit 不够强

**现象**: 即使 labels 修复了, 模型的 lm_head (输出层) 仍然是 base model 权重。LoRA 只覆盖了 attention 和 MLP 的线性层, 对 `<|im_end|>` 的 logit 没有调整。小模型 base 的 lm_head 在 `<|im_end|>` 之后倾向于继续生成内容, LoRA 中间层的调整不足以压制这种倾向。

**修复**: 在 LoRA target_modules 中加入 `lm_head`:

```python
default=["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj", "lm_head"]
```

**注意事项**: Qwen3 的 `lm_head` 和 `embed_tokens` 是 tied (共享权重)。PEFT 会发出警告:

```
Model with `tie_word_embeddings=True` and the tied_target_modules=['lm_head']
are part of the adapter.
```

这在 LoRA 模式下是安全的, 因为 LoRA 不会直接修改 base weight, 只添加旁路矩阵。但 merge adapter 时需要特别注意 (参考 [peft#2018](https://github.com/huggingface/peft/issues/2018))。

### 根因 4 (推理侧): eos_token_id 配置有误

**现象**: 评估脚本中 `build_eos_token_ids` 把 Qwen3 原生 EOS (id=151643) 也加入了 eos 列表。这个 token 在训练数据中从未出现, 模型不知道要在它处停止。

**修复**: 只用 `<|im_end|>` (id=151645) 作为唯一 eos_token_id:

```python
def build_eos_token_ids(tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return None
```

---

## 3. 完整对比数据

| 模型/版本 | max tokens | exact match | format rate | single final | repeat-like | avg hash | avg chars |
|-----------|-----------|-------------|-------------|-------------|-------------|----------|-----------|
| base 0.6B | 512 | 25% | 0% | 0% | — | — | 1545 |
| 0.6B old SFT | 160 | 45% | 70% | 10% | 60% | 2.80 | 373 |
| 0.6B eosfix | 160 | 50% | 65% | 25% | 45% | 2.75 | 374 |
| 0.6B eosfix | 512 | 50% | 85% | 20% | 80% | 17.15 | 1100 |
| **0.6B eosfix2** | **160** | **35%** | **55%** | **55%** | **5%** | **0.55** | **297** |
| **0.6B eosfix2** | **512** | **50%** | **85%** | **85%** | **15%** | **0.85** | **477** |
| 1.7B old SFT | 160 | 45% | 60% | 35% | 50% | 1.20 | 395 |
| 1.7B old SFT | 512 | 50% | 95% | 20% | 100% | 6.00 | 1324 |
| 1.7B eosfix | 160 | 40% | 55% | 45% | 20% | 0.80 | 411 |
| 1.7B eosfix | 512 | 45% | 95% | 40% | 65% | 2.70 | 1557 |
| **1.7B eosfix2** | **160** | **50%** | **60%** | **60%** | **0%** | **0.70** | **281** |
| **1.7B eosfix2** | **512** | **50%** | **100%** | **100%** | **0%** | **1.00** | **308** |

---

## 4. 最佳配置

**1.7B eosfix2 max512** 是目前最优配置:

- exact match: 50% (与之前持平)
- format rate: **100%** (全部输出 `#### 数字` 格式)
- single final: **100%** (全部只输出一个最终答案)
- repeat-like: **0%** (零复读)

---

## 5. 训练产物路径

| model | adapter path |
|---|---|
| 0.6B eosfix2 | `post_training_framework/runs/gsm8k_qwen3_0d6b_len768_lr3e-5_ep1_eosfix2/checkpoints/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2` |
| 1.7B eosfix2 | `post_training_framework/runs/gsm8k_qwen3_1d7b_len768_lr2e-5_ep1_eosfix2/checkpoints/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2` |

---

## 6. 注意事项清单

### 训练数据准备

1. **parquet 无法直接查看**: 用 `validate_sft_data.py` 三阶段校验, 不要依赖文本编辑器
2. **chat template 自动加 `\n`**: 所有 Qwen3 系列 tokenizer 都会在 `<|im_end|>` 后追加 `\n`, 这是 jinja2 template 中的 `{{- '<|im_end|>\n' }}` 行决定的
3. **assistant content 不应包含 `<|im_end|>`**: 数据准备时不需要手动加结束标记, `apply_chat_template` 会自动处理
4. **`####` 后的内容必须干净**: 如果有逗号分隔的大数字 (如 `1,080`), exact match 可能失败

### 训练配置

5. **lm_head 必须加入 LoRA target_modules**: 对 Qwen3 等 tied embedding 模型, 不覆盖 lm_head 就无法有效调整停止行为。PEFT 会警告 `tie_word_embeddings`, 但 LoRA 模式下是安全的
6. **labels 必须 mask `<|im_end|>` 之后的所有 token**: 只做 rstrip 不够, 必须在 labels 中把 `<|im_end|>` 之后的所有 token 设为 `-100`
7. **LoRA rank 和 alpha**: 当前配置 r=16, alpha=32 对 0.6B 和 1.7B 都有效, 但 0.6B 的长输出稳定性仍不如 1.7B

### 推理配置

8. **eos_token_id 只用 `<|im_end|>`**: 不要混入 tokenizer.eos_token_id (151643), 模型从未被训练在该 token 处停止
9. **repetition_penalty 默认 1.0**: 不生效, 需要时设为 1.1-1.3
10. **max_new_tokens 影响复读率**: 短输出 (160) 时复读更容易被截断触发; 长输出 (512) 时模型有空间自然结束, 但需要训练层面确保停止行为

### 模型选择

11. **0.6B 长输出仍然不稳定**: 即使修复了所有根因, 0.6B max512 的 repeat-like 仍有 15%。小模型 attention 容量有限, 长序列后半段的生成质量不可靠
12. **优先用 1.7B**: 相同修复下, 1.7B 的 format rate 和 single final rate 都远高于 0.6B

### 评估方法论

13. **不要只看 train loss**: train loss 下降不代表复读消除, 必须看 format rate、single final rate 和 repeat-like rate
14. **必须对比 max160 和 max512**: 短输出和长输出的行为差异大, 只看一种会遗漏问题
15. **固定验证集**: 评估用固定 20 条样本, 不要用训练集
16. **人工检查样例**: 自动指标 (exact match、format rate) 之外, 逐条看原始输出才能发现乱码词等异常

---

## 7. 下一步

1. 以 **1.7B eosfix2** 为 SFT baseline, 进入 GRPO/PPO 训练阶段
2. 先用 rule reward (GSM8K ground truth 匹配) + GRPO, 验证 rollout → reward → update 闭环
3. 然后构造偏好数据, 训练 Reward Model
4. 用 RM + PPO 做短程 RLHF, 重点监测 reward hacking 和 KL

---

## 8. 工具清单

| 工具 | 路径 | 用途 |
|------|------|------|
| 数据校验 | `dev_tools/sft/validate_sft_data.py` | 三阶段检查 parquet → chat template → tokenization/labels |
| SFT 训练 | `dev_tools/sft/train_lora_sft.py` | 可配置 LoRA SFT, 已含 im_end mask + lm_head |
| SFT 评估 | `dev_tools/sft/evaluate_full_sft_max_tokens.py` | 独立版评估, 已含修正 eos_token_id |
| Base 评估 | `dev_tools/sft/evaluate_base_max_tokens.py` | Base model 评估, 同样修正 eos_token_id |
| 框架版推理 | `post_training_framework/src/ptf/generation.py` | 框架版核心推理模块, 同样修正 |
