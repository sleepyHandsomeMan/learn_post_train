# 后训练练习问题排查与评估复盘笔记

> 用途: 记录练习项目中遇到的问题、排查路径、修复动作、评估指标变化和复盘结论。  
> 任务主线: GSM8K 数学问答。  
> 模型阶段: Base -> SFT -> Reward Model -> RLHF/GRPO/PPO -> Evaluation。  
> 维护方式: 每次新增实验、报错、评估或修复后，按模板追加一条记录。

---

## 1. 使用原则

这份笔记不是单次实验报告，而是长期经验账本。

每条记录都尽量回答 5 个问题:

1. 当时处在哪个阶段: Base、SFT、RM、RLHF 还是 Evaluation。
2. 看到的现象是什么: 日志、样例、指标、报错信息分别是什么。
3. 最初怀疑了什么: 数据、tokenizer、训练配置、推理参数、评估脚本还是模型能力。
4. 实际怎么排查: 用了哪些脚本、命令、对照实验、人工样例。
5. 最终经验是什么: 下次遇到同类现象先看哪里。

记录时不要只写“效果变好了”。要拆成:

- 格式是否变好。
- 正确率是否变好。
- 输出长度是否变化。
- reward 分数是否变化。
- 是否出现重复、乱码、模板化、过长回答。
- 固定验证集和人工样例上是否一致。

---

## 2. 当前固定实验约定

| 项目 | 当前约定 | 备注 |
|---|---|---|
| 主任务 | GSM8K | 先做单一可验证任务 |
| 最终答案格式 | `#### final_answer` | 便于 exact match 和格式检查 |
| 固定验证集 | 待补充 | 例如 `datasets/gsm8k_sft/eval_20.parquet` 或更大的 held-out set |
| 人工观察样例 | 待补充 | 建议固定 20 条 prompt |
| Base checkpoint | 待补充 | 原始 base model |
| SFT checkpoint | 待补充 | 监督微调后模型 |
| RM checkpoint | 待补充 | Reward Model |
| RLHF checkpoint | 待补充 | PPO/GRPO 后 actor |

---

## 3. 总指标台账

每次完成一轮评估，把核心结果汇总到这里。详细报告仍放在 `eval_results/`。

| 日期 | 阶段 | checkpoint | eval set | max tokens | exact match | format rate | avg length | repeat-like | rule reward | rm score | human note | 报告路径 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 待补充 | base | 待补充 | 待补充 | 512 | - | - | - | - | - | - | - | - |
| 待补充 | sft | 待补充 | 待补充 | 512 | - | - | - | - | - | - | - | - |
| 待补充 | rlhf | 待补充 | 待补充 | 512 | - | - | - | - | - | - | - | - |
| 2026-06-21 | sft | qwen3_1d7b old SFT | eval_20 | 512 | 50% | 95% | 1324 chars | 100% | - | - | 格式像样但严重复读 | `eval_results/sft_model/1d7b_old_sft/` |
| 2026-06-21 | sft | qwen3_1d7b eosfix | eval_20 | 512 | 45% | 95% | 1557 chars | 65% | - | - | rstrip 后仍未解决长输出复读 | `eval_results/sft_model/1d7b_eosfix/` |
| 2026-06-21 | sft | qwen3_1d7b eosfix2 | eval_20 | 512 | 50% | 100% | 308 chars | 0% | - | - | 单一 final answer，复读消失 | `eval_results/sft_model/1d7b_eosfix2/` |

指标解释:

- `exact match`: 最终答案是否匹配 ground truth。
- `format rate`: 是否按要求输出 `#### final_answer`。
- `avg length`: 平均输出长度，需警惕 reward 变高但回答变长。
- `repeat-like`: 重复、循环、乱码或多次输出 final answer 的比例。
- `rule reward`: 规则奖励均值，适合数学任务早期闭环。
- `rm score`: Reward Model 给分均值，必须和真实正确率一起看。
- `human note`: 固定人工样例上的主观观察，重点写失败模式。

---

## 4. 阶段性评估方式演进

### 4.1 Base 评估

Base 阶段要回答的问题:

- 原始模型是否理解任务格式。
- 是否能输出可解析的最终答案。
- 错误主要来自格式、推理、知识，还是长输出失控。

建议记录指标:

| 指标 | 必看原因 | 风险信号 |
|---|---|---|
| exact match | 判断数学答案是否正确 | 格式不错但答案经常错 |
| format rate | 判断是否学会 `####` 约定 | 无法解析最终答案 |
| avg length | 判断是否啰嗦或发散 | 输出很长但正确率低 |
| parse fail rate | 判断评估脚本能否稳定解析 | 指标被解析失败污染 |
| manual samples | 看模型真实失败方式 | 指标正常但样例很差 |

### 4.2 SFT 评估

SFT 阶段要回答的问题:

- SFT 是否把 base 拉到了 GSM8K 任务分布。
- 改善的是格式、答案正确率，还是只是输出更像训练集。
- 是否出现过拟合、重复、乱码、停不下来。

建议记录指标:

| 指标 | 必看原因 | 风险信号 |
|---|---|---|
| train loss | 判断训练是否基本收敛 | loss 降低但验证集不变好 |
| exact match | 判断任务正确率 | SFT 后反而下降 |
| format rate | 判断格式学习效果 | `####` 缺失或重复 |
| single final | 判断是否只输出一次最终答案 | 多次 final answer |
| repeat-like | 判断是否复读或乱码 | max tokens 越大越严重 |
| avg length | 判断输出是否变短或变长 | 过长且无效推理增加 |

### 4.3 Rule Reward / GRPO 评估

Rule reward 阶段要回答的问题:

- rollout -> reward -> update 是否真的跑通。
- 规则奖励上升是否带来真实正确率提升。
- 模型是否开始钻规则漏洞。

建议记录指标:

| 指标 | 必看原因 | 风险信号 |
|---|---|---|
| rule reward mean | 判断规则奖励是否提升 | reward 上升但 exact match 下降 |
| exact match | 防止只优化表面格式 | 正确率不升反降 |
| format rate | 数学任务可解析性 | 只输出模板答案 |
| avg length | 检查长度偏置 | 回答越来越长 |
| KL | 观察 actor 是否偏离 reference | KL 快速增大 |
| entropy | 观察探索是否塌缩 | entropy 过快下降 |

### 4.4 Reward Model 评估

RM 阶段要回答的问题:

- RM 是否真的学会区分 chosen/rejected。
- RM 是偏爱正确答案，还是偏爱长答案、固定格式、训练集模板。
- RM 给分是否能和 rule reward、人工判断一致。

建议记录指标:

| 指标 | 必看原因 | 风险信号 |
|---|---|---|
| pairwise accuracy | 判断 chosen 是否高于 rejected | 接近随机 |
| reward margin | `chosen - rejected` 的平均差 | margin 很小或极端大 |
| calibration samples | 手工 sanity check | 错误长答案得分很高 |
| length correlation | 检查长度偏置 | 越长 reward 越高 |
| format-only negative | 检查是否只看格式 | 格式正确但答案错仍高分 |

RM sanity check 固定样例:

| prompt | response 类型 | 期望 reward |
|---|---|---|
| 待补充 | 标准正确答案 | 高 |
| 待补充 | 错误答案 | 低 |
| 待补充 | 空答案 | 低 |
| 待补充 | 很长但错误 | 不应高 |
| 待补充 | 格式正确但答案错 | 低于真正正确答案 |

### 4.5 RLHF / PPO / GRPO 评估

RLHF 阶段要回答的问题:

- actor 是否在 reward model 指导下真正变好。
- reward 上升是否对应真实验证集指标上升。
- KL、entropy、长度、模板化是否处于可控范围。

建议记录指标:

| 指标 | 必看原因 | 风险信号 |
|---|---|---|
| rm score | 判断优化目标是否上升 | rm score 上升但正确率下降 |
| exact match | 判断真实任务收益 | reward hacking |
| format rate | 保持可解析输出 | 格式退化 |
| avg length | 检查长度偏置 | 回答明显变长 |
| KL | 防止 actor 跑偏 | KL 快速增大 |
| entropy | 防止策略塌缩 | entropy 很快变低 |
| clip fraction | 判断 PPO 更新幅度 | 长期过高 |
| critic loss | 判断 value 学习稳定性 | 爆炸或剧烈震荡 |
| human win rate | 人工对比 SFT/RLHF | 主观质量下降 |

---

## 5. 问题排查记录模板

复制下面模板追加新问题。

```markdown
### YYYY-MM-DD 问题标题

- 阶段: Base / SFT / RM / RLHF / Evaluation
- 相关模型: 
- 相关数据: 
- 相关脚本/配置: 
- 相关输出路径: 

#### 现象

- 日志现象:
- 指标现象:
- 样例现象:
- 是否可复现:

#### 初始判断

- 怀疑 1:
- 怀疑 2:
- 怀疑 3:

#### 排查过程

| 步骤 | 动作 | 观察到的证据 | 结论 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

#### 根因

一句话说明真正原因。

#### 修复动作

- 代码/配置修改:
- 重新训练:
- 重新评估:
- 需要保留的产物:

#### 修复前后对比

| checkpoint | exact match | format rate | avg length | repeat-like | rule reward | rm score | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| 修复前 | - | - | - | - | - | - | - |
| 修复后 | - | - | - | - | - | - | - |

#### 经验沉淀

- 下次先检查:
- 不要误判为:
- 后续风险:
```

---

## 6. 单次评估记录模板

```markdown
### YYYY-MM-DD 评估标题

- 阶段: Base / SFT / RM / RLHF
- checkpoint:
- eval set:
- 推理参数: max_new_tokens=, temperature=, top_p=
- 评估脚本:
- 输出文件:

#### 核心结果

| 指标 | 数值 | 解释 |
|---|---:|---|
| exact match | - |  |
| format rate | - |  |
| avg length | - |  |
| repeat-like | - |  |
| rule reward | - |  |
| rm score | - |  |

#### 代表样例

| case id | 结论 | 现象 | 下一步 |
|---|---|---|---|
|  | 正确/错误/格式错/重复 |  |  |

#### 本轮结论

- 真正改善:
- 没有改善:
- 新增风险:
- 下一轮动作:
```

---

## 7. 常见现象排查索引

| 现象 | 优先怀疑 | 检查方法 | 下一步 |
|---|---|---|---|
| SFT 后格式变好但正确率不升 | SFT 只学到格式和任务分布 | 看 exact match 与人工样例 | 增加数据质量检查或后续 rule reward |
| 输出 `####` 多次 | EOS、label mask、stop token、训练数据尾部 | 检查 tokenization 和推理 eos_token_id | 固定停止符并复评 max160/max512 |
| max tokens 越大重复越严重 | 模型不会自然停止 | 对比不同 max_new_tokens | 检查 `<|im_end|>` 学习和 stop 配置 |
| SFT 后出现 base 没有的乱码 | 模型在结束符后继续采样低质量 token | 看 `<|im_end|>` 后 labels、lm_head、原始输出 | mask 结束符后 token，LoRA 覆盖 `lm_head` |
| rstrip 后短输出改善但长输出仍复读 | 只修了文本尾部，没有修 labels 学习目标 | 对比 max160/max512 与 labels 最后有效 token | 在 token 级别 mask `<|im_end|>` 后所有 token |
| EOS 已配置但模型不停 | eos_token_id 与训练中出现的停止符不一致 | 检查 tokenizer.eos_token_id 和 `<|im_end|>` id | 只用训练中学过的 `<|im_end|>` 作为停止符 |
| reward 上升但正确率下降 | reward hacking | 同时看 exact match、长度、样例 | 修改 reward 或加入 hard negative |
| RM 偏爱长答案 | 偏好数据长度偏置 | 计算 reward 与长度相关性 | 构造长但错的 rejected |
| PPO/GRPO 后输出模板化 | entropy 下降或 KL 约束不足 | 看 entropy、KL、人工样例 | 降低 lr、加强 KL、缩短训练 |
| critic loss 爆炸 | value 学习不稳定 | 看 critic loss 曲线 | 降低 critic lr 或检查 reward scale |
| 评估指标异常跳变 | 解析脚本或验证集变化 | 固定 eval set 和 parser | 复跑同一 checkpoint |

---

## 8. 已沉淀经验

### 2026-06-21 SFT 复读/乱码问题复盘

- 阶段: SFT / Evaluation
- 相关模型: Qwen3-0.6B、Qwen3-1.7B
- 相关任务: GSM8K 数学问答
- 相关现象: SFT 后模型会在 `#### final_answer` 后继续复读、输出乱码或多语言碎片。
- 原始记录: `docs/sft_repeat_garble_fix_experience_cn.md`

#### 一句话结论

这次问题不是“模型突然学坏了”，而是 SFT 数据的停止边界、labels、LoRA 覆盖范围和推理停止符没有对齐，导致模型学会了在 `<|im_end|>` 后继续生成。

#### 前因

训练目标是让 base model 学会 GSM8K 的回答格式:

```text
推理过程
#### final_answer
```

Base model 原本格式不稳定、正确率不高，但没有明显复读和乱码。SFT 后，模型虽然更常输出 `####`，却出现了新的异常:

- 在 `#### 18` 后继续输出 `#### 18 beurette aida` 等重复片段。
- 在答案后混入 `erotique`、`KInstruction`、多语言碎片或不可解释文本。
- `max_new_tokens=160` 时问题会被截断掩盖，`max_new_tokens=512` 时复读和乱码暴露得更明显。

这说明 SFT 确实改变了模型行为，但改变的不只是任务格式，还把错误的停止边界也写进了模型。

#### 为什么会出现错误

核心错误链路如下:

```text
chat template 在 <|im_end|> 后追加 \n
  -> tokenization 后 labels 的最后有效 token 可能是 \n
  -> 模型学到 <|im_end|> 后面还要继续生成
  -> LoRA 未覆盖 lm_head，停止 token 的 logit 调整不够
  -> 推理时 eos_token_id 又混入模型没学过的原生 EOS
  -> 长输出时模型越过结束边界，开始复读或采样乱码
```

这不是单点问题，而是训练侧和推理侧共同造成的边界错位。

#### 根因 1: chat template 在 `<|im_end|>` 后自动追加换行

Qwen3 tokenizer 的 `apply_chat_template` 会把样本渲染成:

```text
<|im_end|>\n
```

它不是只有 `<|im_end|>` 一个 token，而是结束符后面还有换行 token。修复前的数据校验显示:

```text
last_label_is_newline: 7473 (100.0%)
```

这意味着模型被训练去预测结束符之后的换行，而不是把 `<|im_end|>` 当作真正终点。推理时它自然不会稳定停在 `<|im_end|>`。

第一次修复是对 `apply_chat_template` 的输出做 `.rstrip()`，也就是 eosfix。它能移除最终文本末尾的换行，所以短输出复读率下降:

```text
1.7B max160 repeat-like: 50% -> 20%
```

但它没有彻底解决问题，因为 chat template 在 user 和 assistant 的每个 `<|im_end|>` 后都可能有换行。只修最后一个文本尾部，不等于修好了所有训练标签里的结束边界。

#### 根因 2: `<|im_end|>` 之后的 token 没有全部 mask

SFT 的训练目标应该让模型学习 assistant response，而不是学习“结束符后还会出现换行、下一个角色标记、更多内容”。

正确做法是在 tokenization 后找到最后一个 `<|im_end|>`，把它之后的所有 label 设为 `-100`:

```python
im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
for j in range(len(labels) - 1, -1, -1):
    if labels[j] == im_end_id:
        labels[j + 1:] = [-100] * len(labels[j + 1:])
        break
```

这里的关键点是 token 级别修复，而不是文本级别修复。文本 `.rstrip()` 只能处理表面空白，labels mask 才能改变模型真正学习的目标。

修复后校验目标应该变成:

```text
last_label_is_im_end: 7473 (100.0%)
```

这个版本记为 eosfix2。它让 1.7B max512 的关键异常直接消失:

```text
repeat-like: 65% -> 0%
single final: 40% -> 100%
```

#### 根因 3: LoRA 未覆盖 `lm_head`

即使 labels 修好了，如果 LoRA 只覆盖 attention 和 MLP，输出层 `lm_head` 仍然保持 base model 的权重。对停止行为来说，这很关键:

- `<|im_end|>` 是否被生成，最终要看输出层 logit。
- 小模型 base 本身不一定强烈偏向在正确位置输出 `<|im_end|>`。
- 中间层 LoRA 的调整可能不足以让停止 token 的概率稳定超过其它 token。

因此在 LoRA target modules 中加入 `lm_head`:

```python
default=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
]
```

Qwen3 的 `lm_head` 和 `embed_tokens` 是 tied weight，PEFT 可能会提示相关警告。这里的经验是: LoRA 模式下添加旁路矩阵，不直接覆盖 base weight；训练可行，但后续 merge adapter 时要特别小心。

#### 根因 4: 推理侧 eos_token_id 与训练目标不一致

评估脚本一开始把 Qwen3 原生 EOS 也放进停止符列表。但训练数据里真正出现并被模型学习的是 `<|im_end|>`，不是原生 EOS。

如果推理时要求模型在一个它没有学过的 token 上停止，就会出现“配置看似有 EOS，但模型还是不停”的错觉。

修复方式是只使用 `<|im_end|>`:

```python
def build_eos_token_ids(tokenizer):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        return im_end_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return None
```

#### 排查过程

| 步骤 | 动作 | 证据 | 结论 |
|---|---|---|---|
| 1 | 对比 base 和 SFT 原始输出 | base 格式差但不复读，SFT 格式更像但复读/乱码 | 问题来自 SFT 链路，不是 base 固有现象 |
| 2 | 对比 max160 和 max512 | 长输出更容易暴露复读，短输出可能只是截断 | 必须同时评估短输出和长输出 |
| 3 | 校验 SFT parquet、chat template、tokenization/labels | labels 最后有效 token 是 `\n` | 训练目标错误地包含结束符后的 token |
| 4 | 做 `.rstrip()` 文本修复 | 短输出改善，长输出仍复读 | 文本尾部修复不够 |
| 5 | 在 labels 中 mask `<|im_end|>` 后所有 token | single final 明显提升，repeat-like 明显下降 | token 级 mask 是关键修复 |
| 6 | 将 `lm_head` 加入 LoRA target modules | 停止行为更稳定 | 输出层需要参与停止 token 调整 |
| 7 | 修正评估 eos_token_id | 推理停止符与训练停止符一致 | 训练和推理边界对齐 |

#### 修复前后指标

| 模型/版本 | max tokens | exact match | format rate | single final | repeat-like | avg hash | avg chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| base 0.6B | 512 | 25% | 0% | 0% | - | - | 1545 |
| 0.6B old SFT | 160 | 45% | 70% | 10% | 60% | 2.80 | 373 |
| 0.6B eosfix | 160 | 50% | 65% | 25% | 45% | 2.75 | 374 |
| 0.6B eosfix | 512 | 50% | 85% | 20% | 80% | 17.15 | 1100 |
| 0.6B eosfix2 | 160 | 35% | 55% | 55% | 5% | 0.55 | 297 |
| 0.6B eosfix2 | 512 | 50% | 85% | 85% | 15% | 0.85 | 477 |
| 1.7B old SFT | 160 | 45% | 60% | 35% | 50% | 1.20 | 395 |
| 1.7B old SFT | 512 | 50% | 95% | 20% | 100% | 6.00 | 1324 |
| 1.7B eosfix | 160 | 40% | 55% | 45% | 20% | 0.80 | 411 |
| 1.7B eosfix | 512 | 45% | 95% | 40% | 65% | 2.70 | 1557 |
| 1.7B eosfix2 | 160 | 50% | 60% | 60% | 0% | 0.70 | 281 |
| 1.7B eosfix2 | 512 | 50% | 100% | 100% | 0% | 1.00 | 308 |

这张表的关键读法:

- old SFT 的 `format rate` 已经很高，但 `repeat-like` 也很高，说明格式指标不能单独代表质量。
- eosfix 只能部分缓解，尤其在 max512 下仍然不稳定。
- eosfix2 解决的是停止边界，所以 `single final` 和 `repeat-like` 改善最大。
- `exact match` 没有同步暴涨，说明这次修复主要解决格式边界和生成稳定性，不等于数学能力大幅提升。

#### 最佳阶段性选择

当前更适合进入 GRPO/PPO 的 SFT baseline:

```text
qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2
```

理由:

- max512 下 `format rate = 100%`。
- max512 下 `single final = 100%`。
- max512 下 `repeat-like = 0%`。
- 平均长度从 old SFT 的 1324 chars 降到 308 chars，更适合作为 RLHF 起点。

0.6B eosfix2 也有明显改善，但 max512 仍有 15% repeat-like。小模型长输出稳定性更弱，后续如果资源允许，优先用 1.7B 继续做 rule reward + GRPO。

#### 工具和路径

| 工具 | 路径 | 用途 |
|---|---|---|
| 数据校验 | `dev_tools/sft/validate_sft_data.py` | 检查 parquet、chat template、tokenization/labels |
| SFT 训练 | `dev_tools/sft/train_lora_sft.py` | LoRA SFT，包含 im_end mask 和 lm_head 配置 |
| SFT 评估 | `dev_tools/sft/evaluate_full_sft_max_tokens.py` | SFT 模型 max tokens 对比评估 |
| Base 评估 | `dev_tools/sft/evaluate_base_max_tokens.py` | Base model 对照评估 |
| 框架版推理 | `post_training_framework/src/ptf/generation.py` | 统一推理停止符逻辑 |

重要产物:

| 模型 | adapter path |
|---|---|
| 0.6B eosfix2 | `post_training_framework/runs/gsm8k_qwen3_0d6b_len768_lr3e-5_ep1_eosfix2/checkpoints/qwen3_0d6b_gsm8k_lora_len768_lr3e-5_ep1_eosfix2` |
| 1.7B eosfix2 | `post_training_framework/runs/gsm8k_qwen3_1d7b_len768_lr2e-5_ep1_eosfix2/checkpoints/qwen3_1d7b_gsm8k_lora_len768_lr2e-5_ep1_eosfix2` |

#### 经验沉淀

- SFT 后出现的新异常，优先检查数据构造和训练标签，不要先归因于模型能力差。
- `train loss` 下降不能证明模型学会停止，必须看 `single final`、`repeat-like` 和原始输出。
- `format rate` 高不代表可用；如果模型输出多个 `####`，格式指标可能会掩盖生成失控。
- stop token 必须训练和推理一致。训练中学的是 `<|im_end|>`，推理就不要指望它自动学会另一个 EOS。
- chat template 是训练数据的一部分。模板多一个换行，模型就可能真的学会多生成一个换行。
- 文本清洗和 token label mask 是两层东西。遇到停止边界问题，要最终检查 labels，而不是只检查渲染文本。
- LoRA target modules 会影响能否改变停止行为。对小模型和 chat stop token，`lm_head` 往往不能漏。
- 每次 SFT 评估至少同时跑 max160 和 max512。短输出可能隐藏复读，长输出更容易暴露停止失败。
- 进入 RLHF 前必须先修好 SFT 的停止边界。否则后续 reward 可能奖励到错误格式，或者 rollout 产生大量无效长输出。

### 2026-06-24 初始化经验账本

- 阶段: Evaluation / 复盘体系建设
- 目标: 建立统一模板，后续把问题、排查、指标演进放到同一份长期笔记中。
- 当前要求: 后续每次记录必须区分 Base、SFT、RM、RLHF，不能只写“模型变好”。

初始经验:

- Base 评估先看任务能不能被稳定解析。
- SFT 评估不能只看 loss，要同时看格式、正确率、长度、重复。
- RM 评估不能只看 pairwise accuracy，要加入错误长答案、格式正确但答案错等 sanity check。
- RLHF 评估不能只看 reward mean，要同时看 exact match、KL、entropy、长度和人工样例。
- 固定验证集不能参与训练，否则后续对比会失真。

相关已有文档:

- `docs/post_training_0_5b_practice_guide_cn.md`
- `docs/sft_repeat_garble_fix_experience_cn.md`
- `eval_results/base_model/`
- `eval_results/sft_model/`
- `eval_results/rule_reward/`

### 2026-06-25 enable_thinking 对 SFT 训练的影响分析

- 阶段: SFT
- 相关模型: Qwen3-0.6B、Qwen3-1.7B
- 相关文件: `post_training_framework/src/ptf/prompting.py`、`train_sft.py`、`rl_dataset.py`

#### 一句话结论

`enable_thinking` 不只是控制"推理时要不要思考"——它直接影响 SFT 训练中 `prompt_text` 与 `full_text` 的 token 对齐。当前 `enable_thinking=False` 是一个 **SFT 训练数据格式与 tokenizer 模板一致性的问题**，不是简单的开关选择。如果改成 `True` 但训练数据里没有 `<think>推理过程</think>`，会导致 SFT 的 mask 边界错位。

#### `<think>` 标签从哪来

空 `<think></think>` 不是我们代码显式写入的，而是 **Qwen3 tokenizer 的 Jinja chat template** 自动生成的。模板末尾的逻辑：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}       ← 这里！
    {%- endif %}
{%- endif %}
```

当 `add_generation_prompt=True` 且 `enable_thinking=False` 时，模板自动在 `<|im_start|>assistant\n` 后面插入空 `<think>` 标签。同时，模板对 assistant 消息的处理（无论 `enable_thinking` 为何值）也会在最终 assistant 回复前包裹 `<think>` 标签。

#### 当前代码中两处触发点

**SFT 训练** (`train_sft.py:48-53`)，构造 prompt 边界：

```python
prompt_text = apply_chat_template_text(
    tokenizer, [messages[0]],
    add_generation_prompt=True,   # ← 触发 assistant 前缀
    enable_thinking=False,        # ← 触发空 <think>
)
```

**RL 数据集** (`rl_dataset.py:47-51`)，构建 GRPO/PPO 的 prompt：

```python
prompt_text = apply_chat_template_text(
    self.tokenizer, [{"role": "user", "content": user_content}],
    add_generation_prompt=True,
    enable_thinking=self.enable_thinking,  # 默认 False → 空 <think>
)
```

#### 为什么当前设置能正确工作

关键在于 `prompt_text` 和 `full_text` 的 token 必须对齐。当前 `enable_thinking=False` 时：

```
prompt_text (add_generation_prompt=True, enable_thinking=False):
  <|im_start|>user\n...题目...<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\n
                            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                            空 think 标签在 prompt 区域

full_text (add_generation_prompt=False):
  <|im_start|>user\n...题目...<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\nJanet sells...#### 18<|im_end|>
                            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                            full_text 的 assistant 部分也同样有这些 token
```

两边都有空 `<think>` 标签，`prompt_len` 正好覆盖到 `</think>\n\n` 结束。mask 边界对齐：
- `<think>` 等 token 的 label 都是 `-100`（不参与 loss）
- `</think>\n\n` 之后第一个 token（`Jan`）的 label 才不等于 `-100`

#### 如果改成 enable_thinking=True 会怎样

```
prompt_text (add_generation_prompt=True, enable_thinking=True):
  <|im_start|>user\n...题目...<|im_end|>\n
  <|im_start|>assistant\n
                          ↑ 结束了！没有 think 标签

full_text (不变，模板对 assistant 消息始终包裹 think):
  <|im_start|>user\n...题目...<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\nJanet sells...
                          ↑ prompt_len 只到这里
                            ← <think>\n\n</think>\n\n 漏进 supervised 区域!
```

后果是 `<think>\n\n</think>\n\n` 这四条 token 的 label 变成 token_id，模型被逼着学习预测这些 token。但训练数据中并没有真实的推理内容填入 `<think>` 标签——它们只是模板塞进去的空壳。

#### 什么时候可以改

| 条件 | 动作 |
|---|---|
| 训练数据没有 `<think>推理过程</think>` | **保持 `enable_thinking=False`**，不要改 |
| 训练数据有带思考链的标注 | 可以改为 `True`，但需要同时验证 mask 对齐 |
| GRPO 阶段想用思考链 | SFT 先改、数据先有，GRPO 再跟进 |

#### 经验沉淀

- `enable_thinking` 控制的是 chat template 行为，不是独立的"训练开关"。改了它等于改了 tokenizer 的输出格式。
- SFT 的 `prompt_text` 和 `full_text` 必须 token 级对齐，否则 mask 边界偏移会导致模型学到预测模板符号。
- 改 tokenizer 的行为参数之前，先看 chat template Jinja 源码，确认改了什么、影响哪些函数调用。
- 与 §8 中 SFT 复读问题的根因一脉相承：**chat template 是训练数据的一部分**，不能把它当成无副作用的渲染器。
