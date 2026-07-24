# 0.5B Base Model 后训练实践指导书

这份指导书用于模拟一个 0.5B 级别 base model 的完整后训练流程: Base -> SFT -> Reward Model -> RLHF -> Evaluation。目标不是追求 SOTA, 而是在实践中理解后训练能改变什么、不能改变什么, 以及训练过程中应该重点监控哪些风险。

建议选一个 0.5B 左右 base model, 例如 `Qwen/Qwen2.5-0.5B`, 而不是 instruct 版。任务建议先选单一、可验证领域: GSM8K 数学问答、简短指令遵循、HH-RLHF helpful/harmless 对话三选一。

推荐学习路线:

```text
Base model
  -> 基线评估
  -> SFT: 学会回答格式和任务分布
  -> SFT 后评估
  -> 构造偏好数据
  -> 训练 Reward Model
  -> 校准 Reward Model
  -> PPO/GRPO/RLHF
  -> RLHF 后评估
  -> 对比 Base / SFT / RLHF
```

在 verl 中, SFT 入口主要是 `verl.trainer.sft_trainer`, 配置在 `verl/trainer/config/sft_trainer_engine.yaml`, 示例脚本可参考 `examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh`。RLHF 入口可参考 `verl.trainer.main_ppo` / 当前同步 PPO 的 `verl.trainer.main_ppo_sync`, PPO 配置在 `verl/trainer/config/ppo_trainer.yaml`, reward 配置在 `verl/trainer/config/reward/reward.yaml`。

## 1. 明确实验目标

不要一开始就做“通用聊天助手”。0.5B 模型容量有限, 建议先做一个窄任务闭环。

推荐目标:

```text
让 0.5B base model 学会按固定格式回答小学数学题:
- 能读懂题目
- 能输出推理过程
- 最终答案放在 #### 后
- SFT 后格式明显变好
- RLHF 后正确率或格式遵循率进一步提高
```

你应该保留三个模型 checkpoint:

- `base`: 原始模型。
- `sft`: 监督微调后模型。
- `rlhf`: PPO/GRPO 后模型。

验收标准:

- base 经常不会按格式答题。
- SFT 后回答格式、任务理解明显改善。
- Reward Model 对 chosen/rejected 的区分准确率高于随机。
- RLHF 后 reward 均值上升, 但不能只看 reward, 要看真实验证集指标。

关键提问:

- 你希望模型学到的是“格式”、“知识”、“推理策略”, 还是“偏好风格”?
- 你的任务是否有客观答案? 如果没有, Reward Model 是否容易学偏?
- 0.5B 模型容量是否足够支持你选择的任务?

## 2. 准备数据

你需要三类数据。

第一类: SFT 数据。

格式上是 prompt-response 或 messages。比如 GSM8K 可以构造成:

```text
user: 问题 + “请逐步思考, 并在最后用 #### 给出答案”
assistant: 标准解题过程 + “#### 答案”
```

verl 的 RLHF 数据通常需要类似字段:

```text
data_source
prompt
ability
reward_model: { ground_truth: ... }
extra_info
```

这在 `docs/preparation/prepare_data.rst` 里也有说明。

第二类: 偏好数据。

用于训练 Reward Model, 每条样本包含:

```text
prompt
chosen
rejected
```

来源可以有三种:

- 使用公开偏好数据, 如 HH-RLHF。
- 用 SFT 模型生成多个答案, 再用规则或人工标注选 chosen/rejected。
- 对数学任务, 用 ground truth 自动构造偏好: 正确答案为 chosen, 错误答案为 rejected。

第三类: RLHF prompt 数据。

RLHF 阶段只需要 prompt, 模型在线 rollout 生成 response, 然后 reward function 或 reward model 打分。

建议规模:

- SFT: 先用 1k-10k 条。
- Reward Model: 先用 5k-20k 对偏好。
- RLHF prompt: 先用 1k-10k 条。
- 验证集: 固定 200-1000 条, 永远不要拿来训练。

关键提问:

- SFT 数据和 RLHF prompt 是否来自同一分布?
- chosen/rejected 的差异是否足够清晰?
- 如果 reward 数据中 chosen 总是更长, Reward Model 会不会学成“越长越好”?

## 3. 做 Base Model 基线评估

在训练前先评估 base model。否则你无法判断 SFT/RLHF 是否真的有效。

评估维度:

- 格式遵循率: 是否输出 `#### final_answer`。
- 正确率: 最终答案是否匹配 ground truth。
- 平均长度: 是否废话很多。
- 拒答率/乱码率。
- 人工样例观察: 固定 20 个 prompt, 对比不同 checkpoint。

对 0.5B base model, 常见现象是:

- 能续写, 但不稳定地遵循指令。
- 输出格式混乱。
- 数学推理弱。
- 对 chat template 很敏感。

关键提问:

- base model 真的理解 chat 格式吗?
- 你看到的错误是知识不足、格式不对, 还是推理失败?
- 如果 base 已经很差, RLHF 是否应该直接上, 还是先 SFT?

## 4. SFT 阶段

SFT 的目的不是“对齐偏好”, 而是先把 base model 拉到任务分布上。它让模型知道: 用户问这种问题时, 应该用什么格式、什么风格、什么长度回答。

建议设置:

- 模型: 0.5B base。
- 后端: 单机用 FSDP 或普通 DDP/FSDP。
- epoch: 1-3 轮先试。
- 学习率: 小模型可从 `1e-5` 到 `1e-4` 试。
- max length: 先控制在 512-1024。
- batch: 按显存调, 先保证稳定。
- 保存每个 epoch checkpoint。

在 verl 中可参考:

- `verl/trainer/config/sft_trainer_engine.yaml`
- `examples/sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh`

SFT 后你应该看到:

- 输出格式显著改善。
- loss 稳定下降。
- 验证集格式遵循率提升。
- 正确率可能提升, 但不一定巨大。

注意事项:

- 不要训练太久。小数据 + 小模型很容易过拟合。
- 不要只看 train loss, 要看 held-out prompt。
- chat template 必须和推理时一致。
- response 部分才应该算 loss, prompt 通常不应该作为训练目标。

关键提问:

- SFT loss 下降是否等价于模型变好?
- 如果模型学会格式但答案仍错, 这说明 SFT 解决了什么、没解决什么?
- 训练数据答案太长时, 模型是否会学会啰嗦?

## 5. 构造 Reward Model 数据

Reward Model 学的是: 给定 prompt 和 response, 判断 response 好不好。最经典做法是 pairwise reward modeling。

一条训练样本:

```text
prompt
chosen_response
rejected_response
```

训练目标:

```text
reward(prompt, chosen) > reward(prompt, rejected)
```

数学任务中可以这样构造:

- chosen: 答案正确且格式正确。
- rejected: 答案错误、格式错误、没有最终答案、胡乱输出。
- hard negative: 格式正确但答案错。
- easy negative: 完全乱答。

不要只用 easy negative, 否则 Reward Model 会很虚高, RLHF 时没有用。

Reward Model 可以从同一个 0.5B base 或 SFT 初始化, 加一个 scalar value head / sequence classification head。verl 中 reward model 推理侧支持 `AutoModelForSequenceClassification` 风格; 数据侧也有 `verl/utils/dataset/rm_dataset.py` 处理 chosen/rejected 格式。但 reward model 训练脚本你可以自己实现, 核心是 pairwise loss。

推荐验收指标:

- pairwise accuracy: chosen reward > rejected reward 的比例。
- validation accuracy: 必须单独验证。
- reward 分布: chosen/rejected 均值要分开, 但不要极端爆炸。
- 长度相关性: reward 是否和 response length 强相关。
- 人工检查 top reward / low reward 样本。

关键提问:

- Reward Model 学到的是“正确性”, 还是“表面格式”?
- 如果 chosen 普遍比 rejected 长, RM 是否会偏爱长答案?
- RM 验证集准确率很高, 但 RLHF 变差, 可能是什么原因?

## 6. Reward Model 训练

Reward Model 训练推荐先独立于 verl 跑通。你可以自己写训练逻辑, 理解会更深。

训练流程:

```text
输入 prompt + chosen -> reward_chosen
输入 prompt + rejected -> reward_rejected
loss = -log sigmoid(reward_chosen - reward_rejected)
```

实践建议:

- 初始化用 SFT 模型通常比 base 更稳。
- 最后一层输出一个 scalar reward。
- 不需要生成, 只做判别。
- 学习率比 SFT 更小, 例如 `1e-6` 到 `1e-5`。
- 训练 1-3 epoch 起步。
- 做 reward normalization 或至少监控 reward scale。

训练完成后保存 reward model。

你需要做一个 RM sanity check:

```text
同一个 prompt:
- 标准答案 reward 高
- 错误答案 reward 低
- 空答案 reward 低
- 很长但错误的答案 reward 不应过高
- 格式正确但答案错的 reward 应低于真正正确答案
```

关键提问:

- RM 输出的绝对值有意义吗, 还是只有相对排序有意义?
- 如果 RM 把所有答案都打高分, PPO 会发生什么?
- RM 过拟合时, RLHF 会怎样利用它的漏洞?

## 7. RLHF 算法选择

完整 RLHF 可以用 PPO: actor、reference、critic、reward model 全部参与。

但对 0.5B 学习实验, 建议分三档:

第一档: Rule Reward + GRPO。

最简单, 适合数学任务。无需训练 critic, 也可以不用 reward model。优点是稳定、容易看懂。

第二档: Reward Model + GRPO。

你自己训练 RM, 然后用 RM 给 rollout 打分, GRPO 做组内相对优势。比 PPO 简单, 但已经包含“用 RM 做 RLHF”的关键体验。

第三档: Reward Model + PPO。

最完整, 有 actor/ref/critic/reward model。复杂度最高, 调参也更难。

如果目标明确包含“奖励模型也要自己训练”, 建议路线是:

```text
先跑 Rule Reward + GRPO/PPO 验证 RL 管线
再替换成自己训练的 Reward Model
最后尝试完整 PPO
```

关键提问:

- PPO 为什么需要 critic, 而 GRPO 可以不用?
- reference model 的作用是什么?
- 如果没有 KL 约束, actor 会怎么钻 RM 的空子?

## 8. 接入 Reward Model 到 RLHF

在 verl 中, reward 配置在:

- `verl/trainer/config/reward/reward.yaml`

关键字段是:

```text
reward.reward_model.enable=True
reward.reward_model.model_path=你的RM路径
reward.reward_model.rollout.name=vllm 或 sglang
reward.reward_model.enable_resource_pool=True/False
```

如果是判别式 RM, 逻辑是: 把 prompt + rollout response 输入 RM, 取 scalar score 作为 reward。

如果是生成式 RM, 则需要自定义 reward function, 把问题和回答包装成评分 prompt, 让 GenRM 生成分数, 再解析分数。`docs/advance/reward_loop.rst` 对 DisRM/GenRM 都有说明。

初学建议用判别式 RM, 不要一开始做 GenRM。

接入前必须验证:

- RM tokenizer/chat template 是否和训练时一致。
- RM 输入是否包含 prompt 和 response, 而不是只看 response。
- RM 输出 reward 是否落在合理范围。
- batch 推理是否能稳定跑完。
- reward 分布是否有区分度。

关键提问:

- RM 应该看完整 prompt+response, 还是只看 response?
- 如果 RM 的 tokenizer 与 actor tokenizer 不同, 会有什么问题?
- reward 是否需要 clip 或 normalize?

## 9. RLHF 训练阶段

以 PPO 为例, 每一步发生:

```text
1. 从 prompt dataset 采样 prompt
2. actor 通过 vLLM/SGLang rollout 生成 response
3. reward model 给 response 打分
4. reference model 计算 ref_log_prob
5. actor 计算 old_log_prob
6. critic 估计 value
7. 计算 advantage / return
8. 更新 critic
9. 更新 actor
10. 把 actor 新权重同步给 rollout engine
```

你需要重点监控:

- `reward mean`
- `actor/ppo_kl`
- `actor/pg_loss`
- `actor/entropy_loss`
- `critic/vf_loss`
- `critic/vpred_mean`
- response length
- clip fraction
- validation accuracy
- 人工样例质量

小模型建议保守设置:

- actor lr: `1e-6` 级别起步。
- critic lr: 可略大, 如 `1e-5`。
- KL: 一定要监控。
- rollout temperature: 不要太高, 先 `0.7-1.0`。
- response length: 先短一些, 降低不稳定性。
- PPO epoch: 先 1。
- rollout n: GRPO 可用 4 或 8; PPO 可先 1。

你希望看到:

- reward 缓慢上升。
- KL 不爆炸。
- 输出格式不崩。
- 验证集指标不下降。
- 人工观察没有 reward hacking。

危险信号:

- reward 快速升高, 但真实正确率下降。
- response 越来越长。
- 模型输出固定模板, 不管题目。
- KL 快速增大。
- critic loss 爆炸。
- entropy 过快下降, 模型变得单一。

关键提问:

- reward 上升一定代表模型更好吗?
- KL 太小和太大分别意味着什么?
- 为什么 RLHF 可能让 SFT 学到的能力退化?

## 10. 评估设计

你至少要做四组对比:

```text
Base
SFT
SFT + RM scoring
RLHF
```

每组在同一批 validation prompts 上生成答案, 记录:

- 任务正确率。
- 格式遵循率。
- 平均 response 长度。
- RM 平均分。
- rule reward 平均分。
- 人工偏好胜率。

对数学任务, 建议用 rule correctness 作为最终指标, 不要只用 RM 分数。RM 分数只能说明模型更会讨好 RM。

推荐输出一个表:

```text
checkpoint | exact match | format rate | avg length | rm score | human win rate
base       | ...
sft        | ...
rlhf       | ...
```

关键提问:

- RLHF 相比 SFT 提升的是正确率、格式, 还是只是 RM 分?
- RM 分和真实指标相关吗?
- 哪些 prompt 上 RLHF 变好了, 哪些变差了?

## 11. 推荐的完整实验路径

建议你按下面顺序做, 不要跳步。

第 1 轮: 最小闭环。

```text
0.5B base
GSM8K 1k SFT
rule reward
GRPO 或 PPO 小步训练
验证 rollout -> reward -> update 是否跑通
```

目标: 理解 verl 数据流和训练日志。

第 2 轮: 加入 Reward Model。

```text
用 SFT 模型生成多个答案
用 ground truth 自动构造 chosen/rejected
训练 RM
用 RM 给固定验证集打分
检查 RM 是否靠谱
```

目标: 理解 RM 的训练和校准。

第 3 轮: RM-RLHF。

```text
SFT checkpoint 作为 actor 初始模型
SFT checkpoint 作为 reference
自己训练的 RM 作为 reward
跑短程 PPO/GRPO
```

目标: 观察 RM 如何影响 actor。

第 4 轮: 分析 reward hacking。

```text
找出 RM 打高分但答案错误的样本
把这些加入 hard negative
重训 RM
再跑 RLHF
```

目标: 理解 RLHF 真正难点。

第 5 轮: 扩展任务。

```text
从数学任务扩展到 helpful QA 或安全拒答
引入人工偏好或公开偏好数据
比较 rule reward 和 learned reward
```

目标: 理解主观偏好任务为什么更依赖 RM。

## 12. 最重要的注意事项

不要把 SFT 和 RLHF 混为一谈:

- SFT 学“模仿答案”。
- RM 学“评价答案”。
- RLHF 学“优化评价器给出的奖励”。

不要只看训练 reward:

- reward 可能被 hack。
- RM 可能过拟合。
- 模型可能变啰嗦。
- KL 可能失控。

不要一开始就多任务、多数据、多模型:

- 先单任务。
- 先短 response。
- 先小数据。
- 先 rule reward 跑通。
- 再上 RM。
- 最后再追求效果。

## 13. 你最终应该形成的理解

完成这套流程后, 你应该能清楚回答:

- Base、SFT、RLHF 的能力边界分别是什么?
- 为什么 SFT 是 RLHF 的前置阶段?
- Reward Model 为什么会成为 RLHF 的瓶颈?
- PPO/GRPO 中 advantage 是怎么把 reward 变成梯度信号的?
- KL 为什么是防止模型跑偏的关键?
- 为什么 reward 上升不一定代表模型真实能力上升?
- verl 里 actor、rollout、ref、critic、reward model 分别对应训练系统中的哪个角色?

## 附录 A. 0.5B 模型和数据集获取路径清单

这一节用于把模型、评估数据、SFT 数据、Reward Model 数据、RLHF prompt 数据的获取方式集中放在一起。第一次实操时, 建议先按这里的命令把路径准备好, 再回到前面的章节跑训练流程。

### A.1 0.5B base model 下载

严格意义上的 0.5B base model 推荐使用:

- Hugging Face: `Qwen/Qwen2.5-0.5B`
- ModelScope: `Qwen/Qwen2.5-0.5B`

Hugging Face 下载:

```bash
pip install -U huggingface_hub

hf download Qwen/Qwen2.5-0.5B \
  --local-dir ~/models/Qwen2.5-0.5B
```

ModelScope 下载:

```bash
pip install -U modelscope

modelscope download --model Qwen/Qwen2.5-0.5B \
  --local_dir ~/models/Qwen2.5-0.5B
```

如果只是想先跑通 verl 示例, 可以临时使用 instruct 版本:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ~/models/Qwen2.5-0.5B-Instruct
```

需要注意: base model 不适合直接做聊天或指令跟随。完整实验应该是 `Qwen2.5-0.5B` -> SFT -> RLHF。如果用 instruct 版, 你是在使用别人已经后训练过的模型, 更适合快速验证代码链路。

### A.2 评估数据和 RLVR 入门数据

推荐先用 GSM8K 跑通。原始数据集路径:

- Hugging Face: `openai/gsm8k`
- 页面: https://huggingface.co/datasets/openai/gsm8k

verl 预处理:

```bash
python examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

生成文件:

```text
~/data/gsm8k/train.parquet
~/data/gsm8k/test.parquet
```

其中 `train.parquet` 用作 RL/PPO/GRPO 训练 prompt, `test.parquet` 用作 `data.val_files` 评估。这个脚本生成的是 RL 数据格式, 字段包含 `prompt`, `ability`, `reward_model`, `extra_info` 等。

更难一些的数学数据可以用 MATH:

- Hugging Face: `DigitalLearningGmbH/MATH-lighteval`
- 页面: https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval

verl 预处理:

```bash
python examples/data_preprocess/math_dataset.py \
  --local_save_dir ~/data/math
```

生成文件:

```text
~/data/math/train.parquet
~/data/math/test.parquet
```

这类数学任务可以先使用 rule-based reward, 不需要训练 Reward Model, 适合第一轮自学。

### A.3 SFT 数据获取

如果用 GSM8K 做 SFT, 不要直接使用 `gsm8k.py` 生成的数据, 因为当前 SFT trainer 默认需要 `messages` 字段。应使用 SFT 专用脚本:

```bash
python examples/data_preprocess/gsm8k_multiturn_sft.py \
  --local_save_dir ~/data/gsm8k_sft
```

生成文件:

```text
~/data/gsm8k_sft/train.parquet
~/data/gsm8k_sft/test.parquet
```

这个数据里每条样本类似:

```text
messages:
  - role: user
    content: 问题 + 输出格式要求
  - role: assistant
    content: 标准解题过程 + 最终答案
```

SFT 训练时可参考:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/gsm8k_sft/train.parquet \
  data.val_files=$HOME/data/gsm8k_sft/test.parquet \
  data.messages_key=messages \
  model.path=$HOME/models/Qwen2.5-0.5B \
  trainer.default_local_dir=$HOME/ckpts/qwen2_5_0_5b_gsm8k_sft \
  trainer.project_name=gsm8k-sft \
  trainer.experiment_name=qwen2_5_0_5b-base-sft \
  trainer.total_epochs=1 \
  trainer.logger=console
```

通用对话 SFT 可选数据集:

- `HuggingFaceH4/ultrachat_200k`: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- `OpenAssistant/oasst1`: https://huggingface.co/datasets/OpenAssistant/oasst1

如果数据集本身不是 `messages` 列, 需要自己转换成:

```text
messages = [
  {"role": "user", "content": prompt},
  {"role": "assistant", "content": response}
]
```

### A.4 Reward Model 训练数据

Reward Model 训练数据需要偏好对, 标准字段是:

```text
prompt
chosen
rejected
```

最适合入门的是 HH-RLHF:

- Hugging Face: `Dahoas/full-hh-rlhf`
- 页面: https://huggingface.co/datasets/Dahoas/full-hh-rlhf

verl 预处理:

```bash
python examples/data_preprocess/full_hh_rlhf.py \
  --split rm \
  --local_save_dir ~/data/full_hh_rlhf
```

生成文件:

```text
~/data/full_hh_rlhf/rm/train.parquet
~/data/full_hh_rlhf/rm/test.parquet
```

这些 parquet 可用于训练判别式 Reward Model。训练 RM 时至少监控:

- `chosen_reward > rejected_reward` 的 pairwise accuracy。
- 验证集 accuracy。
- reward 和 response length 的相关性。
- RM 是否偏爱更长回答。

其他可选偏好数据:

- `Anthropic/hh-rlhf`: https://huggingface.co/datasets/Anthropic/hh-rlhf
- `HuggingFaceH4/ultrafeedback_binarized`: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- `Skywork/Skywork-Reward-Preference-80K-v0.1`: https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1

Reward Model 评估可以看:

- `allenai/reward-bench`: https://huggingface.co/datasets/allenai/reward-bench

### A.5 RLHF prompt 数据

RLHF/PPO/GRPO 阶段通常只需要 prompt。actor 在线生成 response, 再由 rule reward 或 Reward Model 打分。

使用 HH-RLHF 生成 RL prompt 数据:

```bash
python examples/data_preprocess/full_hh_rlhf.py \
  --split rl \
  --local_save_dir ~/data/full_hh_rlhf
```

生成文件:

```text
~/data/full_hh_rlhf/rl/train.parquet
```

如果使用 learned Reward Model, 可参考 verl 中的 reward model 接入示例:

```bash
TRAIN_FILE=$HOME/data/full_hh_rlhf/rl/train.parquet \
MODEL_PATH=$HOME/ckpts/qwen2_5_0_5b_gsm8k_sft \
REWARD_MODEL_PATH=/path/to/your/reward_model \
bash examples/grpo_trainer/run_mistral_nemo_12b_skyworkrm_fsdp.sh
```

这个示例脚本默认是 Mistral actor 和 Skywork reward model。实际做 0.5B 实验时, 需要把 `MODEL_PATH` 换成你的 0.5B SFT checkpoint, 把 `REWARD_MODEL_PATH` 换成你训练好的 RM 或公开 RM。

如果暂时没有 Reward Model, 新手路线应先走:

```text
GSM8K/MATH prompt
  -> actor rollout
  -> rule-based reward
  -> PPO/GRPO
```

这样可以先理解 RL 训练管线, 避免一开始同时调 SFT、RM、PPO 三个不稳定环节。

### A.6 推荐的第一次实操路径

第一次不要直接做完整通用 RLHF。建议按下面顺序:

```text
1. 下载 Qwen/Qwen2.5-0.5B
2. 生成 ~/data/gsm8k_sft/train.parquet 和 test.parquet
3. 用 GSM8K SFT 数据训练一个 SFT checkpoint
4. 生成 ~/data/gsm8k/train.parquet 和 test.parquet
5. 用 SFT checkpoint 在 GSM8K 上跑 rule reward PPO/GRPO
6. 对比 base / sft / rlhf 三个 checkpoint
7. 再准备 full_hh_rlhf/rm 数据训练 Reward Model
8. 最后把 learned RM 接入 RLHF
```

路径检查清单:

```text
~/models/Qwen2.5-0.5B
~/data/gsm8k_sft/train.parquet
~/data/gsm8k_sft/test.parquet
~/data/gsm8k/train.parquet
~/data/gsm8k/test.parquet
~/data/math/train.parquet
~/data/math/test.parquet
~/data/full_hh_rlhf/rm/train.parquet
~/data/full_hh_rlhf/rm/test.parquet
~/data/full_hh_rlhf/rl/train.parquet
```

如果这些路径都准备好了, 后面的训练问题基本就会集中在 batch size、显存、chat template、reward 设计和评估指标上, 而不是“数据到底从哪里来”。
