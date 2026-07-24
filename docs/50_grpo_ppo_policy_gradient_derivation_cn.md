# 从策略梯度到当前 GRPO/PPO Optimizer Update：完整公式推导与代码映射

> 文档定位：本文专门回答“PPO 的公式如何从策略梯度出发、重要性采样为何出现、PPO clipping 在哪里引入、GRPO 如何替换 advantage，以及本项目代码最终怎样执行 LoRA optimizer update”。
>
> 本文是一份独立的算法理论专题。GRPO 训练主线、实验门禁、早停与整改过程仍以 [`45_grpo_v7_preflight_remediation_cn.md`](45_grpo_v7_preflight_remediation_cn.md) 为主入口。

## 目录

1. [结论先行：当前一次 PPO optimizer update 在做什么](#section-1)
2. [符号、语言模型 MDP 与三个策略对象](#section-2)
3. [从期望回报推导策略梯度定理](#section-3)
4. [重要性采样为什么会进入 PPO](#section-4)
5. [从局部替代目标到 TRPO，再到 PPO clipping](#section-5)
6. [GRPO 如何替代传统 PPO 的 critic advantage](#section-6)
7. [把 PPO/GRPO 目标写成当前代码的精确损失](#section-7)
8. [当前代码如何逐步执行 PPO optimizer update](#section-8)
9. [当前配置为什么每个 GRPO step 有 4 次 optimizer update](#section-9)
10. [AdamW 最终如何修改 GRPO LoRA 参数](#section-10)
11. [哪些步骤是严格恒等式，哪些是工程近似](#section-11)
12. [当前实现的两个重要审计边界](#section-12)
13. [完整伪代码与代码阅读路线](#section-13)
14. [常见误解与复盘检查表](#section-14)

---

<a id="section-1"></a>

## 1. 结论先行：当前一次 PPO optimizer update 在做什么

当前代码中的“PPO optimizer update”不是重新生成回答，也不是重新计算规则奖励。它接收已经生成并打分的 rollout，完成以下工作：

```text
同一 prompt 的多条 rollout
→ 根据组内相对 reward 计算 GRPO advantage
→ 比较 current actor 与本批 old policy 的 token 概率
→ 用 PPO clipped surrogate 限制局部更新幅度
→ 计算 current actor 相对冻结 SFT reference 的 KL 惩罚
→ 得到 total loss
→ backward 得到 GRPO LoRA 梯度
→ gradient clipping
→ AdamW optimizer.step 修改 GRPO LoRA 参数
```

当前实现可以概括为：

\[
\boxed{
\text{GRPO 组内 advantage}
+\text{token-level PPO clip}
+\text{direct reference KL penalty}
+\text{AdamW LoRA update}
}
\]

它没有实现传统 actor-critic PPO 中的：

- critic/value model；
- value loss；
- GAE；
- critic optimizer update。

因此，本文所说的“PPO更新”更准确地指：

> 使用 PPO clipped policy objective 实现的 GRPO actor 更新。

---

<a id="section-2"></a>

## 2. 符号、语言模型 MDP 与三个策略对象

### 2.1 语言模型如何写成 MDP

给定数学题 prompt \(x\)，模型生成回答：

\[
y=(y_1,y_2,\ldots,y_T)
\]

在第 \(t\) 个位置：

\[
s_t=(x,y_{<t})
\]

表示当前状态是 prompt 和已经生成的前缀；动作是下一个 token：

\[
a_t=y_t
\]

策略是：

\[
\pi_\theta(a_t|s_t)
=
\pi_\theta(y_t|x,y_{<t})
\]

状态转移就是把新 token 追加到前缀。对语言模型来说，该转移在给定动作后是确定的。

当前 GSM8K reward 主要在完整回答生成后计算，因此属于 outcome-level reward。代码随后把整条回答的同一个 advantage 广播给该回答的所有 response token。

### 2.2 三个不能混淆的策略

| 记号 | 含义 | 当前代码中的表示 | 主要作用 |
|---|---|---|---|
| \(\pi_\theta\) | current policy | 当前可训练 actor | 接受梯度并被 optimizer 修改 |
| \(\pi_{\mathrm{old}}\) | old policy | 本批 rollout 更新前保存的 actor token log-prob | PPO ratio 的分母，限制批内局部更新 |
| \(\pi_{\mathrm{ref}}\) | reference policy | 冻结的 SFT model | 限制整个 GRPO 过程长期偏离 SFT |

必须保持以下边界：

\[
\pi_{\mathrm{old}}\neq\pi_{\mathrm{ref}}
\]

- old policy 是本批数据的行为策略快照，每个新的 GRPO step 都会刷新；
- reference 是长期锚点，在整个训练过程中冻结；
- importance ratio 比较 current 与 old；
- reference KL 比较 current 与 reference。

### 2.3 当前模型参数结构

当前加载链路是：

```text
Base model
→ 合并 SFT LoRA，得到 SFT model
→ deepcopy 一份冻结为 reference
→ 在 SFT model 上加载或创建可训练 GRPO LoRA，得到 actor
```

因此 optimizer 中只有：

```text
actor.parameters() 中 requires_grad=True 的参数
```

也就是 GRPO LoRA 参数，而不是完整 0.6B 基座参数。

对应代码：

- `post_training_framework/src/ptf/train_grpo.py::load_actor_and_reference()`；
- `post_training_framework/src/ptf/train_grpo.py::GRPOTrainer.__init__()`。

---

<a id="section-3"></a>

## 3. 从期望回报推导策略梯度定理

### 3.1 最初目标：最大化期望轨迹回报

设一条轨迹为：

\[
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)
\]

轨迹总回报记为 \(R(\tau)\)。强化学习目标是：

\[
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)]
\]

写成积分或求和：

\[
J(\theta)
=
\int p_\theta(\tau)R(\tau)d\tau
\]

### 3.2 展开轨迹概率

轨迹概率为：

\[
p_\theta(\tau)
=
p(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)
P(s_{t+1}|s_t,a_t)
\]

模型参数 \(\theta\) 只出现在策略 \(\pi_\theta\) 中，不出现在初始状态分布和环境转移概率中。

### 3.3 对目标求梯度

\[
\nabla_\theta J(\theta)
=
\int
\nabla_\theta p_\theta(\tau)
R(\tau)d\tau
\]

使用 log-derivative trick：

\[
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
\]

代入：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
R(\tau)
\nabla_\theta\log p_\theta(\tau)
\right]
\]

### 3.4 轨迹 log-prob 只保留策略项

对轨迹概率取对数：

\[
\log p_\theta(\tau)
=
\log p(s_0)
+
\sum_t\log\pi_\theta(a_t|s_t)
+
\sum_t\log P(s_{t+1}|s_t,a_t)
\]

因为环境项与 \(\theta\) 无关：

\[
\nabla_\theta\log p_\theta(\tau)
=
\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
\]

所以：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
R(\tau)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
\]

### 3.5 用 return-to-go 降低无关方差

动作 \(a_t\) 不可能影响它之前已经发生的 reward。因此，可以把整条轨迹回报替换为从 \(t\) 开始的 return-to-go：

\[
G_t
=
\sum_{k=t}^{T-1}\gamma^{k-t}r_k
\]

得到 REINFORCE 形式：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
G_t
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
\]

对 GSM8K 的有限长度最终结果奖励，可以把它理解为有限时域 episodic 问题；是否显式写 \(\gamma\) 不改变本文关心的 PPO ratio 推导。

### 3.6 引入 baseline 不改变期望梯度

减去一个只依赖状态、不依赖当前动作的 baseline \(b(s_t)\)：

\[
\mathbb{E}_{a_t\sim\pi_\theta}
\left[
b(s_t)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
=0
\]

因为：

\[
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(a|s)]
&=
\sum_a
\pi_\theta(a|s)
\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\\
&=
\sum_a\nabla_\theta\pi_\theta(a|s)\\
&=
\nabla_\theta 1\\
&=0
\end{aligned}
\]

选择 value function 作为 baseline，可以得到 advantage：

\[
A^{\pi_\theta}(s_t,a_t)
=
Q^{\pi_\theta}(s_t,a_t)-V^{\pi_\theta}(s_t)
\]

最终得到策略梯度定理常见形式：

\[
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}_{s_t,a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t|s_t)
A^{\pi_\theta}(s_t,a_t)
\right]
}
\]

### 3.7 为什么这里没有重要性采样

上述公式的期望本来就是：

\[
(s_t,a_t)\sim\pi_\theta
\]

即数据由当前正在求梯度的策略生成。采样分布和目标分布一致，因此没有必要做分布修正。

重要性采样是在“数据已经来自旧策略，但我们还想继续更新当前策略”时才被引入。

---

<a id="section-4"></a>

## 4. 重要性采样为什么会进入 PPO

### 4.1 一批 rollout 生成后，行为策略被固定

在第 \(k\) 轮开始时，用：

\[
\pi_{\theta_k}=\pi_{\mathrm{old}}
\]

生成一批 rollout。

第一次更新之前：

\[
\theta=\theta_k
\]

数据和当前策略一致。

执行一次 optimizer update 后：

\[
\theta\neq\theta_k
\]

但剩余 mini-batch 和下一个 PPO epoch 仍然来自 \(\pi_{\mathrm{old}}\)。此时如果继续优化，就出现：

```text
采样分布：old policy
目标参数：current policy
```

### 4.2 离散动作上的重要性采样恒等式

对任意函数 \(f(a)\)：

\[
\mathbb{E}_{a\sim\pi_\theta}[f(a)]
=
\sum_a\pi_\theta(a|s)f(a)
\]

乘除以旧策略概率：

\[
=
\sum_a
\pi_{\mathrm{old}}(a|s)
\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}
f(a)
\]

因此：

\[
\boxed{
\mathbb{E}_{a\sim\pi_\theta}[f(a)]
=
\mathbb{E}_{a\sim\pi_{\mathrm{old}}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}
f(a)
\right]
}
\]

定义 importance sampling ratio：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}
\]

### 4.3 完整状态—动作分布还包含状态比率

策略梯度中的状态也由策略决定。严格写法应是：

\[
\begin{aligned}
&\mathbb{E}_{s\sim d^\theta,a\sim\pi_\theta}[f(s,a)]\\
&=
\mathbb{E}_{s\sim d^{\mathrm{old}},a\sim\pi_{\mathrm{old}}}
\left[
\frac{d^\theta(s)}{d^{\mathrm{old}}(s)}
\frac{\pi_\theta(a|s)}{\pi_{\mathrm{old}}(a|s)}
f(s,a)
\right]
\end{aligned}
\]

其中 \(d^\theta(s)\) 是策略 \(\pi_\theta\) 的状态访问分布。

PPO 不显式估计：

\[
\frac{d^\theta(s)}{d^{\mathrm{old}}(s)}
\]

而是依赖局部更新假设：

\[
\pi_\theta\approx\pi_{\mathrm{old}}
\quad\Longrightarrow\quad
d^\theta\approx d^{\mathrm{old}}
\]

这是 PPO 理论到工程实现之间的第一层近似。

### 4.4 从 performance difference 到局部替代目标

performance difference lemma 给出：

\[
J(\pi_\theta)-J(\pi_{\mathrm{old}})
=
\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d^\theta,a\sim\pi_\theta}
\left[
A^{\pi_{\mathrm{old}}}(s,a)
\right]
\]

把新策略状态分布近似为旧策略状态分布：

\[
d^\theta(s)\approx d^{\mathrm{old}}(s)
\]

得到：

\[
L_{\pi_{\mathrm{old}}}(\theta)
\propto
\mathbb{E}_{s\sim d^{\mathrm{old}},a\sim\pi_\theta}
\left[
A^{\pi_{\mathrm{old}}}(s,a)
\right]
\]

再用动作概率重要性采样：

\[
\boxed{
L^{\mathrm{CPI}}(\theta)
=
\mathbb{E}_{s,a\sim\pi_{\mathrm{old}}}
\left[
r_t(\theta)
A_t^{\mathrm{old}}
\right]
}
\]

这里的 CPI 指 conservative policy iteration 风格的局部替代目标。

### 4.5 为什么该替代目标在起点匹配策略梯度

在：

\[
\theta=\theta_{\mathrm{old}}
\]

有：

\[
r_t(\theta_{\mathrm{old}})=1
\]

并且：

\[
\begin{aligned}
\nabla_\theta r_t(\theta)
&=
\nabla_\theta
\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}\\
&=
r_t(\theta)
\nabla_\theta\log\pi_\theta(a_t|s_t)
\end{aligned}
\]

所以在旧策略点：

\[
\begin{aligned}
\nabla_\theta L^{\mathrm{CPI}}(\theta)
\big|_{\theta=\theta_{\mathrm{old}}}
&=
\mathbb{E}_{\pi_{\mathrm{old}}}
\left[
A_t^{\mathrm{old}}
\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
\end{aligned}
\]

它与原始策略梯度在 \(\theta_{\mathrm{old}}\) 处的一阶梯度一致。

因此，重要性采样 ratio 没有推翻策略梯度定理，而是在旧数据上构造一个与原始梯度局部一致的替代目标。

### 4.6 完整轨迹重要性采样为什么不直接使用

语言模型回答概率为：

\[
\pi_\theta(y|x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t|x,y_{<t})
\]

完整序列重要性比率是：

\[
\frac{\pi_\theta(y|x)}{\pi_{\mathrm{old}}(y|x)}
=
\prod_{t=1}^{T}
\frac{\pi_\theta(y_t|x,y_{<t})}
{\pi_{\mathrm{old}}(y_t|x,y_{<t})}
\]

写成 log-prob：

\[
=
\exp
\left[
\sum_{t=1}^{T}
\left(
\log\pi_\theta(y_t|\cdot)
-
\log\pi_{\mathrm{old}}(y_t|\cdot)
\right)
\right]
\]

长序列中大量比率连乘极易爆炸或趋近 0，导致极高方差。当前代码采用 token-level ratio：

\[
r_{i,t}
=
\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}
{\pi_{\mathrm{old}}(y_{i,t}|x,y_{i,<t})}
\]

并在 token 层做 clipping。这是常见的 PPO/GRPO 工程近似，不是完整序列级无偏重要性采样。

---

<a id="section-5"></a>

## 5. 从局部替代目标到 TRPO，再到 PPO clipping

### 5.1 未裁剪 ratio 的风险

未裁剪替代目标为：

\[
L^{\mathrm{CPI}}(\theta)
=
\mathbb{E}[r_t(\theta)A_t]
\]

如果：

\[
\pi_{\mathrm{old}}(a_t|s_t)\approx0
\]

而当前策略提高了该动作概率，那么：

\[
r_t(\theta)\gg1
\]

少数样本可能产生过大的目标值和参数变化。策略一旦离开 old policy 太远，前面使用的局部状态分布近似也会越来越不可靠。

### 5.2 TRPO 的显式信赖域约束

TRPO 解决：

\[
\max_\theta
\mathbb{E}
[r_t(\theta)A_t]
\]

满足：

\[
\mathbb{E}_{s\sim d^{\mathrm{old}}}
\left[
D_{\mathrm{KL}}
(\pi_{\mathrm{old}}(\cdot|s)\|\pi_\theta(\cdot|s))
\right]
\le\delta
\]

它通过显式 trust region 限制新旧策略距离，但需要更复杂的二阶近似和约束优化。

### 5.3 PPO 用裁剪近似信赖域

PPO 定义：

\[
\boxed{
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
}
\]

当前配置：

\[
\epsilon=0.2
\]

所以裁剪区间是：

\[
[1-\epsilon,1+\epsilon]=[0.8,1.2]
\]

### 5.4 正 advantage 时如何裁剪

若：

\[
A_t=1,qquad r_t=1.5
\]

则：

\[
r_tA_t=1.5
\]

\[
\operatorname{clip}(r_t,0.8,1.2)A_t=1.2
\]

取较小值：

\[
\min(1.5,1.2)=1.2
\]

说明模型仍会提高好动作概率，但超过 1.2 倍后的额外提高不再获得目标收益。

### 5.5 负 advantage 时如何裁剪

若：

\[
A_t=-1,qquad r_t=0.5
\]

则：

\[
r_tA_t=-0.5
\]

\[
\operatorname{clip}(r_t,0.8,1.2)A_t=-0.8
\]

取较小值：

\[
\min(-0.5,-0.8)=-0.8
\]

对于最大化目标，这避免模型通过无限降低坏动作概率继续获得额外收益。

### 5.6 clipping 的理论边界

必须区分：

- importance sampling identity 是概率分布变换恒等式；
- 忽略状态分布比率是局部近似；
- PPO clipping 是人为加入的稳定化设计。

裁剪后，目标不再是无偏的重要性采样估计。PPO 有意接受一定偏差，换取更小方差和更稳定的多轮更新。

---

<a id="section-6"></a>

## 6. GRPO 如何替代传统 PPO 的 critic advantage

### 6.1 传统 PPO 的 advantage

传统 PPO 通常使用 critic：

\[
A_t
=
G_t-V_\phi(s_t)
\]

或者 GAE：

\[
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
\]

其中：

\[
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
\]

这需要额外训练 value model。

### 6.2 GRPO 的组内相对 reward

当前项目对同一个 prompt \(x\) 生成 \(G=8\) 条回答：

\[
y_1,y_2,\ldots,y_G
\]

规则 reward 为：

\[
R_1,R_2,\ldots,R_G
\]

组内均值：

\[
\bar R_x
=
\frac{1}{G}
\sum_{i=1}^{G}R_i
\]

当前代码使用 NumPy `std` 的总体标准差口径：

\[
\sigma_x
=
\sqrt{
\frac{1}{G}
\sum_{i=1}^{G}
(R_i-\bar R_x)^2
}
\]

当：

\[
\sigma_x>10^{-8}
\]

时：

\[
\boxed{
A_i
=
\frac{R_i-\bar R_x}{\sigma_x}
}
\]

否则：

\[
A_i=R_i-\bar R_x
\]

如果全组 reward 相同：

\[
R_1=\cdots=R_G
\]

则：

\[
A_1=\cdots=A_G=0
\]

该组不会提供 policy gradient 信号。

### 6.3 advantage 的含义是“相对同题其他回答”

因此：

```text
A_i > 0：该回答比同题组平均水平更好
A_i < 0：该回答比同题组平均水平更差
A_i = 0：该回答没有相对优势信号
```

这不要求跨题 reward 绝对可比，是 GRPO 适合 group sampling 的核心原因之一。

### 6.4 response-level advantage 如何进入 token loss

当前每条回答只有一个 advantage：

\[
A_i
\]

代码把它广播给回答中的所有有效 response token：

\[
A_{i,t}=A_i
\]

于是，正确高奖励回答中的所有token整体被提高概率，低奖励回答中的所有token整体被降低概率。

这是一种 outcome-level credit assignment：

- 它能判断整条回答相对好坏；
- 它不能直接定位哪个中间推理 token 导致最终成功或失败。

对应代码：

- `_compute_rewards()`；
- `_compute_advantages()`；
- `_train_mini_batch()` 中的 `adv_expanded`。

---

<a id="section-7"></a>

## 7. 把 PPO/GRPO 目标写成当前代码的精确损失

### 7.1 token log-prob

对第 \(i\) 条回答的第 \(t\) 个 token，定义：

\[
\ell^\theta_{i,t}
=
\log\pi_\theta(y_{i,t}|x_i,y_{i,<t})
\]

\[
\ell^{\mathrm{old}}_{i,t}
=
\log\pi_{\mathrm{old}}(y_{i,t}|x_i,y_{i,<t})
\]

\[
\ell^{\mathrm{ref}}_{i,t}
=
\log\pi_{\mathrm{ref}}(y_{i,t}|x_i,y_{i,<t})
\]

代码通过 causal LM 的 shifted logits 计算“当前位置预测下一个token”的 log-prob，并用 response mask 排除 prompt 和 padding。

对应函数：

```text
compute_sequence_log_probs()
```

### 7.2 当前/旧策略 log-ratio

\[
\Delta_{i,t}
=
\ell^\theta_{i,t}-\ell^{\mathrm{old}}_{i,t}
\]

当前代码先进行数值裁剪：

\[
\tilde\Delta_{i,t}
=
\operatorname{clip}(\Delta_{i,t},-20,20)
\]

再计算：

\[
r_{i,t}=\exp(\tilde\Delta_{i,t})
\]

这个 `[-20, 20]` 裁剪是数值稳定保护，不等于 PPO 的 `[0.8, 1.2]` objective clipping。

### 7.3 当前代码的 policy loss

设 mini-batch 中所有有效 response token 的集合为 \(\mathcal V\)，则：

\[
\begin{aligned}
L_{\mathrm{policy}}(\theta)
=
-\frac{1}{|\mathcal V|}
\sum_{(i,t)\in\mathcal V}
\min\Big(
&r_{i,t}A_i,\\
&\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
\Big)
\end{aligned}
\]

前面有负号，是因为论文通常写“最大化 surrogate objective”，而 PyTorch optimizer 默认执行“最小化 loss”。

### 7.4 current/reference KL 的低方差估计

当前默认配置：

```text
kl_loss_type = low_var_kl
```

定义：

\[
z_{i,t}
=
\ell^{\mathrm{ref}}_{i,t}
-
\ell^\theta_{i,t}
=
\log
\frac{\pi_{\mathrm{ref}}}{\pi_\theta}
\]

代码使用：

\[
k_3(z)=\exp(z)-z-1
\]

如果 token 真正采样自当前策略 \(p=\pi_\theta\)，令 \(q=\pi_{\mathrm{ref}}\)，则：

\[
z=\log\frac{q}{p}
\]

并且：

\[
\begin{aligned}
\mathbb{E}_{a\sim p}
[\exp(z)-z-1]
&=
\mathbb{E}_{a\sim p}
\left[
\frac{q(a)}{p(a)}
-
\log\frac{q(a)}{p(a)}
-1
\right]\\
&=
1
+
D_{\mathrm{KL}}(p\|q)
-1\\
&=
D_{\mathrm{KL}}(p\|q)
\end{aligned}
\]

因此它是 forward KL：

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
\]

的一种非负、低方差单样本估计形式。

当前 loss 为：

\[
L_{\mathrm{KL}}
=
\frac{1}{|\mathcal V|}
\sum_{(i,t)\in\mathcal V}
\left[
\exp(z_{i,t})-z_{i,t}-1
\right]
\]

实际 rollout 来自 old policy；在一次 PPO batch 内，current 与 old 通过 clipping 保持接近，因此这是基于 old rollout 的局部 Monte Carlo 近似。

### 7.5 总损失

当前代码最小化：

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{policy}}
+
\beta L_{\mathrm{KL}}
}
\]

基线配置：

\[
\beta=0.005
\]

其中：

- \(L_{\mathrm{policy}}\) 推动策略追随 GRPO advantage；
- \(L_{\mathrm{KL}}\) 把策略约束在冻结 SFT reference 附近；
- 自适应 KL 控制器可能在后续 step 改变 \(\beta\)；
- 当前 step 内的 PPO mini-batch 使用同一个 `current_kl_loss_coef`。

### 7.6 update KL / approx_kl 不是 reference KL

代码还计算：

\[
\widehat D_{\mathrm{update}}
=
\frac{1}{|\mathcal V|}
\sum_{(i,t)\in\mathcal V}
\left[
r_{i,t}-1-\log r_{i,t}
\right]
\]

在 old policy 采样下，因为：

\[
\mathbb{E}_{\pi_{\mathrm{old}}}[r]=1
\]

所以：

\[
\mathbb{E}_{\pi_{\mathrm{old}}}
[r-1-\log r]
=
D_{\mathrm{KL}}
(\pi_{\mathrm{old}}\|\pi_\theta)
\]

它衡量当前批内更新相对 old policy 的局部变化，代码指标名为 `approx_kl`。

必须区分：

| 指标 | 比较对象 | 是否进入当前 total loss | 回答的问题 |
|---|---|---|---|
| `approx_kl` | current vs old | 否，主要用于诊断与保护 | 本批更新是否过猛 |
| `kl_loss` | current vs reference | 是 | 长期是否偏离 SFT |
| `clip_frac` | current/old ratio 是否越界 | 否 | PPO clipping 介入多少 |

---

<a id="section-8"></a>

## 8. 当前代码如何逐步执行 PPO optimizer update

### 8.1 第一步：采样独立 prompt

当前基线：

```text
train_batch_size = 4
```

每个 GRPO step 采样 4 道独立题目。

对应代码：

```text
GRPOTrainer._sample_prompt_indices()
GRPOTrainer.train_step()
```

### 8.2 第二步：actor 生成 rollout

每题生成：

```text
rollout_n = 8
```

所以：

\[
4\times8=32
\]

条回答。

生成时 actor 被切换到 `eval()`；生成结束后恢复 `train()`。

对应代码：

```text
GRPOTrainer._generate_responses()
```

### 8.3 第三步：规则 reward

每条回答根据：

- 最终答案 exact match；
- `####` 格式；
- 是否只有一个最终答案；
- 重复、过长与截断；
- 其他规则分量；

计算标量 reward：

\[
R_i
\]

对应代码：

```text
GRPOTrainer._compute_rewards()
post_training_framework/src/ptf/reward.py
```

### 8.4 第四步：GRPO advantage

同一个prompt的8条回答组成一组，按组计算：

\[
A_i=(R_i-\bar R)/\sigma_R
\]

对应代码：

```text
GRPOTrainer._compute_advantages()
```

### 8.5 第五步：构造训练 batch 和 response mask

每条训练序列为：

```text
prompt_ids + response_ids
```

mask 定义：

```text
prompt token   → 0
response token → 1
padding token  → 0
```

只有response token参与policy loss和KL loss。

对应代码：

```text
GRPOTrainer._build_training_batch()
```

### 8.6 第六步：冻结本批 old/ref log-prob

在任何 optimizer update 之前，代码分别执行 actor 和 reference 前向：

```python
batch["old_token_log_probs"] = actor(...).detach()
batch["ref_token_log_probs"] = reference(...).detach()
```

old log-prob 在本批所有 PPO epochs 中保持不变。因此，它在数值上承担 old policy 快照，不需要复制一个完整 old model。

对应代码：

```text
GRPOTrainer._compute_token_log_probs_in_chunks()
GRPOTrainer.train_step()
```

### 8.7 第七步：进入 PPO epochs

当前配置：

```text
ppo_epochs = 2
ppo_mini_batch_size = 16
```

每个 PPO epoch 都会：

1. 对32条rollout索引重新随机打乱；
2. 切成两个16条的mini-batch；
3. 对每个mini-batch重新计算current actor log-prob；
4. 计算ratio、PPO clip、reference KL和total loss；
5. backward；
6. gradient clipping；
7. `optimizer.step()`。

### 8.8 第八步：mask-safe ratio

当前实现只取有效 response token 的 log-ratio 检查有限性：

```text
valid_log_ratio = current_logp - old_logp
```

若出现非有限值，则跳过当前 mini-batch，避免污染参数。

随后把无效位置置0，并把有效 log-ratio 数值裁剪到 `[-20,20]` 后取指数。

### 8.9 第九步：backward

代码执行：

```python
(total_loss * loss_scale).backward()
```

得到：

\[
g=\nabla_\theta L_{\mathrm{total}}
\]

其中 \(\theta\) 只包含可训练 GRPO LoRA 参数。

### 8.10 第十步：梯度累积和 optimizer update

如果：

```text
gradient_accumulation_steps = K
```

代码会对连续 \(K\) 个 mini-batch 的 loss 分别除以 \(K\)，累积梯度后执行一次 `optimizer.step()`。

当前：

```text
gradient_accumulation_steps = 1
```

因此每个 mini-batch 都执行一次 optimizer update。

完整顺序为：

```text
optimizer.zero_grad(set_to_none=True)
→ current actor forward
→ policy loss + KL loss
→ backward
→ clip_grad_norm_(max_norm=1.0)
→ optimizer.step()
```

---

<a id="section-9"></a>

## 9. 当前配置为什么每个 GRPO step 有 4 次 optimizer update

当前基线配置：

```text
train_batch_size = 4
rollout_n = 8
ppo_mini_batch_size = 16
ppo_epochs = 2
gradient_accumulation_steps = 1
```

### 9.1 rollout 总数

\[
4\text{个prompt}
\times
8\text{条回答/prompt}
=
32\text{条rollout}
\]

### 9.2 每个 PPO epoch 的 mini-batch 数

\[
32\div16=2
\]

### 9.3 两个 PPO epochs 的更新次数

\[
2\text{个mini-batch/epoch}
\times
2\text{个epochs}
=
4\text{次optimizer.step}
\]

完整轨迹是：

```text
GRPO step 开始
→ 生成32条rollout
→ 保存一份固定old log-prob和ref log-prob

PPO epoch 1
  → shuffle
  → mini-batch 1：forward / backward / optimizer.step
  → mini-batch 2：forward / backward / optimizer.step

PPO epoch 2
  → 重新shuffle同一批32条rollout
  → mini-batch 3：forward / backward / optimizer.step
  → mini-batch 4：forward / backward / optimizer.step

进入下一个GRPO step
→ 重新生成rollout
→ 刷新old policy快照
```

### 9.4 ratio 从什么时候开始真正起作用

理论上，在第一轮更新前：

\[
\pi_\theta=\pi_{\mathrm{old}}
\]

所以：

\[
r_{i,t}=1
\]

此时：

\[
\nabla(rA)
=
A\nabla\log\pi_\theta
\]

与原始策略梯度一致。

第一次 `optimizer.step()` 之后：

\[
\pi_\theta\neq\pi_{\mathrm{old}}
\]

从第二个 mini-batch 开始，importance ratio 和 clipping 才承担“修正旧数据、限制重复利用幅度”的核心作用。

当前实现存在 dropout 导致第一轮 ratio 不一定严格等于1的问题，详见第12.1节。

---

<a id="section-10"></a>

## 10. AdamW 最终如何修改 GRPO LoRA 参数

### 10.1 梯度裁剪

设所有可训练 LoRA 参数梯度拼接后的全局范数为：

\[
\|g\|_2
\]

当前最大范数：

\[
c=1.0
\]

裁剪后：

\[
g
\leftarrow
g\cdot
\min
\left(
1,
\frac{c}{\|g\|_2}
\right)
\]

如果梯度范数小于1，不改变；如果大于1，按比例缩小全部梯度。

### 10.2 Adam 一阶和二阶状态

对第 \(k\) 次 optimizer update：

\[
m_k
=
\beta_1m_{k-1}
+
(1-\beta_1)g_k
\]

\[
v_k
=
\beta_2v_{k-1}
+
(1-\beta_2)g_k^2
\]

偏差修正：

\[
\hat m_k
=
\frac{m_k}{1-\beta_1^k}
\]

\[
\hat v_k
=
\frac{v_k}{1-\beta_2^k}
\]

### 10.3 AdamW 解耦权重衰减

简化写法：

\[
\theta_{k+1}
=
\theta_k
-
\eta
\frac{\hat m_k}
{\sqrt{\hat v_k}+\epsilon_{\mathrm{Adam}}}
-
\eta\lambda\theta_k
\]

其中：

- \(\eta\)：learning rate；
- \(m_k\)：历史梯度一阶动量；
- \(v_k\)：历史梯度平方统计；
- \(\lambda\)：AdamW weight decay；
- \(\epsilon_{\mathrm{Adam}}\)：数值稳定项。

当前代码只显式传入：

```python
torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
```

因此其他参数使用安装环境中 PyTorch 的 AdamW 默认值，通常为：

```text
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.01
```

为了跨 PyTorch 版本做到完全可审计，未来可以考虑在配置和代码中显式固定这些值；本文只记录当前实现，不在此处修改训练代码。

### 10.4 optimizer state 为什么会影响续训

checkpoint 中的 `optimizer.pt` 不只保存更新次数，还保存：

- \(m_k\)：历史梯度方向；
- \(v_k\)：历史梯度尺度；
- optimizer step count；
- 参数组状态。

所以：

- 同一目标的 full resume 应恢复 optimizer；
- reward、KL目标或主要训练目标发生改变时，旧动量可能不再匹配新目标；
- 这就是“改变目标后不应无条件继承旧 optimizer”的数学原因。

---

<a id="section-11"></a>

## 11. 哪些步骤是严格恒等式，哪些是工程近似

| 推导环节 | 性质 | 当前项目中的表现 |
|---|---|---|
| 轨迹期望回报的 log-derivative 推导 | 严格恒等式 | 策略梯度理论起点 |
| 减去只依赖状态的 baseline | 不改变期望梯度 | GRPO改用组内均值作为相对基线思想 |
| 完整分布重要性采样 | 在支持集覆盖时是严格恒等式 | 当前没有使用完整轨迹比率 |
| 用 old 状态分布近似 current 状态分布 | 局部近似 | 依赖新旧策略保持接近 |
| token ratio 替代完整序列 ratio | 工程近似 | 当前逐token计算 current/old ratio |
| PPO clipping | 有意引入偏差的稳定化设计 | 当前 \(\epsilon=0.2\) |
| outcome reward 广播到全部token | 信用分配近似 | 当前每条回答只有一个 \(A_i\) |
| 组内标准化 advantage | GRPO估计设计 | 无critic，按同题8条回答归一化 |
| low-var reference KL | Monte Carlo估计 | 当前使用 \(\exp(z)-z-1\) |
| old rollout估计current/reference KL | 局部近似 | clipping使current与old尽量接近 |
| mini-batch SGD/AdamW | 随机优化近似 | 每步4次LoRA更新 |

### 11.1 为什么“推导”不能写成一串完全等号

从策略梯度到 PPO 的完整逻辑不是：

```text
策略梯度 = importance ratio = clipped PPO
```

更准确的是：

```text
策略梯度定理
→ 为复用old-policy数据，引入importance ratio
→ 为避免显式估计状态分布比率，采用局部策略近似
→ 为限制ratio和策略移动，TRPO使用KL约束
→ PPO用clip替代复杂的约束优化
→ GRPO用同题组内相对reward替代critic advantage
→ 当前项目再加入直接reference KL loss
```

因此，只有部分步骤是恒等变换，后半段包含明确的算法设计和工程折中。

---

<a id="section-12"></a>

## 12. 当前实现的两个重要审计边界

### 12.1 old policy log-prob 与 rollout 行为策略存在 dropout 口径差异

理论上，importance ratio 的分母必须是“真正生成该动作的行为策略概率”：

\[
\pi_{\mathrm{behavior}}(y_{i,t}|x,y_{i,<t})
\]

当前生成流程中：

```text
_generate_responses()
→ self.actor.eval()
→ actor.generate(...)
→ self.actor.train()
```

随后 old log-prob 在 `train()` 模式下重新计算：

```text
_compute_token_log_probs_in_chunks(self.actor, batch)
```

当前 GRPO LoRA 的：

```text
lora_dropout = 0.05
```

因此：

- rollout 生成时 dropout 关闭；
- old log-prob 重算时 LoRA dropout 开启；
- current mini-batch前向时 dropout也开启，但使用另一组随机mask；
- 即使第一次 optimizer update 尚未发生，current/old ratio 也不保证严格等于1；
- old log-prob 也不完全等于生成时 behavior policy 的概率。

这不改变“代码当前如何执行”的事实，但意味着理论上的：

\[
\pi_\theta=\pi_{\mathrm{old}}
\Rightarrow r=1
\]

在当前随机 dropout 前向下只是一种理想口径，而不是严格数值保证。

后续若整改，应把它注册为独立实现修复，例如：

1. 生成与old/current log-prob口径统一；
2. PPO log-prob计算期间禁用dropout；或
3. 直接保存生成行为策略对应的log-prob。

不能在未做控制变量实验前，把修复后的轨迹与旧轨迹直接视为完全同一协议。

### 12.2 当前 loss 是全有效 token 平均，会产生回答长度权重差异

当前 policy loss 聚合方式为：

\[
\frac{1}{|\mathcal V|}
\sum_{(i,t)\in\mathcal V}L_{i,t}
\]

这意味着在同一 mini-batch 中，回答越长，有效token项越多，对总梯度的贡献机会越多。

另一种常见聚合方式是先对每条回答按长度求平均：

\[
\frac{1}{B}
\sum_{i=1}^{B}
\frac{1}{T_i}
\sum_{t=1}^{T_i}L_{i,t}
\]

两种口径不等价：

- 当前实现更接近全token等权；
- 序列先平均更接近每条回答等权；
- 当前reward虽然包含长度和截断惩罚，但这不自动消除loss聚合层面的长度权重差异。

这也是当前实现事实，不应在没有实验的情况下直接改动。若要判断它是否是能力下降或格式退化的主因，需要独立控制变量实验。

---

<a id="section-13"></a>

## 13. 完整伪代码与代码阅读路线

### 13.1 当前单个 GRPO step 的算法伪代码

```text
输入：当前 actor θ、冻结 reference、训练题库

1. 采样4个prompt
2. actor为每题生成8条回答，共32条rollout
3. 对每条回答计算规则reward R_i
4. 按prompt分组：
     mean_R = mean(R_group)
     std_R  = std(R_group)
     A_i    = (R_i - mean_R) / std_R
5. 构造 prompt+response token batch 和 response mask
6. 更新前冻结：
     old_logp = actor(batch).detach()
     ref_logp = reference(batch).detach()

7. 重复 ppo_epochs=2：
     随机打乱32条rollout
     切成2个mini-batch，每个16条

     对每个mini-batch：
       optimizer.zero_grad()
       curr_logp = actor(mini_batch)
       log_ratio = curr_logp - old_logp
       ratio = exp(clamp(log_ratio, -20, 20))

       surr1 = ratio * A
       surr2 = clamp(ratio, 0.8, 1.2) * A
       policy_loss = -mean_valid_tokens(min(surr1, surr2))

       ref_delta = ref_logp - curr_logp
       kl_loss = mean_valid_tokens(exp(ref_delta) - ref_delta - 1)

       total_loss = policy_loss + kl_coef * kl_loss
       total_loss.backward()
       clip_grad_norm(max_norm=1.0)
       optimizer.step()

8. 汇总policy_loss、reference KL、approx_kl、clip_frac、grad_norm
9. 写训练日志和group diagnostics
10. 执行KL guard、signal guard、eval和early stopping
```

### 13.2 推荐代码阅读顺序

| 顺序 | 文件/函数 | 阅读目的 |
|---:|---|---|
| 1 | `post_training_framework/configs/gsm8k_qwen3_0d6b_grpo_v7_causal_base_from169.json` | 确认当前batch、PPO、KL和学习率配置 |
| 2 | `load_actor_and_reference()` | 理解Base、SFT、reference和GRPO LoRA关系 |
| 3 | `_generate_responses()` | 确认rollout行为策略和生成参数 |
| 4 | `_compute_rewards()` | 确认最终reward来自哪些规则分量 |
| 5 | `_compute_advantages()` | 确认GRPO组内归一化公式 |
| 6 | `_build_training_batch()` | 理解response mask和advantage张量 |
| 7 | `compute_sequence_log_probs()` | 理解shift logits和token log-prob |
| 8 | `compute_kl_loss()` | 理解direct reference KL |
| 9 | `train_step()` | 看PPO epochs、shuffle、mini-batch和更新次数 |
| 10 | `_train_mini_batch()` | 看ratio、clip、loss和backward精确实现 |
| 11 | `_check_kl_guard()` | 区分update KL和reference KL保护 |
| 12 | `train()` | 理解optimizer更新后停止规则的调用顺序 |

### 13.3 配置到公式的映射

| 配置 | 公式或执行含义 |
|---|---|
| `train_batch_size=4` | 每步4个独立prompt |
| `rollout_n=8` | 每题8条候选回答 |
| `ppo_mini_batch_size=16` | 每次前向/反向使用16条回答 |
| `ppo_epochs=2` | 同一批rollout完整重复训练2轮 |
| `gradient_accumulation_steps=1` | 每个mini-batch执行一次optimizer.step |
| `clip_ratio=0.2` | PPO ratio区间 `[0.8,1.2]` |
| `norm_adv_by_std=true` | 使用组内标准差归一化advantage |
| `kl_loss_coef=0.005` | \(L_{total}=L_{policy}+0.005L_{KL}\) 的初始系数 |
| `kl_loss_type=low_var_kl` | 使用 \(\exp(z)-z-1\) reference KL估计 |
| `max_grad_norm=1.0` | optimizer.step前裁剪全局梯度范数 |
| `learning_rate=5e-6` | AdamW基础更新步幅；L1分支会覆盖为3e-6 |

---

<a id="section-14"></a>

## 14. 常见误解与复盘检查表

### 14.1 常见误解

| 误解 | 正确理解 |
|---|---|
| 策略梯度定理里没有importance sampling，所以PPO不是策略梯度 | PPO从策略梯度出发，在复用old-policy数据时引入ratio |
| importance ratio是reference/current | PPO ratio是current/old；reference另用于长期KL锚定 |
| ratio是完整回答概率比 | 当前实现是token级ratio，不是完整序列乘积 |
| clipping是重要性采样恒等式的一部分 | clipping是PPO主动引入偏差的稳定化设计 |
| 第一次更新也必须靠ratio修正 | 理论上第一轮在old点与原始策略梯度一致；之后ratio才更关键 |
| PPO epoch 2会再生成一批rollout | 不会，同一批32条rollout被训练两遍 |
| 一个GRPO step只有一次参数更新 | 当前每步有4次AdamW optimizer.step |
| optimizer.step更新整个SFT模型 | 当前只更新GRPO LoRA的可训练参数 |
| `approx_kl`就是reference KL loss | 前者比较current/old，后者比较current/reference |
| advantage能定位错误推理token | 当前一个response-level advantage广播给全部response token |
| old policy一定是完整模型副本 | 当前只保存更新前的token log-prob张量 |

### 14.2 复盘检查表

在分析一次GRPO/PPO更新前，依次回答：

1. 当前的行为策略、old policy和reference分别是谁？
2. rollout是在actor的`eval`还是`train`模式下生成？
3. old log-prob是在optimizer更新前冻结的吗？
4. old log-prob是否与真正生成动作时的行为策略口径一致？
5. advantage来自critic、GAE，还是同题组内reward归一化？
6. advantage是token级还是response级广播？
7. ratio是token级还是sequence级？
8. policy loss按token平均还是先按response长度平均？
9. `ppo_epochs`和mini-batch共同产生多少次optimizer.step？
10. reference KL进入loss，还是只做监控？
11. `approx_kl`和`kl_loss`分别比较哪两个策略？
12. optimizer只更新LoRA还是更新全量参数？
13. resume时是否恢复了与当前目标匹配的optimizer state？
14. 当前策略变化主要由policy advantage、reference KL，还是旧optimizer动量推动？

### 14.3 最终知识链

```text
期望回报 J(θ)
→ log-derivative trick
→ 策略梯度定理 E[∇logπ · A]
→ rollout来自old policy且需要重复使用
→ importance ratio π_current / π_old
→ 忽略状态分布比率，得到局部surrogate
→ TRPO用KL信赖域约束局部更新
→ PPO用clipping近似信赖域
→ GRPO用同题组内相对reward构造advantage
→ outcome advantage广播到response token
→ 加入current/reference直接KL penalty
→ backward得到GRPO LoRA梯度
→ gradient clipping
→ AdamW optimizer.step
→ 多个小更新累积形成新的actor
```

理解这条链之后，就能把“策略梯度、重要性采样、PPO、GRPO、KL、mini-batch、PPO epoch和optimizer update”放进同一套连续的数学与工程框架中，而不是当成互不相干的名词。
