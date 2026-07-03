# GRPO 训练显存占用详解

> 背景：在 RTX 4070 (12 GB VRAM) 上用 Qwen3-0.6B 模型跑 GRPO 训练，
> 任务管理器显示 11.6 GB Dedicated GPU Memory + 3.2 GB Shared GPU Memory = 14.8 GB 总占用。
> 本文逐项拆解每一部分显存占用是什么、为什么需要、有多大。

---

## 1. GRPO 训练的两个阶段

GRPO 训练的每一步分为两个阶段，它们不会同时占用显存峰值：

```
Rollout 阶段 (generate):
  actor.generate(prompt) → 生成多个回答 → 计算 reward → 计算 advantage
  显存大头: actor generate 的临时 tensor
  KV cache 极小 (~36 MB, 不是大头!)

Training 阶段 (forward + backward + optimizer step):
  用 rollout 数据做 PPO clip + KL penalty 更新 actor 的 LoRA 参数
  显存大头: actor 前向传播的中间激活值 (~5 GB, 供反向传播计算梯度)
```

---

## 2. 常驻部分：模型权重

无论在哪个阶段，以下模型权重始终驻留在 GPU 上：

### 2.1 actor 权重 (~1.20 GB)

```
actor = base(0.6B) + SFT(merged) + 新 LoRA(r=16)

参数量: ~600M (Qwen3-0.6B)
存储: 600M × 2 bytes (fp16/bf16) ≈ 1.20 GB

虽然只有 LoRA 参数 (~10M) 可训练，但整个模型的权重都在 GPU 上。
LoRA 是叠加在原模型上的，前向传播需要先算原模型的线性层，
再叠加 LoRA 的结果，所以原模型权重不能卸载。
```

### 2.2 reference 权重 (~1.20 GB)

```
reference = copy.deepcopy(actor 去掉新 LoRA 后的 merged model)

参数量: ~600M
存储: 600M × 2 bytes ≈ 1.20 GB

reference 是冻结的 SFT 模型，用于 KL 惩罚：
  KL loss = π_current 偏离 π_reference 的程度
  计算 KL 需要用 reference 做前向传播得到 ref_log_probs
  所以 reference 的完整权重也必须在 GPU 上

为什么不能共享 actor 的权重？
  actor 的权重在训练过程中会不断更新（梯度更新 LoRA 参数）
  reference 必须保持不变（冻结的 SFT 起点）
  如果共享，reference 也会跟着变 → KL 惩罚失效
```

**常驻模型权重合计: ~2.40 GB**

---

## 3. Training 阶段显存大头详解

### 3.1 为什么 training 阶段是显存瓶颈

训练阶段做 actor 前向传播时，必须保存每一层的**中间激活值**供反向传播使用：

```
前向传播: input → 层1 → 激活1 → 层2 → 激活2 → ... → 层28 → 激活28 → loss

反向传播: loss → ∂loss/∂激活28 → ∂loss/∂层28参数 + ∂loss/∂激活27
                  → ∂loss/∂激活27 → ∂loss/∂层27参数 + ∂loss/∂激活26
                  → ...

要计算 ∂loss/∂层i参数, 需要激活i (前向传播时层i的输出)
所以前向传播时必须保存每一层的激活值, 直到反向传播用完才能释放
```

### 3.2 激活值的精确计算 (Qwen3-0.6B 正确参数)

Qwen3-0.6B 配置:
- hidden_size = 1024
- intermediate_size = 3072
- num_hidden_layers = 28
- num_attention_heads = 16 (query heads)
- num_key_value_heads = 8 (GQA, KV heads)
- head_dim = 128
- vocab_size = 151936

训练时 batch_size=2, rollout_n=2, 总序列数=4, seq_len≈768:

```
每层激活值的子项:

1. Attention 子激活:
   - Q_proj 输出: B×S×n_q×head_dim = 4×768×16×128 × 2B = 12582912 B ≈ 12.0 MB
   - K_proj 输出: 4×768×8×128 × 2B = 6291456 B ≈ 6.0 MB
   - V_proj 输出: 4×768×8×128 × 2B ≈ 6.0 MB
   - Attention probs 矩阵: B×n_q×S×S × 2B
     = 4×16×768×768 × 2B = 75497472 B ≈ 71.5 MB ← S² 增长!
   - Attention output: B×S×hidden_size × 2B = 4×768×1024×2 ≈ 6.0 MB
   合计: ≈ 95.5 MB/层

2. MLP 子激活 (SwiGLU 结构):
   - gate_proj 输出: B×S×intermediate_size × 2B = 4×768×3072×2 ≈ 18.9 MB
   - up_proj 输出:   B×S×intermediate_size × 2B ≈ 18.9 MB
   - SwiGLU 中间 (gate×up): B×S×intermediate_size × 2B ≈ 18.9 MB
   - down_proj 输出: B×S×hidden_size × 2B ≈ 6.0 MB
   合计: ≈ 62.7 MB/层

3. 其他 (RMSNorm input residual, etc): ≈ 8 MB/层

每层总激活: ≈ 166 MB
28 层总激活: ≈ 28 × 166 MB = 4.65 GB ← 显存大头!
```

加上 logits 输出层:
```
logits: B×S×vocab_size × 2B = 4×768×151936×2 = 933 MB ≈ 0.93 GB
```

**actor 前向阶段总占用: ~4.65 GB (激活) + ~0.93 GB (logits) ≈ 5.58 GB**

### 3.3 为什么激活值远大于 KV cache

关键对比——同样处理一个 token，激活值为什么远大于 KV cache：

```
KV cache 每层每个 token 只存 K 和 V 两个向量:
  K: 1 个向量, 维度 = num_kv_heads × head_dim = 8 × 128 = 1024
  V: 1 个向量, 维度 = 1024
  合计: 2048 个 fp16 数 ≈ 4 KB

激活值每层每个 token 存的是整层前向传播的所有中间结果:
  - MLP SwiGLU: intermediate_size=3072 >> hidden_size=1024
    每层存 3 个 3072 维的中间向量 → 每个 token 3×3072×2 = 18 KB
    vs KV cache 每个 token 4 KB → 单这一项就差 4.5×

  - Attention probs: 每个 query head 对所有 seq_len 个 token 计算 attention
    这是 S×S 的矩阵 → 二次增长!
    KV cache 不存 attention 权重, 只存 K/V 向量 → 线性增长

差距来源:
  1. MLP intermediate_size (3072) >> hidden_size (1024),
     每层要存 3 个 3072 维的中间向量
  2. Attention 权重矩阵是 S×S (二次增长),
     而 KV cache 只存 K/V 向量 (线性增长)
  3. 激活值必须全部保存供反向传播; KV cache 只存 K/V 向量供推理时复用
```

---

## 4. KV Cache：极小，不是瓶颈

### 4.1 KV Cache 是什么

Transformer 生成文本是逐 token 进行的——每次只预测下一个 token。
每预测一个新 token 时，attention 需要所有之前 token 的 Key 和 Value 向量。

```
预测第 100 个 token 时:
  需要: 第 1~99 个 token 的 K 和 V
  这些 K/V 在预测第 1~99 个 token 时已经计算过了

两种选择:
  (a) 丢弃 → 第 100 步重新算 99 次 K/V → 极慢，且每步越来越慢
  (b) 缓存 → 把算过的 K/V 存在显存里 → 直接读取 → 只算当前 token 的 K/V
```

KV cache 就是选择 (b)：缓存已处理 token 的 K/V 向量，供后续 attention 计算。

### 4.2 KV Cache 的显存计算

```
KV cache (每个 token) = num_layers × 2(K和V) × num_kv_heads × head_dim × dtype_size

Qwen3-0.6B (GQA, 8 个 KV head):
  28 层 × 2 × 8 KV heads × 128 (head_dim) × 2 bytes (fp16)
  = 28 × 2 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB/token

4 条序列 × 328 token (prompt + response):
  总 KV cache ≈ 4 × 328 × 112 KB ≈ 144 MB

即使序列更长 (768 token):
  总 KV cache ≈ 4 × 768 × 112 KB ≈ 344 MB

KV cache 始终在几十到几百 MB 级别，远不到 GB。
它不是显存瓶颈。
```

### 4.3 为什么训练阶段不需要 KV Cache

```
训练阶段的 compute_sequence_log_probs:
  输入: prompt + response 拼成完整的 input_ids (全部 token 已知)
  处理: 所有 token 并行计算 (矩阵乘法一次性处理整条序列)
  不需要逐 token 生成 → 不需要 KV cache
  模型配置: use_cache=False → 关闭 KV cache
```

---

## 5. Reference 前向传播的显存占用

代码中 reference 的前向传播全部在 `torch.no_grad()` 下执行：

```python
# train_step() 中:
with torch.no_grad():
    _, ref_seq_log_probs = compute_sequence_log_probs(
        self.reference, batch["input_ids"], ...
    )

# _train_mini_batch() 中:
with torch.no_grad():
    ref_token_log_probs, _ = compute_sequence_log_probs(
        self.reference, mb_input_ids, ...
    )
```

`torch.no_grad()` 的效果:
1. 不构建计算图 → 不保存中间激活值供反向传播
2. 每层的中间结果在算完下一层后立即释放
3. 只保留最终输出 (ref_log_probs), 然后 .detach() 赋值给 batch

**reference 不产生持久中间激活值。**

但有一个时序问题: _train_mini_batch() 中,
reference 的前向传播发生在 actor 的 ~5 GB 激活值还没释放的时候：

```
actor 前向 (有梯度) → ~5 GB 中间激活值全部在显存中
reference 前向 (no_grad) → 在 actor 的 5 GB 之上叠加临时峰值
  峰值 = 单层 hidden_states + logits ≈ ~0.3 GB
total_loss.backward() → actor 激活值逐层释放
```

峰值瞬间: actor 5.6 GB + ref 0.3 GB 临时 + 模型权重 2.4 GB ≈ 8.3 GB

---

## 6. CUDA 内存分配器：为什么 8.7 GB 数据占了 11.6 GB VRAM

### 6.1 核心问题

有效数据只有 ~8.7 GB，但 VRAM 实际占用了 11.6 GB。
差额 ~2.9 GB 来自 **CUDA 内存分配器的预留池**，不是"浪费"。

### 6.2 CUDA 分配器的工作原理

```
PyTorch 的 CUDA 分配器不是"按需精确分配":
  它维护一个内存池 (memory pool), 从 CUDA runtime 批量申请大块 VRAM
  然后从池中切分给各个 tensor

行为示例:
  申请 5 GB 存激活值 → 分配器实际从 CUDA runtime 拿 6-7 GB (多拿 1-2 GB)
  激活值释放后 → 分配器不归还给 CUDA runtime → 留在池里等下次用
  下次申请 → 直接从池里切分 → 快 (不需要再向 CUDA runtime 申请)

为什么这样做?
  1. 防止碎片化: 频繁 malloc/free 导致 VRAM 出现大量小块空洞
  2. 性能: 向 CUDA runtime 申请是慢操作, 从池里切分是快操作
  3. 峰值缓冲: 训练循环中激活值反复申请/释放, 池要预留空间供峰值用
```

### 6.3 关键指标

```
torch.cuda.memory_allocated() = 实际数据占用的 VRAM ≈ 8.7 GB
torch.cuda.memory_reserved()  = 分配器从 CUDA runtime 持有的 VRAM 总量 ≈ 11.6 GB

差额 = reserved - allocated ≈ 2.9 GB
这 2.9 GB 是分配器持有但未填数据的预留池空间

不是"浪费"——它是分配器正常工作的必要开销:
  - 内存碎片整理空间
  - 下一步训练循环的峰值缓冲
  - cuBLAS/cuDNN kernel 工作空间
```

### 6.4 Shared GPU Memory：3.2 GB 系统 RAM 溢出

```
Windows WDDM (显示驱动模型) 下的 CUDA:
  当 VRAM 不足时, CUDA 分配器通过 WDDM 向系统请求 Shared GPU Memory
  Shared GPU Memory = 映射到 GPU 地址空间的系统 RAM
  通过 PCIe 访问 → 比显存慢 ~15×

为什么需要 3.2 GB Shared?
  VRAM 已占用 11.6 GB, 只剩 0.4 GB (12 - 11.6)
  分配器需要额外空间做:
    - 激活值释放后重新分配时的缓冲
    - cuBLAS/cuDNN kernel 的临时工作空间
    - 内存碎片整理
  0.4 GB 不够 → WDDM 提供 3.2 GB 系统 RAM 作为备用

  注意: Shared 不意味着有 3.2 GB 数据在系统 RAM 里
  大部分数据仍在 VRAM 内, Shared 只是"可用但未必在用"的备用空间
```

---

## 7. Gradient Checkpointing：用计算换显存

### 7.1 原理

不开 checkpointing 时，前向传播保存所有层的激活值：

```
层1 激活 ✓ 保存 → 反向传播时直接读取
层2 激活 ✓ 保存 → 反向传播时直接读取
...
层28 激活 ✓ 保存 → 反向传播时直接读取

显存: 28 层 × ~166 MB/层 ≈ 4.65 GB
反向传播: 直接读取 → 快
```

开 checkpointing 时，只保存部分层的激活（checkpoint 点），其余层丢弃：

```
层1 激活 ✓ 保存 (checkpoint 点)
层2 激活 ✗ 丢弃
层3 激活 ✗ 丢弃
层4 激活 ✓ 保存 (checkpoint 点)
...
层28 激活 ✓ 保存 (checkpoint 点)

显存: ~7-8 层 × 166 MB ≈ 1.2 GB (只存约 1/4)
反向传播:
  到层3时 → 层2、3 的激活已被丢弃
  → 从层1 (checkpoint 点) 重新做前向传播 → 算出层2、3的激活
  → 再用它们计算梯度 → 多了一次前向计算 → 慢 ~30%, 但省显存
```

### 7.2 效果

| | 不开 checkpointing | 开 checkpointing |
|---|---|---|
| actor 激活值显存 | ~4.65 GB (存所有层) | ~1.2 GB (只存部分层) |
| 训练速度 | 快 | 慢 ~30% (丢弃的层要重新前向) |
| OOM 风险 | 高 (余量 ~0.4 GB) | 低 (多出 ~3.5 GB 余量) |

---

## 8. Optimizer 状态和其他

### 8.1 Optimizer 状态 (~0.08 GB)

```
AdamW 为每个可训练参数存 2 份状态:
  - momentum (一阶矩估计): 与参数同大小
  - variance (二阶矩估计): 与参数同大小

可训练参数 = LoRA 参数 ≈ 10M
optimizer 状态 = 10M × 2 × 4 bytes (fp32) ≈ 80 MB ≈ 0.08 GB

很小，不是显存瓶颈。
```

### 8.2 CUDA 内核与框架开销

```
CUDA runtime 预分配的内存池
PyTorch 的内存分配器碎片
cuBLAS/cuDNN 的工作空间
这些是 CUDA 分配器预留池的一部分 (已包含在第 6 节的 2.9 GB 中)
```

---

## 9. 总汇总

### 9.1 有效数据占用 (allocated)

| 类别 | 估算占用 | 说明 |
|---|---|---|
| actor 权重 | 1.20 GB | 600M × 2B, 常驻 |
| reference 权重 | 1.20 GB | 600M × 2B, 常驻 |
| actor 激活值 (28层) | ~4.65 GB | 真正的显存大头 |
| actor logits | ~0.93 GB | vocab_size=151936 导致 |
| ref 前向临时峰值 | ~0.3 GB | no_grad, 不持久 |
| optimizer 状态 | ~0.08 GB | LoRA 参数的 AdamW |
| **有效数据合计** | **~8.4 GB** | torch.cuda.memory_allocated() |

### 9.2 实际显存占用 (实测)

| 项 | 大小 | 来源 |
|---|---|---|
| 有效数据 (allocated) | ~8.4 GB | 上表 |
| CUDA 分配器预留池 | ~2.9 GB | reserved - allocated |
| **VRAM 实际占用 (reserved)** | **11.6 GB** | 任务管理器 Dedicated |
| Shared GPU Memory | 3.2 GB | WDDM 系统 RAM 溢出 |
| **总占用** | **14.8 GB** | 11.6 + 3.2 |

### 9.3 为什么 8.4 GB 数据占了 14.8 GB

```
核心原因: CUDA 内存分配器的预留池和 WDDM Shared 溢出

8.4 GB (数据)
  + 2.9 GB (CUDA 预留池——分配器持有但未填数据的 VRAM)
  + 2.5 GB (WDDM Shared 中 CUDA 分配器预留的备用空间)
  = 14.8 GB (总占用)

不是"浪费":
  预留池是 CUDA 分配器正常工作的必要开销
  防止碎片化、提供峰值缓冲、给 cuBLAS/cuDNN 工作空间
  Shared 是 WDDM 模式下 VRAM 不足时的自动溢出机制

如果没有预留池 → 频繁 malloc/free → 碎片化 → 还是会 OOM
```

---

## 10. 关键认知总结

> **真正的显存大头是 actor 前向传播的中间激活值 (~4.65 GB)，不是 KV cache (~144 MB)。**

> 激活值远大于 KV cache 的原因:
> 1. MLP 的 intermediate_size (3072) >> hidden_size (1024),
>    每层要存 3 个 3072 维的中间向量
> 2. Attention 权重矩阵是 S×S (二次增长),
>    而 KV cache 只存 K/V 向量 (线性增长)
> 3. 激活值必须全部保存供反向传播; KV cache 只存 K/V 向量供推理时复用

> **CUDA 分配器预留池 (~2.9 GB) + WDDM Shared (~3.2 GB) = 6.1 GB 额外占用。**

> 这不是"浪费"——是 CUDA 分配器正常工作的必要开销。
> 有效数据 8.4 GB + 分配器开销 6.4 GB = 总占用 14.8 GB。

> **解决 OOM 的正确方案是开 gradient checkpointing (激活值从 ~4.65 GB 降至 ~1.2 GB),
> 而不是减小 KV cache (它只有 ~144 MB, 优化它几乎没有效果)。**

> **Windows WDDM 模式下，Shared GPU Memory 是 CUDA 的自动溢出机制:
> 当 VRAM 接近满时，系统 RAM 通过 PCIe 映射为 GPU 可访问的地址空间。
> 大部分数据仍在 VRAM 内, Shared 只是备用。**
