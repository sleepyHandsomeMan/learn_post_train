# verl 源码架构学习笔记

本文面向第一次系统阅读 `verl/` 源码的学习者。写作目标不是复述 README, 而是把当前工作树里的关键实现串成一条可读路径: 先理解 verl 为什么这样设计, 再按训练链路和模块逐层对应到源码。

本文对应的源码根目录是 `D:\learnAI\verl\verl`。重点代码路径:

- `verl/trainer/main_ppo_sync.py`: 当前同步 PPO 入口和主训练循环。
- `verl/trainer/main_ppo.py`: 旧 PPO 入口, 当前代码里标记 deprecated, 仍可作为传统 `DataProto` 主循环参考。
- `verl/protocol.py`: `DataProto`、`DataProtoFuture`、`BatchData`, 是控制器和 worker 之间的数据协议。
- `verl/single_controller/`: 单控制器和 Ray worker group 抽象。
- `verl/workers/engine_workers.py`: 统一 worker 层, 把 actor、rollout、reference、critic、reward 等角色包装成 RPC 可调用对象。
- `verl/workers/engine/`: 训练后端抽象与 FSDP、Megatron、VeOmni、TorchTitan、Automodel 等实现。
- `verl/workers/rollout/`: rollout 推理服务抽象, 包括 vLLM、SGLang、TRT-LLM。
- `verl/trainer/ppo/core_algos.py`: PPO/GRPO/ReMax/RLOO/REINFORCE++ 等算法核心函数。
- `verl/workers/utils/losses.py`: actor/critic/SFT 的 loss 函数接入点。
- `verl/utils/dataset/`: RLHF、SFT、RM 数据集处理。
- `verl/checkpoint_engine/` 与 `verl/utils/checkpoint/`: 训练权重同步和 checkpoint 保存恢复。

## 1. verl 要解决什么问题

verl 的目标是做大语言模型后训练, 尤其是 RLHF/RLAIF/可验证奖励强化学习。它需要同时解决两类问题:

1. 算法控制流: PPO、GRPO、ReMax、RLOO 等算法每一步应该先做什么、后做什么。例如生成样本、打分、算 log probability、算 advantage、更新 critic、更新 actor。
2. 大模型计算流: 每个高成本步骤要在多 GPU、多节点、不同后端上执行。例如 FSDP、FSDP2、Megatron、vLLM、SGLang、TorchTitan、VeOmni。

如果把这两类逻辑写死在一起, 每换一种算法或训练后端都要重写一整套分布式程序。verl 的核心设计是把它们拆开:

- controller/trainer 只描述算法级别的数据流。
- worker/engine 负责分布式计算。
- `DataProto`/`TensorDict`/TransferQueue 负责数据在两层之间流动。
- config 决定角色、资源、后端和算法变体。

这个设计在 `docs/hybrid_flow.rst` 里被称为 HybridFlow。对应源码里, 单进程控制器主要在 `verl/trainer/*`, 多进程计算主要在 `verl/workers/*` 和 `verl/single_controller/*`。

## 2. 从顶到底看一次全局数据流

以当前同步 PPO 入口 `verl/trainer/main_ppo_sync.py` 为主线, 一次训练大致是:

```text
Hydra 配置
  -> main_ppo_sync.main()
  -> main_ppo.run_ppo()
  -> Ray remote TaskRunner.run()
  -> PPOTrainer(...)
  -> PPOTrainer.init_workers()
  -> PPOTrainer.fit()
  -> 每个 step:
       dataset batch
       -> async_rollout_manager.generate_sequences()
       -> TransferQueue / ReplayBuffer 收集 rollout 结果
       -> reward
       -> old_log_probs
       -> ref_log_prob 可选
       -> values 可选
       -> advantages / returns
       -> critic update 可选
       -> actor update
       -> checkpoint_manager.update_weights()
       -> metrics / validation / checkpoint
```

源码对应:

- 入口: `main()` 在 `verl/trainer/main_ppo_sync.py` 底部, 使用 `@hydra.main(config_path="config", config_name="ppo_trainer")`。
- Ray 初始化: `run_ppo()` 在 `verl/trainer/main_ppo.py`, `main_ppo_sync.py` 复用它并传入自己的 `TaskRunner`。
- 远程任务: `TaskRunner.run()` 在 `main_ppo_sync.py`, 它创建 `PPOTrainer`, 调用 `init_workers()` 和 `fit()`。
- 主循环: `PPOTrainer.fit()` 和 `PPOTrainer.step()` 在 `main_ppo_sync.py`。

为什么要有 `TaskRunner`:

- 它是 Ray remote actor, 避免把内存占用较大的 controller 直接放在 Ray head 进程里。
- 它把配置解析、角色注册、资源池注册和 trainer 启动放到一个远程执行环境中。

为什么同步 PPO 里有 TransferQueue:

- `main_ppo_sync.py` 文件头部说明它相对旧 `main_ppo.py` 的差异: 使用 TransferQueue 做 zero-padding/zero-copy 数据传输, 用 ReplayBuffer 从 TransferQueue 采样, 支持每个 prompt 不同的 `n`, 支持 agent loop 多输出。
- 这说明当前代码已经从传统的 `DataProto` 全量往返, 向更细粒度的 KV 批数据流演进。

## 3. 配置系统: 用 YAML 组合算法、角色和后端

主配置是 `verl/trainer/config/ppo_trainer.yaml`。它的 `defaults` 组合了:

- `model_engine: dp`
- `actor@actor_rollout_ref.actor: ${model_engine}_actor`
- `ref@actor_rollout_ref.ref: ${model_engine}_ref`
- `rollout@actor_rollout_ref.rollout: rollout`
- `model@actor_rollout_ref.model: hf_model`
- `critic@critic: ${model_engine}_critic`
- `reward@reward: reward`
- `algorithm@algorithm.rollout_correction: rollout_correction`
- `distillation@distillation: distillation`

这就是为什么同一套 trainer 能切后端: 你不是改 Python 主循环, 而是改 `model_engine` 或各角色配置。例如 `ppo_megatron_trainer.yaml` 只是覆盖 `model_engine: megatron`。

配置会被转成 dataclass:

- `verl/base_config.py` 定义 `BaseConfig`, 让 dataclass 也能像 dict 一样 `get()` 和 `[]` 访问。
- `verl/trainer/config/algorithm.py` 定义 `AlgoConfig`、`KLControlConfig`、`RolloutCorrectionConfig`。
- `verl/workers/config/` 目录定义 actor、critic、engine、optimizer、model、rollout、reward 的 dataclass。
- `verl/utils/config.py` 中的 `omega_conf_to_dataclass` 被多个地方调用, 例如 `ActorRolloutRefWorker.init_model()`、`PPOTrainer.init_workers()`、`SFTTrainer._build_config()`。

初学时可以这样读配置:

1. 从 `ppo_trainer.yaml` 看全局字段。
2. 进入 `trainer/config/actor/dp_actor.yaml` 或 `megatron_actor.yaml` 看 actor 差异。
3. 进入 `trainer/config/engine/*.yaml` 看后端并行、offload、micro batch 参数。
4. 对照 `verl/workers/config/*.py` 看这些 YAML 字段最终变成什么类型。

## 4. 数据协议: DataProto、TensorDict 和 TransferQueue

### 4.1 DataProto 是旧主线的标准批数据结构

`verl/protocol.py` 里的 `DataProto` 是控制器和 worker 之间传递 batch 的通用结构:

```python
@dataclass
class DataProto:
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
```

它拆成三类数据:

- `batch`: 张量数据, 用 `TensorDict` 保存, 要求 batch 维一致。
- `non_tensor_batch`: 非张量数据, 用 `np.ndarray` 保存, 例如 `uid`、`data_source`、`reward_model`。
- `meta_info`: 元信息, 例如 metrics、padding 信息、token 数统计。

关键方法:

- `from_dict()` / `from_single_dict()`: 把普通 dict 分成 tensor 和 non-tensor。
- `pop()`: 从 batch 中取出部分字段, 常用于生成前只取 prompt 相关字段。
- `union()`: 把生成结果、logprob、value、reward、advantage 合回同一个 batch。
- `chunk()`: 按数据并行维度切分给 worker。
- `concat()`: 把多个 worker 输出拼回 controller。
- `DataProtoFuture`: 包装 Ray futures, 允许 controller 不立即 `ray.get`, 为异步/流水提供空间。
- `BatchData`: 统一处理 `DataProto`、`DataProtoFuture`、`TensorDict`、TransferQueue 的 `KVBatchMeta`/`BatchMeta` 的 chunk 和 concat。

为什么这样设计:

- RL 训练每一步都在不断给同一批样本追加字段。`union()` 让主循环可以写成 “batch 加字段” 的方式。
- 分布式调用需要切 batch 和合 batch。`chunk()`/`concat()` 把这个规则放到数据对象里, 不需要每个 worker 方法重复写。
- prompt、reward metadata、工具调用结果不一定是 tensor, 所以需要 `non_tensor_batch`。

### 4.2 当前同步 PPO 更多使用 TensorDict + TransferQueue

`main_ppo_sync.py` 中大量步骤使用 `KVBatchMeta` 和 TransferQueue:

- rollout worker 把结果写入 TransferQueue。
- `ReplayBuffer.sample()` 从 TransferQueue 的 metadata 中等待某个 `global_steps` 的样本全部完成。
- `_compute_old_log_prob()`、`_compute_ref_log_prob()`、`_compute_values()` 从 TransferQueue 读取字段, 让 worker 计算后再写回。
- `_compute_advantage()` 把需要的字段取出, 临时转成 `DataProto`, 调用 `compute_advantage_for_multi_trajectories()`, 再写回 TransferQueue。

这说明当前代码不是抛弃 `DataProto`, 而是在同步 PPO 主链路里用 TransferQueue 优化数据搬运, 在算法计算处仍复用 `DataProto` 和 `core_algos.py`。

## 5. single_controller: 为什么 controller 可以像本地函数一样调用远程 worker

核心文件:

- `verl/single_controller/base/decorator.py`
- `verl/single_controller/base/worker.py`
- `verl/single_controller/base/worker_group.py`
- `verl/single_controller/ray/base.py`

### 5.1 register 装饰器

worker 方法使用 `@register(...)` 标记调度方式。例如:

- `TrainingWorker.reset()` 使用 `Dispatch.ONE_TO_ALL`, 表示每个 worker 都执行。
- `TrainingWorker.train_mini_batch()` 使用 `make_nd_compute_dataproto_dispatch_fn(mesh_name="train")`, 表示按实际 engine 的数据并行拓扑切分。
- `ActorRolloutRefWorker.compute_log_prob()` 使用 `mesh_name="actor"`。
- `ActorRolloutRefWorker.compute_ref_log_prob()` 使用 `mesh_name="ref"`。

`register()` 会给函数挂上一个特殊属性, `RayWorkerGroup` 创建时读取这些属性, 生成同名代理方法。controller 侧调用:

```python
self.actor_rollout_wg.update_actor(batch)
```

实际发生的是:

1. dispatch 函数把 `batch` 切成若干份或复制到所有 worker。
2. Ray 对每个 remote actor 调用对应方法。
3. collect 函数把结果拼回 controller。
4. 如果是 non-blocking, 返回 `DataProtoFuture` 或 future 包装。

代码证据:

- `decorator.py` 中定义 `Dispatch`、`Execute`、`dispatch_dp_compute_data_proto()`、`collect_dp_compute_data_proto()`、`make_nd_compute_dataproto_dispatch_fn()`、`register()`。
- `ray/base.py` 中 `func_generator()` 把 dispatch、execute、collect 串起来。
- `RayWorkerGroup.execute_all_async()` 根据 worker 数量把参数分发给每个 actor。

### 5.2 ResourcePool 和 WorkerGroup

`RayResourcePool` 负责用 Ray placement group 分配 GPU/CPU bundle。`ResourcePoolManager` 根据配置里的:

```python
resource_pool_spec = {
    "global_pool": [n_gpus_per_node] * nnodes
}
```

创建资源池, 并按 role 映射到资源池。

为什么需要 role 到 resource pool 的映射:

- actor/rollout/ref 可以共用 GPU, 这是 on-policy RL 常见的 colocate/hybrid engine。
- reward model 或 teacher model 可以单独资源池, 避免挤占训练/rollout 内存。
- 同一个算法控制流不用关心这些放置细节。

### 5.3 colocated worker

`create_colocated_worker_cls()` 会把多个角色的 worker 类放进同一个 Ray actor 里。`main_ppo_sync.py` 的 `PPOTrainer.init_workers()` 中:

```python
worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

这样 controller 仍然看到多个 worker group, 但底层可以共享同一批进程和 GPU context。

为什么这样做:

- actor 和 rollout 频繁同步权重, colocate 可以减少跨进程/跨节点传输。
- reference policy 和 actor 可能共享基础模型, LoRA 场景下可以通过禁用 adapter 得到 ref logprob。
- critic/reward/teacher 可以按配置决定是否同池或独立池。

## 6. worker 层: 统一封装训练、推理和 rollout

核心文件是 `verl/workers/engine_workers.py`。它定义两个关键类:

- `TrainingWorker`: 一个通用的 “模型引擎 + optimizer + profiler + loss_fn” worker。
- `ActorRolloutRefWorker`: 一个混合 worker, 根据 role 组合 actor、rollout、ref 和 checkpoint engine。

### 6.1 TrainingWorker

构造逻辑:

1. 初始化 Ray 分布式进程组: `initialize_global_process_group_ray()`。
2. 保存 `TrainingWorkerConfig` 中的 model、engine、optimizer、checkpoint、profiler 配置。
3. 如果没有显式 engine_config, 可以通过 `auto_select_engine_optim_fn` 自动选择。
4. 调用 `EngineRegistry.new(model_type, backend, ...)` 创建具体 engine。
5. 注册 dispatch 信息: `mesh_name="train"`, 记录 engine 的 data parallel rank 和哪些 rank 负责输出。

关键方法:

- `reset()`: 调 `engine.initialize()`, 初始化或重置模型、优化器、scheduler。
- `set_loss_fn()`: 安装训练/推理要用的 loss closure。
- `train_mini_batch()`: 把一个全局 batch 切成 PPO mini-batch, 迭代多 epoch, 每个 mini-batch 调 `train_batch()`。
- `train_batch()`: 调 `engine.train_batch(data, loss_function=self.loss_fn)`, 更新一次参数。
- `infer_batch()`: eval/no_grad 下调 `engine.infer_batch()`, 用于 logprob、value、reward、teacher。
- `save_checkpoint()` / `load_checkpoint()`: 委托给具体 engine。

为什么 `TrainingWorker` 不直接写 FSDP/Megatron:

- controller 只需要 “训练一个 batch” 和 “推理一个 batch”。
- FSDP 和 Megatron 的模型构建、forward/backward、checkpoint 差异很大, 这些差异放到 engine 层。
- 同一个 `TrainingWorker` 可用于 actor、critic、reward model、SFT、DPO、distillation。

### 6.2 ActorRolloutRefWorker

`ActorRolloutRefWorker.__init__()` 根据 `role` 决定是否包含:

- actor: `self.actor: TrainingWorker`
- rollout: `self.rollout: BaseRollout`
- ref: `self.ref: TrainingWorker`

`init_model()` 分四段:

1. 如果 role 含 `ref`, 构建 ref `TrainingWorker`, 强制 ref 不启用 MTP, 设置 `mesh_name="ref"`。
2. 如果 role 含 `actor`, 构建 actor `TrainingWorker`, 安装 `ppo_loss` 或 distillation PPO loss, 设置 `mesh_name="actor"`。
3. 如果 role 含 `rollout`, 根据 `RolloutConfig` 创建 device mesh, 通过 `get_rollout_class()` 选择 vLLM/SGLang/TRT-LLM server adapter。
4. 如果 role 含 `actor`, 创建 checkpoint engine, 用于 trainer 到 rollout 的权重同步。

对 controller 暴露的关键方法:

- `compute_log_prob()`: actor 做 inference, 输出当前 actor 对响应 token 的 log probability。
- `compute_ref_log_prob()`: ref 做 inference, 输出参考策略 log probability。
- `update_actor()`: actor 调 `train_mini_batch()` 更新参数。
- `update_weights()`: 把 actor 新权重同步给 rollout engine。
- `save_checkpoint()` / `load_checkpoint()`: 保存/恢复 actor。

为什么 actor、rollout、ref 放在一个类:

- on-policy PPO 中 actor 训练后马上要把权重推给 rollout。
- actor/ref/rollout 经常共享同一模型结构和 tokenizer。
- LoRA 场景下 ref 可以是 actor 禁用 adapter 后的输出, 避免单独加载完整 ref 模型。

## 7. engine 层: 把训练后端差异隔离起来

`verl/workers/engine/base.py` 定义 `BaseEngine` 和 `EngineRegistry`。

`BaseEngine` 要实现:

- `initialize()`: 构建模型、优化器、scheduler、checkpoint manager。
- `train_mode()` / `eval_mode()`: 训练/推理上下文, 包括 offload 进出 GPU。
- `forward_backward_batch()`: 后端特定的 forward/backward。
- `train_batch()`: 基类实现了 zero_grad -> forward_backward -> optimizer_step。
- `infer_batch()`: 基类实现了 no_grad -> forward_backward(forward_only=True)。
- `get_per_tensor_param()`: 导出一组 `(name, tensor)`, 用于同步到 rollout。
- `save_checkpoint()` / `load_checkpoint()`。
- data parallel rank/size/group 和 `is_mp_src_rank_with_outputs()`。

`EngineRegistry.register(model_type, backend, device)` 把具体 engine 注册进去。当前可见注册点:

- `FSDPEngineWithLMHead`: `model_type="language_model"`, `backend=["fsdp", "fsdp2"]`, `device=["cuda", "npu"]`。
- `FSDPEngineWithValueHead`: `model_type="value_model"`, `backend=["fsdp", "fsdp2"]`。
- `MegatronEngineWithLMHead`: `model_type="language_model"`, `backend="megatron"`。
- `MegatronEngineWithValueHead`: `model_type="value_model"`, `backend="megatron"`。
- `VeOmniEngineWithLMHead` / `VeOmniEngineWithValueHead`: `backend="veomni"`。
- `TorchTitanEngineWithLMHead`: `backend="torchtitan"`。
- `AutomodelEngineWithLMHead`: `backend="automodel"`。
- MindSpeed 相关 engine 通过 `workers/engine/mindspeed/` 支持 NPU/Megatron 变体。

为什么 actor 和 critic 用 `model_type` 区分:

- actor 是 language model with LM head, 输出 token log probabilities 和 entropy。
- critic 是 value model, 对 response token 输出 values。
- 它们可以共用大量输入准备、并行和 checkpoint 逻辑, 但输出后处理不同。

如果你要理解某个后端:

- FSDP: 看 `verl/workers/engine/fsdp/transformer_impl.py`。
- Megatron: 看 `verl/workers/engine/megatron/transformer_impl.py`。
- VeOmni: 看 `verl/workers/engine/veomni/transformer_impl.py`。
- TorchTitan: 看 `verl/workers/engine/torchtitan/transformer_impl.py`。
- Automodel: 看 `verl/workers/engine/automodel/transformer_impl.py`。

每个后端都围绕同几个方法展开: `initialize()`、`forward_backward_batch()`、`get_per_tensor_param()`、`save_checkpoint()`、`load_checkpoint()`。

## 8. rollout 层: 用推理服务生成样本

核心文件:

- `verl/workers/rollout/base.py`
- `verl/workers/rollout/replica.py`
- `verl/workers/rollout/llm_server.py`
- `verl/workers/rollout/vllm_rollout/`
- `verl/workers/rollout/sglang_rollout/`
- `verl/workers/rollout/trtllm_rollout/`

`BaseRollout` 定义 rollout adapter 的最小接口:

- `resume(tags)`: 恢复 weights 或 kv_cache。
- `update_weights(weights, **kwargs)`: 接收训练 actor 权重。
- `release()`: 释放 weights 和 kv cache。
- `generate_sequences(prompts)`: 同步生成接口, 当前 server mode 更多由 agent loop/LLM client 调用。

`get_rollout_class()` 根据 `(rollout_name, mode)` 选择:

- `("vllm", "async") -> verl.workers.rollout.vllm_rollout.ServerAdapter`
- `("sglang", "async") -> verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter`
- `("trtllm", "async") -> verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter`

`RolloutReplica` 表示一个具体推理服务副本, 支持三种模式:

- `HYBRID`: rollout engine 和训练 engine 融合在同一个 worker 进程, 适合 on-policy 训练。
- `COLOCATED`: rollout server 和 hybrid engine 在同一个 placement group, 但不同进程。
- `STANDALONE`: rollout server 用独立 GPU 资源池, 适合解耦/异步/off-policy。

为什么 rollout 单独成层:

- vLLM、SGLang、TRT-LLM 的服务启动、权重更新、KV cache 管理完全不同。
- 训练 actor 用 FSDP/Megatron 等后端, 生成样本用推理引擎, 两者生命周期不同。
- PPO 每步需要先生成、再训练、再把新权重同步到生成服务。

## 9. checkpoint engine: 训练权重如何同步给 rollout

这里有两类 checkpoint:

1. 训练 checkpoint: 保存/恢复模型、优化器、scheduler、dataloader, 主要在 `verl/utils/checkpoint/` 和 engine 实现里。
2. rollout 权重同步: 每步 actor 更新后把权重传给 rollout, 主要在 `verl/checkpoint_engine/`。

`verl/checkpoint_engine/base.py` 定义:

- `CheckpointEngineRegistry`
- `CheckpointEngine`
- `ColocatedCheckpointEngine`, 注册名是 `"naive"`
- `CheckpointEngineWorker`
- `CheckpointEngineManager`

`ActorRolloutRefWorker.update_weights()` 有两条路径:

- `backend == "naive"`: actor 和 rollout colocated, 直接 `engine.get_per_tensor_param()` 后调用 `rollout.update_weights()`。
- 非 naive: 通过 `checkpoint_engine.send_weights()` 传给 rollout 侧的 `CheckpointEngineWorker`, 可用于 NCCL/NIXL 等异步或解耦传输。

`CheckpointEngineManager.update_weights()` 协调多 replica:

1. naive 时直接调用 trainer worker 的 `update_weights()`。
2. 非 naive 时先 abort/保存未完成 rollout 请求。
3. 临时组装 rollout worker group。
4. 释放 kv cache。
5. 构建 trainer 和 rollout 之间的通信 topology。
6. trainer 发送权重, rollout 接收并更新。
7. finalize, 恢复 kv cache 和未完成请求。

为什么这层必要:

- 训练后端的权重通常是分片的, rollout 引擎需要可加载的权重视图。
- colocated 和 disaggregated 的最优同步方式不同。
- 大模型权重同步很重, 需要 bucket、RDMA/NCCL、KV cache 释放等工程细节。

## 10. PPO 同步主流程逐步拆解

主类是 `PPOTrainer` in `verl/trainer/main_ppo_sync.py`。

### 10.1 初始化

`PPOTrainer.__init__()` 做:

- 保存 config、role mapping、resource pool manager。
- 判断是否需要 critic/reference/teacher。
- 创建 `ReplayBuffer()`。
- 如果 `algorithm.use_kl_in_reward=True`, 创建 KL controller。
- `_init_tokenizer()`: 从 actor 模型路径加载 tokenizer/processor。
- `_init_dataloader()`: 创建 RLHF train/val dataset 和 StatefulDataLoader。
- `_init_dump_executor()`: 创建 rollout/validation 数据 dump 的线程池。

### 10.2 init_workers

`init_workers()` 是从配置到远程 worker 的关键:

1. `resource_pool_manager.create_resource_pool()`: 创建 Ray placement groups。
2. 根据 role 创建 actor_rollout worker:
   - 如果需要 reference policy 且 ref 不在 actor 内, 用 `Role.ActorRolloutRef`。
   - 否则用 `Role.ActorRollout`。
3. 如果需要 critic, 构建 `TrainingWorkerConfig(model_type="value_model")`。
4. 用 `create_colocated_worker_cls()` 和 `RayWorkerGroup` 创建 colocated workers。
5. critic 调 `reset()` 初始化 engine, 再 `set_loss_fn(value_loss)`。
6. actor_rollout 调 `init_model()` 初始化 actor/ref/rollout/checkpoint engine。
7. 如果 LoRA 使 ref 在 actor 内, 设置 `ref_in_actor`。
8. 初始化 reward loop manager。
9. 如果需要 distillation, 初始化 teacher model manager。
10. 初始化 LLM server manager 和 agent loop manager。
11. 初始化 `CheckpointEngineManager`。
12. 先让 rollout replicas sleep, 为加载 checkpoint 和同步权重留内存。

### 10.3 fit

`fit()` 的结构:

1. 创建 logger。
2. `_load_checkpoint()`: 根据 resume 配置恢复 actor/critic/dataloader。
3. `checkpoint_manager.update_weights()`: 先把当前 actor 权重同步给 rollout。
4. 可选初始 validation。
5. 外层 epoch, 内层 train dataloader。
6. 每步调用 `self.step(batch_dict, metrics, timing_raw)`。
7. 按频率保存 checkpoint。
8. 每步结束再次 `checkpoint_manager.update_weights()`。
9. 按频率 validation。
10. 计算 metrics、dump rollout、清理 TransferQueue、logger.log。

### 10.4 step

`step()` 是 PPO 数据流最清楚的地方:

1. 给每条 prompt 分配 `uid`。
2. 如果是 ReMax, 额外构造 greedy baseline batch。
3. `async_rollout_manager.generate_sequences(batch)`: 把 prompt 发给 agent loop/LLM server。
4. `replay_buffer.sample(partition_id="train", global_steps=...)`: 等 rollout 结果写入 TransferQueue。
5. `checkpoint_manager.sleep_replicas()`: 释放 rollout 内存, 让训练/奖励等计算有显存。
6. 如果 reward loop 没有远程 worker, 调 `_compute_reward_colocate()`。当前同步 PPO 这里是未实现/NotImplemented, 实际主线更倾向 reward loop manager 写好 reward。
7. ReMax 时 `_add_remax_reward_baselines()`。
8. `_balance_batch()`: 按序列长度重排, 让各 DP rank token 量更均衡。
9. `_compute_old_log_prob()`: 用 actor 重新算 old logprob, 或 bypass mode 直接使用 rollout logprob。
10. `_compute_ref_log_prob()` 可选。
11. `_compute_values()` 可选。
12. `_compute_advantage()`: KL reward、rollout correction、advantage/return。
13. `_update_critic()` 可选。
14. `_update_actor()`。

为什么 old logprob 要重算:

- rollout 引擎和训练引擎可能数值不完全一致。
- PPO 的 ratio 通常需要一个稳定的旧策略 `pi_old` 作为近端约束。
- 代码里支持 bypass mode, 即直接把 rollout logprob 当 old logprob, 用于减少一次 actor inference。

### 10.5 reward 和 advantage

`_compute_advantage()` 从 TransferQueue 取:

- `uid`
- `response_mask`
- `rm_scores`
- `rollout_log_probs`
- `old_log_probs`
- `ref_log_prob`
- `values`
- ReMax 额外的 `reward_baselines`

然后:

1. 转成 `DataProto`。
2. `token_level_scores = rm_scores`。
3. 如果 `algorithm.use_kl_in_reward`, 调 `apply_kl_penalty()`:
   - `kld = kl_penalty(old_log_probs, ref_log_prob)`
   - `token_level_rewards = token_level_scores - beta * kld`
   - KL controller 更新 beta。
4. 否则 `token_level_rewards = token_level_scores`。
5. 如果开启 rollout correction, 调 `compute_rollout_correction_and_add_to_batch()`。
6. 调 `compute_advantage_for_multi_trajectories()`。
7. 把 `advantages`、`returns`、可选 `token_level_rewards`/`rollout_is_weights` 写回 TransferQueue。

`compute_advantage_for_multi_trajectories()` 对 GRPO 有特殊处理:

- agent loop 可能一个 session 产生多个输出。
- GRPO 只用每个 session 的最终输出算 group relative advantage。
- 算完再把 final score broadcast 回同 session 的其他输出。
- 非 GRPO 直接调用传统 `compute_advantage()`。

### 10.6 actor/critic update

`_update_critic()`:

- 设置 `global_batch_size`、`mini_batch_size`、`epochs`、`seed`、dataloader shuffle。
- 调 `critic_wg.train_mini_batch(batch)`。
- 返回的 metrics 加 `critic/` 前缀。

`_update_actor()`:

- 设置是否计算 entropy。
- distillation 时设置 topk。
- 设置 PPO mini-batch size、epochs、temperature。
- 调 `actor_rollout_wg.update_actor(batch)`。
- 返回的 metrics 加 `actor/` 前缀。

真正 loss 在 worker 内执行:

- actor loss: `verl/workers/utils/losses.py::ppo_loss()`
- critic loss: `verl/workers/utils/losses.py::value_loss()`
- worker 调度: `TrainingWorker.train_mini_batch()` -> `TrainingWorker.train_batch()` -> `BaseEngine.train_batch()` -> 后端 `forward_backward_batch()`。

## 11. 算法核心: core_algos.py 如何支持多种 PPO-like 算法

`verl/trainer/ppo/core_algos.py` 有两个注册表:

- `ADV_ESTIMATOR_REGISTRY`: advantage/return 估计器。
- `POLICY_LOSS_REGISTRY`: actor policy loss。

它们的设计意义是: trainer 不关心具体算法公式, trainer 只负责把 rollout、logprob、reward、mask、values 等字段准备好, 然后按配置名调一个函数。这样 PPO、GRPO、RLOO、ReMax、OTB 等算法可以复用同一套分布式 worker、rollout 和 checkpoint 代码。

### 11.1 advantage estimator

枚举 `AdvantageEstimator` 在 `verl/trainer/ppo/core_algos.py:88` 定义, 当前包含:

- `gae`
- `grpo`
- `reinforce_plus_plus`
- `reinforce_plus_plus_baseline`
- `remax`
- `rloo`
- `opo`
- `grpo_passk`
- `gpg`
- `rloo_vectorized`
- `grpo_vectorized`
- `optimal_token_baseline`
- `tir_optimal_token_baseline`
- `gdpo`

注册方式:

```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(...):
    ...
```

调度入口有两个:

- 旧 `DataProto` 主线: `verl/trainer/ppo/ray_trainer.py:185` 的 `compute_advantage()`。
- 当前同步 PPO 主线: `verl/trainer/main_ppo_sync.py:1405` 的 `_compute_advantage()` 先从 TransferQueue 取字段, 再调用 `compute_advantage_for_multi_trajectories()`。

同步 PPO 多了一层 `compute_advantage_for_multi_trajectories()` (`verl/trainer/main_ppo_sync.py:122`): 如果不是 GRPO, 直接转给旧 `compute_advantage()`; 如果是 GRPO, 它只拿每个 `{uid}_{session_id}` 的最后一条输出做组内 advantage, 再把最终分数 broadcast 回同一 session 的所有输出 token。这是为了适配 agent loop 可能产生多轮输出的场景。

所有 estimator 的输入字段基本来自 `_compute_advantage()`:

- `token_level_rewards`: reward 或扣过 KL 的 reward。
- `response_mask`: 只让 response 有效 token 参与计算。
- `uid`: 同一个 prompt 的多条采样分组。
- `values`: GAE 需要 critic value。
- `reward_baselines`: ReMax 需要 greedy baseline reward。
- `old_log_probs` 和 `sum_pi_squared`: OTB 需要路径方差代理量。
- `rollout_is_weights`: 解耦 rollout correction 可能产生 importance sampling 权重。

逐个 estimator 对照源码:

| 配置值 | 源码函数 | 核心代码逻辑 | 为什么这样做 |
| --- | --- | --- | --- |
| `gae` | `compute_gae_advantage_return()` (`core_algos.py:216`) | 从 response 最后一个 token 反向循环。每步计算 `delta = reward_t + gamma * next_value - value_t`, 再递推 `lastgaelam = delta + gamma * lam * lastgaelam`; `response_mask` 用来跳过 EOS 后 padding; `returns = advantages + values`; 最后 `masked_whiten(advantages, response_mask)`。 | 这是标准 PPO + critic。critic 给每个 token 一个 baseline, GAE 在 bias 和 variance 之间折中。 |
| `grpo` | `compute_grpo_outcome_advantage()` (`core_algos.py:268`) | 先 `scores = token_level_rewards.sum(dim=-1)` 把 outcome reward 变成每条 response 一个标量; 按 `index/uid` 分组; 组内算 mean/std; 每条样本用 `(score - group_mean) / (group_std + eps)` 或只减 mean; 再乘 `response_mask` broadcast 到所有 response token。 | GRPO 不用 critic, 让同一 prompt 的多条回答互相当 baseline, 适合数学/代码等 outcome reward。 |
| `grpo_vectorized` | `compute_grpo_vectorized_outcome_advantage()` (`core_algos.py:335`) | 和 GRPO 公式相同, 但通过 `as_torch_index()` 与 `group_mean_std()` 一次性算 group mean/std。 | 避免 Python dict 循环, 大 batch 时更快。 |
| `gdpo` | `compute_gdpo_outcome_advantage()` (`core_algos.py:362`) | 如果配置了 `algorithm.gdpo_reward_keys`, 就从 `non_tensor_batch` 取多个 reward 维度, 把每个维度放到对应 response 的最后有效 token; 每个维度分别调用 GRPO 归一化; 再按 `gdpo_reward_weights` 加权求和; 最后 `masked_whiten()`。 | 多维 reward 直接相加会让大尺度 reward 淹没小尺度 reward。GDPO 先逐维归一化, 再聚合。 |
| `grpo_passk` | `compute_grpo_passk_outcome_advantage()` (`core_algos.py:472`) | 每个 prompt 组要求至少 2 条 response; 取组内 top2 reward; 只有最优 response 获得 `r_max - r_second_max` 的 advantage, 可选除以组内 std; 其他 response 为 0。 | Pass@k 优化关心“组里最好的一条是否更好”, 不是平均提升所有样本。 |
| `reinforce_plus_plus_baseline` | `compute_reinforce_plus_plus_baseline_outcome_advantage()` (`core_algos.py:536`) | outcome reward 求和成 `scores`; 按 `uid` 算组内均值; 每条 response 减组均值; 把标量平铺到 response 长度; 再做 `masked_whiten()`。 | 这是不用 critic 的 REINFORCE++ baseline 版本, 用组均值降方差。 |
| `rloo` | `compute_rloo_outcome_advantage()` (`core_algos.py:588`) | outcome reward 求和; 按组算均值和组大小 `n`; 当 `n > 1` 时使用 `score * n/(n-1) - mean * n/(n-1)`, 等价于减去其他样本的 leave-one-out 均值; broadcast 到 token。 | 当前样本不能参与自己的 baseline, 可以减少 baseline 偏差。 |
| `rloo_vectorized` | `compute_rloo_vectorized_outcome_advantage()` (`core_algos.py:832`) | 用 `np.unique(..., return_inverse=True)` 和 `torch.bincount()` 计算同样的 leave-one-out advantage。 | 与 `rloo` 含义一致, 但向量化实现更适合大 batch。 |
| `opo` | `compute_opo_outcome_advantage()` (`core_algos.py:640`) | 先算每条 response 的长度和 score; 组内 baseline 是 `sum(length * score) / sum(length)`; 每条 response 的 score 减这个长度加权 baseline。 | OPO 把 response 长度纳入 baseline, 避免长短回答在 outcome reward 下产生系统偏差。 |
| `reinforce_plus_plus` | `compute_reinforce_plus_plus_outcome_advantage()` (`core_algos.py:694`) | 反向做 reward-to-go: `running_return = reward_t + gamma * running_return`; EOS 后用 `response_mask` 重置; `returns` 做 `masked_whiten()` 得到 advantages。 | 不依赖 critic 或组采样, 是 token 级 REINFORCE++ 回报估计。 |
| `remax` | `compute_remax_outcome_advantage()` (`core_algos.py:733`) | 对 token reward 做反向累加得到 `returns`; 用 `returns - reward_baselines.unsqueeze(-1) * response_mask` 得到 advantage。 | ReMax 用 greedy decoding 的 reward 当 baseline, 对比“采样回答比贪心回答好多少”。 |
| `gpg` | `compute_gpg_outcome_advantage()` (`core_algos.py:769`) | outcome score 按 `uid` 分组; 统计非零 reward 数 `m`, 令 `alpha = batch_size / max(m, 1)`; advantage 是 `alpha * (score - group_mean) / f_norm`, 再 broadcast。 | GPG 对稀疏非零 reward 做缩放, 减少多数 0 reward 把梯度冲淡。 |
| `optimal_token_baseline` | `compute_optimal_token_baseline_advantage()` (`core_algos.py:870`) | 先算 token-level reward-to-go `returns`; 用 `pi_t = exp(old_log_probs)` 和 `sum_pi_squared` 构造 `w_t = 1 - 2*pi_t + sum_pi_squared`; 可乘 `rollout_is_weights ** 2`; 累加成 `W_t`; 每个 prompt 组在每个 timestep 算 `baseline_t = sum(G_t * W_t * mask) / sum(W_t * mask)`; advantage 是 `returns - baseline`。 | baseline 不再是整条 trajectory 一个数, 而是每个 token 一个最优 baseline, 用路径方差给高方差位置更合理的权重。 |
| `tir_optimal_token_baseline` | `compute_multi_turn_optimal_token_baseline_advantage()` (`core_algos.py:989`) | 先把多轮 response 中 `response_mask=True` 的 token 压缩成连续序列, 在压缩序列上按 OTB 算 baseline, 再散回原 token 位置。 | Tool-integrated / multi-turn rollout 的有效 response token 可能不连续, 需要先映射到有效 token 轨迹再算 OTB。 |

为什么 advantage estimator 独立成注册表:

- 同一 PPO 训练骨架中, rollout、logprob、reward、actor update 逻辑高度相似。
- 不同算法主要差在如何把 reward/value 转成 advantage/return。
- 新算法可以注册 estimator, 不必复制 trainer。

### 11.2 KL controller 和 KL penalty

KL controller:

- `FixedKLController`: beta 固定。
- `AdaptiveKLController`: 根据当前 KL 和目标 KL 调整 beta。
- `get_kl_controller()` 根据 `algorithm.kl_ctrl.type` 选择。

KL penalty:

- `kl_penalty()` 和 `kl_penalty_forward()` 支持 `kl`/`k1`、`abs`、`mse`/`k2`、`low_var_kl`/`k3`、`full`。
- `apply_kl_penalty()` 在 `ray_trainer.py` 中把 KL 从 reward 里扣掉。

为什么 KL 可以在 reward 里, 也可以在 actor loss 里:

- `algorithm.use_kl_in_reward=True` 时, KL 进入 `token_level_rewards`。
- `actor.use_kl_loss=True` 时, `ppo_loss()` 会额外把 KL loss 加到 policy loss。
- 两种方式对应不同 RLHF 配方。

### 11.3 policy loss

`POLICY_LOSS_REGISTRY` 在 `verl/trainer/ppo/core_algos.py:50` 定义, `get_policy_loss_fn()` 在 `core_algos.py:70` 根据 `actor.policy_loss.loss_mode` 找函数。

`workers/utils/losses.py::ppo_loss()` 做:

1. 从 model output 取当前 `log_probs` 和 entropy。
2. 从 batch 取 `response_mask`、`old_log_probs`、`advantages`、可选 `rollout_is_weights`、`ref_log_prob`。
3. 根据 `config.policy_loss.loss_mode` 调 `get_policy_loss_fn(loss_mode)`。
4. policy loss 返回 `pg_loss` 和 metrics。
5. 可选加 entropy loss。
6. 可选加 KL loss。

这就是 actor update 与算法 loss 的连接点。

actor loss 的完整调用链是:

```text
PPOTrainer._update_actor()
  -> ActorRolloutRefWorker.update_actor()
  -> TrainingWorker.train_mini_batch()
  -> TrainingWorker.train_batch()
  -> BaseEngine.train_batch()
  -> 后端 forward_backward_batch()
  -> workers/utils/losses.py::ppo_loss()
  -> core_algos.py::get_policy_loss_fn(loss_mode)
```

逐个 policy loss 对照源码:

| `loss_mode` | 源码函数 | 核心代码逻辑 | 为什么这样做 |
| --- | --- | --- | --- |
| `vanilla` | `compute_policy_loss_vanilla()` (`core_algos.py:1279`) | `negative_approx_kl = log_prob - old_log_prob`; `ratio = exp(clamp(negative_approx_kl))`; 两个 loss 分别是 `-A * ratio` 和 `-A * clamp(ratio, 1-low, 1+high)`; 对负 advantage 还支持 dual-clip `clip_ratio_c`; 可乘 `rollout_is_weights`; 最后 `agg_loss()`。 | 标准 PPO clipped objective, 限制新旧策略比值, 防止一次更新过大。 |
| `dppo_tv` | `compute_policy_loss_dppo_tv()` (`core_algos.py:1373`) | 先算 `ratio`, 再用 `clip_ratio_c` 做 truncated importance sampling; 用当前 token 概率和旧概率的差值构造 TV-style valid mask: 正 advantage 限制 `prob - old_prob` 不超过上界, 负 advantage 限制不低于下界; loss 是 `-A * truncated_ratio * log_prob * valid_mask`。 | DPPO 的 TV 版本把“允许更新的 token”变成二值 mask, 超出 TV 阈值的 token 不给梯度。 |
| `dppo_kl` | `compute_policy_loss_dppo_kl()` (`core_algos.py:1454`) | 与 `dppo_tv` 类似, 但 valid mask 来自 binary KL: `old_prob * (old_log_prob - log_prob) + (1-old_prob) * log((1-old_prob)/(1-prob))`; 正负 advantage 分别有不同放行条件。 | 用 KL 阈值而不是概率差阈值控制单 token 的更新幅度。 |
| `gspo` | `compute_policy_loss_gspo()` (`core_algos.py:1539`) | 先对每条 response 求序列级平均 log-ratio: `sum(log_prob-old_log_prob)/length`; 用 stop-gradient 序列 ratio 和当前 token `log_prob - log_prob.detach()` 组合; 再按 PPO clipping; 强制用 `seq-mean-token-mean` 聚合。 | GSPO 把重要性比值提升到序列级, 更贴近 outcome reward 对整条回答打分的场景。 |
| `sapo` | `compute_policy_loss_sapo()` (`core_algos.py:1615`) | 用 `tau_pos/tau_neg` 区分正负 advantage; `gate = sigmoid(tau * (ratio - 1)) * (4/tau)`; loss 是 `-gate * advantage`, 并用 `seq-mean-token-mean` 聚合。 | SAPO 用平滑 gate 代替硬 clipping, 让正负样本有不同更新温度。 |
| `gpg` | `compute_policy_loss_gpg()` (`core_algos.py:1700`) | 不用 PPO ratio, 直接 `pg_losses = -log_prob * advantages`, 可乘 rollout IS 权重, 再聚合。 | 与 GPG advantage 配套, 是更直接的 policy gradient 形式。 |
| `clip_cov` | `compute_policy_loss_clip_cov()` (`core_algos.py:1736`) | 先算 vanilla PPO 的 clipped loss; 再计算 `(advantages - mean_adv) * (log_prob - mean_log_prob)` 的协方差指标; 对未被原始 PPO clip、且协方差落在 `[clip_cov_lb, clip_cov_ub]` 的 token 随机选 top 比例, 把 `corr` 置 0; 最终 loss 乘 `corr`。 | 对高协方差 token 额外裁剪, 控制可能导致过强更新的 token。 |
| `kl_cov` | `compute_policy_loss_kl_cov()` (`core_algos.py:1841`) | 基础 loss 是 `-A * ratio`; 计算 advantage 与 logprob 的协方差, 选 top 比例 token; 这些 token 的 loss 改成 `-A * ratio + ppo_kl_coef * abs(log_prob-old_log_prob)`。 | 不直接丢弃高协方差 token, 而是对它们额外加 KL 惩罚。 |
| `geo_mean` | `compute_policy_loss_geo_mean()` (`core_algos.py:1921`) | 对 token log-ratio 先按 advantage 符号裁剪, 再对整条 response 做几何平均: `ratio = exp(sum(clipped_log_ratio * mask)/length)`; advantage 也聚合成序列平均; loss 是 `-sequence_advantage * sequence_ratio`; rollout IS 权重也按几何平均聚合。 | GMPO/geo-mean loss 在序列级控制更新, 适合 sequence-level reward。 |
| `cispo` | `compute_policy_loss_cispo()` (`core_algos.py:2007`) | 计算 `ratio`, clip 后立刻 `detach`; loss 是 `-stopgrad(clipped_ratio) * advantage * log_prob`; clipping 只影响权重, 梯度只从 `log_prob` 走。 | CISPO 把重要性权重当作无梯度权重, 避免 ratio clipping 本身引入梯度路径。 |
| `bypass_mode` | `compute_policy_loss_bypass_mode()` (`core_algos.py:2352`) | 约定 `old_log_prob` 实际是 `rollout_log_prob`; 在 loss 内调用 `compute_rollout_correction_and_rejection_mask()` 计算 IS 权重和 rejection mask; `loss_type="reinforce"` 时调 `compute_policy_loss_reinforce()` 并显式乘 IS; `loss_type="ppo_clip"` 时调 `compute_policy_loss_vanilla()` 且不再额外乘 IS。 | 用于训练策略与 rollout 策略解耦时的修正。PPO clip 已经包含 `pi_current / pi_rollout`, 所以不能再重复乘 IS。 |

`agg_loss()` (`core_algos.py:1138`) 决定 token loss 如何变成标量:

- `token-mean`: 所有有效 token 求平均。
- `seq-mean-token-sum`: 每条序列先 token sum, 再对序列平均。
- `seq-mean-token-sum-norm`: 在上面基础上除以固定 horizon 或 `loss_scale_factor`。
- `seq-mean-token-mean`: 每条序列先 token mean, 再对序列平均。

这一步很重要, 因为 FSDP/Megatron/data parallel 下, loss 的归一化必须和全局 batch/token 数一致, 否则同一配置在不同并行度下梯度尺度会变。

### 11.4 value loss

critic loss 在 `workers/utils/losses.py::value_loss()`:

1. engine 输出 `vpreds`。
2. batch 里取旧 `values`、`returns`、`response_mask`。
3. 调 `core_algos.compute_value_loss()`。
4. 它实现 PPO clipped value loss:
   - `vpredclipped = clip(vpreds, values - cliprange, values + cliprange)`
   - 取 unclipped MSE 和 clipped MSE 的 max
   - mask 后聚合。

## 12. reward: 函数奖励、reward model 和 reward loop

旧/通用 reward manager 在:

- `verl/workers/reward_manager/abstract.py`
- `verl/workers/reward_manager/registry.py`
- `verl/workers/reward_manager/naive.py`
- `verl/trainer/ppo/reward.py`

`NaiveRewardManager` 的逻辑:

1. 如果 batch 已有 `rm_scores`, 直接返回。
2. 否则逐条样本:
   - 用 tokenizer 解码 prompt 和 response。
   - 取 `reward_model["ground_truth"]`。
   - 取 `data_source`。
   - 调 `compute_score(data_source, solution_str, ground_truth, extra_info)`。
3. reward 写到 response 最后一个有效 token 上。
4. 如果 compute_score 返回 dict, 额外信息进入 `reward_extra_info`。

当前 `main_ppo_sync.py` 引入了 `verl.experimental.reward_loop.RewardLoopManager`, rollout agent 可以在生成后把 reward 写入 TransferQueue。`_compute_reward_colocate()` 在同步 PPO 类里仍未实现, 说明当前同步主线更依赖 reward loop worker 或 agent loop postprocess 完成 reward 写入。

自定义 reward 入口:

- `get_custom_reward_fn(config)` 从 `config.reward.custom_reward_function.path/name` 动态加载函数。
- `load_reward_manager()` 可按注册表或 importlib 加载 reward manager。
- 默认 reward score 在 `verl/utils/reward_score/` 下。

## 13. 数据集: RLHF 和 SFT 如何进训练

### 13.1 RLHFDataset

`verl/utils/dataset/rl_dataset.py::RLHFDataset`:

- 支持 parquet/json/jsonl。
- 本地缓存远程路径: `_download()` 调 `copy_to_local()`。
- 读成 HuggingFace Dataset: `_read_files_and_tokenize()`。
- 支持 `max_samples`、shuffle、seed。
- 支持文本、图片、视频、音频字段。
- 支持 tool schema, 让长度过滤和 rollout 看到同样 prompt 模板。
- `maybe_filter_out_long_prompts()` 按 tokenizer/processor 过滤超长 prompt。
- `__getitem__()` 返回:
  - `raw_prompt`
  - `dummy_tensor`
  - `index`
  - `tools_kwargs`
  - `interaction_kwargs`
  - 原始 reward/model/source 等字段。

当前同步 PPO 的注释明确说: apply_chat_template 已移动到 AgentLoop, 所以 dataset 返回 raw prompt。

`collate_fn()`:

- tensor 用 `torch.stack`。
- 非 tensor 转成 `np.ndarray(dtype=object)`。

### 13.2 SFT dataset

SFT 使用:

- `verl/trainer/sft_trainer.py`
- `verl/utils/dataset/multiturn_sft_dataset.py`
- `verl/utils/dataset/dataset_utils.py::SFTTensorCollator`

`SFTTrainer` 是 SPMD 风格, 不走 Ray controller:

1. `initialize_global_process_group()`。
2. 构建 config、dataset、engine、dataloader。
3. 用 `TrainingWorker` 创建 engine, 但 `self.engine = self.training_client.engine`, 本地直接用。
4. DistributedSampler 按 engine data parallel rank/size 划分数据。
5. `fit()` 每步把 batch 转成 `TensorDict`, 加 meta info, 调 `training_client.train_batch()`。
6. validation 调 `infer_batch()`。
7. checkpoint 由 `CheckpointHandler` 管理。

为什么 SFT 不需要 PPO 那套复杂 controller:

- SFT 是普通监督学习, 没有 rollout、reward、advantage、actor/ref/critic 多角色数据流。
- 只需要分布式训练一个 language model。
- 因此可以复用 `TrainingWorker` 和 engine, 但不需要 Ray worker group。

## 14. 模型目录: models 和 backend-specific 模型适配

`verl/models/` 主要处理模型实现和权重转换:

- `verl/models/transformers/`: HuggingFace/Transformers 模型适配, 如 llama、qwen2、qwen2_vl、qwen3_vl、glm4v、kimi_vl 等。
- `verl/models/mcore/`: Megatron Core 相关 bridge、loader、saver、config converter、weight converter。
- `weight_loader_registry.py`: 权重加载注册。
- `registry.py`: 模型注册相关逻辑。

它不是训练主循环, 而是 engine 构建模型时需要的底层模型/权重适配层。阅读顺序建议放在 engine 之后。

## 15. utils 目录怎么读

`verl/utils/` 很大, 可以按用途分组:

- 配置和导入: `config.py`, `import_utils.py`。
- 数据: `dataset/`, `tokenizer.py`, `chat_template.py`。
- 张量和 mask: `torch_functional.py`, `tensordict_utils.py`, `attention_utils.py`。
- 分布式: `distributed.py`, `ray_utils.py`, `megatron_utils.py`, `fsdp_utils.py`, `ulysses.py`。
- checkpoint: `checkpoint/`。
- profiling 和 metrics: `profiler/`, `metric/`, `tracking.py`, `debug/`。
- rollout/后端 patch: `vllm/`, `trtllm/`, `modelopt/`, `qat/`。
- reward score: `reward_score/`。

初学不必从 `utils` 开始。先看主链路, 遇到函数再回到 `utils`。

## 16. 旧 PPO 主线: main_ppo.py 和 ray_trainer.py

当前 `main_ppo.py` 和 `RayPPOTrainer` 都有 deprecated 注释, 但仍非常适合学习传统 HybridFlow:

- `main_ppo.py::TaskRunner.run()` 展示了如何创建 tokenizer、dataset、resource pool、`RayPPOTrainer`。
- `ray_trainer.py::RayPPOTrainer.fit()` 展示了纯 `DataProto` 风格主循环。
- `_compute_old_log_prob()`、`_compute_ref_log_prob()`、`_compute_values()` 展示 padded TensorDict 与 no-padding 之间的转换。
- `apply_kl_penalty()`、`compute_advantage()` 是同步 PPO 仍复用的算法函数。

旧主循环的可读伪代码:

```python
for batch in train_dataloader:
    gen_batch = batch.pop(["input_ids", "attention_mask", "position_ids"])
    gen_output = actor_rollout_wg.generate_sequences(gen_batch)
    batch = batch.union(gen_output)
    old_log_prob = actor_rollout_wg.compute_log_prob(batch)
    ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)
    values = critic_wg.compute_values(batch)
    reward = reward_fn(batch)
    batch = compute_advantage(batch)
    critic_wg.update_critic(batch)
    actor_rollout_wg.update_actor(batch)
```

它比当前同步 PPO 少 TransferQueue 和 agent loop 细节, 适合先理解 HybridFlow 的抽象。

## 17. 逐模块阅读路线

建议按这个顺序读源码:

1. `README.md` 和 `docs/hybrid_flow.rst`: 先知道 HybridFlow 的动机。
2. `verl/trainer/config/ppo_trainer.yaml`: 看默认配置如何组合角色。
3. `verl/trainer/main_ppo_sync.py`: 读 `main()`、`TaskRunner.run()`、`PPOTrainer.init_workers()`、`fit()`、`step()`。
4. `verl/single_controller/base/decorator.py`: 理解 `@register`、dispatch 和 collect。
5. `verl/single_controller/ray/base.py`: 理解 `RayWorkerGroup` 如何把远程方法代理成本地调用。
6. `verl/protocol.py`: 理解 `DataProto`、`DataProtoFuture`、`BatchData`。
7. `verl/workers/engine_workers.py`: 读 `TrainingWorker` 和 `ActorRolloutRefWorker`。
8. `verl/workers/engine/base.py`: 理解 engine 统一接口。
9. 选择一个训练后端读, 推荐先 FSDP: `workers/engine/fsdp/transformer_impl.py`。
10. `verl/workers/rollout/base.py` 和 `rollout/replica.py`: 理解 rollout 服务抽象。
11. `verl/checkpoint_engine/base.py`: 理解每步权重同步。
12. `verl/trainer/ppo/core_algos.py`: 读 advantage estimator 和 policy loss。
13. `verl/workers/utils/losses.py`: 看 actor/critic/SFT loss 如何接入 engine。
14. `verl/utils/dataset/rl_dataset.py`: 看数据如何进入 rollout。
15. `verl/trainer/sft_trainer.py`: 对比 SFT 和 PPO 架构差异。

## 18. 如何扩展 verl

### 18.1 加一个新的 advantage estimator

在 `core_algos.py` 或外部模块里:

```python
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_estimator")
def compute_my_advantage(token_level_rewards, response_mask, config=None, **kwargs):
    advantages = ...
    returns = ...
    return advantages, returns
```

配置中设置:

```yaml
algorithm:
  adv_estimator: my_estimator
```

为什么只要这样:

- `ray_trainer.compute_advantage()` 会通过 `get_adv_estimator_fn()` 找函数。
- trainer 的其他步骤不关心 advantage 是怎么来的。

### 18.2 加一个新的 policy loss

注册:

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def compute_my_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights=None):
    ...
    return loss, metrics
```

配置:

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: my_loss
```

`ppo_loss()` 会自动通过 `get_policy_loss_fn()` 调到你的 loss。

### 18.3 加一个新的训练后端

实现一个 `BaseEngine` 子类, 注册:

```python
@EngineRegistry.register(model_type="language_model", backend="my_backend", device="cuda")
class MyEngineWithLMHead(BaseEngine):
    ...
```

最少需要实现:

- `initialize()`
- `train_mode()` / `eval_mode()`
- `forward_backward_batch()`
- `get_per_tensor_param()`
- `save_checkpoint()` / `load_checkpoint()`
- data parallel rank/size/group
- `is_mp_src_rank_with_outputs()`

然后在配置中设置 actor/critic engine strategy 为你的 backend。

### 18.4 加一个新的 rollout 后端

实现 `BaseRollout` 或 `RolloutReplica` 相关 adapter, 并在 registry 里加入 `(name, mode)` 到类路径的映射。核心要支持:

- 启动服务。
- 根据 prompt 生成 response。
- 接收 actor 新权重。
- sleep/resume/release KV cache 和 weights。

## 19. 关键源码定位速查表

下面这张表可以当作读源码时的地图。行号来自当前工作区源码, 如果上游代码更新, 行号可能轻微漂移, 但文件和类/函数名仍是定位入口。

| 主题 | 源码位置 | 看什么 |
| --- | --- | --- |
| 同步 PPO 入口 | `verl/trainer/main_ppo_sync.py:1844` `main()` | Hydra 入口, 设置运行时配置并启动 TaskRunner。 |
| 同步 PPO TaskRunner | `verl/trainer/main_ppo_sync.py:1757` `TaskRunner`; `:1818` `run()` | 初始化 TransferQueue, 构建 tokenizer/dataset/resource pool/trainer。 |
| PPOTrainer 主类 | `verl/trainer/main_ppo_sync.py:501` | 当前同步 PPO 的主 controller。 |
| PPOTrainer 初始化 | `verl/trainer/main_ppo_sync.py:510` `__init__()` | 保存 config, 初始化 tokenizer/dataloader/replay buffer/KL controller。 |
| 创建 worker | `verl/trainer/main_ppo_sync.py:599` `init_workers()` | 创建 actor/ref/critic/reward/teacher worker group 和 colocated worker。 |
| 训练循环 | `verl/trainer/main_ppo_sync.py:1589` `fit()` | epoch/global step/checkpoint/validation 的外层循环。 |
| 单步训练 | `verl/trainer/main_ppo_sync.py:1688` `step()` | rollout -> reward -> logprob/value -> advantage -> critic/actor update -> weight sync。 |
| old logprob | `verl/trainer/main_ppo_sync.py:1302` `_compute_old_log_prob()` | actor inference 后把 `old_log_probs` 写回 TransferQueue。 |
| ref logprob | `verl/trainer/main_ppo_sync.py:1363` `_compute_ref_log_prob()` | reference policy inference。 |
| critic value | `verl/trainer/main_ppo_sync.py:1389` `_compute_values()` | critic 输出 `values`。 |
| advantage | `verl/trainer/main_ppo_sync.py:1405` `_compute_advantage()` | 取 reward/logprob/value, 调 core_algos, 写回 `advantages/returns`。 |
| critic update | `verl/trainer/main_ppo_sync.py:1468` `_update_critic()` | 设置 critic mini-batch/epoch, 调 critic worker。 |
| actor update | `verl/trainer/main_ppo_sync.py:1490` `_update_actor()` | 设置 actor loss 元信息, 调 actor worker。 |
| TransferQueue 适配 | `verl/trainer/main_ppo_sync.py:44`; `verl/utils/transferqueue_utils.py:35` | `transfer_queue` 外部包导入、bridge 和 fallback。 |
| ReplayBuffer | `verl/trainer/main_ppo_sync.py:195` | 从 TransferQueue 轮询可训练样本元信息。 |
| 旧 PPO 入口 | `verl/trainer/main_ppo.py:39` `main()`; `:111` `TaskRunner`; `:223` `run()` | 旧 RayPPOTrainer/DataProto 主线, 适合对照学习。 |
| 旧 RayPPOTrainer | `verl/trainer/ppo/ray_trainer.py:286` | 旧分布式 PPO trainer。 |
| KL penalty | `verl/trainer/ppo/ray_trainer.py:76` `apply_kl_penalty()` | 把 policy/ref KL 从 reward 中扣除。 |
| advantage 调度 | `verl/trainer/ppo/ray_trainer.py:185` `compute_advantage()` | 按 `adv_estimator` 调 `core_algos.py` 中的函数。 |
| 数据协议 | `verl/protocol.py:318` `DataProto`; `:1174` `DataProtoFuture`; `:1231` `BatchData` | tensor/non-tensor/meta 的批数据封装和 future 化。 |
| controller 装饰器 | `verl/single_controller/base/decorator.py:26` `Dispatch`; `:50` `Execute`; `:398` `register()` | 远程方法如何声明 dispatch、execute、blocking。 |
| ResourcePool | `verl/single_controller/base/worker_group.py:27`; `verl/single_controller/ray/base.py:182` | GPU/placement group 资源组织。 |
| WorkerGroup | `verl/single_controller/base/worker_group.py:123`; `verl/single_controller/ray/base.py:416` `RayWorkerGroup` | 把远程 worker 方法代理成本地可调用方法。 |
| colocated worker | `verl/single_controller/ray/base.py:986` `create_colocated_worker_cls()` | 多个角色共进程/共 placement group 的动态类生成。 |
| TrainingWorker | `verl/workers/engine_workers.py:76` | 训练/推理 worker 通用封装。 |
| mini-batch 训练 | `verl/workers/engine_workers.py:234` `train_mini_batch()`; `:325` `train_batch()` | PPO epoch/mini-batch 切分和单次 engine 训练。 |
| worker 推理 | `verl/workers/engine_workers.py:380` `infer_batch()` | actor/ref/critic/reward/teacher inference 通路。 |
| ActorRolloutRefWorker | `verl/workers/engine_workers.py:434`; `:500` `init_model()` | actor、rollout、ref、checkpoint engine 的组合 worker。 |
| actor/ref API | `verl/workers/engine_workers.py:637` `compute_ref_log_prob()`; `:644` `compute_log_prob()`; `:652` `update_actor()` | controller 调用 actor/ref 的主要远程方法。 |
| engine 抽象 | `verl/workers/engine/base.py:29` `BaseEngine`; `:267` `EngineRegistry` | 训练后端统一接口和注册表。 |
| engine 配置 | `verl/workers/config/engine.py:601` `TrainingWorkerConfig` | 每个 TrainingWorker 如何选择 model/engine/optimizer/checkpoint。 |
| actor 配置 | `verl/workers/config/actor.py:103` `ActorConfig`; `:78` `PolicyLossConfig` | PPO actor loss、KL、entropy、clip、mini-batch 等配置。 |
| critic 配置 | `verl/workers/config/critic.py:46` `CriticConfig` | critic value loss、cliprange、mini-batch 等配置。 |
| rollout 配置 | `verl/workers/config/rollout.py:161` `RolloutConfig` | rollout 引擎、采样、多轮 agent loop、server、checkpoint engine 配置。 |
| rollout 抽象 | `verl/workers/rollout/base.py:29` `BaseRollout`; `:83` `_ROLLOUT_REGISTRY` | vLLM/SGLang/TRT-LLM server adapter 选择。 |
| rollout replica | `verl/workers/rollout/replica.py:70` `RolloutReplica` | hybrid/colocated/standalone rollout 服务副本。 |
| checkpoint engine | `verl/checkpoint_engine/base.py:323` `CheckpointEngineManager` | actor 权重同步到 rollout 的协调器。 |
| actor/critic/SFT loss | `verl/workers/utils/losses.py:28` `sft_loss`; `:57` `ppo_loss`; `:147` `value_loss` | engine backward 前真正调用的 loss closure。 |
| 算法注册表 | `verl/trainer/ppo/core_algos.py:50` `POLICY_LOSS_REGISTRY`; `:88` `AdvantageEstimator` | policy loss 和 advantage estimator 的注册入口。 |
| value loss | `verl/trainer/ppo/core_algos.py:2084` `compute_value_loss()` | PPO clipped value loss。 |
| KL 形式 | `verl/trainer/ppo/core_algos.py:2126` `kl_penalty()` | `kl/abs/mse/low_var_kl/full` 等 KL 计算。 |
| RLHF dataset | `verl/utils/dataset/rl_dataset.py:71` `RLHFDataset` | parquet/jsonl prompt 数据如何 tokenization、padding、过滤。 |
| reward manager | `verl/workers/reward_manager/naive.py:27` `NaiveRewardManager` | 函数奖励和已有 `rm_scores` 的通用处理。 |
| SFT trainer | `verl/trainer/sft_trainer.py:50` `SFTTrainer`; `:464` `create_sft_dataset()` | SFT 如何复用 TrainingWorker/engine 但不使用 PPO controller。 |
| SFT dataset | `verl/utils/dataset/multiturn_sft_dataset.py:73` `MultiTurnSFTDataset` | 多轮 SFT 数据如何构造 loss mask。 |

## 20. 关键设计总结

verl 的架构可以压缩成一句话:

> controller 写算法数据流, worker group 负责远程调度, engine 负责分布式训练后端, rollout 负责高吞吐生成, DataProto/TransferQueue 负责数据流, registry/config 负责把这些组件组合起来。

为什么这样设计:

- 算法可换: PPO、GRPO、ReMax、RLOO 等主要换 advantage estimator 和 policy loss。
- 后端可换: FSDP、Megatron、VeOmni、TorchTitan 通过 `BaseEngine` 隔离。
- 推理可换: vLLM、SGLang、TRT-LLM 通过 `BaseRollout`/`RolloutReplica` 隔离。
- 放置可换: resource pool 和 colocated worker 让 actor/critic/ref/reward/teacher 可以共用或独占 GPU。
- 数据可扩展: `DataProto` 支持 tensor/non-tensor/meta, TransferQueue 支持更高性能的 KV 数据流。

读源码时最重要的是不要陷入某个后端细节。先抓住这一条主链:

```text
PPOTrainer.step()
  -> rollout 生成
  -> reward
  -> actor/ref/critic inference
  -> core_algos 计算 advantage
  -> workers/utils/losses.py 计算 loss
  -> TrainingWorker
  -> BaseEngine
  -> 具体 FSDP/Megatron/...
  -> checkpoint_manager.update_weights()
```

只要这条链路清楚, 之后看任意算法或后端都是在这条链上替换一个节点。
