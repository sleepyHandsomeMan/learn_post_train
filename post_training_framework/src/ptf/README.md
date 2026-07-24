# ptf 源码模块说明

`ptf` 是当前项目自包含后训练实验框架的 Python 包。

## 模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | 配置读取、路径解析、命令行覆盖 |
| `data.py` | messages parquet 读取和评估样本规范化 |
| `prompting.py` | prompt 构造和 chat template 渲染 |
| `metrics.py` | 答案抽取、exact match、格式和复读指标 |
| `reports.py` | JSONL、summary、Markdown 报告写出 |
| `generation.py` | 模型加载、推理、评估主流程 |
| `train_sft.py` | LoRA SFT 训练 |
| `rl_dataset.py` | 自定义 GRPO/PPO 的 RLPromptDataset |
| `reward.py` | GSM8K rule reward |
| `train_grpo.py` | 自定义 GRPO trainer |
| `stopping.py` | 统一停止决策、优先级调度、停止事件与终态摘要 |
| `train_ppo.py` | 自定义 PPO trainer |
| `grpo_data.py` | RL prompt 预览辅助 |
| `compare.py` | Base/SFT/GRPO 对比报告 |

## 数据流

```text
messages parquet
  -> data.py / rl_dataset.py
  -> prompting.py
  -> generation.py 或 train_grpo.py
  -> metrics.py / reward.py
  -> reports.py / compare.py
```

## 维护规则

- 新增正式能力优先放入 `src/ptf/`，脚本只做薄封装。
- 训练逻辑改动后同步更新 `scripts/README.md` 和相关阶段文档。
- 新增停止判定时必须返回结构化停止决定，由统一控制器负责优先级、留档和收尾；不得在判定器内部直接退出训练循环。
- 数据格式改动后同步更新 `datasets/README.md`。
