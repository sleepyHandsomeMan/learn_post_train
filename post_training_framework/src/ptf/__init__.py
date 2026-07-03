"""面向自学后训练闭环的轻量实验框架。

模块：
  - config: 实验配置读取
  - data: 评估数据加载与规范化
  - rl_dataset: GRPO/PPO 训练数据集
  - prompting: prompt 构造与 chat template
  - metrics: 自动评估指标
  - reward: GSM8K 规则奖励
  - reports: JSONL/JSON/Markdown 报告
  - generation: 模型加载与推理
  - train_sft: LoRA SFT 训练
  - train_grpo: 自包含 GRPO 训练器
  - train_ppo: 自包含 PPO 训练器
  - compare: 模型对比报告
  - grpo_data: RL 数据预览工具
"""

__all__ = [
    "config",
    "data",
    "rl_dataset",
    "generation",
    "metrics",
    "prompting",
    "reports",
    "reward",
    "train_sft",
    "train_grpo",
    "train_ppo",
    "compare",
    "grpo_data",
]
