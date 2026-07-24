# dev_tools 目录说明

本目录放一次性诊断、手工检查和环境工具。长期训练/评估入口应优先放到 `post_training_framework/scripts/`。

## 顶层工具

| 文件 | 作用 |
|---|---|
| `download_hf_model.py` | 下载 Hugging Face 模型 |
| `check_training_gpu.py` | 检查训练进程和 GPU |
| `check_training_gpu_external.py` | 外部 GPU 检查 |
| `cuda_allocator_demo.py` | CUDA allocator 行为演示 |
| `grpo_rollout_diagnosis.py` | GRPO rollout 多样性和 reward 信号诊断 |

## 子目录

| 目录 | 说明 |
|---|---|
| `sft/` | SFT 数据、EOS、生成、报告诊断脚本 |
| `grpo/` | GRPO 临时工具或后续阶段专用工具 |

## 维护规则

- 如果脚本会成为正式流程入口，迁移到 `post_training_framework/scripts/`。
- 如果脚本只用于一次排错，留在 `dev_tools/` 并在文件头写用途。
- 输出文件不要放在 `dev_tools/`，应进入 `logs/`、`eval_results/` 或 `datasets/*/previews/`。
