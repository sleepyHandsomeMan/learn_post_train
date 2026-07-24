# notebooks 目录说明

本目录保存手工实验 notebook。

## 文件

| 文件 | 说明 |
|---|---|
| `base_model_manual_test.ipynb` | base model 手工推理测试 |
| `sft_experiment_notes.ipynb` | SFT 实验记录和手工分析 |

## 维护规则

- notebook 适合探索，不作为最终训练入口。
- 稳定流程应沉淀到 `post_training_framework/scripts/`。
- 重要结论应整理到 `docs/`，不要只留在 notebook 输出里。
- 避免提交大体积输出单元。
