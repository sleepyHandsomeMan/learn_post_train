# GSM8K SFT 数据说明

本目录是 SFT 阶段的固定数据入口。

## 文件

| 文件 | 用途 |
|---|---|
| `train.parquet` | GSM8K SFT 全量训练集，messages 格式 |
| `train_500.parquet` | 早期 500 条小规模实验 |
| `test.parquet` | GSM8K 测试集 |
| `eval_20.parquet` | 固定 20 条人工观察验证集，不用于训练 |
| `preview_eval_20.md` | eval_20 人工预览 |
| `preview_eval_20_tokenized.md` | token/label mask 预览 |

## 使用入口

- SFT 训练: `post_training_framework/scripts/run_sft_train.py`
- SFT 评估: `post_training_framework/scripts/run_sft_eval.py`
- 数据检查: `dev_tools/sft/validate_sft_data.py`

## 注意

固定验证集只能用于评估和人工观察，不能混入训练。
