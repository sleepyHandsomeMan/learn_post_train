"""生成 GSM8K SFT 数据预览 Markdown。"""
import pandas as pd, re
from pathlib import Path

df = pd.read_parquet('datasets/gsm8k_sft/eval_20.parquet')

lines = []
lines.append('# GSM8K SFT 训练数据预览')
lines.append('')
lines.append(f'- 文件: `datasets/gsm8k_sft/eval_20.parquet`')
lines.append(f'- 样本数: {len(df)}')
lines.append(f'- 列: {list(df.columns)}')
lines.append('')
lines.append('## 数据格式说明')
lines.append('')
lines.append('每行 `messages` 包含两轮：')
lines.append('1. **user** — 数学题 + 格式指令 `Let\'s think step by step and output the final answer after "####".`')
lines.append('2. **assistant** — 逐步推理过程，包含：')
lines.append('   - 自然语言分步推导')
lines.append('   - `<<算式>>` 标注每步计算（训练时这些也会被模型学习）')
lines.append('   - `#### 最终答案` 作为结束标记')
lines.append('')
lines.append('SFT 训练时，user 部分被 mask 掉（labels=-100），只对 assistant 部分计算 loss。')
lines.append('模型学到的是：**给定题目，生成和标准答案风格一致的推理过程**。')
lines.append('')
lines.append('---')
lines.append('')

for i, (_, row) in enumerate(df.iterrows()):
    msgs = row['messages']
    if hasattr(msgs, 'tolist'):
        msgs = msgs.tolist()
    msgs = [dict(m) for m in msgs]

    user = msgs[0]
    assistant = msgs[1] if len(msgs) > 1 else None

    lines.append(f'## 样本 {i}')
    lines.append('')

    # user 部分
    uc = user['content']
    lines.append('**User:**')
    lines.append('')
    lines.append('```')
    lines.append(uc)
    lines.append('```')
    lines.append('')

    # assistant 部分
    if assistant:
        ac = assistant['content']
        lines.append('**Assistant (SFT 监督目标):**')
        lines.append('')
        lines.append('```')
        lines.append(ac)
        lines.append('```')
        lines.append('')

        # 解析
        calcs = re.findall(r'<<(.+?)>>', ac)
        final = re.search(r'####\s*(.+)', ac)
        lines.append(f'| 指标 | 值 |')
        lines.append(f'|---|---|')
        lines.append(f'| 计算步骤数 | {len(calcs)} |')
        lines.append(f'| 最终答案 | `{final.group(1).strip() if final else "未找到"}` |')
        for j, c in enumerate(calcs):
            lines.append(f'| 步骤 {j+1} | `{c}` |')
        lines.append('')

    lines.append('---')
    lines.append('')

md = '\n'.join(lines)
out = Path('datasets/gsm8k_sft/preview_eval_20.md')
out.write_text(md, encoding='utf-8')
print(f'OK: {out} ({len(lines)} 行)')
