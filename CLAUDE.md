# yhy 后训练学习工作区

本文件是兼容入口。当前项目协作规则以 `AGENTS.md` 为准。

请优先查看:

```text
AGENTS.md
docs/00_workspace_architecture_map_cn.md
```

关键约定:

- 默认中文沟通。
- 代码注释使用中文。
- 长文档不设固定行数上限；优先通过结构化目录维护单一主文档，避免仅因篇幅拆分。
- 当前项目自定义 GRPO 训练器读取 `messages` parquet。
- 模型、checkpoint、日志、评估大文件和派生数据默认不进入 git。
