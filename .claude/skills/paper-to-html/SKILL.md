---
name: paper-to-html
description: 将学术论文转化为交互式HTML网页，采用苹果官网设计美学。当用户提到"论文可视化"、"paper to HTML"、"学术论文网页"、"论文转HTML"、"可视化论文"、"论文解读网页"、"interactive paper"或想要创建论文展示页面时使用此skill。即使用户只是说"帮我做个论文网页"或"把这篇论文做成网页"，也要使用这个skill。
compatibility:
  tools:
    - Read
    - Write
    - Bash
    - Grep
  optional:
    - pdfplumber (Python库，用于PDF解析)
    - PyPDF2 (Python库，备用PDF解析)
---

# 学术论文转交互式HTML网页

## 概述

将学术论文转化为单一、完整的`index.html`文件，深度解析并展示论文的核心内容，达到读者能够理解论文90%内容并具备复现能力的程度。

## 设计原则

### 苹果美学标准
- **极简主义**：大量留白，内容呼吸感强
- **流畅动画**：平滑的滚动效果和渐入动画
- **层次分明**：清晰的视觉层级，从标题到正文
- **优雅配色**：浅色背景配深色文字，或深色模式支持
- **响应式设计**：适配桌面、平板、移动设备

### 技术栈
- **LaTeX渲染**：MathJax 3.x（支持完整LaTeX语法）
- **字体**：San Francisco / -apple-system / system-ui
- **动画**：CSS transitions + Intersection Observer API
- **布局**：CSS Grid + Flexbox

## 工作流程

### 第一步：理解输入

支持多种输入格式：

1. **PDF文件**：使用`scripts/parse_pdf.py`提取文本和结构
2. **arXiv链接**：自动下载PDF并解析
3. **文本内容**：用户直接粘贴的论文文本
4. **LaTeX源码**：如果用户提供.tex文件

**处理策略**：
- 首先检查输入类型
- 如果是PDF，调用解析脚本提取内容
- 如果是链接，先下载再解析
- 提取论文的章节结构、公式、图表、表格

### 第二步：内容深度解析

必须包含以下核心部分，每部分都要事无巨细：

#### 1. 研究动机 (Research Motivation)
- **问题发现**：作者发现了什么问题？
- **重要性**：为什么这个问题值得研究？
- **研究意义**：本文的contribution和significance是什么？
- **与现有工作的关系**：填补了什么gap？

#### 2. 数学表示及建模 (Mathematical Formulation)
- **符号定义**：所有数学符号的含义（使用LaTeX inline公式）
- **核心公式**：关键的数学公式（使用LaTeX display公式）
- **公式推导**：重要公式的推导过程（逐步展示）
- **算法流程**：伪代码或算法描述

**LaTeX渲染要求**：
- 行内公式使用`\(...\)`或`$...$`，**绝对不能换行**
- 块级公式使用`\[...\]`或`$$...$$`
- 确保所有公式都能被MathJax正确渲染
- 复杂公式使用`align`或`equation`环境

#### 3. 实验方法与设计 (Experimental Setup)
这部分要达到**可复现**的程度：
- **数据集**：名称、规模、来源、预处理方法
- **模型架构**：详细的模型结构（参考论文和appendix）
- **超参数**：学习率、batch size、优化器等所有超参数
- **训练细节**：训练轮数、early stopping策略、硬件配置
- **Prompt设计**（如果是LLM相关）：完整的prompt模板
- **评估指标**：使用的所有评估指标及其定义

#### 4. 实验结果及核心结论 (Results & Insights)
- **Baseline对比**：与哪些方法对比，各自的性能
- **主要结果**：关键的实验结果（表格要渲染，不只是截图）
- **消融实验**：各个组件的贡献分析
- **核心洞察**：实验揭示了什么规律或结论
- **可视化**：重要的图表（能提取就提取，复杂的用占位符）

#### 5. 你的评论 (Critical Review)
作为一个犀利的reviewer，提供：
- **优势**：这篇工作做得好的地方（2-3点）
- **不足**：存在的问题或局限性（2-3点）
- **改进方向**：可能的改进思路（1-2点）
- **整体评价**：一句话总结这篇工作的价值

#### 6. One More Thing
自由发挥，可以包括：
- 相关工作的对比
- 论文的历史背景
- 作者团队介绍
- 代码实现链接
- 其他你认为重要的内容

### 第三步：图表处理

采用**混合策略**：

1. **表格**：
   - 关键实验表格：转换为HTML表格，保留所有数据
   - 使用LaTeX表格语法作为参考，渲染为美观的HTML表格
   - 添加表格标题和注释

2. **图片**：
   - 简单图表：尝试从PDF提取并嵌入（base64或外部文件）
   - 复杂图表：使用占位符`[Figure X: 图表描述]`
   - 占位符格式：`<div class="figure-placeholder" data-figure="X">Figure X: 具体描述</div>`

3. **公式**：
   - 所有公式必须使用LaTeX语法
   - 确保MathJax能正确渲染

### 第四步：生成HTML

使用`references/apple-template.html`作为基础模板，生成单一完整的`index.html`文件。

**HTML结构**：
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>论文标题</title>
    <!-- MathJax配置 -->
    <script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <!-- 样式 -->
    <style>
        /* 苹果风格CSS - 参考references/apple-styles.css */
    </style>
</head>
<body>
    <!-- 内容区域 -->
</body>
</html>
```

**关键CSS样式**：
- 使用CSS变量定义颜色主题
- 平滑滚动：`scroll-behavior: smooth`
- 渐入动画：使用Intersection Observer
- 响应式断点：1200px, 768px, 480px

### 第五步：自我校正

在输出最终代码前，进行彻底的自我检查：

1. **完整性检查**：
   - [ ] 是否包含所有六个核心部分？
   - [ ] 实验部分是否足够详细（可复现）？
   - [ ] 是否有遗漏的重要内容？

2. **LaTeX检查**：
   - [ ] 所有公式都能被MathJax渲染？
   - [ ] 行内公式没有换行？
   - [ ] 特殊符号正确转义？

3. **设计检查**：
   - [ ] 是否符合苹果美学？
   - [ ] 响应式设计是否完善？
   - [ ] 动画效果是否流畅？

4. **语言检查**：
   - [ ] 除公式和术语外，是否使用中文？
   - [ ] 技术名词是否保留英文？

## 输出格式

生成单一的`index.html`文件，包含：
- 完整的HTML结构
- 内联CSS样式
- 内联JavaScript（MathJax配置、动画脚本）
- 所有论文内容

**文件大小**：通常在200-500KB之间（取决于论文长度）

## 注意事项

1. **不要分批输出**：虽然用户的CLAUDE.md要求长文档分批输出，但这个skill生成的是单一HTML文件，应该一次性完整输出。

2. **公式渲染优先**：确保所有LaTeX公式都能正确渲染，这是最容易出错的地方。

3. **深度优先**：宁可内容详细到冗余，也不要遗漏关键信息。目标是90%的论文内容。

4. **评论要犀利**：不要客套话，要有真知灼见，指出真正的问题和价值。

5. **可复现性**：实验部分要详细到别人能根据你的描述复现论文。

## 示例触发场景

- "把这篇论文转成网页"
- "帮我做个论文可视化"
- "生成这篇paper的HTML展示页面"
- "我想要一个交互式的论文解读网页"
- "把这个PDF论文做成苹果风格的网页"

## 相关脚本

- `scripts/parse_pdf.py`：PDF解析脚本
- `scripts/download_arxiv.py`：arXiv论文下载脚本
- `references/apple-template.html`：HTML模板
- `references/apple-styles.css`：CSS样式参考
