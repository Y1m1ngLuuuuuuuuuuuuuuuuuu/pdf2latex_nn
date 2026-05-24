# 版式感知 LaTeX 重建目标

**最后更新**：2026-05-24

本文档用于固定项目目标和评测哲学，避免系统继续被一个不可能的目标牵着走：从渲染后的 PDF 反推出作者原始 TeX 源码树。

## 1. 正式目标

本项目的正式目标是：

```text
从渲染后的科研 PDF 中，重建可编译、版式感知、保留块级语义结构的 LaTeX 文档
```

本项目的目标不是：

```text
源码级 TeX AST 复原
```

PDF 是排版结果，TeX 源码是生成程序。多个不同的 TeX 源码可以生成视觉等价的 PDF，LaTeX 的浮动体机制也会让图表视觉位置远离源码中的书写位置。因此，用作者原始 TeX AST 作为全局强 gold，会不公平地惩罚一个从 PDF 出发的系统。

## 2. 核心原则

必须区分“PDF 中可观察的事实”和“只存在于源码程序里的事实”。

```text
PDF 中可观察：
  文本块
  块级标题
  视觉阅读区域
  图/表/caption 的位置
  references 区域
  页面几何和样式

PDF 中不能唯一确定：
  作者具体用了什么宏
  run-in \paragraph 还是 \textbf 前缀
  float 在源码中的真实位置
  期刊模板内部实现细节
  原始文件拆分和宏 AST
```

系统应该重建一个稳定、可读、可编辑，并且尽量保留视觉版式和块级结构的 LaTeX 文档，而不是假装 PDF 能唯一决定作者原始源码。

## 3. IR 分层

重建栈中必须分开三类 IR。

### Layout IR

描述视觉内容出现在哪里：

```text
page
bbox
column / band
visual order
reading order
font / style statistics
float visual slot
```

### Semantic IR

描述可恢复的文档结构：

```text
块级标题树
段落
列表
行间公式
references
captions
脚注 / 边注
交叉引用
```

### Float IR

浮动体必须拆成三件事：

```text
visual float slot:
  图/表在 PDF 上出现在哪里

caption-float pairing:
  哪个 caption 属于哪个图/表

semantic anchor:
  这个 float 语义上更接近哪个 section 或 paragraph
```

这三者不是同一种关系，不能混在一起评。

## 4. GNN 的责任边界

GNN 仍然是被维护的学习式关系预测分支，但它不是当前默认
reconstruction authority。当前默认 v8 / layout-first 生成链路不加载
GNN checkpoint。GNN 分支保持当前三分类任务：

```text
MERGE
PARENT_CHILD
NONE
```

`PARENT_CHILD` 仍然是从 TeX 标签中学习到的结构挂载关系，作用于 graph-visible view。它不改名成 local-only label，也不拆成大量稀疏多头任务。

Decoder 可以在 GNN 输出之上加入确定性约束：

```text
heading evidence / stack priors
section-scope safety gates
float/caption grouping
physical impossibility vetoes
renderer layout policies
```

这些约束是当前生产默认的 heading scope 与渲染安全机制。它们用于阻止物理上不可能的结构，并让生成的 LaTeX 更稳定。只有在显式训练或审计可选 GNN 分支时，监督任务才是原来的三分类关系预测。

也就是说，可选 GNN 分支保留之前的设计：

```text
candidate graph -> GNN relation probabilities -> constrained decoder -> IR renderer
```

默认面向论文的 reconstruction 不应过度声称 GNN 的 E2E 影响。除非后续 GNN-sensitive 验证集证明下游生成有清晰提升，GNN 结果应作为辅助 ablation / diagnostic 证据汇报。

## 5. Run-In Heading

Run-in heading 在 PDF 中视觉上非常暧昧。

例如：

```text
Summary. The method consists of ...
Linear Relationship in Predictors: The linear predictor ...
```

它们可能来自：

```tex
\paragraph{Summary.} The method consists of ...
```

也可能只是：

```tex
\textbf{Summary.} The method consists of ...
```

PDF 通常没有足够证据稳定地区分二者。

策略：

```text
run-in heading 默认作为 paragraph inline label
```

它们不进入 block-level heading evaluation。generator 可以把它们渲染成加粗行内前缀。只有整篇文章存在强一致证据时，才考虑升格为 `\paragraph{}`。

## 6. 评测哲学

评测必须分层，不能用单一 AST 分数概括。

### 视觉重建

```text
compile_success
page_count_match
layout_similarity
block_bbox_iou
float visual slot recovery
```

### 文本和内容

```text
paragraph_text_coverage
normalized edit distance
formula recovery
caption text recall
reference item recall
```

### 段落和块边界

```text
paragraph_boundary_f1
paragraph/list/reference/caption split-merge tolerance
block_type_f1
```

### 块级标题

只评估块级标题：

```text
\section
\subsection
\subsubsection
```

以下内容不进入 heading tree 指标：

```text
\paragraph
\subparagraph
加粗 run-in paragraph 前缀
caption label
reference item
list item
front matter metadata，除非单独评测
```

### 正文 Section Attachment

`section_attachment_f1` 不应该作为全局成功指标。更公平的主指标是：

```text
section_attachment_body_no_float_f1
```

它只评估正文类内容：

```text
paragraph
list item
display equation
ordinary body text
algorithm/code block，当它作为正文内容使用时
```

它排除：

```text
figure
table
caption
footnote
margin note
reference item
run-in heading
header/footer/page number
front matter
```

### Float 指标

Float 单独评：

```text
caption detection
caption label accuracy
caption-float pairing
float visual slot recovery
float semantic anchor
```

其中 semantic anchor 权重应该最低，因为它天然受 LaTeX float 机制影响。

## 7. Baseline 对比

Nougat 这类系统应被视为面向学术文档的 markup transcription baseline，而不是完整的版式保留 LaTeX reconstruction system。

共享比较维度：

```text
text coverage
formula/caption/reference recovery
block-level heading tree
body section attachment
reading order
```

我们的重建特有维度：

```text
可编译 LaTeX 输出
页面 layout similarity
float visual slots
crop-based figure/table preservation
style and column reconstruction
```

我们不应该声称“复原作者原始 TeX 比 Nougat 更好”。合理主张是：

```text
我们重建可编译、版式感知、保留块级语义结构的 LaTeX，
并在普通 markup transcription 之外保留页面版式、float 和 references 结构。
```

## 8. 实际影响

这个目标定义意味着：

```text
1. 不再为了 raw TeX AST 等价优化整个系统。
2. 不让 section_attachment 单独决定系统好坏。
3. run-in headings 不进入 block-level heading 指标。
4. float visual position 和 semantic anchor 分开。
5. heading skeleton / state stack 是当前默认 decoder prior 和安全约束。
6. GNN 保持三分类关系预测器（`MERGE / PARENT_CHILD / NONE`），但属于显式可选关系学习分支；默认 v8 reconstruction 不依赖它。
7. 视觉、文本、标题、正文挂载、float、references 分开汇报。
```

这是后续模型、generator、evaluation 修改的项目级契约。
