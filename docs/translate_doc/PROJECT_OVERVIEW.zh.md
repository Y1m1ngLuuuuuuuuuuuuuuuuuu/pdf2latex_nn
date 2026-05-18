# PDF2LaTeX-NN 项目概览

**最后更新**：2026-05-18

本文总结当前 v7 架构，是项目的高层解释。更底层的合约在 schema、labeling 和 frontend/backend 文档中。完整架构、数据流、判断规则、代码地图和指标体系见 `docs/PROJECT_ARCHITECTURE_FULL.md`。面向论文写作的长文档见 `docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md`。

## 1. 目标

本项目为学术论文构建结构感知的 PDF-to-LaTeX 系统。规范目标是：

```text
从渲染后的科研论文 PDF 中，重建可编译、版式感知、保留块级结构的 LaTeX。
```

目标不是源码级 TeX AST 恢复。PDF 是渲染结果，不是作者 TeX 程序的一一对应编码；同一个 PDF 可能由多种 TeX 源码生成，LaTeX float 也会让图表远离源码位置。因此系统追求稳定、可读、可编辑、保留页面版式和块级语义组织的 LaTeX。

抽象流程：

```text
PDF visual facts + matching TeX source
  -> learned relation model
  -> structured IR
  -> compilable LaTeX
```

目标不是普通 OCR，也不是单一端到端语言模型。系统将感知、真值生成、关系学习、解码和渲染拆开，使每一层都可以独立替换。

## 2. 核心模块

```text
PDF Frontend
  MinerU + PyMuPDF + v7 reading/layout cleanup

GNN View Adapter
  full v7 fact layer -> graph-visible node view + reversible v7 mapping
  metadata/noise/annotation exclusion + float proxies

Graph Builder
  SciBERT + geometry + style + layout-flow features

TeX Truth Generator
  LaTeX flattener + TexSoup parser + sliding-window alignment

GNN
  EdgeRelationGAT / Y-Network predicts MERGE / PARENT_CHILD / NONE

Decoder
  merge contraction + structure constraints + tree/IR assembly

Generator
  OriginalLikeIRLatexRenderer + style/citation/float adapters
```

## 3. v7 前端原则

v7 前端描述视觉事实，不提前写死最终结构。

它保留 MinerU block 粒度和原始 bbox，同时增加：

```text
reading order metadata
band / column / layout layer hints
toc/header/footer/noise flags
list marker probes
duplicate-continuation detection
PyMuPDF style spans
reference item preservation
table / figure grouping metadata
footnote and margin-note candidates
```

跨页、跨段落的逻辑合并属于 decoder/generator 层，不属于 v7 JSON 预处理。

完整 v7 JSON 不会因为某个节点对 GNN 训练没用而被缩减。Metadata、floats、annotations、headers/footers 和 references 都保留在事实层。`GNNViewAdapter` 构建更窄的图可见视图，并记录映射回完整 v7 node ids。

当前实验 adapter 策略：

```text
metadata / true page furniture / annotations -> excluded from GNN view
figure / table / algorithm -> included as float proxies
caption text -> used as proxy semantics when available
raw table body / raw figure OCR -> not embedded as normal paragraph text
float -> text message passing -> masked
skip-over-float candidate edges -> added for paragraph continuation
```

## 4. GNN 任务

图模型的标签空间刻意保持很小：

```text
MERGE        物理延续
PARENT_CHILD 逻辑挂载
NONE         无关系
```

当前锁定基线模型是 GATv2/Y-Network 混合结构。PARENT_CHILD 和 NONE 使用经过 GAT 传播的节点状态；MERGE 绕过 message passing，使用原始 projected node-pair features，避免段落缝合被邻近 floats、tables 和无关文本过平滑。

float-proxy adapter 路径正在单独 rebuild，应该与锁定基线比较，而不是盲目替换。

深层边预测头接收有方向的节点项：

```text
concat([Hu, Hv, Hu-Hv, Hu*Hv, Euv])
```

这样可以学习 parent-child 的方向性，避免对称误报。

## 5. 为什么需要 TeX 标签

训练时，匹配的 TeX 源码是真值来源。Labeler 会：

```text
flattens TeX
parses structural nodes
aligns TeX nodes to PDF blocks
generates edge labels over graph candidate edges
enforces quality gates
```

这是训练数据生成器，不是推理依赖。推理时模型只看到 PDF 派生的图特征。

## 6. Generator 方向

当前规范 generator 是 IR renderer：

```text
OriginalLikeIRLatexRenderer
  -> IRRendererRegistry
    -> FrontMatterRenderer / HeadingRenderer / TextRenderer
    -> MathRenderer / FigureRenderer / TableRenderer
    -> ListRenderer / ReferenceRenderer / NoteRenderer
```

它支持：

```text
page style profiling
single/two/mixed-column approximation
front matter and abstract handling
caption and citation repair
reference rendering fallback
figure/table crop assets
footnote and margin-note rendering
inline-math protection
display equation rendering fallback
```

Generator 仍然是可扩展表面。期刊模板渲染和学习式风格复原应接入同一 IR 合约之后。

## 7. 当前数据策略

生产样本必须来自闭环编译：

```text
arXiv TeX source -> compiled PDF -> MinerU -> graph -> TeX labels
```

只通过 arXiv id 配对官方 PDF 的方式不适合生产训练，因为 source/PDF revision 可能不同。

## 8. 评估策略

使用分层检查。视觉重建、文本覆盖、段落边界、块级标题、body section attachment、float/caption 和 references 回答的是不同问题。

```text
1. edge metrics on MERGE and PARENT_CHILD
2. candidate-edge recall and label quality gates
3. visual QA of generated PDFs against originals
```

Accuracy 和 NONE F1 不是主要指标，因为 NONE 类极度占优。

块级标题指标只包括：

```text
\section
\subsection
\subsubsection
```

run-in `\paragraph` / `\subparagraph` 在比较时归一化为 paragraph inline label，因为 PDF 中往往无法可靠区分它们和粗体段首。

`section_attachment_f1` 是辅助结构指标。更公平的主变体是 `section_attachment_body_no_float_f1`，它排除 floats、captions、references、footnotes、page furniture 和 run-in headings。Float/caption 单独评价。

最新锁定基线结果：

```text
M07_y_network_plus_gaussian_edge_feature
MERGE F1        0.6331
PARENT_CHILD F1 0.9620
Positive Macro  0.7976
```

当前面向论文的评测轨道：

```text
active data/model family: v7_floatproxy_adapter_20260516_205926
current ablation matrix: configs/ablation_matrix_current.json
full evaluation suite: scripts/pipeline/run_current_full_eval_suite.py
result rollup: scripts/pipeline/collect_current_eval_results.py
```

Generator-only 修改不需要 relabel 或 retrain；它们需要重新跑 E2E 和对比评测。
