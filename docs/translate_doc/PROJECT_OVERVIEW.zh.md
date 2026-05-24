# PDF2LaTeX-NN 项目概览

**最后更新**：2026-05-24

本文是当前项目状态的中文高层说明。更完整的模块、代码地图和指标
定义见 `docs/PROJECT_ARCHITECTURE_FULL.md`，目录契约见
`docs/PROJECT_FILE_LAYOUT.md`。

## 1. 项目目标

项目目标是：

```text
从渲染后的科研论文 PDF 中，重建可编译、版式感知、保留块级语义结构的 LaTeX。
```

它不是普通 OCR，也不是作者源码级 TeX AST 复原。PDF 是渲染结果，
不是 TeX 程序的一一对应表示；同一个 PDF 可能由不同源码生成，float
也会远离源码位置。因此当前主线追求的是稳定、可读、可编译、版式近似、
结构清晰的 reconstruction TeX。

## 2. 当前默认主线：v8 / Layout-First

当前默认重建链路是 v8 layout-first，不依赖 GNN checkpoint：

```text
compiled PDF
  -> MinerU middle.json + content_list.json
  -> v8 middle-derived reflow
  -> page-local reading-order repair
  -> DocumentIR
  -> FrontMatterIR
  -> heading style registry + stack skeleton
  -> RenderTreeIR
  -> StyleProfile / v8 style detector
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

v8 存在的原因是：MinerU 的 `content_list.json` 有时会先按错误阅读顺序
把跨页、跨栏或同页不同栏的内容合并。v8 从 `middle.json` 的行级 bbox
和原始 fragment 重新还原可排序的文本片段，先修阅读顺序，再生成与
旧 v7 接口兼容的逻辑内容。

v8 不修改历史 v7 JSON，不进入 GNN graph，不改变 graph tensor schema。

## 3. 保留的 v7 / GNN 分支

GNN 分支仍然保留，但现在是显式实验分支，而不是默认 E2E 生成依赖：

```text
content_list_v7_styles.json
  -> GNNViewAdapter
  -> graph.pt
  -> TeX-derived MERGE / PARENT_CHILD / NONE labels
  -> GNN training / ablation / diagnostics
```

GNN view 不能作为 renderer source。任何带 GNN 的实验都必须把 graph
edge 映射回完整 v7 / DocumentIR / RenderTreeIR，再由 renderer 消费完整
事实层。

当前结论是：stack skeleton 和 layout rules 对 heading/section scope
贡献更稳定；GNN parent 主要作为可选 hint。MERGE 方向仍可用于 relation
learning 研究，但不再支配默认 reconstruction path。

## 4. 主要模块边界

```text
src/perception/    PDF/v7 感知、阅读顺序、样式探针、GNNViewAdapter
src/adapters/      MinerU/v7 -> DocumentIR 适配
src/reasoning/     v8 reflow、heading skeleton、decoder、label/GNN 分支
src/ir/            DocumentIR / RenderTreeIR schema
src/generation/    front matter、style、float/table/list/reference 渲染
src/evaluation/    comparison structure 和指标
tools/             转换、审计、外部 baseline 工具
scripts/pipeline/  数据、训练、E2E、ablation 入口
```

当前默认生成入口：

```bash
python scripts/pipeline/run_v8_layout_reconstruction.py ...
```

可选 GNN 数据/训练入口：

```bash
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
python scripts/pipeline/train_edge_gnn_full.py ...
```

## 5. 当前 generator 能力

v8/default generator 侧当前重点包括：

```text
middle-derived reading-order repair
source PDF page-size preservation
document-local heading style registry
stack skeleton heading hierarchy
FrontMatter Phase 0 title/author/affiliation/email/abstract preservation
single/two/mixed-column approximation
wide figure* / table* rendering
ordered enumerate / itemize recovery
citation and bibliography repair
```

后续精确 author-affiliation-email linking 计划见：

```text
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md
```

## 6. 评估策略

评估分层进行，不能用单一 AST 分数概括：

```text
compile_success
layout_similarity / page_count_score
paragraph_text_coverage
paragraph_boundary_f1
heading_tree_accuracy
section_attachment_body_no_float_f1
float_caption_attachment_accuracy
reference_section_completeness
generated_structure_validity
```

`section_attachment_f1` 是辅助指标。论文主口径优先使用 body-no-float
变体，float/caption、references、layout 需要分开报告。

## 7. 数据策略

生产数据应来自闭环编译：

```text
arXiv TeX source -> compiled PDF -> MinerU -> v8/v7 fact layer -> labels/eval
```

下载时不需要保存 arXiv 原始 PDF；我们需要保留 TeX source 和由 TeX
本地编译出的 PDF，因为后续 MinerU、layout evaluation 和视觉对齐都依赖
这对 source/PDF。

## 8. 文档入口

```text
docs/PROJECT_FILE_LAYOUT.md
docs/PROJECT_SOURCE_OF_TRUTH.md
docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md
docs/generator_logic_audit_2026_05_17.md
docs/v7_training_and_monitoring.md
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md
```
