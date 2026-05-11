# Frontend / Backend Contract v1

本文档固定 PDF 前端、TeX 真值生成器、GNN、TreeDecoder 和 LaTeX Renderer 之间的解耦接口。目标是让每一层只依赖稳定 IR，而不是依赖上游的具体实现。

## 总体边界

```text
PDF Frontend
compiled PDF + MinerU v7 + PyMuPDF + SciBERT
→ *_content_list_v7_styles.json
→ DocumentIR.json
→ GraphInput.pt + GraphInput.json

TeX Label Backend
same compile-record TeX source
→ TeX AST
→ GraphLabels.json / graph_labeled.pt

GNN
GraphInput + GraphLabels
→ trained relation model

Inference
GraphInput + model
→ PredictedRelations.json

Decoder + Generator
DocumentIR + PredictedRelations + StyleProfile
→ RenderTreeIR.json
→ generated.tex
→ generated.pdf
```

## 接口铁律

```text
1. PDF 前端只描述“看见了什么”，不写真值标签，不写模型预测。
2. TeX 真值生成器只描述“源码结构如何对应图边”，不修改 PDF 视觉节点。
3. GNN 只读 GraphInput / GraphLabels，输出 PredictedRelations。
4. Decoder 只把 PredictedRelations 变成 RenderTreeIR，不负责最终 LaTeX 排版。
5. Generator 只读 DocumentIR + RenderTreeIR + StyleProfile + CitationResolution。
6. 表格/图片截图属于显式资产生成，不是默认渲染行为。
```

所有跨层数据都必须携带 `schema_version`。当前接口定义在：

```text
src/ir/schema.py
src/ir/serialization.py
src/ir/validators.py
```

数据生产阶段必须保留 `pdf_origin` 和 compile manifest provenance。训练用样本的
正常值是 `pdf_origin="compiled_from_tex"`，表示 PDF 和 TeX 来自同一条
compile accepted 记录。官方 PDF 或同 ID 猜配对只能作为调试输入，不能混入
production label 数据。

## 1. DocumentIR

`DocumentIR` 是 PDF 前端的最终边界。它描述视觉事实，不描述训练标签。

来源：

```text
MinerU v7 raw output
PyMuPDF style spans
v7 reading-order / toc filter / header-footer filter
```

v7 适配入口：

```text
src/adapters/mineru_v7_document_ir.py
scripts/pipeline/convert_v7_to_document_ir.py
```

原则：

```text
v7 JSON 是前端内部格式
DocumentIR 是前后端稳定接口
Renderer / StyleProfileExtractor / CitationResolver 不直接依赖 v7 私有字段
后续若出现 v8，只新增或修改 adapter，不改生成后端
```

核心字段：

```text
doc_id
source_pdf
pages
nodes
reading_order
provenance
metadata
```

`nodes` 中的每个 `DocumentNode` 必须保留：

```text
node_id
node_type
text
page_idx
bboxes
reading_index
spans
flags
features
source_refs
```

当前稳定的 `node_type` 至少包括：

```text
text / title / equation / inline_math / table / figure / algorithm / list / code
footnote / margin_note / reference / toc / header_footer / other
```

表格节点的附加约定：

```text
MinerU 可能把一个视觉宽表切成多个左右相邻 table block。
adapter 必须在 DocumentNode.metadata 中写入 table_group_* 字段：

table_group_id
table_group_member_node_ids
table_group_member_index
table_group_size
table_group_primary
table_group_bbox
table_group_caption
table_group_render_strategy

同一 group 只允许 primary 节点在 renderer 中输出 table 环境。
批量渲染默认不主动为每个 bbox 生成 PDF crop，避免大规模 QA/训练
产物占用过多磁盘。figure 会优先复用 MinerU 已经给出的
`img_path` / `image_path` 等位图资产；如果没有现成资产，只有显式打开
`--render-table-crops` 或给 IR renderer 配置 `table_asset_output_dir`
或 `figure_asset_output_dir` 时，才根据 table_group_bbox / figure bbox
从 source_pdf 重新裁一整块图并用 `\includegraphics` 渲染。
两者都不存在时才保留 TODO placeholder。
MinerU table_body/HTML 只作为后续结构化重建的候选证据。
```

约束：

```text
不提前做跨段落合并
不回写 generator 的合并结果
不把 TeX label 写入 DocumentIR
toc/header/footer/noise 只做标记或过滤策略记录
table/figure/image 只保留 OCR、bbox、caption、asset hint，不默认截图
```

稳定字段与扩展字段的边界：

```text
稳定字段:
  node_id / node_type / text / page_idx / bboxes / reading_index
  spans / flags / features / source_refs / metadata

允许在 metadata 中扩展:
  table_group_* / figure_caption / reference_items / toc_role
  raw_mineru_type / raw_block / source_pdf / image_path

不允许在 metadata 中扩展:
  y / predicted_label / logits / loss / train_split
```

## 2. GraphInput

`GraphInput` 是 GNN 的纯输入契约。它可以引用 PyTorch `.pt` 张量文件，也可以附带轻量 JSON manifest。

核心字段：

```text
doc_id
document_ir_path
graph_path
node_ids
edge_ids
x
edge_index
edge_attr
feature_schema_version
graph_schema_version
```

`GraphInput` 不包含真值标签。这样改 TeX labeler 时不需要重跑 PDF 前端和 SciBERT。

## 3. GraphLabels

`GraphLabels` 是 TeX 真值生成器的输出契约。

固定三分类：

```text
0 MERGE
1 PARENT_CHILD
2 NONE
```

核心字段：

```text
doc_id
graph_input_path
edge_ids
y
alignments
quality_report
```

约束：

```text
edge_ids 必须与 GraphInput.edge_ids 对齐
PARENT_CHILD 是有向标签
SIBLING 不再作为类别出现
alignment_dict 只作为可审计证据，不作为 PDF 前端字段
```

## 4. PredictedRelations

`PredictedRelations` 是模型推理输出。TreeDecoder 读取它，不读取模型内部对象。

核心字段：

```text
doc_id
graph_input_path
edge_ids
predicted_labels
probabilities
logits
threshold_config
model_version
```

约束：

```text
阈值校准写入 threshold_config
概率矩阵类别顺序固定为 MERGE / PARENT_CHILD / NONE
```

## 5. RenderTreeIR

`RenderTreeIR` 是 LaTeX Generator 的唯一结构输入。

核心字段：

```text
doc_id
root_id
nodes
document_ir_path
predicted_relations_path
style_profile_path
```

每个 `RenderTreeNode`：

```text
render_id
role
source_node_ids
text
latex
children
attributes
```

Generator 不应该直接消费 GNN logits，也不应该直接遍历 graph edge。

RenderTreeIR 的 sibling 顺序不是最终可信顺序。Generator 在渲染
children 前必须使用源 `DocumentNode.reading_index` 做兜底排序。这样
MST 或 decoder 的插入顺序不会破坏正文流。

对于同一个父节点下连续出现的 list-like children：

```text
text="1. ..." / "a. ..." / "• ..." / node_type=list
```

Generator 必须动态合成为 `enumerate` 或 `itemize`。如果列表项之间
夹有 display equation / inline_math / table / figure / algorithm / code，
这些节点默认作为上一条 item 的内部内容，不闭合列表环境。

## 6. StyleProfile

`StyleProfile` 决定同一棵 RenderTree 如何被包装成 LaTeX。

支持三种模式：

```text
original_like
journal_template
learned_style
```

当前代码入口：

```text
src/generation/style_profile.py
```

`StyleProfileExtractor` 从 `DocumentIR` 统计全局排版画像，不参与
GNN 关系判断。第一版提取：

```text
page_layout:
  page_width / page_height
  aspect_ratio
  margins
  margin_ratios
  text_width / text_height
  column_count
  column_gap
  column_gap_ratio
  column_mode: single / two_column / mixed
  two_column_page_ratio
  mixed_columns

role_styles:
  body / heading / section / subsection / subsubsection
  bibliography / list / math / table / figure
  font_size
  relative_font_size
  font_family
  font_class: serif / sans / mono / math
  bold / italic

renderer_options:
  body_font_size
  body_font_family
  body_font_class
  font_clusters:
    font_size
    relative_to_body
    char_weight
    dominant_role
    role_weights
    dominant_font_family
    bold_ratio / italic_ratio
  role_font_clusters
  font_setup:
    enabled
    requires_engine
    body_pdf_font
    main_font / sans_font / mono_font
    role_pdf_fonts
    role_font_classes
  paragraph_indent
  paragraph_spacing
  display_math_spacing.above / below
  list_spacing.itemsep / topsep
  geometry_options
  column_mode
  bibliography.strip_source_labels
  bibliography.citation_key_strategy
  header_footer:
    render_by_default
    page_number.enabled / position / confidence
    header.enabled / text / position / confidence
    footer.enabled / text / position / confidence
```

StyleProfile 是生成后端的版式控制面，不参与 GNN 特征或训练标签。
字体与字号判断必须来自 `DocumentIR.nodes[].spans`、节点类型、
节点 bbox 和 v7/PyMuPDF 已提取的 feature 字段。Generator 不应根据
标题关键词、正文语义或固定期刊名称臆造样式。

同一个 RenderTreeIR 可以搭配不同 StyleProfile 生成不同目标：

```text
original_like: 尽量复原原 PDF 的页面、字号、缩进、双栏、列表、公式间距
journal_template: 显式套用 IEEE/ACM/NeurIPS/Elsevier 等模板
learned_style: 从一组论文统计出 StyleProfile，再显式渲染
```

### original_like

目标是尽量接近原 PDF 的排版。可使用：

```text
字号聚类
标题层级
单双栏/局部 band
列表缩进
caption 样式
公式/图表位置
```

### journal_template

目标是将结构内容放入显式模板：

```text
IEEE
ACM
NeurIPS
AAAI
Elsevier
article
```

在此模式下，模板优先于原 PDF 样式。

### learned_style

目标是从论文簇中学习样式配置，但输出仍然是显式 `StyleProfile`，不是让模型直接生成任意 LaTeX。

## 7. CitationResolution

引用和参考文献不应该作为普通文本直接渲染。PDF OCR 会把正文引用
识别成 `[1]`，也会把 bibliography 条目前缀识别成 `[1]` 或 `1.`。
LaTeX 生成阶段必须把它们转成语义结构：

```text
正文 [1-3]        -> \cite{ref_1,ref_2,ref_3}
参考文献 [1] ...  -> \bibitem{ref_1} ...
正文 (Smith, 2020) -> \cite{Smith2020}
参考文献 Smith...2020 -> \bibitem[Smith, 2020]{Smith2020} ...
```

当前代码入口：

```text
src/generation/citations.py
```

`CitationResolver` 输出：

```text
entries:
  key
  label
  display_label
  authors / year
  text
  source_node_id

occurrences:
  node_id
  raw_marker
  keys
  citation_style
  start / end

text_by_node_id:
  修复后的正文文本
```

如果 TeX / `.bbl` 能提供真实 citation key，后续可以覆盖 `ref_1`
这类 PDF-only 伪 key。没有真实 key 时，`ref_<number>` 是稳定降级方案。
当前实现会优先读取 `reference_items` 内的 `citation_key` / `bib_key`
/ `bibkey` / `bibtex_key` / `tex_key`。如果这些字段存在，正文 `[1]` 会被修复成
真实 key，例如 `\cite{smith2024}`，参考文献条目会输出为
`\bibitem{smith2024}`。

如果没有真实 key，resolver 会从 author-year reference 文本中推断稳定 key
和 optional label，例如 `Smith, J. (2020)` -> `Smith2020` 与
`\bibitem[Smith, 2020]{Smith2020}`。正文里的 `(Smith, 2020)`、
`Smith et al. (2020)`、`;` 分隔的多 author-year 引用会被映射成同一组
`\cite{...}`。数字引用仍然展开范围；在 numeric citation style 下，IR
renderer 会加载 `cite` package，由 LaTeX 负责压缩连续编号显示。

OCR 识别出的参考文献序号 `[1]` / `1.` / `【1】` 不进入最终
bibliography 正文。

## 8. Original-Like IR Renderer

当前新后端入口：

```text
src/generation/ir_renderer.py
src/generation/render_surface.py
```

它只读取稳定接口：

```text
DocumentIR + RenderTreeIR + StyleProfile + CitationResolution -> generated.tex
```

推荐脚本入口是 `render_original_like_document(document, tree, ...)`。它会补齐
缺失的 `StyleProfile` / `CitationResolution`，然后调用
`OriginalLikeIRLatexRenderer`。`src/generation/latex_renderer.py` 不再作为整
文档生产入口，只保留 escape、inline math、algorithm、table/figure block 等
底层 helper；其中 `render_latex_document()` 是 legacy 兼容面。

已支持的 original-like 生成能力：

```text
全局页面包装:
  \documentclass 选项
  geometry 页面/边距设置
  body font size / baseline
  parindent / parskip
  display equation spacing
  list spacing
  title spacing

局部 span 渲染:
  bold -> \textbf{...}
  italic -> \textit{...}
  inline_code -> \texttt{...}
  inline_math -> $...$
  citation marker -> \cite{...}

	引用/参考文献:
	  正文 citation marker 修复
	  reference label stripping
	  \begin{thebibliography} + \bibitem

	脚注/边注:
	  footnote / margin_note 节点先从正文流收容
	  再锚定到最近的前置正文节点
	  渲染为 \footnote{...} / \marginpar{...}

	表格:
	  table fragment grouping
	  默认 table placeholder
	  显式开启后 union-bbox PDF crop -> \includegraphics
	  table_body/HTML 作为弱证据保留，不作为默认渲染主路径
		图片:
		  优先复用 MinerU img_path / image_path -> \includegraphics
		  无现成图片且显式开启 crop assets 后 bbox PDF crop -> \includegraphics
		  最后才 fallback 为 figure placeholder + caption
	  仍保留 caption 作为结构化文本
	```

IR renderer 额外保证：

```text
children 渲染前按 source DocumentNode.reading_index 排序
连续 sibling 列表自动合成 itemize/enumerate
列表项中间的公式/表格/图片/算法作为上一条 item 的内部内容渲染
RenderRole.TABLE / FIGURE / ALGORITHM / CODE / ABSTRACT / TOC_PLACEHOLDER 有独立分发
算法用 algorithmic 环境，不再默认 verbatim
	表格默认输出 TODO_TABLE_RECONSTRUCT placeholder，不生成截图资产
	图片优先使用 MinerU 现成图片资产；缺失资产时才进入 crop/placeholder fallback
	显式开启 crop assets 后，table / figure 都会从 source_pdf 裁剪 bbox
并输出 \includegraphics
重复 REFERENCES / REFERENCE_ITEM sibling 会折叠成一个 bibliography
孤立 symbol-font 大括号不会被渲染成 inline math
inline math 中的 unicode 数学符号会转成 LaTeX 命令
普通 text/span 中裸露的 `\mathrm{...}` / `\frac{...}` 等数学命令会被
保护成行内公式；display equation 中的单个 `\tag{}` 或尾随 `(1)` 编号
会升格为 equation 环境，多行 `&` 对齐会升格为 align 环境。
```

保守策略与未完备能力：

```text
页眉/页脚全局复刻:
  前端可保留 header_footer 节点，StyleProfile 会单独统计
  header_footer 字号、示例、边缘位置和跨页重复度。
  Generator 不把 OCR 页眉页脚逐页写回正文，避免污染论文内容。
  如果统计显示存在稳定页眉/页脚/页码，则用 fancyhdr 生成全局
  \fancyhead / \fancyfoot。低置信边缘文本仍然只保留为 metadata。

脚注/边注:
  前端显式标成 footnote / margin_note 后，Generator 会结构化输出
  \footnote{...} / \marginpar{...}。
  当前仍然不会把普通底部文本强行猜成脚注。

表格结构化重建:
  当前只保留 MinerU table_body/HTML 与 union-bbox placeholder。
  未来可接入 CV 表格重建或显式开启 PDF crop。
```

旧的 `TreeDecoder.render_document()` 和
`latex_renderer.render_latex_document()` 暂时继续保留，便于现有推理脚本和
历史测试不被打断。新生成后端必须逐步切到
`render_original_like_document()` / `OriginalLikeIRLatexRenderer`。

## 推荐目录

```text
data/
  03_document_ir/
  04_tex_ast_ir/
  05_alignments/
  06_graph_inputs/
  07_graph_labels/
  08_predictions/
  09_render_tree/
  10_generated_tex/
  11_generated_pdf/
```

## 当前兼容策略

现有 v7 生产脚本仍然可以继续写 `.pt`：

```text
data/06_graph_features_v7...
```

新接口层是 additive 的。后续可以逐步增加 sidecar JSON：

```text
graph_input.json
graph_labels.json
predicted_relations.json
render_tree.json
style_profile.json
```

这样不影响当前 AutoDL 后台生产任务，也不会破坏现有训练入口。
