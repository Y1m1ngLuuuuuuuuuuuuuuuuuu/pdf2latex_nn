# Feature Schema v0

本文档固定 PDF 前端抽取的第一版数据契约。目标是让 MinerU、PyMuPDF、SciBERT、GNN、validator 和 generator 都面向同一套 IR，而不是各自猜字段。

## 版本

```text
schema_version: feature_schema_v0
coordinate_space: page_normalized_1000
node_feature_dim: 813
edge_attr_dim: 11
```

坐标统一使用 MinerU 当前输出的页面归一化坐标，页面左上角是 `(0, 0)`，右下角近似是 `(1000, 1000)`。如果后续需要保留 PDF 原始点坐标，应新增字段，不覆盖现有归一化坐标。

## 顶层结构

```text
Document
Page
Block
Line
Span
ReferenceItem
EdgeCandidate
FeatureTensorSchema
```

Python 结构定义在 `src/perception/schema.py`。文档说明和代码定义必须同步更新。

## Document

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | string | 固定为 `feature_schema_v0` |
| `document_id` | string | PDF/论文的稳定 id，优先使用 arXiv id 或文件 stem |
| `source_pdf` | string or null | 源 PDF 路径或外部标识 |
| `coordinate_space` | string | 固定为 `page_normalized_1000` |
| `pages` | list[Page] | 页面列表，`page_idx` 从 0 连续递增 |
| `blocks` | list[Block] | 全文视觉阅读顺序下的节点列表 |
| `edges` | list[EdgeCandidate] | 候选边；训练前可以为空 |
| `feature_schema` | FeatureTensorSchema | 节点特征张量切片说明 |
| `metadata` | object | 非训练关键的运行信息、commit hash、工具版本 |

缺失值约定：未知标量使用 `null`，未知列表使用 `[]`，未知对象使用 `{}`。不要用空字符串表示未知；空字符串只表示该字段确实没有文本。

## Page

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `page_idx` | int | 0-based page index，必须连续 |
| `width` | float | 归一化页面宽，当前默认 `1000.0` |
| `height` | float | 归一化页面高，当前默认 `1000.0` |
| `blocks` | list[string] | 本页 block_id 列表 |
| `metadata` | object | 页级扩展信息 |

## BBox

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `x0` | float | 左边界 |
| `y0` | float | 上边界 |
| `x1` | float | 右边界 |
| `y1` | float | 下边界 |

合法 bbox 必须满足 `x0 <= x1`、`y0 <= y1`。坐标允许因为解析误差略微超出 `[0, 1000]`，validator 应报警但不直接静默截断。

当前 v7 JSON 使用 MinerU 原始单块 bbox，不在预处理阶段做跨段或跨页合并：

```text
bbox: [x0, y0, x1, y1]
```

稳定 IR 中等价字段是：

```text
bboxes: list[BBox]
```

后续如果 generator 决定合并多个节点，应在生成阶段维护 `bboxes: list[BBox]`，不回写污染 v7 输入。

## Block

Block 是 GNN 的主节点候选，也是 generator 还原文档结构的核心输入。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `block_id` | string | 稳定节点 id，建议 `b000001` |
| `global_order` | int | 全文视觉阅读顺序 |
| `block_type` | enum | 见 Block Type |
| `raw_type` | string or null | MinerU 原始类型，如 `paragraph`, `list`, `algorithm` |
| `list_type` | string or null | MinerU list 子类型，如 `reference_list` |
| `page_idx` | int | 节点起始页 |
| `bboxes` | list[BBox] | 一个逻辑块可对应多个视觉 bbox |
| `column_id` | int or null | 页内栏编号，未知为 null |
| `is_full_width` | bool | 是否跨栏或全宽块 |
| `text` | string | generator 使用的完整文本 |
| `reference_items` | list[ReferenceItem] | reference 块的结构化条目 |
| `lines` | list[Line] | 行级结构，可为空 |
| `spans` | list[Span] | 样式 span 顺序表，可为空 |
| `merge_count` | int | 聚合了多少源块 |
| `source_page_idxs` | list[int] | 每个 bbox 对应页 |
| `source_visual_orders` | list[int] | 来源视觉顺序 |
| `source_original_indexes` | list[int] | MinerU 原始 index |
| `metadata` | object | 特定抽取阶段的附加标记 |

### Block Type

固定枚举：

```text
text
title
equation
table
figure
algorithm
list
code
reference
other
```

`reference` 不等价于普通 `list`。当 MinerU 输出 `type=list` 且 `list_type=reference_list` 时，IR 必须设为：

```json
{
  "block_type": "reference",
  "raw_type": "list",
  "list_type": "reference_list",
  "reference_items": [...]
}
```

## ReferenceItem

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `text` | string | 一篇参考文献的完整题录 |
| `raw_index` | int or null | 在 MinerU reference_list 中的原始序号 |
| `bbox` | BBox or null | 如果能定位到单条 reference 的框，则填写 |

Reference 的完整题录必须保留给 generator。训练侧的 SciBERT 输入可以使用 `[REFERENCE]` 占位符，避免参考文献文本污染正文语义特征。

## Line

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `line_id` | string | 行 id |
| `page_idx` | int | 所在页 |
| `bbox` | BBox | 行框 |
| `text` | string | 行文本 |
| `spans` | list[Span] | 行内样式片段 |

Line 是可选层。当前 pipeline 可以先只产出 Block 和 Span。

## Span

Span 是 PyMuPDF 样式状态机合并后的文本片段，不允许机械保存每个原始 span。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `text` | string | 保留空格后的文本 |
| `font_name` | string or null | 去掉 PDF subset 前缀后的字体名 |
| `font_size` | float or null | 字号，建议 0.5 pt bucket |
| `is_bold` | bool | 粗体 |
| `is_italic` | bool | 斜体 |
| `is_inline_math` | bool | 行内数学字体/符号 |
| `is_inline_code` | bool | 行内代码字体 |
| `bbox` | BBox or null | span 框 |
| `source` | enum | 当前通常是 `pymupdf` |

## EdgeCandidate

EdgeCandidate 是训练前的候选关系，不等同于最终预测结构。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `source_block_id` | string | 起点 block |
| `target_block_id` | string | 终点 block |
| `relation` | enum | 候选关系类型 |
| `features` | object | 边特征，如距离、跨页、同栏、字体差异 |

固定关系枚举：

```text
next_reading_order
same_page_next
cross_page_next
cross_column_next
float_separated_continuation
parent_child
caption_of
reference_item_of
```

## Node Feature Tensor

当前节点特征维度为：

```text
768 SciBERT semantic
10 type one-hot
4 geometry
3 derived stats
6 style stats
16 sequence position
3 column-aware features
3 title structure features
= 813
```

切片固定为：

| 名称 | 范围 | 维度 | 说明 |
| --- | --- | --- | --- |
| `semantic` | `[0, 768)` | 768 | SciBERT `[CLS]` sliding-window mean |
| `type_onehot` | `[768, 778)` | 10 | Block Type 独热 |
| `geometry` | `[778, 782)` | 4 | 首尾锚点几何 |
| `derived_stats` | `[782, 785)` | 3 | 宏观位置、宽高比、文本密度 |
| `style_stats` | `[785, 791)` | 6 | PyMuPDF 字号、粗体、斜体、数学、代码样式摘要 |
| `sequence_position` | `[791, 807)` | 16 | 基于单双栏状态机扫描线阅读序的正弦位置编码 |
| `column_features` | `[807, 810)` | 3 | 左栏、右栏、单栏/跨栏 one-hot |
| `title_structure` | `[810, 813)` | 3 | 相对字号、一级标题编号探针、二级及以下标题编号探针 |

### Geometry Fields

```text
x_start_local
y_start_page
x_end_local
y_end_page
```

`x_*_local` 使用动态局部栏坐标：`(x - column_x_min) / column_width`。`y_*_page` 使用页面高度归一化。

### Derived Stats

```text
macro_position = global_order / (node_count - 1)
aspect_ratio = sum_bbox_height / sum_bbox_width
text_density = char_count / sum_bbox_area
```

`equation/table/figure/algorithm/code` 的 `text_density` 强制为 `0.0`。`reference` 的 BERT 输入使用 `[REFERENCE]`，但 JSON 仍保留完整 `reference_items`。

### Style Stats

```text
baseline_font_size_norm = baseline_font_size / 100
font_size_vs_doc_body = (baseline_font_size - document_body_font_size) / document_body_font_size
bold_char_ratio = bold_chars / styled_chars
italic_char_ratio = italic_chars / styled_chars
inline_math_char_ratio = inline_math_chars / styled_chars
inline_code_char_ratio = inline_code_chars / styled_chars
```

这些字段来自 PyMuPDF 的状态机合并 `style_spans`。如果某个节点没有样式抽取结果，对应 style stats 使用 `0.0`。`document_body_font_size` 从正文类节点的 char-weighted baseline size 中推断。

## Edge Attribute Tensor

当前 graph builder 输出：

```python
Data(x, edge_index, edge_attr)
```

`edge_attr` 是候选边的 directed relation feature。建图使用双重视角邻居采样：

```text
sequential_window = 3
spatial_k = 3
```

候选边来自两类视角：

```text
forced sequential neighbors: 按单双栏状态机阅读序强制连接 `i -> i+1`，默认同时连接 `i+1 -> i`
reading-order neighbors: 每个节点连接单双栏状态机阅读序列前后各 k 个节点
line-of-sight neighbors: 每个节点在同页向下、向右各寻找最近 k 个空间邻居
```

重复边只保留第一条，优先级为 `sequential_forced`、reading-order，再是 `spatial_down`，最后是 `spatial_right`。`Data.edge_source_types` 与 `edge_index` 列对齐，用于记录候选边来源。

固定维度：

```text
edge_attr_dim = 11
```

| index | 字段 | 说明 |
| --- | --- | --- |
| 0 | `semantic_cosine` | 源节点和目标节点 SciBERT 768 维向量的余弦相似度 |
| 1 | `delta_y_gap` | `(target.y_min - source.y_max) / page_height`，允许负数表达跨栏/回跳 |
| 2 | `delta_x_left` | `(target.x_min - source.x_min) / page_width` |
| 3 | `left_alignment` | 若 `abs(delta_x_left) < 0.01` 则为 `1.0`，否则 `0.0` |
| 4 | `center_distance` | 源/目标在逻辑阅读坐标系中的中心距离。双栏区域会被展开：左栏占该区域阅读进度前半段，右栏占后半段；X 使用栏内相对坐标，避免左栏底部与右栏顶部被物理沟壑误判为远距离 |
| 5 | `font_size_delta` | `target_font_size - source_font_size`，缺失样式时为 `0.0` |
| 6 | `bold_to_regular` | 源节点多数文本为粗体且目标节点不是粗体时为 `1.0` |
| 7 | `line_height_ratio` | `target_bbox_height / source_bbox_height` |
| 8 | `index_delta` | `target_index - source_index` |
| 9 | `is_next` | 若 `target_index - source_index == 1` 则为 `1.0` |
| 10 | `is_sequential_edge` | 若边来自强制/窗口阅读流连边则为 `1.0` |

这 11 维严格只表达语义连续性、空间相对性、排版阶跃性和序列跨度。独立公式、图表、算法等类别信息保留在节点 type one-hot 中，不再额外塞入边特征，避免边张量过度膨胀。

## Model-Side Projection

原始 `.pt` 继续保存完整 768 维 SciBERT 节点语义，不在数据层降维。降维和归一化属于模型层：

```text
semantic_768 -> Linear -> semantic_64 -> L2 normalize
layout/type/stats/sequence/column/title_45 -> Linear -> layout_32 -> LayerNorm
concat -> model_input_96
```

当前 `src/reasoning/gnn_model.py` 提供 `FeatureProjector` 作为这个瓶颈层的最小实现。后续 GNN 层应优先选择支持 `edge_attr` 的 PyG 层，例如 `GATv2Conv(edge_dim=11)`、`TransformerConv(edge_dim=11)` 或 `GINEConv`。

## Validator 最低要求

validator 至少检查：

```text
schema_version 是否匹配
page_idx 是否从 0 连续
global_order 是否从 0 连续
block_type 是否在枚举内
bbox 数量是否合法
bbox 坐标是否满足 x0 <= x1, y0 <= y1
reference block 是否有 reference_items
非 reference block 是否不误带 reference_items
feature_schema.node_feature_dim 是否等于实际 tensor 宽度
edge_attr_schema.dim 是否等于实际 edge_attr 宽度
edge_attr 行数是否等于 edge_index 列数
```

## 当前兼容层

当前生产 JSON 文件固定为：

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

其中 v7 保留 MinerU 原始 bbox 和文本块粒度，只新增 list marker、列修复阅读序和 PyMuPDF 样式字段：

```text
type
raw_type
bbox
text_for_embedding
reference_items
list_marker
column_fix_span
column_fix_column
style_spans
block
```

后续 adapter/validator 应以 v7 为唯一输入格式，把 v7 JSON 明确转换成 `Document` IR，并用小批量 PDF 验证所有字段可稳定产出。
