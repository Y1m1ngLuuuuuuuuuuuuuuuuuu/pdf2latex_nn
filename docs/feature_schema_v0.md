# Feature Schema v0

本文档固定 PDF 前端抽取的第一版数据契约。目标是让 MinerU、PyMuPDF、SciBERT、GNN、validator 和 generator 都面向同一套 IR，而不是各自猜字段。

## 版本

```text
schema_version: feature_schema_v0
coordinate_space: page_normalized_1000
node_feature_dim: 785
edge_attr_dim: 16
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

当前 v4 JSON 为了兼容已有脚本仍使用扁平字段：

```text
bbox: [x0, y0, x1, y1, x0, y0, x1, y1, ...]
```

稳定 IR 中等价字段是：

```text
bboxes: list[BBox]
```

后续适配器负责两者转换。

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
= 785
```

切片固定为：

| 名称 | 范围 | 维度 | 说明 |
| --- | --- | --- | --- |
| `semantic` | `[0, 768)` | 768 | SciBERT `[CLS]` sliding-window mean |
| `type_onehot` | `[768, 778)` | 10 | Block Type 独热 |
| `geometry` | `[778, 782)` | 4 | 首尾锚点几何 |
| `derived_stats` | `[782, 785)` | 3 | 宏观位置、宽高比、文本密度 |

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

## Edge Attribute Tensor

当前 graph builder 输出：

```python
Data(x, edge_index, edge_attr)
```

`edge_attr` 是 directed sequential edge 的关系特征。双向边会各自计算一份 directed feature，所以 `i -> i+1` 和 `i+1 -> i` 的 `delta_*`、文本信号和 `is_forward_edge` 不相同。

固定维度：

```text
edge_attr_dim = 16
```

| index | 字段 | 说明 |
| --- | --- | --- |
| 0 | `semantic_cosine` | 源节点和目标节点 SciBERT 768 维向量的余弦相似度 |
| 1 | `delta_x_start` | `target.x_start_local - source.x_start_local` |
| 2 | `delta_y_start` | `target.y_start_page - source.y_start_page` |
| 3 | `delta_x_end` | `target.x_end_local - source.x_end_local` |
| 4 | `delta_y_end` | `target.y_end_page - source.y_end_page` |
| 5 | `vertical_gap` | 同页时 `target.y0 - source.y1`，小于 0 记 0，再除以页面高 |
| 6 | `horizontal_overlap` | 源/目标 bbox 的水平重叠比例 |
| 7 | `same_page` | 源尾页和目标首页是否相同 |
| 8 | `same_column` | `column_id` 是否相同且已知 |
| 9 | `cross_page` | 源尾页和目标首页是否不同 |
| 10 | `cross_column` | `column_id` 是否不同且已知 |
| 11 | `same_type` | canonical block type 是否相同 |
| 12 | `source_ends_with_hyphen` | 源文本是否以 `-` 结尾 |
| 13 | `source_has_terminal_punctuation` | 源文本是否以句号、问号、感叹号、分号等终止符结尾 |
| 14 | `target_starts_lowercase` | 目标文本首个非空字符是否为小写 |
| 15 | `is_forward_edge` | 目标 index 是否大于源 index |

这些边特征不替代节点特征，而是显式告诉 GNN “相邻块之间的关系”。尤其是 `semantic_cosine`、`delta_*`、`same_column/cross_page` 和断词标记，服务于 merge、continuation、parent-child、caption 等结构判断。

## Model-Side Projection

原始 `.pt` 继续保存完整 768 维 SciBERT 节点语义，不在数据层降维。降维和归一化属于模型层：

```text
semantic_768 -> Linear -> semantic_64 -> LayerNorm
layout/type/stats_17 -> Linear -> layout_32 -> LayerNorm
concat -> model_input_96
```

当前 `src/reasoning/gnn_model.py` 提供 `FeatureProjector` 作为这个瓶颈层的最小实现。后续 GNN 层应优先选择支持 `edge_attr` 的 PyG 层，例如 `GATv2Conv(edge_dim=16)`、`TransformerConv(edge_dim=16)` 或 `GINEConv`。

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

当前生产 JSON 文件仍是：

```text
*_content_list_v2_visual_order.json
*_content_list_v3.json
*_content_list_v4.json
```

其中 v4 是最接近本契约的结构，但仍保留若干兼容字段：

```text
type
raw_type
bbox
text_for_embedding
reference_items
block
```

下一步应增加 adapter/validator，把 v4 JSON 明确转换成 `Document` IR，并用小批量 PDF 验证所有字段可稳定产出。
