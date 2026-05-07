# Ground Truth Labeling v0

本文档固定当前自动化真值生成器的工程契约。目标是把 arXiv TeX 源码和 MinerU/PyMuPDF 视觉节点对齐，给 PyG 图生成三分类边标签：

```text
MERGE = 0
PARENT_CHILD = 1
NONE = 2
```

当前实现入口：

```text
src/reasoning/label_generator.py
scripts/pipeline/step3_label_graph.py
scripts/pipeline/build_mini_dataset.py
```

## Inputs

单篇论文标注需要三份输入：

```text
content_v7_styles.json
main.tex 或等价 TeX 入口文件
unlabeled graph.pt
```

`content_v7_styles.json` 是视觉侧序列，必须已经完成 v7 阅读序修复和 PyMuPDF 样式注入。`graph.pt` 必须与 JSON 节点数一致，否则直接报错，不尝试静默重排。

## TeX AST Parser

解析器使用 TexSoup。正则只用于清洗文本、display math 规范化和节点类型嗅探，不用于整体解析 LaTeX 嵌套结构。

解析流程：

1. `latex_flattener.py` 展平 `\input` / `\include`，注入 `.bbl`，处理简单宏。
2. 优先只遍历 `\begin{document}...\end{document}` 内部，避免 preamble 宏定义污染视觉正文。
3. 用 TexSoup 展开 AST，并按深度优先顺序生成一维 TeX 序列 `T=[T_1,...,T_n]`。

每个 TeX 节点导出字段：

| 字段 | 说明 |
| --- | --- |
| `tex_id` | 全局递增 id，如 `T_000001` |
| `node_type` | 语义类型 |
| `text` | 剥壳后的原始文本 |
| `clean_text` | 对齐用清洗文本 |
| `parent_id` | 原始 TeX 语义父节点 |
| `path_ids` | 从 `ROOT` 到当前节点的路径 |
| `source_name` | 原 TeX 命令或环境名 |

当前白名单类型：

```text
section
paragraph
equation_display
list_container
list_item
figure_caption
table_caption
algorithm
```

其中 `list_container` 只参与 parent-child 结构，不直接参与 fuzzy 文本对齐。

## Unknown TeX Fallbacks

未知结构按保守策略处理：

```text
non-visual node without text -> silent drop
unknown wrapper macro with text -> unwrap into paragraph
unknown environment with text -> downgrade to paragraph/container traversal
```

已知会破坏视觉-文本一致性的绘图环境直接熔断：

```text
tikzpicture
pgfpicture
pgfplots
axis
pspicture
```

触发后抛出 `LayoutBreakerException`，批处理应跳过该论文或该样本。

## Text Normalization

PDF 和 TeX 使用同一套强清洗规则：

```text
lowercase
remove TeX wrappers and math delimiters
remove whitespace and punctuation
keep alphanumeric / CJK payload
```

display equation 使用专门的 `clean_equation_text()`。它会保留 `\alpha`、`\beta` 这类命令名，避免纯命令公式被清洗成空串。

行内公式和 display 公式必须分开处理：

```text
\(...\)、$...$、\begin{math}...\end{math} -> 留在 paragraph/list_item 文本内
\[...\]、$$...$$、equation/align/gather/multline/flalign/displaymath -> equation_display
```

这条规则是为了避免列表项被行内公式打碎。例如 `Euclidean distance (\(d_E\))` 必须仍是同一个 `list_item`，不能被拆成多个 enumerate item。

## Alignment Engine

对齐器使用顺序双指针，不做 `O(N^2)` 全局搜索。

核心逻辑：

1. 指针 `i` 指向 PDF 节点 `V_i`，指针 `j` 指向 TeX 节点 `T_j`。
2. 从 `V_i` 开始累加 buffer。
3. 用 Levenshtein ratio 比较 `buffer.clean` 和 `T_j.clean_text`。
4. 达到阈值后记录 `T_j -> [V_i,...,V_k]`，两个指针继续向前。
5. 如果当前 TeX 不匹配，允许有限 lookahead 找更好的 TeX 候选。

默认主要参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `similarity_threshold` | `65.0` | fuzzy 对齐阈值 |
| `max_window_nodes` | `8` | 一个 TeX 节点最多吃入多少个 PDF bbox |
| `tex_lookahead_nodes` | `4` | TeX 侧错位时向前查找多少节点 |
| `tail_absorption_nodes` | `3` | 长段落尾部碎片吸收窗口 |
| `equation_blind_alignment_window` | `2` | 公式盲对齐的 PDF 邻近窗口 |
| `caption_fallback_threshold` | `80.0` | caption 全局弱匹配阈值 |

### Equation Blind Alignment

当 `T_j.node_type == equation_display` 时，优先在当前位置附近寻找 PDF 公式类 bbox：

```text
equation
equation_interline
display_formula
formula
```

如果找到，直接赋予 `score=100.0`。这解决 `\frac{a}{b}` 和视觉 OCR `a/b` 不可比的问题。

### Caption Global Fallback

浮动体 caption 允许一轮窄域全局弱匹配。顺序滑动窗口结束后，对仍未匹配的 PDF 节点，只和未匹配的 `figure_caption` / `table_caption` TeX 节点比较：

```text
score = max(partial_ratio, levenshtein_ratio)
threshold = caption_fallback_threshold (default 80.0)
```

该兜底只处理 caption，不扩大到普通 paragraph，避免把主文顺序约束打碎。

## Label Rules

图标签写入：

```text
graph.y
graph.edge_label
graph.pdf_to_tex
graph.pdf_to_tex_scores
graph.label_counts
graph.alignment_schema
```

### MERGE

两个 PDF 节点映射到同一个 TeX 节点时，只有类型同构才允许 `MERGE=0`：

```text
text + text -> MERGE
reference + reference -> MERGE
equation + equation -> MERGE
text + equation/table/figure/algorithm -> NONE
```

长标题跨行时，如果多个 bbox 映射到同一个 `section`，仍按同构规则打 `MERGE`。

列表 marker 是硬边界。如果目标节点以 `1.`、`a.`、bullet 等列表符号开头，不与前一节点合并。

### PARENT_CHILD

父子关系来自 TeX `parent_id` 或视觉标题栈兜底：

```text
parent_tex_id -> child_tex_id
```

只在父节点映射的第一个 PDF bbox 与子节点映射的第一个 PDF bbox 之间打 `PARENT_CHILD=1`，避免一个逻辑父节点对所有碎片重复连边。

### NONE

其它所有情况都是 `NONE=2`：

```text
missing alignment
cross-type same-TeX fragments
ordinary sibling paragraphs
bad visual relation
non-anchor duplicate edge
```

当前不再使用 `SIBLING` 类。兄弟顺序由 v7 reading order 和 renderer 排序保证。

## Quality Gates

严格模式由 `--abort-on-bad-alignment` 开启。失败时抛出 `AlignmentQualityError`，批处理跳过样本。

质量统计使用“有效 PDF 节点”口径，而不是把模板噪声和正文同等计算：

```text
expected visual orphans:
  page_header / page_footer / page_number / header / footer / footnote / watermark
  或页面极靠上/靠下、文本很短的边缘节点

document root scoped:
  成功匹配、TeX parent_id=None、且不是 section 的前置元数据节点
  例如 title/authors/abstract/keywords 这类第一个 section 之前的内容
```

前者不参与 orphan ratio 和 isolated ratio；后者不参与 isolated ratio。Graph 里暂不额外插入 `DOCUMENT_ROOT` 节点，以保持 `graph.pt` 节点维度契约稳定；TreeDecoder 端已经有虚拟 root 来承接这些顶级节点。

默认熔断阈值：

| 指标 | 默认值 | 说明 |
| --- | ---: | --- |
| `max_orphan_ratio` | `0.15` | PDF 节点未映射比例上限 |
| `max_unmapped_tex_ratio` | `0.30` | TeX alignable 节点未映射比例上限 |
| `max_isolated_node_ratio` | `0.85` | MERGE/PARENT_CHILD 完全不连通节点比例上限 |
| `min_section_nodes` | `1` | 正文较多时至少需要解析出 section |

2501.00050 在当前纯逻辑真值生成器下可用于诊断，但严格模式会被拒绝：

```text
orphan_ratio=65.65% > 15.00%
unmapped_tex_ratio=81.28% > 30.00%
```

这说明熔断阀已经生效：低覆盖样本不会静默进入训练集。

## Commands

单篇标注：

```bash
python scripts/pipeline/step3_label_graph.py \
  --content-json data/02_mineru_outputs/mineru_output/2501.00050/auto/2501.00050_content_list_v7_styles.json \
  --tex data/03_tex_source_pool/2501.00050/aaai25.tex \
  --graph data/06_graph_features_v7/2501.00050_v7_graph.pt \
  --output data/06_graph_features_v7/2501.00050_v7_truthgen_labeled_graph.pt \
  --mapping-output data/04_ground_truth_ir/2501.00050_v7_alignment_mapping.json \
  --similarity-threshold 65
```

严格模式：

```bash
python scripts/pipeline/step3_label_graph.py \
  --content-json ... \
  --tex ... \
  --graph ... \
  --abort-on-bad-alignment
```

小批量构建：

```bash
python scripts/pipeline/build_mini_dataset.py \
  --target 10 \
  --similarity-threshold 65 \
  --max-orphan-ratio 0.15 \
  --max-unmapped-tex-ratio 0.30 \
  --max-isolated-node-ratio 0.85
```

## Tests

核心测试：

```text
tests/test_alignment_labeler.py
tests/test_label_generator.py
tests/test_tex_relation_labeler.py
tests/test_document_dataset.py
```

AST 框架对照工具：

```bash
python tools/compare_tex_ast_framework.py \
  --tex data/03_tex_source_pool/2501.00050/aaai25.tex \
  --output data/09_eval_reports/ast_framework_compare/2501.00050_ast_framework_compare.json
```

该工具把源码中的 section/list/equation/algorithm/caption 框架与 `AlignmentLabeler.parse_tex_nodes()` 的 TexSoup AST 框架做顺序对比，只比较结构节点，不比较公式内部 TeX 字符串。

当前 AutoDL 验证：

```text
144 passed
```
