# 最新测试结果与指标说明（2026-05-18）

本文档整理当前项目中最新可引用的测试结果、Ablation 结论、Nougat 对比结果，以及每个指标的具体含义。

重要口径：截图里的 `Final Ablation Table` 属于 **locked registry-adapter baseline / M07 家族旧基线表**，不是最新 clean hard20 rollup 的唯一结果。它仍然有价值，因为它是稳定可回滚的模型家族；但如果论文中引用“当前最新端到端结果”，应优先引用 2026-05-18 的 clean metrics rollup。

---

## 1. 当前结果文件位置

最新 clean hard20 + Nougat 汇总：

```text
local_outputs/final_eval_20260518/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

核心文件：

```text
current_eval_rollup.md
current_eval_rollup.json
ablation_summary.csv
e2e_documents.csv
nougat_paired_documents.csv
```

远程 AutoDL 对应路径：

```text
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

当前 Ablation 总说明文档：

```text
docs/ablation_results_current.md
```

该文件同时记录：

```text
1. locked registry-adapter baseline / M07 家族旧基线结果
2. v7_floatproxy_adapter / 当前 clean rollup 结果
```

所以后续写论文时需要区分“模型结构旧基线 ablation”和“当前 E2E/Nougat clean rollup”。

---

## 2. 当前最新 hard20 E2E 与 Nougat 对比

样本数：

```text
paired_documents = 20
```

这 20 篇是 hard cases，用来快速验证当前系统在真实难例上的结构表现。它不是最终大规模统计，只是当前最新、最干净的一组配对比较。

| 指标 | 我们的方法 | Nougat | 差值（我们 - Nougat） |
| --- | ---: | ---: | ---: |
| `generated_structure_validity` | 0.9976 | 0.9542 | +0.0434 |
| `macro_structure_score` | 0.8614 | 0.7925 | +0.0689 |
| `heading_tree_accuracy` | 0.7540 | 0.6350 | +0.1190 |
| `paragraph_boundary_f1` | 0.8125 | 0.6003 | +0.2122 |
| `paragraph_text_coverage_f1` | 0.8798 | 0.8341 | +0.0457 |
| `reading_order_accuracy` | 0.9789 | 0.9873 | -0.0084 |
| `reference_section_completeness` | 0.9908 | 0.6985 | +0.2923 |
| `float_caption_attachment_accuracy` | 0.7246 | 0.5878 | +0.1368 |
| `section_attachment_body_no_float_f1` | 0.9018 | 0.9280 | -0.0262 |

补充：

```text
compile_success_rate = 0.9500
compiled = 19 / 20
layout_similarity = 0.8045
```

当前结论：

```text
1. 我们在 macro structure、heading tree、paragraph boundary、references、
   float/caption 和 generated validity 上领先 Nougat。
2. Nougat 在 reading order 和 body-only section attachment 上略优。
3. 我们的核心优势集中在版式感知的块结构保留、references、可编译结构、
   以及 float/caption 组织。
4. 目前主要短板是正文块挂载到正确 heading scope 的精细程度仍略低于 Nougat。
```

---

## 3. 指标含义说明

### 3.1 `generated_structure_validity`

衡量生成出的结构 IR / LaTeX 是否满足基本结构合法性。

它主要检查：

```text
1. 是否存在非法树结构
2. 是否存在明显断裂的节点引用
3. heading / paragraph / float / reference 等结构是否可解析
4. 生成 LaTeX 是否接近可编译、可消费的状态
```

越高越好。该指标不是视觉相似度，也不是 OCR 全文准确率，而是结构合法性。

### 3.2 `macro_structure_score`

多个结构指标的综合平均分，用于快速概览整体结构表现。

通常综合：

```text
heading tree
reading order
paragraph boundary
section attachment
references
float/caption
validity
```

越高越好。论文中可以作为总览指标，但最好同时列出分项指标。

### 3.3 `heading_tree_accuracy`

衡量块级标题树是否恢复正确。

主要关注：

```text
\section
\subsection
\subsubsection
```

是否形成正确的大纲层级。

注意：run-in heading 不应作为该主指标的强约束。例如 `Summary. The method consists of ...` 这类视觉上和正文在同一行的粗体短语，不应强制要求系统输出 `\paragraph{}`。当前任务目标是 block-level heading，而不是作者源码级 AST 复原。

### 3.4 `reading_order_accuracy`

衡量块级阅读顺序是否正确。

它关注：

```text
1. 单栏/双栏/混合版面下块顺序是否正确
2. 左栏读完再右栏，或者 full-span block 切换是否合理
3. references / appendix / float 等位置是否没有严重乱序
```

越高越好。它不等于视觉排版完全一致，只衡量结构阅读顺序。

### 3.5 `paragraph_boundary_f1`

衡量段落边界是否切分正确。

它关心：

```text
1. 应该合并的文本块是否被合并
2. 独立段落是否没有被错误合并
3. 被公式、图表、分页、分栏打断的正文是否保持合理段落边界
```

越高越好。

旧报告中曾出现 `paragraph_merge_f1`，现在它是废弃别名。论文表格中应只使用：

```text
paragraph_boundary_f1
```

不要再同时报告 `paragraph_merge_f1`，否则会出现两个数完全一样、概念重复的问题。

### 3.6 `paragraph_text_coverage_f1`

衡量正文文本内容是否覆盖到，而不是严格要求块边界完全一致。

它使用 many-to-one / one-to-many 滑动窗口匹配。比如 gold 是一个大段，而生成结果拆成两个连续小段，只要文本覆盖完整，该指标仍然可以给较高分。

它回答：

```text
“内容有没有被覆盖？”
```

而 `paragraph_boundary_f1` 回答：

```text
“段落边界切得像不像？”
```

### 3.7 `section_attachment_body_no_float_f1`

只在正文类块上评估它们是否挂到了正确章节下。

评估范围包括：

```text
ordinary paragraph
list item
display formula
ordinary body text block
algorithm body（视当前 IR 类型而定）
```

排除：

```text
figure
table
caption
footnote
reference item
front matter
appendix/references 中不适合作为正文 attachment 的部分
```

为什么要使用这个指标：原始 `section_attachment_f1` 会把 float、caption、references、appendix、front matter 等混进去，容易惩罚 PDF 重建系统。PDF 中图表位置由 LaTeX float 机制决定，视觉位置和源码位置天然可能不一致。

因此论文中应优先报告：

```text
section_attachment_body_no_float_f1
```

而不是把 raw `section_attachment_f1` 当主指标。

### 3.8 `reference_section_completeness`

衡量参考文献部分是否完整。

主要看：

```text
1. References / Bibliography 区域是否识别出来
2. reference item 是否丢失
3. 是否重复
4. 是否混入正文
5. 是否保持大体顺序和条目完整性
```

越高越好。当前我们的方法在该指标上明显优于 Nougat，这是论文中可以强调的优势点。

### 3.9 `float_caption_attachment_accuracy`

衡量图、表、算法等浮动体是否和正确 caption 配成一组。

它不只是看 caption 文本有没有出现，而是看：

```text
1. Figure/Table/Algorithm caption 是否被识别为 caption
2. caption 是否挂到正确的 figure/table/algorithm
3. 组图、多图块、跨栏图表是否能够合成一个 float group
4. caption 是否没有被当成普通正文乱插
```

越高越好。

当前 hard20：

```text
ours = 0.7246
Nougat = 0.5878
delta = +0.1368
```

说明当前 float/caption grouping 在这组难例中有优势，但仍需继续关注组图合并、caption 去重、跨栏 float slot 等问题。

### 3.10 `layout_similarity`

衡量生成 PDF 与原始 PDF 的版式相似程度。

该指标主要对我们的方法有意义，因为 Nougat 默认输出 Markdown / Markup，不直接追求 PDF 版式复原。

它主要反映：

```text
1. 单栏/双栏模式是否合理
2. 页边距、正文宽度、图表槽位是否近似
3. 是否能保持大致页面结构
```

越高越好。它不是 OCR 指标，也不是源码 AST 指标。

### 3.11 `compile_success_rate`

衡量生成 LaTeX 是否能成功编译成 PDF。

当前 hard20：

```text
compile_success_rate = 0.9500
compiled = 19 / 20
```

这对我们的任务非常关键，因为我们的目标是：

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction
```

而不只是 Markdown 结构转录。

---

## 4. 最新 current ablation 结果

当前 clean rollup 使用：

```text
local_outputs/final_eval_20260518/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/ablation_summary.csv
```

共有：

```text
ablation runs = 12
```

按 `calibrated_test_positive_macro_f1_mean` 选出的最佳项：

```text
E01_no_gutter_overlap
positive macro F1 = 0.7609
MERGE F1 = 0.5812
PARENT F1 = 0.9405
tau_merge = 0.6000
tau_parent = 0.7500
```

这里的 `positive macro F1` 是：

```text
(MERGE F1 + PARENT_CHILD F1) / 2
```

它不把 NONE 类放进均值，避免 NONE 由于数量巨大而掩盖 MERGE/PARENT 的真实表现。

### 4.1 current GNN-only ablation 表

这张表只评估 **GNN 边关系预测模型**，不包含 generator、LaTeX 编译、版式渲染等后处理效果。  
为了避免把实验日志名直接写进论文，下面给出“论文可读名称 + 原始实验 ID”。最高值用 **加粗** 标出，当前主模型基准用 `Baseline` 标注。

| 分组 | 论文可读名称 | 原始 ID | MERGE P | MERGE R | MERGE F1 | PARENT F1 | Positive Macro F1 | tau_merge | tau_parent |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge feature | 去掉 gutter/overlap 特征 | `E01_no_gutter_overlap` | 0.5773 | 0.5853 | 0.5812 | 0.9405 | **0.7609** | 0.60 | 0.75 |
| Propagation | 去掉类型感知消息屏蔽 | `A02_no_type_aware_message_mask` | 0.5928 | 0.5300 | 0.5596 | **0.9536** | 0.7566 | 0.77 | 0.73 |
| Baseline | 当前主模型：Y-network + merge gate | `M06_current_main_merge_gate` | 0.5475 | 0.5576 | 0.5525 | 0.9444 | 0.7485 | 0.62 | 0.74 |
| Training | 不使用 OHEM | `T00_no_ohem` | 0.6121 | 0.6037 | **0.6079** | 0.8802 | 0.7441 | 0.39 | 0.74 |
| Edge feature | 加入高斯距离特征 | `M07_gaussian_edge_feature` | **0.6886** | 0.5300 | 0.5990 | 0.8722 | 0.7356 | 0.70 | 0.62 |
| Edge feature | 去掉标点探针 | `E00_no_punctuation` | 0.6441 | 0.5253 | 0.5787 | 0.8795 | 0.7291 | 0.47 | 0.78 |
| Architecture | 去掉 merge gate | `M05_no_merge_gate` | 0.6067 | 0.4194 | 0.4959 | 0.9393 | 0.7176 | 0.80 | 0.83 |
| Flow feature | 去掉 v7 reading-flow 修正 | `F02_no_v7_reading_flow` | 0.4974 | 0.4470 | 0.4709 | 0.9229 | 0.6969 | 0.83 | 0.78 |
| Architecture | 不做 GNN message passing | `A01_no_message_passing` | 0.5764 | **0.6083** | 0.5919 | 0.7884 | 0.6901 | 0.42 | 0.67 |
| Semantic feature | 去掉 SciBERT 语义向量 | `F00_no_scibert` | 0.4458 | 0.4931 | 0.4683 | 0.8791 | 0.6737 | 0.58 | 0.61 |
| Legacy baseline | 旧共享 GAT 基线 | `A00_old_shared_gat` | 0.5714 | 0.2765 | 0.3727 | 0.9461 | 0.6594 | 0.58 | 0.70 |
| Geometry feature | 去掉几何/版式特征 | `F01_no_geometry_layout` | 0.3032 | 0.3088 | 0.3059 | 0.6516 | 0.4788 | 0.74 | 0.54 |

表中应重点看三列：

```text
MERGE F1:
  段落/文本块缝合能力。它衡量模型能不能判断两个 bbox 是否属于同一逻辑段落。

PARENT F1:
  层级挂载能力。它衡量模型能不能判断标题、正文、公式、图表等之间的父子关系。

Positive Macro F1:
  (MERGE F1 + PARENT F1) / 2。它是当前 GNN 关系预测的主排序指标。
```

基准说明：

```text
Baseline:
  M06_current_main_merge_gate 是当前 clean schema 下的主模型基准。

Legacy baseline:
  A00_old_shared_gat 是旧共享 GAT 基线，用于证明当前 Y-network 结构确实优于早期设计。
```

### 4.2 current ablation 解读

```text
1. 几何/排版特征非常重要。
   F01_no_geometry_layout 的 positive macro F1 只有 0.4788，是最差项。

2. SciBERT 有贡献，但不是唯一核心。
   F00_no_scibert 降到 0.6737，说明语义向量帮助 MERGE/PARENT，
   但几何排版仍是主轴。

3. v7 reading flow 修正仍然重要。
   F02_no_v7_reading_flow 只有 0.6969，低于当前主模型和最佳项。

4. message passing 对 PARENT 有帮助，但会影响 MERGE。
   A01_no_message_passing 的 MERGE F1 不低，但 PARENT F1 掉到 0.7884。

5. 旧共享 GAT 明显不如当前 Y-network / merge-gate 家族。
   A00_old_shared_gat 的 MERGE F1 只有 0.3727。

6. 标点探针对 MERGE 很重要。
   E00_no_punctuation 的表现低于最佳项，说明 hyphen、终止标点等特征
   确实参与了段落合并判断。

7. 不是所有“去掉某个特征”的结果都一定下降。
   E01_no_gutter_overlap 在本轮 clean set 中最高，说明 gutter/overlap 特征在当前实现下可能带入噪声或过约束。
   这不等价于“跨栏沟壑信息无用”，而是说明该特征仍需要结合 E2E hard cases 做安全性验证。
```

---

## 5. 截图中旧 `Final Ablation Table` 的解释

截图表格对应 `docs/ablation_results_current.md` 中的旧基线表：

```text
tag = v7_registry_adapteraware_20260515_181724
documents = 1851
node_feature_dim = 832
edge_attr_dim = 22
```

旧基线表中的主要结果：

| 实验 | MERGE F1 | PARENT F1 | Positive Macro F1 |
| --- | ---: | ---: | ---: |
| `M06_y_network_plus_merge_gate` | 0.6625 | 0.9353 | 0.7989 |
| `M07_y_network_plus_gaussian_edge_feature` | 0.6331 | 0.9620 | 0.7976 |
| `M05_y_network_dual_head` | 0.6304 | 0.9534 | 0.7919 |

当时决策：

```text
Primary production/E2E model: M07_y_network_plus_gaussian_edge_feature
Best MERGE-only model: M06_y_network_plus_merge_gate
Balanced architecture baseline: M05_y_network_dual_head
```

为什么截图里 M06/M07 分数比 current clean ablation 高？

```text
截图表：registry-adapter baseline，旧 schema，edge_attr_dim=22
current clean rollup：floatproxy/current schema，edge_attr_dim=26，并结合后续 generator/eval 修正
```

所以不能直接把截图表和 current hard20/Nougat delta 混成一个实验。论文写作时应明确：

```text
1. locked baseline ablation：用于证明模型结构设计
2. current clean hard20/Nougat：用于展示当前端到端系统效果
```

---

## 6. 模型实验名解释

### `M06_y_network_plus_merge_gate` / `M06_current_main_merge_gate`

Y-network 双路径结构上加入 merge gate。目标是让 MERGE 分支更依赖局部原始特征，减少 GNN message passing 对段落合并边界的污染。

### `M07_y_network_plus_gaussian_edge_feature` / `M07_gaussian_edge_feature`

在边特征中加入高斯距离/接近度特征。目标是让模型知道两个节点的物理接近程度，帮助处理远距离幻觉连接和局部结构判断。

### `M05_y_network_dual_head`

Y-network 双头模型。MERGE 分支和 PARENT/NONE 分支分离，避免二者互相干扰。

### `A01_no_message_passing`

去掉 GNN message passing。用于验证消息传递是否真的有帮助。结论是：MERGE 可能不太依赖 message passing，但 PARENT_CHILD 明显需要上下文传播。

### `A02_no_type_aware_message_mask`

去掉 type-aware message mask。用于验证图、表、脚注、正文之间的信息传播屏蔽是否必要。

### `F00_no_scibert`

去掉 SciBERT 语义特征。用于验证语义向量是否有贡献。

### `F01_no_geometry_layout`

去掉几何/排版特征。用于验证 bbox、宽度、pseudo-y、列信息、字体字号等排版特征的重要性。

### `F02_no_v7_reading_flow`

去掉 v7 修正后的阅读流特征。用于验证我们对 MinerU 原始阅读顺序的修正是否有意义。

### `F03_raw_mineru_flow`

保留 reading-flow 相关特征，但使用 MinerU 原始顺序，而不是 v7 修正后的顺序。旧表中该项 MERGE F1 明显下降，说明 v7 reading-order repair 是真实贡献。

### `E00_no_punctuation`

去掉标点探针。例如：

```text
source 是否以句号/问号/感叹号结束
source 是否以 hyphen 结束
```

用于验证段落合并时标点线索是否重要。

### `E01_no_gutter_overlap`

去掉 gutter / overlap 类特征或相关干预后的一组对照。在 current clean rollup 中它成为 positive macro F1 最优项。这个结果提示：当前数据/阈值下，某些 gutter 特征可能带来噪声或过约束。但这不等于“永远不需要 gutter”，因为 generator 和 hard-case 视觉验收仍需关注跨栏误合并风险。

### `T00_no_ohem`

不使用 OHEM（Online Hard Example Mining）。用于验证困难负样本挖掘是否有帮助。

---

## 7. 评价口径注意事项

### 7.1 不再把 `paragraph_merge_f1` 当正式指标

旧结果中 `paragraph_merge_f1` 和 `paragraph_boundary_f1` 数值完全一致，是因为前者已经退化为后者别名。

正式写作时使用：

```text
paragraph_boundary_f1
paragraph_text_coverage_f1
```

不要使用：

```text
paragraph_merge_f1
```

### 7.2 `section_attachment_f1` 不是主指标

raw `section_attachment_f1` 会混入：

```text
float
caption
front matter
references
appendix
footnote
source AST placement effects
```

因此它只作为诊断项。正式比较时更推荐：

```text
section_attachment_body_no_float_f1
```

### 7.3 与 Nougat 对比时的任务边界

Nougat 是 markup-oriented scientific document transcription baseline，主要输出 Markdown/Markup。  
我们的方法目标是：

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction
```

因此对比应分为共享指标与我们专属指标。

共享指标：

```text
heading_tree_accuracy
reading_order_accuracy
paragraph_boundary_f1
paragraph_text_coverage_f1
reference_section_completeness
float_caption_attachment_accuracy
generated_structure_validity
```

我们专属或更适合我们的方法：

```text
compile_success_rate
layout_similarity
LaTeX 可编译性
figure/table crop fallback 的版式槽位还原
```

---

## 8. 论文写作可引用结论

英文表述：

```text
On the 20-document hard-case paired comparison, our method outperforms Nougat
on macro structure score, heading tree accuracy, paragraph boundary F1,
paragraph text coverage, reference section completeness, float-caption
attachment, and generated structure validity. Nougat is slightly better on
reading order and body-only section attachment. This suggests that our
layout-aware reconstruction pipeline is particularly effective for block
boundary preservation, references, compilable structure, and float/caption
organization, while heading-to-body section attachment remains a focused
future improvement target.
```

中文解释：

```text
在 20 篇 hard-case 配对比较中，我们的方法在宏观结构、标题树、段落边界、
正文覆盖、参考文献完整性、图表 caption 配对和结构合法性上优于 Nougat。
Nougat 在阅读顺序和正文到章节挂载上略优。说明我们的优势集中在版式感知的
块结构保留、references、可编译结构和 float/caption 组织；后续仍需继续优化
heading/body section attachment。
```

---

## 9. 其他仍需保留的辅助测试结果

这一节记录不是“当前 clean hard20 主表”的结果，但仍然对论文写作、失败分析和模型选择有参考价值。

### 9.1 Float-proxy M06 E2E smoke：20 篇

路径：

```text
data/09_eval_reports/e2e_v7_floatproxy_adapter_20260516_205926_M06_best_20_20260517_031023/
```

模型：

```text
M06_y_network_plus_merge_gate
tau_merge = 0.37
tau_parent = 0.45
```

结果：

| 指标 | 数值 |
| --- | ---: |
| `compile_success_rate` | 1.0000 |
| `macro_structure_score` | 0.7095 |
| `heading_tree_accuracy` | 0.5752 |
| `reading_order_accuracy` | 0.9560 |
| `paragraph_boundary_f1` | 0.5602 |
| `section_attachment_f1` | 0.6366 |
| `reference_section_completeness` | 0.8908 |
| `float_caption_attachment_accuracy` | 0.3564 |
| `generated_structure_validity` | 0.9910 |
| `layout_similarity` | 0.8004 |

解释：

```text
1. 编译成功率很高，说明 generator 的基础安全性不错。
2. reading order 不差，但 heading / section / float-caption 明显偏低。
3. 这说明 float-proxy schema 可以跑通，但不能直接替代锁定的 registry-adapter 主路线。
```

### 9.2 Section-scope diagnostic：body-no-float 指标验证

该测试用于证明 raw `section_attachment_f1` 不适合作为主指标，因为它混入了 float/caption/front matter 等不应该参与正文挂载评估的内容。

结果：

| 系统 / 模式 | macro | heading | reading order | paragraph boundary | section all | section body-no-float | oracle heading-flow | references | float-caption | validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skeleton only，无 GNN parent 覆盖 | 0.7095 | 0.5752 | 0.9560 | 0.5602 | 0.6366 | 0.7412 | 0.7449 | 0.8908 | 0.3564 | 0.9910 |
| full M06 pipeline | 0.7095 | 0.5752 | 0.9560 | 0.5602 | 0.6366 | 0.7412 | 0.7449 | 0.8908 | 0.3564 | 0.9910 |
| Nougat 20 | 0.7286 | 0.6542 | 0.9875 | 0.5486 | 0.7438 | 0.8790 | 0.8289 | 0.7074 | 0.5051 | 0.9535 |

解释：

```text
1. skeleton only == full pipeline，说明这批样本里 GNN parent edge 没有明显偷走 section scope。
2. ours 的 section all 从 0.6366 到 body-no-float 0.7412 明显提升，证明 float/caption 不应混入正文 attachment 主指标。
3. oracle heading-flow 和 body-no-float 很接近，说明剩余 section 差距主要来自 heading detection/tree 和 reading-flow 对齐，而不是 GNN parent 覆盖。
```

### 9.3 100 篇 Nougat pilot：历史对比结果

该结果来自 2026-05-16 的 100 篇 pilot，不是最新 clean hard20，但适合说明大样本趋势。

当前较好的历史 decoder variant：

```text
model = M07_y_network_plus_gaussian_edge_feature
decoder = --heading-skeleton-mode stack
report = data/09_eval_reports/ours_vs_nougat_compare100_no20_filteredstack_m07_20260516/summary.json
documents = 100
compile_success = 100 / 100
```

| 指标 | 我们的方法 | Nougat | 差值 |
| --- | ---: | ---: | ---: |
| `macro_structure_score` | 0.7289 | 0.7455 | -0.0166 |
| `heading_tree_accuracy` | 0.6337 | 0.7315 | -0.0978 |
| `reading_order_accuracy` | 0.9565 | 0.9835 | -0.0271 |
| `paragraph_boundary_f1` | 0.6048 | 0.5777 | +0.0271 |
| `section_attachment_f1` | 0.6608 | 0.7403 | -0.0795 |
| `reference_section_completeness` | 0.8713 | 0.7256 | +0.1456 |
| `float_caption_attachment_accuracy` | 0.3798 | 0.5134 | -0.1336 |
| `generated_structure_validity` | 0.9952 | 0.9461 | +0.0491 |
| `layout_similarity` | 0.8104 | n/a | n/a |

解释：

```text
1. 100 篇 pilot 中，我们在 references、paragraph boundary、generated validity 上有优势。
2. Nougat 在 heading tree、reading order、section attachment、float-caption 上更强。
3. 最新 hard20 clean rollup 已经显示 float_caption_attachment 反超 Nougat，说明后续 generator / float grouping 修复是有效的，但还需要更大样本重新验证。
```

### 9.4 Merge risk audit

测试集：

```text
test documents = 185
accepted_merges = 174
risk_edges = 14
docs_with_risk = 12
risk_per_merge = 0.0805
long_distance_merges = 0
non_text_endpoint_merges = 0
crosses_float_merges = 14
```

解释：

```text
1. 当前模型在测试集上没有明显长距离 MERGE 泄漏。
2. 没有非文本端点 MERGE，说明类型屏蔽基本有效。
3. 剩余风险集中在 crosses_float_merges，即段落被 float 打断时的合并风险。
4. 因此 E2E hard cases 仍应持续包含 float-heavy 文档。
```

### 9.5 PARENT_CHILD 组成审计

目的：检查当前 `PARENT_CHILD` 类到底由什么关系组成，判断是否需要立刻改成多头任务。

locked registry-adapter baseline：

```text
docs = 1851
PARENT_CHILD edges = 193827
NONE edges = 5887048
MERGE edges = 1769
```

Top PARENT_CHILD families：

| family | count | ratio |
| --- | ---: | ---: |
| `heading_to_body` | 128795 | 0.6645 |
| `title/heading -> equation/display_math` | 16485 | 0.0851 |
| `heading_to_heading` | 15803 | 0.0815 |
| `title/heading -> figure/figure_caption` | 7887 | 0.0407 |
| `same_text` | 7853 | 0.0405 |
| `title/heading -> figure/chart` | 4990 | 0.0257 |
| `title/heading -> table/table_caption` | 4169 | 0.0215 |

float-proxy experimental set：

```text
docs = 1829
PARENT_CHILD edges = 190142
NONE edges = 5772940
MERGE edges = 1750
```

Top PARENT_CHILD families：

| family | count | ratio |
| --- | ---: | ---: |
| `heading_to_body` | 126819 | 0.6670 |
| `title/heading -> equation/display_math` | 15993 | 0.0841 |
| `heading_to_heading` | 15535 | 0.0817 |
| `title/heading -> figure/figure_caption` | 7947 | 0.0418 |
| `same_text` | 7716 | 0.0406 |
| `title/heading -> figure/chart` | 4669 | 0.0246 |
| `title/heading -> table/table_caption` | 4212 | 0.0222 |

解释：

```text
1. PARENT_CHILD 并不是主要由 caption/formula/list/table 这类微关系构成。
2. 大约 2/3 是 heading_to_body。
3. 因此现在不适合把主模型强行拆成多个稀疏 auxiliary heads。
4. 当前主线仍保留三分类：MERGE / PARENT_CHILD / NONE。
5. 多头任务可以作为未来 ablation/branch，但不替代当前主模型。
```

---

## 10. 当前最重要的结论

```text
1. 截图表是旧稳定基线 ablation，不是最新 clean rollup 的唯一结论。
2. 最新 hard20 对比中，我们整体 macro_structure_score 高于 Nougat。
3. 当前系统最大的优势是 references、paragraph boundary、generated validity、float/caption。
4. 当前弱项是 body section attachment 仍略低于 Nougat。
5. 正式论文中应使用 body-no-float section attachment，而不是 raw section_attachment_f1。
6. paragraph_merge_f1 已废弃，不应再独立报告。
```
