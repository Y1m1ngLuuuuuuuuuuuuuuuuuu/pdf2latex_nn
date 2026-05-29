# AutoDL pdf2latex_nn Directory Guide

生成时间：2026-05-26  
远端路径：`/root/autodl-tmp/pdf2latex_nn`  
本文件用途：说明 AutoDL 上 `pdf2latex_nn` 目录中各个文件夹和主要子文件夹的含义，方便后续迁移、同步、归档和清理。本文档只做解释，不代表已经删除或移动任何远端文件。

## 当前主线结论

当前生产主线已经固定为：

```text
MinerU middle.json
  -> v8 reflow / reading-order correction / style registry
  -> deterministic merge
  -> DocumentIR / RenderTreeIR
  -> LaTeX / PDF generation
```

GNN / learned merge 相关内容已不作为默认生产路径，只保留为历史实验和可追溯材料。后续同步远端代码时，应优先同步 v8 主线、renderer、evaluation 和 audit 工具；GNN 相关内容只在明确需要复现实验时再使用。

## 顶层目录

| 路径 | 状态 | 含义 |
|---|---|---|
| `.git/` | ACTIVE_KEEP | 远端 Git 仓库元数据。用于代码版本追踪。 |
| `.ipynb_checkpoints/` | DELETE_CANDIDATE_AFTER_CONFIRM | Jupyter 自动生成缓存。通常不影响项目，可确认后清理。 |
| `configs/` | ACTIVE_KEEP | Prompt、API baseline、外部评估配置。 |
| `data/` | ACTIVE_KEEP | 数据、MinerU 输出、TeX 源、PDF、graph、eval reports 的主目录。最大、最重要。 |
| `docs/` | ACTIVE_KEEP | 项目架构、接口契约、实验结论和 runbook。 |
| `logs/` | HISTORICAL_KEEP | 远端后台任务日志和 pid/runner 脚本。用于追溯下载、MinerU、v8 audit、GNN 旧实验。 |
| `models/` | ACTIVE_KEEP | 本地缓存模型，目前主要是 SciBERT。 |
| `scripts/` | ACTIVE_KEEP | pipeline 和 debug 脚本。 |
| `src/` | ACTIVE_KEEP | 生产代码主目录。 |
| `tests/` | ACTIVE_KEEP | 单测目录。 |
| `tools/` | ACTIVE_KEEP | 转换、评估、audit、MinerU v8、API baseline 等命令行工具。 |
| `tools_upload_tmp/` | DELETE_CANDIDATE_AFTER_CONFIRM | 旧同步临时目录。若无当前引用，可清理。 |
| `deterministic` / `full` / `v8` | DELETE_CANDIDATE_AFTER_CONFIRM | 顶层零字节文件，疑似早期 shell/backtick 误生成；不是当前主线目录。确认无引用后可删除。 |
| `README.md` | ACTIVE_KEEP | 项目入口说明。 |
| `requirements.txt` / `requirements_server.txt` | ACTIVE_KEEP | 本地/远端依赖说明。 |
| `.env.example` | ACTIVE_KEEP | 环境变量样例，不应包含 secret。 |
| `deploy_to_server.sh` / `upload_to_server.sh` | HISTORICAL_KEEP | 早期同步脚本；当前更推荐 Git 或 targeted sync。 |
| `verify_environment.py` | ACTIVE_KEEP | 环境检查脚本。 |
| `test_overfit.py` | HISTORICAL_KEEP | 早期实验脚本，不是当前主线入口。 |

## `configs/`

```text
configs/
  api_baselines/
  external_eval/
  prompts/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `configs/api_baselines/` | HISTORICAL_KEEP | API/VLM baseline 配置样例，例如 provider/model/input mode。当前不主动运行 API。 |
| `configs/external_eval/` | HISTORICAL_KEEP | 外部评估相关配置。 |
| `configs/prompts/` | HISTORICAL_KEEP | API/VLM prompt 模板，例如 full-document / multipage-to-LaTeX prompt。 |

## `src/`

```text
src/
  adapters/
  datasets/
  evaluation/
  generation/
    ir_renderers/
  ir/
  perception/
  pipeline/
  reasoning/
  _archive/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `src/adapters/` | ACTIVE_KEEP | 外部/前端数据适配层。包含 MinerU v7/v8 进入内部 IR 的适配逻辑。 |
| `src/datasets/` | HISTORICAL_KEEP | 旧 GNN/graph 数据集加载与训练数据接口。当前默认主线不依赖它，但保留用于复现实验。 |
| `src/evaluation/` | ACTIVE_KEEP | comparison structure、结构指标、source coverage / paragraph preservation 等评估核心。 |
| `src/generation/` | ACTIVE_KEEP | LaTeX 生成后端。包含 style profile、front matter、float/table/citation、renderer 主逻辑。 |
| `src/generation/ir_renderers/` | ACTIVE_KEEP | RenderTreeIR 各类节点的专门 renderer，例如 heading、figure、table、paragraph 等。 |
| `src/ir/` | ACTIVE_KEEP | DocumentIR / RenderTreeIR schema、序列化、校验。 |
| `src/perception/` | ACTIVE_KEEP | PDF/MinerU 观测层处理。当前 v8 主线重点是 `mineru_v8_reflow.py` 等 reading-order/style 相关逻辑。 |
| `src/pipeline/` | ACTIVE_KEEP | Pipeline contract / shared pipeline utilities。 |
| `src/reasoning/` | ACTIVE_KEEP | 结构推理、heading skeleton、postprocess、旧 GNN label/decoder 相关逻辑。当前生产默认不以 GNN 为核心。 |
| `src/_archive/v8_gnn_merge_experiments_20260526/` | HISTORICAL_KEEP | 已封存的 v8 GNN/learned merge 实验代码，避免污染当前主线。 |

### `src/generation/` 重点

| 文件/子目录 | 含义 |
|---|---|
| `ir_renderer.py` | 当前 LaTeX 生成入口之一，消费 RenderTreeIR，而不是 GNN view。 |
| `ir_renderers/` | 不同 block/float/table/heading 的渲染实现。 |
| `style_profile.py` | 页面尺寸、字号、段落间距、单双栏等 style profile 逻辑。 |
| `front_matter_extractor.py` 或相关模块 | front matter Phase 0：title/author/affiliation/email/abstract 的确定性粗粒度处理。 |
| `table_assets.py` / `source_float_layout.py` | 表格、图片、跨栏/浮动位置等输出辅助。 |

### `src/perception/` 重点

| 文件/子目录 | 含义 |
|---|---|
| `mineru_v8_reflow.py` | 当前 v8 核心：从 MinerU middle/content 信息重建阅读顺序、style registry、logical owner 和 deterministic merge 输入。 |
| `reading_order.py` | 通用阅读顺序逻辑。 |
| `style_spans.py` / `layout_probes.py` | 样式、bbox、版面特征抽取。 |
| `gnn_view_adapter.py` | 旧 GNN view 适配器。当前不作为生产渲染源。 |

### `src/reasoning/` 重点

| 文件/子目录 | 含义 |
|---|---|
| `heading_skeleton.py` | 标题栈和 section/subsection 层级推断，仍是结构主线的重要部分。 |
| `postprocess.py` | RenderTreeIR 前的结构后处理。 |
| `label_generator.py` / `graph_builder.py` / `gnn_model.py` | GNN/label/graph 旧路线。当前默认不启用，但保留可追溯。 |

## `scripts/`

```text
scripts/
  debug/
  pipeline/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `scripts/pipeline/` | ACTIVE_KEEP | 主 pipeline 脚本，包括批量推理、生成、训练/旧 graph 流程、结果收集。当前使用时应优先确认是否是 v8 主线路径。 |
| `scripts/debug/` | HISTORICAL_KEEP | 调试脚本。 |

注意：旧 GNN 训练脚本还在 `scripts/pipeline/` 中，但它们不代表当前默认生产路径。运行前需要显式确认实验目的和输出目录。

## `tools/`

```text
tools/
  api_baselines/
    providers/
  audit/
  comphrdoc/
  mineru_v8/
  _archive/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `tools/audit/` | ACTIVE_KEEP | 当前最常用的评估与诊断工具。包括 reading order、paragraph preservation、visible prose order、v8 batch refresh 等。 |
| `tools/mineru_v8/` | ACTIVE_KEEP | v8 从 MinerU 原始输出生成/检查/渲染的工具入口。当前主线相关。 |
| `tools/api_baselines/` | HISTORICAL_KEEP | API/VLM baseline pipeline。当前不主动调用真实 API，保留为后续 external baseline。 |
| `tools/api_baselines/providers/` | HISTORICAL_KEEP | API provider abstraction。 |
| `tools/comphrdoc/` | HISTORICAL_KEEP | CompHRDoc smoke/eval 相关工具。当前不再主动运行。 |
| `tools/_archive/v8_gnn_merge_experiments_20260526/` | HISTORICAL_KEEP | 已封存的 v8 GNN/learned merge 工具，避免和当前 deterministic v8 主线混用。 |

### 当前关键 audit 工具

| 工具 | 含义 |
|---|---|
| `tools/audit/check_paragraph_preservation_against_tex.py` | 对 generated.tex 与 source TeX 做 paragraph/source coverage、merge、order、visible prose metrics。 |
| `tools/audit/refresh_paragraph_order_audits.py` | 批量刷新已有 generated.tex 的 order/coverage 指标，不重新生成 PDF。 |
| `tools/audit/check_reading_order_against_tex.py` | 阅读顺序与 source 对齐检查。 |

## `docs/`

```text
docs/
  _archive/
  translate_doc/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `docs/PROJECT_SOURCE_OF_TRUTH.md` | ACTIVE_KEEP | 当前项目事实源，后续优先读它。 |
| `docs/PROJECT_ARCHITECTURE_FULL.md` | ACTIVE_KEEP | 完整架构文档。 |
| `docs/PROJECT_FILE_LAYOUT.md` | ACTIVE_KEEP | 本地/远端文件结构约定。 |
| `docs/V8_MAINLINE_RECONSTRUCTION_PATH.md` | ACTIVE_KEEP | 当前 v8 deterministic 主线说明。 |
| `docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md` | ACTIVE_KEEP | v8 middle reflow、style detector、阅读顺序和版面逻辑。 |
| `docs/MINERU_ADAPTER_CONTRACT.md` | ACTIVE_KEEP | MinerU 接口契约。后续 MinerU 更新但接口不变时，应以此为替换边界。 |
| `docs/TABLE_ENGINE_CONTRACT.md` | ACTIVE_KEEP | 表格引擎替换接口。 |
| `docs/STYLE_TEMPLATE_CONTRACT.md` | ACTIVE_KEEP | sci/IEEE 等风格模板接口预留。 |
| `docs/_archive/` | HISTORICAL_KEEP | 已归档的旧路线/旧报告文档。 |
| `docs/translate_doc/` | HISTORICAL_KEEP | 翻译/说明材料。 |

## `tests/`

| 路径 | 状态 | 含义 |
|---|---|---|
| `tests/` | ACTIVE_KEEP | 单测目录。当前本地和远端应尽量保持一致。未来同步后建议优先跑 v8、renderer、evaluation、front matter、style detector 相关测试。 |

## `models/`

```text
models/
  huggingface/
    allenai/
      scibert_scivocab_uncased/
```

| 路径 | 状态 | 含义 |
|---|---|---|
| `models/huggingface/allenai/scibert_scivocab_uncased/` | ACTIVE_KEEP | SciBERT 本地缓存。旧 GNN/embedding/audit 可能会复用；即使当前主线不依赖训练，也建议保留。 |

## `logs/`

`logs/` 是远端后台任务的日志集合，主要包含：

| 文件模式 | 含义 |
|---|---|
| `arxiv2025_compilable_tex8000_*.log/.pid/.sh` | arXiv TeX 下载/编译筛选任务。 |
| `arxiv2025_tex8000_mineru_only_20260523*.log/.pid/.sh` | 8000 TeX 编译 PDF 后的 MinerU 处理任务。 |
| `pilot500_v7_mineru_scibert_strict_20260522*.log/.pid/.sh` | pilot500 / selected200 前期材料处理任务。 |
| `v8_atomic_merge_selected200_*.log/.pid/.sh` | 已归档 GNN/atomic merge 实验日志。 |
| `v8_gnn_rerun_selected200_20260526*.log/.pid` | 已关闭的 v8 GNN rerun。 |
| `v8_ordered_coverage_refresh_20260526*.log/.pid` | ordered coverage 指标刷新。 |
| `v8_visible_prose_order_refresh_20260526.log` | 最新 visible prose order 指标刷新日志。 |

这些日志不属于生产输入，但对复盘很有价值。清理前应先确认对应报告已经存在且不再需要复现。

## `data/`

远端 `data/` 是主体数据目录。当前大小分布约为：

| 路径 | 约大小 | 状态 | 含义 |
|---|---:|---|---|
| `data/02_mineru_outputs/` | 121G | ACTIVE_KEEP | MinerU 原始输出。v8 主线的重要输入。 |
| `data/09_eval_reports/` | 108G | ACTIVE_KEEP + HISTORICAL_KEEP | 所有评估、实验、生成输出、报告。 |
| `data/03_tex_sources/` | 40G | ACTIVE_KEEP | 编译通过的 TeX 源码目录，一篇一个 arXiv ID 子目录。 |
| `data/01_raw_pdfs/` | 30G | ACTIVE_KEEP | 由 TeX 编译出的 PDF，供 MinerU 和视觉评估使用。 |
| `data/07_v8_atomic_merge/` | 23G | HISTORICAL_KEEP / ARCHIVE_CANDIDATE | v8 atomic/GNN merge 旧实验数据。当前不再是生产主线。 |
| `data/06_graph_features/` | 3.0G | HISTORICAL_KEEP | 旧 GNN graph/labeled graph。 |
| `data/08_training_cache/` | 1.3G | HISTORICAL_KEEP | 训练/embedding 缓存。 |
| `data/_tmp_compilable_arxiv_build/` | 868M | DELETE_CANDIDATE_AFTER_CONFIRM | TeX 编译筛选临时构建残留。 |
| `data/_tmp_v7_staged_builder/` | 44M | DELETE_CANDIDATE_AFTER_CONFIRM | 旧 staged builder 临时目录。 |
| `data/_archive/` | 4.5G | HISTORICAL_KEEP | 已归档历史数据。 |
| `data/00_manifests/` | 93M | ACTIVE_KEEP | 所有 manifest，总表、选样、实验输入列表。 |
| `data/04_ground_truth_ir/` | 70M | HISTORICAL_KEEP | TeX-derived gold/alignment/label 映射。 |
| `data/05_observed_ir/` | 0 | PLACEHOLDER | 当前为空，预留 observed IR。 |
| `data/07_predicted_ir/` | 0 | PLACEHOLDER | 当前为空，预留 predicted IR。 |
| `data/08_output_latex/` | 0 | PLACEHOLDER | 当前为空，旧输出占位。 |

### `data/00_manifests/`

用途：保存各种数据集、选样、实验输入输出 manifest。

常见内容：

```text
data/00_manifests/
  *.json
  .ipynb_checkpoints/
```

含义：

- 8000 TeX/PDF/MinerU 总表。
- pilot500 / selected200 列表。
- v8 mainline/evaluation 输入列表。
- 旧 GNN/ablation 输入列表。

清理建议：manifest 是可复现实验的索引，不建议删除。

### `data/01_raw_pdfs/`

用途：保存由 TeX 源码编译得到的 PDF。不是从 arXiv 仓库直接下载的原始 PDF，而是项目自己从成功编译的 TeX 生成的 PDF。

含义：

- MinerU 输入。
- rendered/layout QA 的 gold PDF。
- 后续如果重新跑 MinerU，需要从这里读取 PDF。

清理建议：ACTIVE_KEEP。

### `data/02_mineru_outputs/`

```text
data/02_mineru_outputs/
  arxiv2025_tex8000_mineru_only_20260523/
  pilot500_v7_mineru_scibert_strict_20260522/
```

| 子目录 | 状态 | 含义 |
|---|---|---|
| `arxiv2025_tex8000_mineru_only_20260523/` | ACTIVE_KEEP | 当前 8000 规模 TeX 编译 PDF 的 MinerU 输出。v8 主线后续全量处理的基础。 |
| `pilot500_v7_mineru_scibert_strict_20260522/` | ACTIVE_KEEP / HISTORICAL_KEEP | 早期 pilot500，selected200 来自这里。仍用于对照和小实验复现。 |

MinerU 输出内部通常按 doc_id 或批次组织，含 `middle.json`、`content_list.json`、图片/表格/asset 等。当前 v8 主线以 `middle.json` 为 canonical text / reading-order 修复起点，`content_list` 只作为辅助 metadata/asset hint。

### `data/03_tex_sources/`

用途：保存编译通过的 TeX 源码。目录模式为：

```text
data/03_tex_sources/
  2501.00009/
  2501.00011/
  ...
  2502.xxxxx/
```

每个子目录是一篇论文的源码树，可能包含：

- `main.tex` 或入口 TeX；
- 图片资源；
- `.bib`；
- style/class 文件；
- 编译辅助文件。

清理建议：ACTIVE_KEEP。不要因为它看起来像 source pool 就误删，这是后续 gold/source coverage/evaluation 的基础。

### `data/04_ground_truth_ir/`

```text
data/04_ground_truth_ir/
  pilot500_v7_mineru_scibert_strict_20260522_mappings/
```

用途：TeX-derived alignment / mapping / label gold 中间层。当前 production 不使用 TeX source 做 inference，但 evaluation、label audit、旧 GNN 实验需要它。

清理建议：HISTORICAL_KEEP。

### `data/06_graph_features/`

```text
data/06_graph_features/
  pilot500_v7_mineru_scibert_strict_20260522_graphs/
  v8_atomic_merge_selected200_20260524_graphs/
  v8_atomic_merge_selected200_20260524_v11_graphs/
  v8_atomic_merge_selected200_20260525_v13_graphs/
  v8_atomic_merge_selected200_20260525_v14_graphs/
  v8_gnn_rerun_selected200_20260526_graphs/
```

用途：旧 GNN/atomic merge graph families。当前 v8 deterministic 主线不依赖这些 graph。

清理建议：不要删除。若空间紧张，可整体归档到历史盘或压缩，但应保留 family 名称与报告对应关系。

### `data/07_v8_atomic_merge/`

```text
data/07_v8_atomic_merge/
  v8_atomic_merge_selected200_20260524/
  v8_atomic_merge_selected200_20260524_v11/
  v8_atomic_merge_selected200_20260525_v13/
  v8_atomic_merge_selected200_20260525_v14/
  v8_gnn_rerun_selected200_20260526/
```

用途：v8 atomic/fragment GNN/learned merge 实验数据和输出。已经不作为默认生产路径。

状态：HISTORICAL_KEEP / ARCHIVE_CANDIDATE。

建议：保留直到论文/报告中不再需要复现 GNN 结论。若要整理，可迁入 `data/_archive/` 或外部盘，不建议直接删除。

### `data/08_training_cache/`

```text
data/08_training_cache/
  middlefrag_merge_training_20260523/
```

用途：旧 middle fragment / merge 训练缓存。当前不再是主线。

状态：HISTORICAL_KEEP。

### `data/09_eval_reports/`

这是最复杂的目录，包含当前结果和历史实验。主要子目录如下。

#### 当前主线结果

| 路径 | 状态 | 含义 |
|---|---|---|
| `v8_mainline_final_20260526/` | ACTIVE_KEEP | 当前 v8 deterministic 主线最终报告与总表。最重要。 |
| `v8_visible_prose_order_refresh_20260526/` | ACTIVE_KEEP | 最新 visible prose order metrics 刷新结果。当前评价口径使用这里。 |
| `v8_ordered_coverage_refresh_20260526/` | HISTORICAL_KEEP | ordered coverage 前一版刷新结果，已被 visible prose 指标扩展。 |
| `selected200_eval_rerun_v4_contentlist_merge_hint_20260526/` | ACTIVE_KEEP | v8 + contentlist merge hint 的 deterministic 对照。 |
| `selected200_eval_rerun_v3_same_evaluator_20260526/` | ACTIVE_KEEP | contentlist direct 与 v8 layout batch 使用同一 evaluator 的控制实验。 |

#### 数据准备与验收

| 路径 | 状态 | 含义 |
|---|---|---|
| `arxiv2025_compilable_tex8000_idscan_20260522/` | ACTIVE_KEEP | 8000 TeX 源下载/编译筛选记录。 |
| `mineru_acceptance_20260526/` | ACTIVE_KEEP | MinerU 输出验收统计。 |
| `pilot500_v7_mineru_scibert_strict_20260522/` | HISTORICAL_KEEP | pilot500/selected200 早期检查、merge audit、stage 文件。 |
| `selected200_eval_standard_20260525/` | HISTORICAL_KEEP | selected200 早期标准化链接/基准组织。 |

#### 已归档旧结果

| 路径 | 状态 | 含义 |
|---|---|---|
| `_archive/v8_gnn_closed_20260526/` | HISTORICAL_KEEP | 已关闭的 GNN/learned merge 路线最终归档。 |
| `_archive_selected200_pre_rerun_20260525/` | HISTORICAL_KEEP | selected200 rerun 前旧结果。 |
| `_archive_v8_gnn_pre_rerun_20260526/` | HISTORICAL_KEEP | v8 GNN rerun 前旧结果。 |

#### report 目录下的常见内部结构

| 子目录/文件模式 | 含义 |
|---|---|
| `generated_roots/` | 某一轮生成结果根目录集合。 |
| `logs/` | 该 report 自己的运行日志。 |
| `contentlist_direct/` | 直接从 MinerU content_list 硬编码生成的对照。 |
| `v8_layout_batch/` | v8 middle reflow + renderer 生成结果。 |
| `v8_contentlist_merge_hint/` | v8 上叠加 contentlist merge hint 的 deterministic 实验。 |
| `summary.csv/json` | 汇总指标。 |
| `*_REPORT.md` | 人类可读报告。 |

### `data/_archive/`

```text
data/_archive/
  legacy_non_v8_20260524/
```

用途：旧非 v8 主线数据归档。保留用于追溯，不参与当前默认流程。

### `data/_tmp_compilable_arxiv_build/`

用途：TeX 编译筛选过程中的临时构建目录，内部是若干 doc_id + 临时后缀目录。

状态：DELETE_CANDIDATE_AFTER_CONFIRM。

清理前确认：

1. `data/03_tex_sources/` 已保留成功源码；
2. `data/01_raw_pdfs/` 已保留编译 PDF；
3. `data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/` 已有日志/summary；
4. 当前没有下载/编译任务仍引用该目录。

### `data/_tmp_v7_staged_builder/`

用途：旧 v7 staged builder 临时目录。

状态：DELETE_CANDIDATE_AFTER_CONFIRM。

当前主线为 v8 middle reflow，不建议继续依赖这里。

## 当前推荐保留/归档策略

### ACTIVE_KEEP

这些目录是当前主线或后续必需输入，不应清理：

```text
src/
scripts/
tools/
configs/
docs/
tests/
models/huggingface/allenai/scibert_scivocab_uncased/
data/00_manifests/
data/01_raw_pdfs/
data/02_mineru_outputs/
data/03_tex_sources/
data/09_eval_reports/v8_mainline_final_20260526/
data/09_eval_reports/v8_visible_prose_order_refresh_20260526/
data/09_eval_reports/selected200_eval_rerun_v3_same_evaluator_20260526/
data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/
data/09_eval_reports/mineru_acceptance_20260526/
data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/
```

### HISTORICAL_KEEP

这些目录不在生产主线，但保留用于论文实验、审计、复现：

```text
data/04_ground_truth_ir/
data/06_graph_features/
data/07_v8_atomic_merge/
data/08_training_cache/
data/09_eval_reports/_archive/
data/09_eval_reports/_archive_selected200_pre_rerun_20260525/
data/09_eval_reports/_archive_v8_gnn_pre_rerun_20260526/
data/09_eval_reports/pilot500_v7_mineru_scibert_strict_20260522/
logs/
src/_archive/
tools/_archive/
docs/_archive/
```

### ARCHIVE_CANDIDATE

这些目录可以进一步整体打包或迁移到冷存储，但不建议直接删除：

```text
data/07_v8_atomic_merge/
data/06_graph_features/v8_atomic_merge_*
data/08_training_cache/middlefrag_merge_training_20260523/
data/09_eval_reports/_archive/v8_gnn_closed_20260526/
```

### DELETE_CANDIDATE_AFTER_CONFIRM

这些看起来是临时或误生成内容，但删除前仍应由用户确认：

```text
.ipynb_checkpoints/
tools_upload_tmp/
data/00_manifests/.ipynb_checkpoints/
data/09_eval_reports/*/.ipynb_checkpoints/
data/_tmp_compilable_arxiv_build/
data/_tmp_v7_staged_builder/
deterministic
full
v8
```

## 以后迁移到笔记本时建议保存什么

如果空间有限，最小可复现包应包含：

```text
代码与文档：
  .git/
  src/
  tools/
  scripts/
  configs/
  docs/
  tests/
  requirements*.txt
  README.md

当前主线数据索引：
  data/00_manifests/

当前主线报告：
  data/09_eval_reports/v8_mainline_final_20260526/
  data/09_eval_reports/v8_visible_prose_order_refresh_20260526/
  data/09_eval_reports/selected200_eval_rerun_v3_same_evaluator_20260526/
  data/09_eval_reports/selected200_eval_rerun_v4_contentlist_merge_hint_20260526/
```

如果要继续跑全流程，还需要：

```text
data/01_raw_pdfs/
data/02_mineru_outputs/
data/03_tex_sources/
models/huggingface/allenai/scibert_scivocab_uncased/
```

如果只看报告和写论文，不需要搬完整的 `data/02_mineru_outputs/`、`data/03_tex_sources/`、`data/07_v8_atomic_merge/`。

## 注意事项

1. 不要把 `data/02_mineru_outputs/` 当成可清理缓存，它是 v8 主线的原始观测层。
2. 不要误删 `data/03_tex_sources/`，它不是旧 source pool，而是当前 source coverage、evaluation 和 PDF 重编译的基础。
3. `data/07_v8_atomic_merge/` 和 `data/06_graph_features/` 现在不是生产主线，但仍是“为什么不以 GNN 为默认路径”的证据。
4. 清理任何 `_tmp_*` 前都应确认当前没有后台任务引用。
5. 后续同步代码建议使用 Git 或 targeted source sync，不要 broad rsync 整个 `data/`、`logs/`、`models/`。
