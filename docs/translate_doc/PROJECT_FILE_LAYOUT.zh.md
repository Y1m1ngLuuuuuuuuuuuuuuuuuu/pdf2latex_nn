# 项目文件布局

**最后更新**：2026-05-22

这个文档是本地和 AutoDL 两边统一后的目录契约。相同路径在两边应该表示同一类内容。AutoDL 可以有完整大数据和运行产物，本地可以只有小子集，但语义不能变。

## 根目录

```text
local:  /Users/lu/Code/Project/pdf2latex_nn/test_4_19
AutoDL: /root/autodl-tmp/pdf2latex_nn
```

代码修改优先通过 GitHub 或定向源码同步。不要把本地目录大范围递归覆盖到 AutoDL；运行时产物应该留在远程。

## 顶层源码目录

| 路径 | 含义 | 是否进 Git | 备注 |
| --- | --- | ---: | --- |
| `src/` | 生产 Python 包。 | 是 | 核心流水线代码。 |
| `src/perception/` | PDF/v7 感知层、GNN 可见视图、阅读顺序、样式和标题探针。 | 是 | 不负责最终渲染。 |
| `src/adapters/` | 前端适配器，尤其 MinerU v7 到 `DocumentIR`。 | 是 | 可替换前端边界。 |
| `src/reasoning/` | graph builder、label、GNN、decoder/postprocess、heading/float 逻辑。 | 是 | GNN view 和 decoder 在这里。 |
| `src/ir/` | 中间表示 schema、序列化、校验器。 | 是 | 语义/布局 IR 契约。 |
| `src/generation/` | IR renderer、style profile、表格/图片/citation 渲染 helper。 | 是 | 生产渲染必须回到 full-v7-first。 |
| `src/evaluation/` | comparison structure 和指标实现。 | 是 | 论文指标逻辑。 |
| `src/datasets/` | 图训练 dataset wrapper。 | 是 | 不应写死机器路径。 |
| `src/pipeline/` | pipeline contract 和 v7 校验 helper。 | 是 | 包括 v7 contract。 |
| `scripts/pipeline/` | 数据、训练、E2E、ablation、汇总入口。 | 是 | AutoDL 任务从这里启动。 |
| `scripts/debug/` | 手动可视化/调试脚本。 | 是 | bbox、reading order 检查。 |
| `tools/` | 转换、评测、审计、外部 benchmark 工具。 | 是 | 默认输出到 `data/09_eval_reports/`。 |
| `tools/audit/` | label、MERGE/PARENT、heading/float/layout 诊断。 | 是 | 默认只读或写报告。 |
| `tools/api_baselines/` | API/VLM baseline 脚手架。 | 是 | 真实 API 调用必须显式开关。 |
| `tools/comphrdoc/` | CompHRDoc/HRDH 外部适配 smoke 工具。 | 是 | 不是主训练目标。 |
| `configs/` | ablation、prompt、外部评测、API baseline 配置。 | 是 | 不能放 secret。 |
| `tests/` | 单测和回归测试。 | 是 | 逻辑修改优先加合成测试。 |
| `docs/` | 当前契约、架构、runbook、历史结果说明。 | 是 | 本文档是目录说明总入口。 |
| `third_party/` | 第三方代码/数据占位。 | 部分 | 大型第三方数据留在 AutoDL。 |
| `_legacy_reference/` | 老代码/旧实验参考快照。 | 小文件可进 | 不是生产路径。 |

## 顶层运行产物目录

| 路径 | 含义 | Git 策略 | 清理策略 |
| --- | --- | --- | --- |
| `data/` | 统一数据和产物根目录。 | 大多忽略 | 当前和可追溯历史 run 要保留。 |
| `logs/` | AutoDL 长任务日志和 pid。 | 忽略 | 活跃任务不能删。 |
| `local_outputs/` | 本地检查/可视化输出。 | 忽略 | 本地用，可单独归档。 |
| `e2e_outputs/` | 较早本地 E2E 输出。 | 忽略 | 历史/本地，不是当前生产数据。 |
| `.venv*/` | 本地 Python 环境。 | 忽略 | 本地专用。 |
| `audit_bundle_*.zip` | 外部审计包。 | 通常忽略 | 只保留主动打包的版本。 |

## `data/` 目录契约

| 路径 | 含义 | 由谁产生 | 被谁消费 |
| --- | --- | --- | --- |
| `data/00_manifests/` | 数据/运行 manifest。 | 下载、build、relabel 脚本。 | 所有后续阶段。 |
| `data/01_raw_pdfs/` | MinerU 输入 PDF。2026-05-22 重建中，这是我们从 TeX 本地编译出的 PDF，不是下载的 arXiv 原始 PDF。 | `step0_build_compilable_arxiv_dataset.py` 或外部导入。 | MinerU / v7 builder / rendered-output 评测。 |
| `data/02_mineru_outputs/` | 原始 MinerU 输出和前端抽取产物。 | MinerU/v7 staged builder。 | v7 normalization、`DocumentIR`。 |
| `data/03_tex_sources/` | 编译通过的 TeX 源码树，每篇一个目录。 | arXiv source downloader/compiler。 | TeX AST、label、复现。 |
| `data/03_tex_source_pool/` | 源码池 staging/archive。 | 旧 ingestion/repass。 | 只有 manifest 引用时才使用。 |
| `data/04_ground_truth_ir/` | TeX-derived mapping、gold/comparison IR、label report。 | label/eval 转换。 | relabel、audit、evaluation。 |
| `data/05_observed_ir/` | 前端观测 IR 或 sidecar。 | v7/MinerU adapter、诊断导出器。 | generator/eval 诊断。 |
| `data/06_graph_features/` | 当前 graph/labeled graph 家族。 | graph builder / relabel。 | GNN 训练和推理。 |
| `data/06_graph_features_v7/` | 历史 v7 graph 家族。 | 旧 v7 rebuild。 | 仅历史对比，除非 manifest 明确指向。 |
| `data/06_graph_features_v5/` | 历史 v5 graph。 | 旧实验。 | 非生产。 |
| `data/06_graph_features_oracle/` | oracle/debug graph。 | 诊断。 | 非生产训练，除非明确说明。 |
| `data/07_predicted_ir/` | 模型预测 sidecar / predicted relation IR。 | inference/decoder。 | decoder/generator 审计。 |
| `data/08_output_latex/` | 受控生成的 TeX/PDF。 | E2E/generator。 | evaluation、视觉 QA。 |
| `data/08_runs/` | 批处理 shell 脚本和命令包。 | `prepare_ablation_suite.py` 等。 | AutoDL 执行。 |
| `data/09_eval_reports/` | 报告、汇总、评测输出、审计输出、部分 checkpoint。 | train/eval/audit。 | 论文、分析、后续决策。 |
| `data/10_checkpoints/` | 可选 checkpoint 镜像。 | training/manual curation。 | 推理或 resume。 |
| `data/external/` | 外部 benchmark 资料和预测。 | external bridge。 | 外部评测。 |
| `data/_tmp_*` | 临时工作目录。 | 长任务 builder。 | 确认无任务使用后才可清理。 |

## 当前 2026-05-22 新数据重建布局

```text
run_name: arxiv2025_compilable_tex8000_idscan_20260522

候选 id:
  data/00_manifests/arxiv_2025_idscan_candidates_360000.jsonl

编译通过的 TeX:
  data/03_tex_sources/{doc_id}/

本地编译出的 PDF:
  data/01_raw_pdfs/{doc_id}.pdf

运行状态:
  data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/
  logs/arxiv2025_compilable_tex8000_idscan_20260522.log
```

跑完后，应把 `accepted.jsonl` 冻结成正式 manifest，例如：

```text
data/00_manifests/arxiv2025_compilable_tex_pdf_clean8000_20260522.jsonl
```

每一行至少保留：

```json
{
  "doc_id": "2501.12345",
  "source_dir": "data/03_tex_sources/2501.12345",
  "main_tex": "main.tex",
  "pdf_path": "data/01_raw_pdfs/2501.12345.pdf"
}
```

## 模型/实验家族命名

完整数据/模型实验应该使用稳定 tag：

```text
<purpose>_<schema_or_adapter>_<YYYYMMDD_HHMMSS>
```

典型产物：

```text
data/00_manifests/${TAG}_rebuilt.json
data/00_manifests/${TAG}_labeled.json
data/00_manifests/${TAG}_trainable_recall98.json
data/06_graph_features/${TAG}_graphs/
data/06_graph_features/${TAG}_labeled_graphs/
data/04_ground_truth_ir/${TAG}_mappings/
data/09_eval_reports/${TAG}/
logs/${TAG}_run.log
```

不要混用 graph/checkpoint schema。尤其 `edge_attr_dim=22` 的 registry-adapter 家族和 `edge_attr_dim=26` 的 float-proxy 家族不兼容。

## 什么东西放哪里

| 内容 | 正确位置 | 不要放在 |
| --- | --- | --- |
| 编译通过源码树 | `data/03_tex_sources/{doc_id}/` | `data/09_eval_reports/`, `local_outputs/` |
| 编译得到的输入 PDF | `data/01_raw_pdfs/{doc_id}.pdf` | `data/09_eval_reports/` |
| 原始 MinerU 输出 | `data/02_mineru_outputs/<tag>/...` | `src/`, `docs/` |
| v7 content JSON | run tag 下、由 manifest 引用的位置 | 随机根目录 |
| graph tensor | `data/06_graph_features/${TAG}_graphs/` | `data/09_eval_reports/` |
| labeled graph tensor | `data/06_graph_features/${TAG}_labeled_graphs/` | 覆盖 unlabeled graph 目录 |
| checkpoint | `data/09_eval_reports/<run>/<model>/seed_*/best_model.pth` 或 `data/10_checkpoints/` | `src/` |
| 评测生成 TeX/PDF | `data/08_output_latex/` 或 `data/09_eval_reports/<run>/` | `data/01_raw_pdfs/` |
| 汇总报告 | `data/09_eval_reports/<run>/` | 随机顶层目录 |
| 临时缓存 | `data/_tmp_*` | 长期 manifest |

## 清理规则

删除 AutoDL 文件前，先分类：

```text
ACTIVE_KEEP                  当前数据、当前 run、当前 checkpoint
HISTORICAL_KEEP              论文/审计可追溯产物
ARCHIVE_CANDIDATE            旧的大型 run 输出
DELETE_CANDIDATE_AFTER_CONFIRM tmp/stage/cache/重复渲染 PDF
DO_NOT_TOUCH_RUNNING         活跃 run 日志/tmp/output
```

不能删：

```text
data/03_tex_sources/
data/01_raw_pdfs/
data/00_manifests/ 下的活跃 manifest
活跃 graph/checkpoint/eval run family
活跃或刚完成长任务日志
```

通常可以考虑清理：

```text
data/_tmp_*
过期 smoke 输出
重复本地渲染 PDF
没有任何 manifest 引用的旧 cache/stage 目录
```

## 文档分工

| 问题 | 当前入口 |
| --- | --- |
| 每个目录/文件家族做什么？ | `docs/PROJECT_FILE_LAYOUT.md` |
| 架构和模块职责？ | `docs/PROJECT_ARCHITECTURE_FULL.md` |
| 项目目标和指标哲学？ | `docs/layout_aware_reconstruction_target.md` |
| 本地/GitHub/AutoDL 边界？ | `docs/PROJECT_SOURCE_OF_TRUTH.md` |
| 数据、训练、评测怎么跑？ | `docs/v7_training_and_monitoring.md` |
| label 怎么生成？ | `docs/ground_truth_labeling_v0.md` |
| generator 怎么消费 v7/GNN/IR？ | `docs/generator_logic_audit_2026_05_17.md` |
| MinerU/table/style 如何替换？ | `docs/MINERU_ADAPTER_CONTRACT.md`, `docs/TABLE_ENGINE_CONTRACT.md`, `docs/STYLE_TEMPLATE_CONTRACT.md` |

