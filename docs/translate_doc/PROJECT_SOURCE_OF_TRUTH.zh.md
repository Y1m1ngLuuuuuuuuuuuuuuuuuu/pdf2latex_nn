# 项目事实源

**最后更新**：2026-05-18

本仓库是 v7 PDF-to-LaTeX 系统的源码控制位置。AutoDL 是数据集、MinerU 输出、图张量、checkpoint、生成 PDF 和长时间运行任务的运行时位置。

## 源码流

```text
local source edits -> GitHub -> AutoDL git pull / targeted sync
```

避免从本地向 AutoDL 做大范围递归覆盖。如果必须定向同步，只同步源码文件。运行时产物应留在远程。

## 根目录

本地：

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL：

```text
/root/autodl-tmp/pdf2latex_nn
```

GitHub：

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

## 生产流水线

生产流水线只使用 v7：

```text
compiled PDF + matching TeX
  -> MinerU content_v2
  -> content_v7 + style spans
  -> GNNViewAdapter
  -> graph.pt
  -> TeX AST alignment labels
  -> GATv2/Y-Network training / inference
  -> TreeDecoder
  -> RenderTreeIR
  -> OriginalLikeIRLatexRenderer
```

旧 v3/v4/v5 JSON 是历史实验。不要把它们喂给训练或评测。

完整 v7 JSON 是完整事实层。它不能因为 GNN 不直接使用某些节点，就删除或改写 metadata、figures、tables、footnotes、headers、captions 或 references。图可见视图由 `src/perception/gnn_view_adapter.py` 单独构建。

当前模型/数据轨道：

```text
locked baseline/results:
  tag: v7_registry_adapteraware_20260515_181724
  raw edge_attr_dim: 22
  保留所有 reports/checkpoints/generator outputs

active experimental rebuild:
  tag: v7_floatproxy_adapter_20260516_205926
  raw edge_attr_dim: 26
  float proxy + skip-over-float features
```

在新路径验证完成之前，不要删除之前的测试结果或权重。

## 活跃入口

从 PDF + TeX 生成新数据：

```text
scripts/pipeline/build_v7_dataset_staged.py
```

基于已有 v7 内容重建并重标注：

```text
scripts/pipeline/run_current_v7_rebuild_relabel.sh
scripts/pipeline/rebuild_graphs_from_manifest.py
scripts/pipeline/relabel_manifest.py
```

训练：

```text
scripts/pipeline/train_edge_gnn_full.py
```

Ablation：

```text
configs/ablation_matrix_v7_adapteraware_20260514_2109.json
scripts/pipeline/prepare_ablation_suite.py
data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh
```

E2E 推理和视觉 QA：

```text
scripts/pipeline/batch_visual_qa_inference.py --renderer ir
scripts/pipeline/run_e2e_inference.py --renderer ir
scripts/pipeline/step5_generate_tex.py --renderer ir
```

实验性 float-proxy rebuild/relabel：

```bash
TAG=v7_floatproxy_adapter_$(date +%Y%m%d_%H%M%S) \
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

`--renderer ir` 是当前 E2E 脚本暴露的唯一生产渲染面。legacy TreeDecoder renderer 只保留给历史单测和底层 helper 兼容；生产脚本不再接受 `--renderer tree`。

Decoder heading 模式：

```text
--heading-skeleton-mode legacy   基线 decoder 行为
--heading-skeleton-mode stack    当前生产候选：layout heading detector 提供候选/提示；
                                 确定性 stack 提供 outline prior 和 section-scope 安全约束；
                                 GNN parent edges 仍然进入 relation bridge，
                                 但受物理/heading 约束保护
--heading-skeleton-mode off      不使用 heading skeleton，仅用于回归/调试基线
```

当前 E2E 生成和 section-scope A/B 实验应使用 `stack`。它不需要重新跑 MinerU、不需要 rebuild graph、不需要 relabel、不需要重新训练。stack 模式会显式过滤 front-matter paper title、长数学/OCR 残片等错误 heading evidence，再构建大纲。

## 当前 Manifest 家族

锁定基线训练集和 checkpoint 家族：

```text
data/00_manifests/v7_registry_adapteraware_20260515_181724_labeled.json
data/06_graph_features/v7_registry_adapteraware_20260515_181724_labeled_graphs
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/
```

当前正在 rebuild/relabel 的实验性 float-proxy 集：

```text
data/00_manifests/v7_floatproxy_adapter_20260516_205926_rebuilt.json
data/00_manifests/v7_floatproxy_adapter_20260516_205926_labeled.json
data/06_graph_features/v7_floatproxy_adapter_20260516_205926_graphs
data/06_graph_features/v7_floatproxy_adapter_20260516_205926_labeled_graphs
```

当前面向论文的完整评测套件：

```text
configs/ablation_matrix_current.json
scripts/pipeline/run_current_full_eval_suite.py
scripts/pipeline/collect_current_eval_results.py

data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current/
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.json
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.csv
data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

`run_current_full_eval_suite.py` 是当前结果收集的顶层可复现实验命令。它可以跳过已经完成的阶段并复用输出。`collect_current_eval_results.py` 是只读汇总器，可以随时生成 pending 或 final 报告。

ablation matrix 文件名仍然包含 `20260514` 是为了复现实验；新实验必须显式传入 manifest 和 graph root。不要仅凭 matrix 文件名推断当前数据家族。

当前锁定关系模型方向：

```text
M05_current_y_network
```

M05 为 PARENT_CHILD 保留 type-aware GAT message passing，同时让 MERGE 绕过 message passing，直接使用 raw projected edge-pair features 预测 MERGE logit。hard MERGE gate 是当前主路径的一部分，不是额外的后处理补丁。

## 运行边界

应提交：

```text
source code
configs
docs
tests
lightweight manifests when useful
```

不要提交：

```text
PDF corpora
TeX corpora
MinerU outputs
graph .pt caches
model checkpoints
generated PDFs
secrets
AutoDL passwords
Kaggle tokens
```

## 当前维护文档

```text
README.md
docs/PROJECT_ARCHITECTURE_FULL.md
docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md
docs/PROJECT_SOURCE_OF_TRUTH.md
docs/PROJECT_OVERVIEW.md
docs/frontend_backend_contract_v1.md
docs/feature_schema_v0.md
docs/ground_truth_labeling_v0.md
docs/ablation_plan_v2.md
docs/ablation_results_current.md
docs/v7_training_and_monitoring.md
docs/interface_audit_2026_05_14.md
docs/LOCAL_CONFIGURATION.md
```

此列表之外的内容，要么是源码注释、生成报告，要么是 legacy reference material。
