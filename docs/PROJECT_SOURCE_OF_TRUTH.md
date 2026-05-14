# Project Source Of Truth

**Last updated**: 2026-05-14

This repository is the source-control home for the v7 PDF-to-LaTeX system. AutoDL is the runtime home for datasets, MinerU outputs, graph tensors, checkpoints, generated PDFs, and long-running jobs.

## Source Flow

```text
local source edits -> GitHub -> AutoDL git pull / targeted sync
```

Avoid broad recursive overwrites from local to AutoDL. If targeted sync is necessary, sync source files only. Runtime artifacts should stay remote.

## Roots

Local:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL:

```text
/root/autodl-tmp/pdf2latex_nn
```

GitHub:

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

## Production Pipeline

The production pipeline is v7-only:

```text
compiled PDF + matching TeX
  -> MinerU content_v2
  -> content_v7 + style spans
  -> graph.pt
  -> TeX AST alignment labels
  -> GATv2/Y-Network training / inference
  -> TreeDecoder
  -> RenderTreeIR
  -> OriginalLikeIRLatexRenderer
```

Old v3/v4/v5 JSON variants are historical experiments. Do not feed them into training or evaluation.

## Active Entrypoints

New data from PDF + TeX:

```text
scripts/pipeline/build_v7_dataset_staged.py
```

Rebuild and relabel existing v7 content:

```text
scripts/pipeline/run_current_v7_rebuild_relabel.sh
scripts/pipeline/rebuild_graphs_from_manifest.py
scripts/pipeline/relabel_manifest.py
```

Training:

```text
scripts/pipeline/train_edge_gnn_full.py
```

Ablation:

```text
configs/ablation_matrix_v3.json
scripts/pipeline/prepare_ablation_suite.py
data/08_runs/run_ablation_matrix_v3.sh
```

E2E inference and visual QA:

```text
scripts/pipeline/batch_visual_qa_inference.py --renderer ir
scripts/pipeline/run_e2e_inference.py --renderer ir
```

## Current Manifest Families

Current active trainable clean set:

```text
data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
data/06_graph_features_v7_ablation_epigraph_20260514_0238
```

Current ablation matrix expects:

```text
data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
```

Current locked relation model direction:

```text
M05_y_network_dual_head
```

M05 keeps GAT message passing for PARENT_CHILD while bypassing it for MERGE, using raw projected edge-pair features for the MERGE logit. M06 adds a hard MERGE gate and is retained as a high-precision conservative variant.

## Runtime Boundaries

Commit:

```text
source code
configs
docs
tests
lightweight manifests when useful
```

Do not commit:

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

## Current Maintained Docs

```text
README.md
PROJECT_TUTORIAL.md
docs/PROJECT_SOURCE_OF_TRUTH.md
docs/PROJECT_OVERVIEW.md
docs/frontend_backend_contract_v1.md
docs/feature_schema_v0.md
docs/ground_truth_labeling_v0.md
docs/ablation_plan_v2.md
docs/ablation_results_current.md
docs/v7_training_and_monitoring.md
docs/LOCAL_CONFIGURATION.md
```

Anything outside this list is either source-code comments, generated reports, or legacy reference material.
