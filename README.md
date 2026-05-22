# PDF2LaTeX NN

**Last updated**: 2026-05-22

PDF2LaTeX NN is a structure-aware PDF-to-LaTeX pipeline for born-digital research papers. It does not treat PDF conversion as plain OCR. The current system extracts visual facts from PDF, derives graph relation labels from matching TeX source, trains a GNN to predict document relations, and reconstructs compilable LaTeX through a decoupled IR renderer.

## Current Deployment

The production path is v7-only:

```text
compiled PDF + matching TeX
  -> MinerU content_v2
  -> v7 reading/layout cleanup
  -> PyMuPDF style spans
  -> GNNViewAdapter float-proxy graph view
  -> SciBERT + geometry/style/sequence graph features
  -> TeX AST alignment labels
  -> GATv2/Y-Network edge-relation model
  -> TreeDecoder / RenderTreeIR
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

Old v3/v4/v5 preprocessing variants are no longer production inputs. They are historical experiments only.

The current checked-in code keeps two data/model tracks separate:

```text
locked baseline/results:
  v7_registry_adapteraware_20260515_181724
  edge_attr_dim=22
  existing M05/M07 checkpoints and reports stay untouched

current experimental rebuild:
  v7_floatproxy_adapter_20260516_205926
  edge_attr_dim=26
  figure/table/algorithm nodes enter GNN as caption/placeholder float proxies
```

Do not delete the locked baseline checkpoints or reports while evaluating the
new float-proxy path.

## Main Relation Task

The graph model predicts three directed edge classes:

```text
MERGE        = 0  physical continuation / paragraph stitching
PARENT_CHILD = 1  logical hierarchy / attachment
NONE         = 2  no structural relation
```

`SIBLING` is not a learned class. Sibling order is recovered from v7 reading order and renderer sorting.

## Active Interfaces

```text
content_v7_styles.json  complete PDF fact layer
GNNViewAdapter          filtered/proxied graph-visible view + v7 mapping
GraphInput.pt           node/edge tensors
GraphLabels             TeX-derived edge labels over the GNN view
PredictedRelations      GNN output probabilities
RenderTreeIR            decoder output bridged back to full v7 ids
StyleProfile            global/local layout profile
CitationResolution      citation/reference repair state
```

See [docs/frontend_backend_contract_v1.md](docs/frontend_backend_contract_v1.md).

## Key Scripts

```bash
# Continue building new production data from PDF + TeX sources
python scripts/pipeline/build_v7_dataset_staged.py ...

# Rebuild graph tensors and relabel existing v7 content
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh

# Train the current GATv2/Y-Network relation model
python scripts/pipeline/train_edge_gnn_full.py ...

# Generate ablation commands
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v7_adapteraware_20260514_2109.json \
  --output-sh data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh

# Batch visual QA / E2E inference with the current IR renderer
python scripts/pipeline/batch_visual_qa_inference.py --renderer ir ...
```

Current experimental rebuild/relabel command pattern:

```bash
TAG=v7_floatproxy_adapter_$(date +%Y%m%d_%H%M%S) \
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

Current paper-facing full evaluation suite:

```bash
# Run current ablation matrix, E2E generator QA, Nougat paired comparison,
# and final rollup report. Use --skip-* flags to reuse completed stages.
python scripts/pipeline/run_current_full_eval_suite.py

# Collect existing outputs only, without training or generation.
python scripts/pipeline/collect_current_eval_results.py
```

Current evaluation outputs are expected under:

```text
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current/
data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

## Current Docs

```text
docs/PROJECT_FILE_LAYOUT.md          local/AutoDL directory and artifact map
docs/PROJECT_ARCHITECTURE_FULL.md     complete architecture, logic, metrics, and code map
docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md paper-facing full project description
docs/PROJECT_SOURCE_OF_TRUTH.md      local / GitHub / AutoDL boundary
docs/PROJECT_OVERVIEW.md             architecture and implementation summary
docs/frontend_backend_contract_v1.md decoupled IR contracts
docs/feature_schema_v0.md            graph tensor feature contract
docs/ground_truth_labeling_v0.md     TeX-to-PDF truth-label generation
docs/ablation_plan_v2.md             current ablation protocol
docs/ablation_results_current.md     latest locked ablation results
docs/v7_training_and_monitoring.md   production data/training runbook
docs/interface_audit_2026_05_14.md   current interface audit and stale-path check
docs/LOCAL_CONFIGURATION.md          private local configuration notes
```

## Important Paths

Local project root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Large artifacts stay on AutoDL under:

```text
/root/autodl-tmp/pdf2latex_nn/data
```

The canonical folder contract is now:

```text
docs/PROJECT_FILE_LAYOUT.md
```

Do not commit datasets, checkpoints, generated PDFs, secrets, or AutoDL credentials.
