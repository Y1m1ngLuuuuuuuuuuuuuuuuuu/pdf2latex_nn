# V7 Training And Monitoring

**Last updated**: 2026-05-14

This is the current operational runbook for v7 data production, relabeling, training, ablation, and monitoring.

## New Data Production

Use the staged builder for remaining unprocessed PDF/TeX samples:

```bash
python -u scripts/pipeline/build_v7_dataset_staged.py \
  --require-compiled-accepted \
  --target-total 2000 \
  --process-workers 2 \
  --preflight-workers 4 \
  --embedding-device cpu \
  --min-candidate-recall 1.0
```

The staged builder performs:

```text
candidate scan from compile accepted manifests
TeX preflight
MinerU batch extraction
content_v7 generation
PyMuPDF style enrichment
SciBERT graph build
TeX label generation
candidate-edge recall profiling
quality-gated manifest writing
```

## Rebuild / Relabel Existing V7 Content

When MinerU/v7 content already exists, do not rerun MinerU. Use:

```bash
TAG=v7_dupcont_crossref_20260513 \
INPUT_MANIFEST=data/00_manifests/<input_manifest>.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

Outputs:

```text
data/00_manifests/${TAG}_rebuilt.json
data/00_manifests/${TAG}_labeled.json
data/06_graph_features/${TAG}_graphs
data/06_graph_features/${TAG}_labeled_graphs
data/04_ground_truth_ir/${TAG}_mappings
logs/${TAG}_run.log
logs/${TAG}_delta.json
```

## Monitoring

Check current run:

```bash
tail -n 40 logs/<tag>_run.log
ps -p "$(cat logs/<tag>.pid)" -o pid,etime,pcpu,pmem,cmd
```

Health indicators:

```text
failed count near zero during rebuild
labeled manifest appears after relabel
candidate_edge_recall close to required threshold
effective orphan ratio below gate
label distribution not all NONE
```

## Training

Train on a clean labeled manifest:

```bash
python scripts/pipeline/train_edge_gnn_full.py \
  --root data/06_graph_features_v7_ablation_epigraph_20260514_0238 \
  --manifest data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
  --output-dir data/09_eval_reports/<run_name> \
  --epochs 30 \
  --batch-size 8 \
  --prediction-architecture y_network \
  --message-edge-mode all \
  --loss cross_entropy \
  --class-weights none \
  --ohem-negative-ratio 5.0 \
  --ohem-min-negatives 64 \
  --selection-metric val_positive_macro_f0_5
```

The split is document-level. Never use page-level random splitting.

Current locked default:

```text
M05_y_network_dual_head
MERGE branch: raw projected node-pair features, no message passing
PARENT/NONE branch: propagated GAT states
thresholds: tau_merge=0.37, tau_parent=0.45
```

Conservative mode:

```text
M06_y_network_plus_merge_gate
same architecture as M05, with hard MERGE physical gate
thresholds: tau_merge=0.41, tau_parent=0.49
```

## Ablation

Prepare scripts:

```bash
python scripts/pipeline/prepare_ablation_suite.py
```

Run after the labeled manifest exists:

```bash
nohup bash data/08_runs/run_ablation_matrix_v3.sh \
  > logs/ablation_matrix_v3_20260514.log 2>&1 &
```

## Visual QA

Use the current IR renderer:

```bash
python scripts/pipeline/batch_visual_qa_inference.py \
  --renderer ir \
  --manifest data/00_manifests/<test_manifest>.json \
  --checkpoint data/09_eval_reports/<run>/best_model.pth
```

Generated PDFs should be reviewed for:

```text
compile success
reading order
section hierarchy
paragraph continuation
formula rendering
figure/table placement
references and appendix handling
front matter / abstract / author block
```

## Concurrency Notes

Current safe defaults:

```text
SciBERT embedding_device=cpu during multiprocessing builds
process_workers=2 for graph/label workers
preflight_workers=4
OMP/MKL/OPENBLAS/NUMEXPR threads = 1
```

Raise parallelism only after a short monitored run.
