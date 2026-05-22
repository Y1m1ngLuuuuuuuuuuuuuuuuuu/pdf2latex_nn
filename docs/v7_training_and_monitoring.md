# V7 Training And Monitoring

**Last updated**: 2026-05-22

This is the current operational runbook for v7 data production, relabeling, training, ablation, and monitoring.

For the meaning of every repository/data directory, use:

```text
docs/PROJECT_FILE_LAYOUT.md
```

## New Data Production

### 2026-05-22 Fresh TeX-Source Rebuild

After the remote runtime reset, the current rebuild target is a fresh
compile-success TeX-source pool:

```text
run_name: arxiv2025_compilable_tex8000_idscan_20260522
target_successes: 8000
candidate source: deterministic 2025 arXiv id scan
candidate_manifest: data/00_manifests/arxiv_2025_idscan_candidates_360000.jsonl
```

Important: this run does **not** download arXiv-hosted original PDFs. It
downloads arXiv e-print TeX sources, compiles them locally, and keeps the TeX
source plus our compiled PDF.

Storage:

```text
data/03_tex_sources/{doc_id}/       accepted TeX source tree
data/01_raw_pdfs/{doc_id}.pdf       locally compiled PDF for MinerU
data/09_eval_reports/{run_name}/    progress, accepted/rejected logs, compile logs
logs/{run_name}.log                 long-running process log
```

Monitor:

```bash
cd /root/autodl-tmp/pdf2latex_nn

cat data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/progress.json

tail -f logs/arxiv2025_compilable_tex8000_idscan_20260522.log

ps -eo pid,etime,pcpu,pmem,cmd \
  | grep -E 'arxiv2025_compilable|step0_build_compilable' \
  | grep -v grep

wc -l data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/accepted.jsonl

find data/03_tex_sources -mindepth 1 -maxdepth 1 -type d | wc -l
find data/01_raw_pdfs -type f -name '*.pdf' | wc -l
```

Do not start MinerU/v7 processing directly from this source rebuild until a
small acceptance sanity check has confirmed that the compiled PDFs and source
directories are paired correctly.

### V7 Dataset Builder

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
TAG=<new_experiment_tag> \
INPUT_MANIFEST=data/00_manifests/<input_manifest>.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

Current float-proxy experiment:

```bash
TAG=v7_floatproxy_adapter_$(date +%Y%m%d_%H%M%S) \
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

This produces a new manifest/graph family and intentionally does not overwrite
the locked adapter-aware baseline results.

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
node_dim / edge_dim match the intended experiment
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

Historical locked default:

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

Current locked E2E model for generator experiments:

```text
M07_y_network_plus_gaussian_edge_feature
checkpoint: data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/M07_y_network_plus_gaussian_edge_feature/seed_7/best_model.pth
thresholds: tau_merge=0.44, tau_parent=0.45
```

The float-proxy graph schema changes edge attributes, so it requires a fresh
training run before it can replace or challenge the locked M07 baseline.

Float-proxy run status, 2026-05-17:

```text
tag: v7_floatproxy_adapter_20260516_205926
trainable manifest: data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json
trainable docs: 1829
edge_attr_dim: 26
best model by positive macro F1: M06_y_network_plus_merge_gate
checkpoint: data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926/M06_y_network_plus_merge_gate/seed_7/best_model.pth
thresholds: tau_merge=0.37, tau_parent=0.45
E2E smoke: 20 / 20 compiled
```

Do not mix checkpoints between the registry-adapter schema and the float-proxy
schema.  Their edge feature contracts differ.

## Ablation

Prepare scripts:

```bash
python scripts/pipeline/prepare_ablation_suite.py
```

Run after the labeled manifest exists:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v7_adapteraware_20260514_2109.json \
  --output-sh data/08_runs/run_ablation_matrix_<tag>.sh \
  --output-json data/09_eval_reports/ablation_matrix_<tag>_commands.json

nohup bash data/08_runs/run_ablation_matrix_<tag>.sh \
  > logs/ablation_matrix_<tag>.log 2>&1 &
```

When testing a new adapter/schema family, prepare a new ablation command file
instead of reusing historical M05/M07 checkpoints.

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
