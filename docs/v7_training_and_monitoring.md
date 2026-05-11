# V7 Training And Monitoring

This document records the post-extraction steps for the v7 pipeline. These
steps consume existing batch outputs and do not modify the `.pt` graph schema or
the running batch-builder interface.

## Live Batch Monitoring

Use the watcher while `build_mini_dataset.py` is running in the background:

```bash
python tools/watch_v7_batch.py --current-dir logs --tail 10
```

The watcher reads:

- `logs/current_v7_build1000_log.txt`
- `logs/current_v7_build1000_errors.txt`
- `logs/current_v7_build1000_manifest.txt`

It reports success count, skip count, skip error types, pass rate, projected
success count, and rough ETA. It is read-only.

## PDF/TeX Pairing Contract

Production data must use a closed compile loop:

```text
arXiv TeX source -> latexmk compiled PDF -> MinerU -> TeX AST labels
```

The v7 builders now prefer compile `accepted.jsonl` manifests from
`step0_compile_arxiv_source_pool.py` or
`step0_build_compilable_arxiv_dataset.py`. Those records contain the exact
`pdf`, `source_dir`, and `main_tex` used to produce each sample, so the builder
does not silently pair an official PDF with a different TeX revision. When such
manifests exist under `data/09_eval_reports/*compile*/accepted.jsonl`, they are
auto-discovered. The fallback same-ID scan remains only for local smoke tests
and older one-off datasets.

To force strict compiled-only pairing:

```bash
python -u scripts/pipeline/build_v7_dataset_staged.py \
  --require-compiled-accepted \
  --compiled-accepted-manifest data/09_eval_reports/arxiv_2025_source_pool_round3_compile_fixed/accepted.jsonl \
  ...
```

Each produced manifest document now records `pdf_origin`, `compile_manifest`,
and `compile_status`. Production samples should have
`pdf_origin="compiled_from_tex"`.

## Source Pool Backfill

Legacy tools may still scan `data/03_tex_source_pool`. If compiled and
PDF-matched sources exist in `data/03_tex_sources` but are missing from that
pool, backfill them without overwriting existing pool entries:

```bash
python tools/sync_compiled_sources_to_pool.py \
  --raw-pdf-dir data/01_raw_pdfs \
  --compiled-source-dir data/03_tex_sources \
  --source-pool-dir data/03_tex_source_pool \
  --report-json data/09_eval_reports/source_pool_sync_YYYYMMDD_HHMMSS.json
```

The default copy mode is `hardlink`, so files appear in the pool without
duplicating storage when both directories are on the same filesystem. The tool
writes each document through a temporary directory and then renames it into
place, so interrupted runs do not leave half-synced samples. A running batch
builder will not see newly added samples because its candidate list is created
at startup; use the expanded pool for the next or supplement run.

## Staged Dataset Builder

For large runs, prefer the staged builder over the original single-document
loop:

```bash
python -u scripts/pipeline/build_v7_dataset_staged.py \
  --target 1000 \
  --require-compiled-accepted \
  --manifest-output data/00_manifests/v7_staged_1000_YYYYMMDD_HHMMSS.json \
  --error-log data/00_manifests/v7_staged_1000_YYYYMMDD_HHMMSS_errors.jsonl \
  --mineru-batch-size 16 \
  --mineru-batch-max-pages 160 \
  --preflight-workers 4 \
  --process-workers 2 \
  --max-orphan-ratio 0.30 \
  --max-unmapped-tex-ratio 0.60 \
  --max-isolated-node-ratio 0.90 \
  --min-candidate-recall 1.0
```

The staged builder keeps the final graph, label, manifest, and recall contracts
from `build_mini_dataset.py`, but changes scheduling for throughput:

- Candidate discovery prefers compile manifests and `data/03_tex_sources`, so
  PDF and TeX stay paired through the original compile record.
- TeX preflight rejects obvious parser-breaking sources before MinerU work.
- MinerU receives directories of staged PDFs, amortizing service/model startup
  over many documents. Batches are capped by both document count and estimated
  page count so a few long PDFs do not create a 900-page low-utilization batch.
- Existing MinerU/v7/graph/label artifacts are reused unless a force flag is
  supplied.
- Graph building and label quality gates run through a bounded worker pool.

Use `--dry-run` first to inspect candidate counts, already cached MinerU
outputs, and existing valid labeled graphs. If another script is currently
writing to the same MinerU output directory, do not start the staged MinerU
stage at the same time; use `--skip-mineru-stage` to process only already cached
outputs.

## Full GNN Training

After the batch manifest is written, first freeze a strict document-level
training manifest. For the current v7 run we keep only documents whose orphan
ratio is at most `0.30`, whose candidate edge recall is complete, and whose
graph contains at least one non-NONE relation:

```bash
python scripts/pipeline/filter_split_manifest.py \
  --input data/00_manifests/v7_reprocess_1776_scibert_cpu_YYYYMMDD_HHMMSS.json \
  --output data/00_manifests/v7_reprocess_1776_scibert_cpu_orphan030.json \
  --max-orphan-ratio 0.30 \
  --min-candidate-recall 1.0 \
  --min-non-none-edges 1 \
  --split-dir data/00_manifests/v7_reprocess_1776_scibert_cpu_orphan030_splits \
  --seed 7
```

The emitted train/val/test files are document-level splits. Do not split edges
or pages independently, because that leaks a paper's layout style across train
and validation.

Then start full training from the generated labeled graph `.pt` files:

```bash
python scripts/pipeline/train_edge_gnn_full.py \
  --root data/06_graph_features_v7/full_train_dataset \
  --manifest data/00_manifests/v7_reprocess_1776_scibert_cpu_orphan030.json \
  --output-dir data/09_eval_reports/full_train_v7_YYYYMMDD_HHMMSS \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-4 \
  --loss focal \
  --class-weights inverse \
  --positive-weight-multiplier 2.0 \
  --train-negative-dropout 0.80
```

Outputs:

- `split_summary.json`
- `history.json`
- `training_report.json`
- `best_model.pth`
- `last_model.pth`

The default checkpoint selection metric is `val_positive_macro_f1`, the mean F1
of MERGE and PARENT_CHILD. This keeps the focus on structural relations instead
of the dominant NONE class.

`--train-negative-dropout` is applied only inside the training loader. Validation
and test metrics are always computed on the original edge distribution.

## MERGE Label Contract

MERGE is a physical stitch relation, not a broad "same TeX object" relation.
The labeler now applies these gates before assigning `MERGE=0`:

- Both PDF nodes must map to the same TeX node.
- They must be adjacent fragments within that TeX node's aligned PDF span.
- Titles, page noise, document-root metadata, figures, tables, algorithms, and
  mixed-type text/equation pairs are never MERGE.
- List markers block backward merge into a previous item.
- References are not globally merged just because they live under the
  References section; reference structure is rendered from `reference_items`
  and section scope.

The residual global fallback is intentionally orphan-only. It recovers high
confidence text fragments displaced by floats/tables, then lets the same
adjacent-fragment MERGE gate decide stitch edges.
