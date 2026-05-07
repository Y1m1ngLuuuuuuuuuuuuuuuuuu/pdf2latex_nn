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

## Source Pool Backfill

`build_mini_dataset.py` scans `data/03_tex_source_pool` for TeX sources. If
compiled and PDF-matched sources exist in `data/03_tex_sources` but are missing
from the pool, backfill them without overwriting existing pool entries:

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

## Full GNN Training

After the batch manifest is written, start full training from the generated
labeled graph `.pt` files:

```bash
python scripts/pipeline/train_edge_gnn_full.py \
  --root data/06_graph_features_v7/full_train_dataset \
  --manifest data/00_manifests/v7_build1000_YYYYMMDD_HHMMSS.json \
  --output-dir data/09_eval_reports/full_train_v7_YYYYMMDD_HHMMSS \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-4 \
  --loss cross_entropy \
  --class-weights none
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
