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
