# Ablation Plan

**Last updated**: 2026-05-14

This is the current ablation protocol for the v7 graph-relation model. The filename remains `ablation_plan_v2.md`, but the executable matrix for the adapter-aware run is `configs/ablation_matrix_v7_adapteraware_20260514_2109.json`.

## Controls

All ablations must share:

```text
same labeled manifest
same graph .pt files
same document-level split rule
same random seed unless explicitly repeated
same calibration grid
same metric definitions
```

Feature removal is runtime-only. Ablations zero node/edge feature groups during training/evaluation and never rewrite `.pt` graph tensors.

## Current Matrix

```text
configs/ablation_matrix_v7_adapteraware_20260514_2109.json
```

Current expected manifest:

```text
data/00_manifests/v7_adapteraware_20260514_2109_clean_trainable.json
```

Generate commands:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v7_adapteraware_20260514_2109.json \
  --output-sh data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh \
  --output-json data/09_eval_reports/ablation_matrix_v7_adapteraware_20260514_2109_commands.json
```

Generated AutoDL script:

```text
data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh
```

## Experiment Families

```text
M05_current_y_network              current main model
A00_old_shared_gat                 old shared-head GAT baseline
A01_no_message_passing             remove GAT propagation
A02_no_type_aware_message_mask     let all candidate edges propagate
F00_no_scibert                     zero semantic node features
F01_no_geometry_layout             remove geometry/layout/style/flow signals
F02_no_v7_reading_flow             remove repaired v7 flow cues
E00_no_punctuation                 remove terminal punctuation and hyphen probes
E01_no_gutter_overlap              remove overlap/gutter features
M07_y_network_gaussian_edge_feature runtime gaussian proximity edge feature
```

## Current Adapter-Aware Relabel Snapshot

Run:

```text
data/09_eval_reports/train_v7_adapteraware_20260514_2109_m05
```

Clean dataset:

```text
documents: 1851
labels: MERGE=1769, PARENT_CHILD=193827, NONE=5887048
node_feature_dim: 832
edge_attr_dim: 22
candidate_recall_min: 0.9887
candidate_recall_median: 1.0000
orphan_ratio_median: 0.0909
orphan_ratio_max: 0.3000
```

Locked M05 smoke result on the adapter-aware set:

```text
best_epoch: 60
test_f1: 0.8557
test_positive_macro_f1: 0.7840
test_positive_macro_f0.5: 0.7915
test_merge_precision: 0.6250
test_merge_recall: 0.5700
test_merge_f0.5: 0.6130
```

## Previous Locked Results

Run:

```text
data/09_eval_reports/gnn_y_network_compare_20260514
```

Clean dataset:

```text
documents: 1857
train/val/test: 1486 / 186 / 185
labels: MERGE=1816, PARENT_CHILD=194300, NONE=6243086
```

| experiment | MERGE P | MERGE R | MERGE F1 | PARENT F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M00_full_ce_ohem | 0.2949 | 0.7151 | 0.4176 | 0.9459 | 0.6817 | 0.08 | 0.47 |
| M01_no_message_passing | 0.7305 | 0.5538 | 0.6300 | 0.8052 | 0.7176 | 0.47 | 0.37 |
| M04_type_aware_message_mask | 0.5000 | 0.4677 | 0.4833 | 0.9415 | 0.7124 | 0.33 | 0.39 |
| M05_y_network_dual_head | 0.6740 | 0.6559 | 0.6649 | 0.9412 | 0.8030 | 0.37 | 0.45 |
| M06_y_network_plus_merge_gate | 0.6923 | 0.6290 | 0.6592 | 0.9412 | 0.8002 | 0.41 | 0.49 |
| M07_y_network_plus_gaussian_edge_feature | 0.6879 | 0.5806 | 0.6297 | 0.9715 | 0.8006 | 0.43 | 0.72 |

Decision:

```text
The older epigraph run remains useful as a historical architecture comparison,
but new tables should be generated from the adapter-aware matrix above.
```

## Metrics

Primary:

```text
positive_macro_f0.5 = mean(MERGE_F0.5, PARENT_CHILD_F0.5)
```

Secondary:

```text
MERGE precision / recall / F1 / F0.5
PARENT_CHILD precision / recall / F1 / F0.5
threshold-calibrated test metrics
visual QA compile success
generated PDF structural sanity
```

Accuracy and NONE F1 are diagnostic only.

## Calibration

The matrix runs post-training calibration:

```text
tau_merge / tau_parent grid search
threshold_priority mode
merge physical gates enabled
precision floors from 0.55 to 0.90
```

Calibration is selected on validation data and then locked for test evaluation.

## Launch

Only launch after the labeled manifest exists:

```bash
cd /root/autodl-tmp/pdf2latex_nn
nohup bash data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh \
  > logs/ablation_matrix_v7_adapteraware_20260514_2109.log 2>&1 &
```

The generated script sets tokenizer and BLAS thread guards to reduce CPU contention.
