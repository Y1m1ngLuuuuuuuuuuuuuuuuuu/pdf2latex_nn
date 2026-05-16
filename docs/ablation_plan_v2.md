# Ablation Plan

**Last updated**: 2026-05-16

This is the current ablation protocol for the v7 graph-relation model. The
filename remains `ablation_plan_v2.md`, but the active locked results now come
from the registry/adapter-aware run tagged `v7_registry_adapteraware_20260515_181724`.

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

The executable matrix is maintained in the current project configs. Historical
matrix names from 2026-05-14 remain in the repo for reproducibility.

Current expected manifest:

```text
data/00_manifests/v7_registry_adapteraware_20260515_181724_labeled.json
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
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724
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

## Current Locked Results

Run:

```text
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724
```

| experiment | MERGE P | MERGE R | MERGE F1 | PARENT F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M06_y_network_plus_merge_gate | 0.6688 | 0.6564 | 0.6625 | 0.9353 | 0.7989 | 0.37 | 0.35 |
| M07_y_network_plus_gaussian_edge_feature | 0.6114 | 0.6564 | 0.6331 | 0.9620 | 0.7976 | 0.44 | 0.45 |
| M05_y_network_dual_head | 0.5914 | 0.6748 | 0.6304 | 0.9534 | 0.7919 | 0.22 | 0.52 |
| M01_no_message_passing | 0.6731 | 0.6442 | 0.6583 | 0.7995 | 0.7289 | 0.43 | 0.34 |
| F03_raw_mineru_flow | 0.6835 | 0.2813 | 0.3985 | 0.9581 | 0.6783 | 0.37 | 0.39 |
| F02_no_reading_flow | 0.1807 | 0.1840 | 0.1824 | 0.9508 | 0.5666 | 0.08 | 0.52 |

Decision:

```text
M07 is the current production/E2E checkpoint.
M06 remains the best MERGE-only result.
M05 remains the architectural baseline.
```

The full table is in `docs/ablation_results_current.md`.

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
