# Ablation Plan

**Last updated**: 2026-05-14

This is the current ablation protocol for the v7 graph-relation model. The filename remains `ablation_plan_v2.md`, but the executable matrix is `configs/ablation_matrix_v3.json`.

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
configs/ablation_matrix_v3.json
```

Current expected manifest:

```text
data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
```

Generate commands:

```bash
python scripts/pipeline/prepare_ablation_suite.py
```

Generated AutoDL script:

```text
data/08_runs/run_ablation_matrix_v3.sh
```

## Experiment Families

```text
M00_full_ce_ohem          full current model
M01_no_message_passing   remove GAT propagation
M02_no_symmetry_terms    remove Hu-Hv and Hu*Hv directional terms
M03_shallow_predictor    replace deep head with a shallow predictor
M04_type_aware_message_mask
                          restrict GAT propagation with type/layout mask
M05_y_network_dual_head  MERGE bypasses GNN; PARENT/NONE use GAT states
M06_y_network_plus_merge_gate
                          M05 plus hard physical MERGE gate
M07_y_network_plus_gaussian_edge_feature
                          M05 plus runtime gaussian proximity edge feature
F00_no_scibert           zero semantic node features
F01_semantic_only        keep semantic features only
F02_no_reading_flow      remove scroll/sequence/column/flow cues
F03_raw_mineru_flow      keep flow features but derive them from MinerU order
E00_no_punctuation       remove terminal punctuation and hyphen probes
E01_no_gutter_overlap    remove overlap/gutter features
T00_no_ohem              train without online hard negative mining
```

## Latest Locked Results

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
M05_y_network_dual_head is the current main model.
M06_y_network_plus_merge_gate is retained as a conservative high-precision mode.
M07_y_network_plus_gaussian_edge_feature is retained as a hierarchy-stability variant, not the default.
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
nohup bash data/08_runs/run_ablation_matrix_v3.sh \
  > logs/ablation_matrix_v3_20260514.log 2>&1 &
```

The generated script sets tokenizer and BLAS thread guards to reduce CPU contention.
