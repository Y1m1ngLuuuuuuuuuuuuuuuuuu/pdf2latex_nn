# Current Ablation Results

**Last updated**: 2026-05-14

This file records the current locked GNN ablation state. The old epigraph
ablation remains below as historical evidence; the active adapter-aware matrix is
now:

```text
configs/ablation_matrix_v7_adapteraware_20260514_2109.json
data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh
data/09_eval_reports/ablations_v7_adapteraware_20260514_2109
```

The latest single M05 training run on the adapter-aware relabel is stored in:

```text
data/09_eval_reports/train_v7_adapteraware_20260514_2109_m05
```

Previous raw machine-readable ablation results are stored in:

```text
data/09_eval_reports/gnn_y_network_compare_20260514/summary.csv
data/09_eval_reports/gnn_y_network_compare_20260514/summary.json
data/09_eval_reports/gnn_m07_gaussian_20260514/summary.csv
data/09_eval_reports/gnn_m07_gaussian_20260514/summary.json
```

## Active Dataset

```text
manifest: data/00_manifests/v7_adapteraware_20260514_2109_clean_trainable.json
graph root: data/06_graph_features/v7_adapteraware_20260514_2109_labeled_graphs
documents: 1851
labels: MERGE=1769, PARENT_CHILD=193827, NONE=5887048
edge_attr_dim: 22
node_feature_dim: 832
```

The dataset includes `message_edge_mask`, `merge_candidate_mask`, and the
GNN-view to full-v7 mapping sidecars. Mapping sidecars are excluded from PyG
batching and used only by inference/generation bridges.

Quality gate summary:

```text
candidate edge recall: min=0.9887, median=1.0000, mean≈1.0000
orphan ratio: median=0.0909, p90=0.2364, max=0.3000
```

## Current M05 Adapter-Aware Smoke Result

```text
best_epoch: 60
selection_metric: val_positive_macro_f0_5 = 0.8046
test_f1: 0.8557
test_positive_macro_f1: 0.7840
test_positive_macro_f0.5: 0.7915
test_merge_precision: 0.6250
test_merge_recall: 0.5700
test_merge_f0.5: 0.6130
```

This result is the sanity check that the new adapter-aware label set trains
cleanly. The full adapter-aware ablation matrix is queued under
`data/09_eval_reports/ablations_v7_adapteraware_20260514_2109`.

## Previous Locked Comparison

| experiment | MERGE precision | MERGE recall | MERGE F1 | PARENT_CHILD F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M00_full_ce_ohem | 0.2949 | 0.7151 | 0.4176 | 0.9459 | 0.6817 | 0.08 | 0.47 |
| M01_no_message_passing | 0.7305 | 0.5538 | 0.6300 | 0.8052 | 0.7176 | 0.47 | 0.37 |
| M04_type_aware_message_mask | 0.5000 | 0.4677 | 0.4833 | 0.9415 | 0.7124 | 0.33 | 0.39 |
| M05_y_network_dual_head | 0.6740 | 0.6559 | 0.6649 | 0.9412 | 0.8030 | 0.37 | 0.45 |
| M06_y_network_plus_merge_gate | 0.6923 | 0.6290 | 0.6592 | 0.9412 | 0.8002 | 0.41 | 0.49 |
| M07_y_network_plus_gaussian_edge_feature | 0.6879 | 0.5806 | 0.6297 | 0.9715 | 0.8006 | 0.43 | 0.72 |

## Decision

```text
Primary model family: M05 Y-network with type-aware propagation and hard MERGE gate
Historical conservative model: M06_y_network_plus_merge_gate
```

M05 remains the current main architecture because it removes the old tradeoff:
MERGE uses unpolluted local edge-pair features, while PARENT_CHILD still uses
propagated section/layout context.

M07 adds a runtime Gaussian proximity feature derived from `center_distance`. It improves PARENT_CHILD substantially (`0.9715`) but reduces MERGE recall enough that positive macro F1 stays slightly below M05. Keep M07 as evidence that proximity hints are useful for hierarchy, but do not promote it over M05 unless the downstream priority is PARENT_CHILD stability.

## Interpretation

The ablation validates the current architectural diagnosis:

```text
message passing helps PARENT_CHILD
message passing can pollute MERGE
MERGE benefits from raw local edge-pair features
PARENT_CHILD benefits from propagated section/layout context
```

The Y-Network keeps these two signals separate:

```text
MERGE logit        = raw projected node features + edge_attr
PARENT/NONE logits = GAT-propagated node features + edge_attr
```

The next optional experiment is M08: Gaussian proximity as an actual attention bias. It should only be attempted if we explicitly want to trade engineering complexity for a stronger hierarchy-focused message-passing prior.
