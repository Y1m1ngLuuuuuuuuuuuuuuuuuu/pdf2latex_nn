# Current Ablation Results

**Last updated**: 2026-05-14

This file records the current locked GNN ablation state. Raw machine-readable results are stored in:

```text
data/09_eval_reports/gnn_y_network_compare_20260514/summary.csv
data/09_eval_reports/gnn_y_network_compare_20260514/summary.json
```

## Dataset

```text
manifest: data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
graph root: data/06_graph_features_v7_ablation_epigraph_20260514_0238
documents: 1857
split: document-level 1486 / 186 / 185
labels: MERGE=1816, PARENT_CHILD=194300, NONE=6243086
edge_attr_dim: 22
node_feature_dim: 832
```

The dataset includes `message_edge_mask` and `merge_candidate_mask` in graph tensors. Candidate MERGE gating was verified over all 1857 documents:

```text
total edges: 6,439,202
gate allowed: 529,465 (8.22%)
true MERGE: 1,816
true MERGE allowed: 1,816
MERGE gate oracle recall: 1.0000
```

## Locked Comparison

| experiment | MERGE precision | MERGE recall | MERGE F1 | PARENT_CHILD F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M00_full_ce_ohem | 0.2949 | 0.7151 | 0.4176 | 0.9459 | 0.6817 | 0.08 | 0.47 |
| M01_no_message_passing | 0.7305 | 0.5538 | 0.6300 | 0.8052 | 0.7176 | 0.47 | 0.37 |
| M04_type_aware_message_mask | 0.5000 | 0.4677 | 0.4833 | 0.9415 | 0.7124 | 0.33 | 0.39 |
| M05_y_network_dual_head | 0.6740 | 0.6559 | 0.6649 | 0.9412 | 0.8030 | 0.37 | 0.45 |
| M06_y_network_plus_merge_gate | 0.6923 | 0.6290 | 0.6592 | 0.9412 | 0.8002 | 0.41 | 0.49 |

## Decision

```text
Primary model: M05_y_network_dual_head
Conservative model: M06_y_network_plus_merge_gate
```

M05 is the current main architecture because it removes the old tradeoff: MERGE recovers beyond the no-message-passing model while PARENT_CHILD stays near the full GAT model. M06 is useful when downstream rendering needs stricter merge precision and can tolerate slightly lower merge recall.

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

The next optional experiment is Gaussian proximity as an additional edge feature, not as a direct PyG `edge_weight`, because the active model uses `GATv2Conv(edge_attr=...)`.
