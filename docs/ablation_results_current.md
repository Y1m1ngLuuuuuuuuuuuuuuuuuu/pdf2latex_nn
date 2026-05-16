# Current Ablation Results

**Last updated**: 2026-05-16

This file records the current locked GNN ablation state after the
adapter-aware v7 relabel and final M07 E2E safety run.

## Active Dataset

```text
tag: v7_registry_adapteraware_20260515_181724
manifest: data/00_manifests/v7_registry_adapteraware_20260515_181724_labeled.json
documents: 1851
labels: MERGE=1769, PARENT_CHILD=193827, NONE=5887048
edge_attr_dim: 22
node_feature_dim: 832
```

Quality gate summary:

```text
candidate edge recall: min=0.9887, mean≈0.99999
orphan ratio: mean≈0.1117, max=0.3000
```

The dataset includes the full v7-to-GNN adapter mapping. Full v7 remains the
document truth layer for generation, while the GNN view filters or masks only
the nodes that would pollute graph training.

## Machine-Readable Reports

```text
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/summary.csv
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/summary.json
data/09_eval_reports/merge_risk_audit_m07_v7_registry_adapteraware_20260515_181724_test.json
```

Local copies:

```text
local_outputs/final_eval_20260516/summary.csv
local_outputs/final_eval_20260516/summary.json
local_outputs/final_eval_20260516/merge_risk_audit_m07_test.json
local_outputs/final_eval_20260516/e2e/
```

## Final Ablation Table

| experiment | MERGE P | MERGE R | MERGE F1 | PARENT P | PARENT R | PARENT F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M06_y_network_plus_merge_gate | 0.6688 | 0.6564 | 0.6625 | 0.9377 | 0.9329 | 0.9353 | 0.7989 | 0.37 | 0.35 |
| M07_y_network_plus_gaussian_edge_feature | 0.6114 | 0.6564 | 0.6331 | 0.9637 | 0.9602 | 0.9620 | 0.7976 | 0.44 | 0.45 |
| M05_y_network_dual_head | 0.5914 | 0.6748 | 0.6304 | 0.9548 | 0.9520 | 0.9534 | 0.7919 | 0.22 | 0.52 |
| M02_no_symmetry_terms | 0.6230 | 0.4663 | 0.5333 | 0.9744 | 0.9742 | 0.9743 | 0.7538 | 0.39 | 0.59 |
| F01_semantic_only | 0.5664 | 0.4969 | 0.5294 | 0.9389 | 0.9549 | 0.9468 | 0.7381 | 0.26 | 0.39 |
| T00_no_ohem | 0.4674 | 0.5276 | 0.4957 | 0.9665 | 0.9712 | 0.9688 | 0.7322 | 0.16 | 0.48 |
| M01_no_message_passing | 0.6731 | 0.6442 | 0.6583 | 0.7747 | 0.8259 | 0.7995 | 0.7289 | 0.43 | 0.34 |
| M00_full_ce_ohem | 0.4854 | 0.5092 | 0.4970 | 0.9574 | 0.9519 | 0.9547 | 0.7258 | 0.28 | 0.52 |
| M04_type_aware_message_mask | 0.5806 | 0.4417 | 0.5017 | 0.9449 | 0.9516 | 0.9482 | 0.7250 | 0.34 | 0.20 |
| E01_no_gutter_overlap | 0.4670 | 0.5215 | 0.4928 | 0.9519 | 0.9580 | 0.9550 | 0.7239 | 0.19 | 0.45 |
| M03_shallow_predictor | 0.4438 | 0.4847 | 0.4633 | 0.9754 | 0.9731 | 0.9742 | 0.7188 | 0.19 | 0.45 |
| F00_no_scibert | 0.5726 | 0.4356 | 0.4948 | 0.9255 | 0.9504 | 0.9378 | 0.7163 | 0.21 | 0.44 |
| E00_no_punctuation | 0.3433 | 0.6319 | 0.4449 | 0.9513 | 0.9436 | 0.9474 | 0.6962 | 0.24 | 0.53 |
| F03_raw_mineru_flow | 0.6835 | 0.2813 | 0.3985 | 0.9531 | 0.9631 | 0.9581 | 0.6783 | 0.37 | 0.39 |
| F02_no_reading_flow | 0.1807 | 0.1840 | 0.1824 | 0.9509 | 0.9506 | 0.9508 | 0.5666 | 0.08 | 0.52 |

## Decision

```text
Primary production/E2E model: M07_y_network_plus_gaussian_edge_feature
Best MERGE-only model: M06_y_network_plus_merge_gate
Balanced architecture baseline: M05_y_network_dual_head
```

M07 is promoted for the current E2E route because it keeps PARENT_CHILD highly
stable while preserving M05-level MERGE quality. M06 has slightly better MERGE
F1, but its weaker PARENT_CHILD score is riskier for tree reconstruction and
section/caption attachment.

The Raw-MinerU-Flow ablation is important evidence: keeping all features but
deriving flow/index/pseudo-y/bins from MinerU's original order drops MERGE F1 to
`0.3985`. Removing reading flow almost collapses MERGE to `0.1824`. This
supports the v7 reading-order repair as a real contribution rather than a
cosmetic preprocessing step.

## E2E Compile QA

M07 checkpoint:

```text
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/M07_y_network_plus_gaussian_edge_feature/seed_7/best_model.pth
tau_merge=0.44
tau_parent=0.45
```

Compiled outputs:

```text
hard cases: 10 / 10 compiled
test30: 30 / 30 compiled
```

Local E2E output folders:

```text
local_outputs/final_eval_20260516/e2e/m07_final_hardcases_v7_registry_adapteraware_20260515_181724_20260516_025808
local_outputs/final_eval_20260516/e2e/m07_final_test30_mathsafe_v7_registry_adapteraware_20260515_181724_20260516_031250
```

One random test case originally failed because MinerU produced malformed display
math OCR. The generator now detects obviously unsafe display math and falls back
to a safe escaped block, bringing test30 compile success to 30/30. This is a
generator safety fallback, not a GNN change.

## Merge Risk Audit

Audit split: test documents, limit 200 requested, 185 available.

```text
documents: 185
accepted_merges: 174
risk_edges: 14
docs_with_risk: 12
risk_per_merge: 0.0805
long_distance_merges: 0
non_text_endpoint_merges: 0
crosses_float_merges: 14
```

Interpretation: the current model no longer shows long-distance merge leakage
on the audited test split. The remaining risky accepted merges are float-cross
cases, so E2E visual QA should continue to include float-heavy papers.

## Interpretation

The ablation validates the current architectural diagnosis:

```text
message passing helps PARENT_CHILD
message passing can pollute MERGE
MERGE benefits from raw local edge-pair features
PARENT_CHILD benefits from propagated section/layout context
reading-flow repair is essential for MERGE recall
punctuation probes materially improve MERGE precision/recall balance
```

The current Y-network keeps these two signals separate:

```text
MERGE logit        = raw projected node features + edge_attr
PARENT/NONE logits = GAT-propagated node features + edge_attr
```

M08 Gaussian attention bias is not currently necessary. M07 already captures the
useful proximity signal as an edge feature without adding a custom attention
kernel.
