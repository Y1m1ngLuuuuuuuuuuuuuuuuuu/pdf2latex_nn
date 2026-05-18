# Current Ablation Results

**Last updated**: 2026-05-18

This file records the current locked GNN ablation state, the float-proxy
experimental branch, and the latest paper-facing hard20/Nougat comparison.

The newest paper-facing evaluation collection is being run through:

```text
configs/ablation_matrix_current.json
scripts/pipeline/run_current_full_eval_suite.py
scripts/pipeline/collect_current_eval_results.py
```

Its outputs are collected under:

```text
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current/
data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

The rollup above is the current clean metrics copy.  It removes the deprecated
`paragraph_merge_f1` alias from paper-facing tables; use
`paragraph_boundary_f1` for strict paragraph-like block boundary fidelity and
`paragraph_text_coverage_f1` for content coverage under many-to-one /
one-to-many window matching.

## Current Clean Hard20 / Nougat Delta, 2026-05-18

Inputs:

```text
E2E:    data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
Nougat: data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
Rollup: data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
Local:  local_outputs/final_eval_20260518/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

| metric | ours | Nougat | delta |
| --- | ---: | ---: | ---: |
| generated_structure_validity | 0.9976 | 0.9542 | +0.0434 |
| macro_structure_score | 0.8614 | 0.7925 | +0.0689 |
| heading_tree_accuracy | 0.7540 | 0.6350 | +0.1190 |
| paragraph_boundary_f1 | 0.8125 | 0.6003 | +0.2122 |
| paragraph_text_coverage_f1 | 0.8798 | 0.8341 | +0.0457 |
| reading_order_accuracy | 0.9789 | 0.9873 | -0.0084 |
| reference_section_completeness | 0.9908 | 0.6985 | +0.2923 |
| float_caption_attachment_accuracy | 0.7246 | 0.5878 | +0.1368 |
| section_attachment_body_no_float_f1 | 0.9018 | 0.9280 | -0.0262 |

Notes:

```text
compile_success_rate = 0.95
paired_documents = 20
paragraph_merge_f1 is deprecated and intentionally absent from the clean rollup.
raw section_attachment_f1 remains diagnostic only because it mixes floats,
front matter, references, appendix, and source-AST placement effects.
```

## Active Dataset

```text
tag: v7_registry_adapteraware_20260515_181724
manifest: data/00_manifests/v7_registry_adapteraware_20260515_181724_labeled.json
documents: 1851
labels: MERGE=1769, PARENT_CHILD=193827, NONE=5887048
edge_attr_dim: 22
node_feature_dim: 832
```

This is the locked baseline/results family. It must be kept for rollback and
paper tables.

## Current Experimental Rebuild

```text
tag: v7_floatproxy_adapter_20260516_205926
status: rebuild/relabel + ablation complete; E2E smoke complete
input manifest: data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
rebuilt manifest: data/00_manifests/v7_floatproxy_adapter_20260516_205926_rebuilt.json
labeled manifest: data/00_manifests/v7_floatproxy_adapter_20260516_205926_labeled.json
trainable manifest: data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json
node_feature_dim: 832
edge_attr_dim: 26
```

Purpose:

```text
figure/table/algorithm become float proxies in the GNN view
caption/placeholder semantics replace raw table body or figure OCR
float-skip/intervening-float edge features test paragraph continuation across floats
old checkpoints are not reused for this schema
```

Quality gate summary:

```text
rebuilt: 1857 / 1857
labeled success: 1829
quality-gate failures: 28
candidate edge recall: min=0.99197, mean=0.99999, median=1.00000
orphan ratio: mean=0.1073, median=0.0877, p95=0.2623, max=0.3000
label distribution: MERGE=1750, PARENT_CHILD=190142, NONE=5772940
document split: train=1463, val=183, test=183
```

The dataset includes the full v7-to-GNN adapter mapping. Full v7 remains the
document truth layer for generation, while the GNN view filters or masks only
the nodes that would pollute graph training.

### Float-Proxy Ablation Results, 2026-05-17

Report files:

```text
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_summary.json
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_summary.csv
local_outputs/remote_reports/v7_floatproxy_adapter_20260516_205926/
```

| experiment | MERGE P | MERGE R | MERGE F1 | PARENT F1 | positive macro F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M06_y_network_plus_merge_gate | 0.7481 | 0.4654 | 0.5739 | 0.9325 | 0.7532 | 0.37 | 0.45 |
| M05_y_network_dual_head | 0.7090 | 0.4378 | 0.5413 | 0.9500 | 0.7456 | 0.50 | 0.50 |
| M07_y_network_plus_gaussian_edge_feature | 0.6984 | 0.4055 | 0.5131 | 0.9579 | 0.7355 | 0.44 | 0.62 |
| M00_full_ce_ohem | 0.5192 | 0.3733 | 0.4343 | 0.9603 | 0.6973 | 0.26 | 0.33 |
| M01_no_message_passing | 0.6557 | 0.5530 | 0.6000 | 0.7912 | 0.6956 | 0.19 | 0.43 |
| M04_type_aware_message_mask | 0.5760 | 0.3318 | 0.4211 | 0.9228 | 0.6719 | 0.30 | 0.32 |
| F00_no_scibert | 0.6374 | 0.2673 | 0.3766 | 0.9293 | 0.6530 | 0.36 | 0.46 |
| E00_no_punctuation | 0.4012 | 0.3088 | 0.3490 | 0.9306 | 0.6398 | 0.35 | 0.43 |
| F02_no_reading_flow | 0.3133 | 0.2166 | 0.2561 | 0.9539 | 0.6050 | 0.18 | 0.39 |

Interpretation:

```text
Best float-proxy model by positive macro F1: M06_y_network_plus_merge_gate
Best PARENT_CHILD among top models: M07_y_network_plus_gaussian_edge_feature
Best MERGE recall: M01_no_message_passing, but parent hierarchy collapses
```

The result preserves the earlier architectural conclusion: message passing is
useful for hierarchy, but can blur MERGE boundaries.  The Y-network variants
remain the right family.  In the float-proxy schema, M06 is the best candidate
for E2E smoke because it recovers MERGE precision without collapsing parent
F1.  M07 remains useful when parent stability is the primary objective.

### Float-Proxy M06 E2E Smoke, 20 Documents

Checkpoint and thresholds:

```text
checkpoint: data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926/M06_y_network_plus_merge_gate/seed_7/best_model.pth
tau_merge=0.37
tau_parent=0.45
output: data/09_eval_reports/e2e_v7_floatproxy_adapter_20260516_205926_M06_best_20_20260517_031023/
```

| metric | value |
| --- | ---: |
| compile_success_rate | 1.0000 |
| macro_structure_score | 0.7095 |
| heading_tree_accuracy | 0.5752 |
| reading_order_accuracy | 0.9560 |
| paragraph_boundary_f1 | 0.5602 |
| section_attachment_f1 | 0.6366 |
| reference_section_completeness | 0.8908 |
| float_caption_attachment_accuracy | 0.3564 |
| generated_structure_validity | 0.9910 |
| layout_similarity | 0.8004 |

Lowest-structure cases in this smoke run:

```text
2501.17281  macro=0.4597  heading=0.3051  section=0.3128  float_caption=0.3556
2502.06474  macro=0.4997  heading=0.3333  section=0.8182  float_caption=0.2500
2502.07416  macro=0.5547  heading=0.4000  section=0.2667  float_caption=0.0000
2501.06236  macro=0.6501  heading=0.1786  section=0.9474  float_caption=0.0000
2502.14099  macro=0.6697  heading=0.2692  section=0.6379  float_caption=0.4667
```

This confirms that the new schema compiles reliably and keeps reading order
strong.  The remaining quality bottlenecks are heading-tree reconstruction,
section attachment, and float/caption grouping.  Those should be improved in
postprocess/generator logic before treating the float-proxy path as a
production replacement for the locked registry-adapter M07 route.

### Section-Scope Diagnostic With Float-Masked Metric, 2026-05-17

The section attachment metric was extended with:

```text
section_attachment_body_no_float_f1:
  section attachment over body text, abstract, list items, display math,
  and algorithms only; figure/table/caption are excluded.

section_attachment_oracle_heading_flow_f1:
  diagnostic upper-bound style metric.  It ignores predicted parent edges,
  walks the predicted reading order, and uses matched gold headings as the
  active heading identity.

section_attachment_breakdown:
  separate body / float / references / appendix attachment scores.
```

Diagnostic run:

```text
tag: v7_floatproxy_adapter_20260516_205926
model: M06_y_network_plus_merge_gate
checkpoint: data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926/M06_y_network_plus_merge_gate/seed_7/best_model.pth
documents: 20 test split documents
B output: data/09_eval_reports/section_scope_diag_v7_floatproxy_adapter_20260516_205926_20260517_102359_B_skeleton_only/
C output: data/09_eval_reports/section_scope_diag_v7_floatproxy_adapter_20260516_205926_20260517_102359_C_full_pipeline/
Nougat output: data/09_eval_reports/nougat_compare20_masked_ckpt_v7_floatproxy_adapter_20260516_205926_20260517_103919/
local copy: local_outputs/remote_reports/section_scope_diag_v7_floatproxy_20260517/
```

| system / mode | macro | heading | reading order | paragraph boundary | section all | section body-no-float | oracle heading-flow | references | float-caption | validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B: skeleton only, no GNN parent overwrite | 0.7095 | 0.5752 | 0.9560 | 0.5602 | 0.6366 | 0.7412 | 0.7449 | 0.8908 | 0.3564 | 0.9910 |
| C: full M06 pipeline | 0.7095 | 0.5752 | 0.9560 | 0.5602 | 0.6366 | 0.7412 | 0.7449 | 0.8908 | 0.3564 | 0.9910 |
| Nougat 20 | 0.7286 | 0.6542 | 0.9875 | 0.5486 | 0.7438 | 0.8790 | 0.8289 | 0.7074 | 0.5051 | 0.9535 |

Interpretation:

```text
B == C:
  GNN parent edges are not currently stealing section scope in this sample.
  The low all-content section F1 is mainly heading/flow/float-evaluation related.

body-no-float improves ours:
  0.6366 -> 0.7412
  This confirms that figure/table/caption placement should not be mixed into
  the body section-attachment score.

oracle heading-flow is close to body-no-float:
  0.7449 vs 0.7412
  The remaining section gap is not mostly from GNN parent overwrite.  It is
  dominated by heading detection/tree quality and reading-flow alignment.

Nougat remains stronger on heading/section/float-caption in this 20-doc pilot,
while ours remains stronger on references and generated LaTeX validity.
```

Next work item: keep body-no-float as a reported metric and continue improving
deterministic heading evidence plus float/caption grouping.  Do not use the raw
all-content section attachment alone as the headline section metric, because it
conflates body attachment with float placement.

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

## Nougat Structure Comparison, 100-Document Pilot

The current comparison evaluates generated documents through the neutral
comparison-structure IR.  It is a structure/layout benchmark, not an OCR text
accuracy benchmark.

Current best decoder variant:

```text
model: M07_y_network_plus_gaussian_edge_feature
decoder: --heading-skeleton-mode stack
postprocess variant: layout-heading candidates + global stack scope + false-heading filters
report: data/09_eval_reports/ours_vs_nougat_compare100_no20_filteredstack_m07_20260516/summary.json
local copy: local_outputs/final_eval_20260516/nougat_compare/ours_vs_nougat_compare100_no20_filteredstack_m07_20260516/summary.json
documents: 100
compile success: 100 / 100
```

| metric | ours | Nougat | delta |
| --- | ---: | ---: | ---: |
| macro_structure_score | 0.7289 | 0.7455 | -0.0166 |
| heading_tree_accuracy | 0.6337 | 0.7315 | -0.0978 |
| reading_order_accuracy | 0.9565 | 0.9835 | -0.0271 |
| paragraph_boundary_f1 | 0.6048 | 0.5777 | +0.0271 |
| section_attachment_f1 | 0.6608 | 0.7403 | -0.0795 |
| reference_section_completeness | 0.8713 | 0.7256 | +0.1456 |
| float_caption_attachment_accuracy | 0.3798 | 0.5134 | -0.1336 |
| generated_structure_validity | 0.9952 | 0.9461 | +0.0491 |
| layout_similarity | 0.8104 | n/a | n/a |

Interpretation: our strongest comparative wins are references, paragraph
boundary, generated structure validity, and direct LaTeX/PDF compilability.  The
remaining large gap is outline/caption structure, especially heading tree and
float-caption attachment.  The stack decoder improved heading accuracy over
the prior stack run (`0.6222 -> 0.6337`) but does not yet close the Nougat
heading gap.

Strict active-scope check:

```text
postprocess variant: strict active heading scope + font-size gate + numbering override
report: data/09_eval_reports/ours_vs_nougat_compare100_no20_activestack_m07_20260516/summary.json
local copy: local_outputs/final_eval_20260516/nougat_compare/ours_vs_nougat_compare100_no20_activestack_m07_20260516/summary.json
documents: 100
compile success: 100 / 100
```

| metric | strict active-stack ours | Nougat | delta |
| --- | ---: | ---: | ---: |
| macro_structure_score | 0.7178 | 0.7455 | -0.0276 |
| heading_tree_accuracy | 0.5581 | 0.7315 | -0.1734 |
| reading_order_accuracy | 0.9587 | 0.9835 | -0.0249 |
| paragraph_boundary_f1 | 0.6046 | 0.5777 | +0.0269 |
| section_attachment_f1 | 0.6633 | 0.7403 | -0.0769 |
| reference_section_completeness | 0.8713 | 0.7256 | +0.1456 |
| float_caption_attachment_accuracy | 0.3732 | 0.5134 | -0.1403 |
| generated_structure_validity | 0.9959 | 0.9461 | +0.0498 |
| layout_similarity | 0.8101 | n/a | n/a |

Interpretation: the strict implementation follows the intended active-scope
contract more faithfully: once a body node is attached by reading flow to the
current heading, GNN parent edges cannot steal it. It also requires unnumbered
headings to pass a document-local font-size step (`>= 1.15x` body) while strong
numbering patterns can still promote headings. This slightly improves section
attachment over the filtered stack run (`0.6608 -> 0.6633`), but the current
font gate is too conservative for heading-tree recall (`0.6337 -> 0.5581`).
The next heading-skeleton improvement should learn per-document heading style
clusters more aggressively rather than lowering the strict active-scope
contract.

Scored heading evidence + float-caption matcher check:

```text
postprocess variant: active heading scope + multi-dimensional heading evidence score + caption/float candidate matching
report: data/09_eval_reports/ours_vs_nougat_compare100_no20_scoredstack_m07_20260516/summary.json
local copy: local_outputs/final_eval_20260516/nougat_compare/ours_vs_nougat_compare100_no20_scoredstack_m07_20260516/summary.json
documents: 100
compile success: 100 / 100
```

| metric | scored-stack ours | Nougat | delta |
| --- | ---: | ---: | ---: |
| macro_structure_score | 0.7210 | 0.7455 | -0.0244 |
| heading_tree_accuracy | 0.5778 | 0.7315 | -0.1536 |
| reading_order_accuracy | 0.9588 | 0.9835 | -0.0248 |
| paragraph_boundary_f1 | 0.6052 | 0.5777 | +0.0274 |
| section_attachment_f1 | 0.6659 | 0.7403 | -0.0744 |
| reference_section_completeness | 0.8713 | 0.7256 | +0.1456 |
| float_caption_attachment_accuracy | 0.3732 | 0.5134 | -0.1403 |
| generated_structure_validity | 0.9952 | 0.9461 | +0.0491 |
| layout_similarity | 0.8104 | n/a | n/a |

Interpretation: the scored evidence gate recovers part of the strict
active-stack heading loss (`0.5581 -> 0.5778`) and gives the best section
attachment among the stack variants (`0.6659`), but it is still substantially
below the filtered-stack heading score (`0.6337`) and does not improve
float-caption attachment.  Relative to filtered-stack, scored-stack changes are
not production-worthy yet: macro mean delta is `-0.0078`, heading mean delta is
`-0.0558`, heading improves on only 20/100 documents and worsens on 31/100.
For paper metrics, filtered-stack remains the best current structure decoder;
scored-stack is retained as an experimental ablation and diagnostic path.

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

## PARENT_CHILD Composition Audit

Date: 2026-05-17.

Purpose: check whether the current `PARENT_CHILD` class is mostly a mixed bag of
local relations that should be split into auxiliary heads, or whether it is
still dominated by the original structural attachment task.  This audit is
read-only: it does not change labels, graphs, checkpoints, or model code.

Script:

```text
tools/audit_parent_child_composition.py
```

Local reports:

```text
local_outputs/final_eval_20260517/label_audits/parent_child_audit_registry_adapteraware_20260515_181724.json
local_outputs/final_eval_20260517/label_audits/parent_child_audit_floatproxy_20260516_205926.json
```

### Locked registry-adapter baseline

```text
docs: 1851
PARENT_CHILD edges: 193827
NONE edges: 5887048
MERGE edges: 1769
```

Top PARENT_CHILD families:

| family | count | ratio |
| --- | ---: | ---: |
| heading_to_body | 128795 | 0.6645 |
| title/heading -> equation/display_math | 16485 | 0.0851 |
| heading_to_heading | 15803 | 0.0815 |
| title/heading -> figure/figure_caption | 7887 | 0.0407 |
| same_text | 7853 | 0.0405 |
| title/heading -> figure/chart | 4990 | 0.0257 |
| title/heading -> table/table_caption | 4169 | 0.0215 |

### Float-proxy experimental set

```text
docs: 1829
PARENT_CHILD edges: 190142
NONE edges: 5772940
MERGE edges: 1750
```

Top PARENT_CHILD families:

| family | count | ratio |
| --- | ---: | ---: |
| heading_to_body | 126819 | 0.6670 |
| title/heading -> equation/display_math | 15993 | 0.0841 |
| heading_to_heading | 15535 | 0.0817 |
| title/heading -> figure/figure_caption | 7947 | 0.0418 |
| same_text | 7716 | 0.0406 |
| title/heading -> figure/chart | 4669 | 0.0246 |
| title/heading -> table/table_caption | 4212 | 0.0222 |

Interpretation:

```text
PARENT_CHILD is not currently dominated by caption/formula/list/table local
micro-relations. Roughly two thirds of the class is heading -> body, and another
large slice is heading -> equation/figure/table/title. Splitting the main task
into many sparse auxiliary heads would therefore not match the current label
distribution and would likely create severe positive-sample starvation.
```

Decision:

```text
keep the current three-class GNN mainline:
  MERGE / PARENT_CHILD / NONE

do not rename PARENT_CHILD to LOCAL_RELATION for the main model
do not open a production auxiliary-head migration yet
keep auxiliary heads as a possible future ablation only
```
