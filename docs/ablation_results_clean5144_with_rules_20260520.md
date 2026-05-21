# Clean5144 Ablation With Rules-Only E2E Controls

## Scope

- Dataset: clean5144 canonical manifest.
- Edge ablation: trained 3-class GNN relation models.
- E2E relation-source ablation: compares M06 GNN relations with no-GNN deterministic rules.
- M06 main checkpoint is reused from the completed clean5144 main training run.

## Artifacts

- Matrix: `configs/ablation_matrix_clean5144_20260520.json`
- Edge ablation root: `data/09_eval_reports/clean5144_mainline_20260520/edge_ablation_with_rules_20260520`
- Edge summary JSON: `data/09_eval_reports/clean5144_mainline_20260520/edge_ablation_with_rules_20260520/summary.json`
- Edge summary CSV: `data/09_eval_reports/clean5144_mainline_20260520/edge_ablation_with_rules_20260520/summary.csv`
- E2E relation-source root: `data/09_eval_reports/clean5144_mainline_20260520/e2e_relation_source_ablation_20260520`

## Edge Ablation Summary

| experiment | positive macro F1 | MERGE F1 | PARENT F1 | tau_merge | tau_parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| M06_current_main_merge_gate | 0.8247 | 0.6776 | 0.9719 | 0.7600 | 0.7900 |
| T00_no_ohem | 0.8229 | 0.6706 | 0.9752 | 0.7900 | 0.7600 |
| M07_gaussian_edge_feature | 0.8187 | 0.6595 | 0.9780 | 0.8500 | 0.7400 |
| A02_no_type_aware_message_mask | 0.8148 | 0.6553 | 0.9743 | 0.7200 | 0.7800 |
| E01_no_gutter_overlap | 0.8073 | 0.6535 | 0.9611 | 0.6000 | 0.8300 |
| M05_no_merge_gate | 0.8059 | 0.6527 | 0.9590 | 0.6100 | 0.7100 |
| E00_no_punctuation | 0.7935 | 0.6296 | 0.9573 | 0.6600 | 0.6700 |
| A00_old_shared_gat | 0.7738 | 0.5684 | 0.9791 | 0.5600 | 0.7300 |
| F02_no_v7_reading_flow | 0.7684 | 0.5972 | 0.9397 | 0.5000 | 0.7000 |
| F00_no_scibert | 0.7568 | 0.5595 | 0.9540 | 0.7000 | 0.6600 |
| A01_no_message_passing | 0.7445 | 0.6625 | 0.8265 | 0.5800 | 0.6800 |
| F01_no_geometry_layout | 0.6024 | 0.4630 | 0.7417 | 0.6300 | 0.6400 |

## E2E Relation-Source Ablation

| relation source | documents | macro | heading | reading order | paragraph coverage | section body/no-float | float-caption | refs | validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M06_gnn_relation_source | 20 | 0.8975 | 0.8428 | 0.9851 | 0.8716 | 0.9756 | 0.8326 | 0.9963 | 0.9996 |
| R00_rules_only_no_merge | 20 | 0.8969 | 0.8390 | 0.9851 | 0.8715 | 0.9756 | 0.8326 | 0.9963 | 0.9996 |
| R01_rules_only_deterministic_merge | 20 | 0.8969 | 0.8390 | 0.9851 | 0.8714 | 0.9756 | 0.8326 | 0.9963 | 0.9996 |

## Interpretation Notes

- `A01_no_message_passing` is still a learned edge classifier; it is not a pure no-GNN document reconstruction baseline.
- `R00_rules_only_no_merge` disables learned relation predictions entirely and uses heading-stack/full-v7 rendering only.
- `R01_rules_only_deterministic_merge` adds conservative adjacent text merge edges, still without loading a GNN checkpoint.
- Use the edge table for model-design claims and the relation-source table for final generator dependency claims.
- In this 20-doc relation-source sample, the full renderer/heading-stack path already explains most structure metrics; the GNN contributes only a small gain in heading/macro through local relation predictions.
