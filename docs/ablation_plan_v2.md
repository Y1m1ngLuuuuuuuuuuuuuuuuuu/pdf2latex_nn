# Ablation Plan v2

This is the fine-grained ablation protocol for the v7 graph-relation model.  The purpose is to answer precise questions, not merely produce many runs.

## Non-Negotiable Controls

All experiments must share:

```text
same clean v7 manifest
same graph .pt files
same document-level train/val/test split rule
same threshold-calibration grid
same random seeds
same visual-QA sample set
```

Feature ablation is runtime-only.  We zero feature ranges during model forward and never rewrite source graph tensors.

The executable matrix is:

```text
configs/ablation_matrix_v2.json
```

Each experiment is repeated with:

```text
seed = 7, 13, 29
```

The repeated-seed result is summarized by mean and population standard deviation.

## Metrics

Primary:

```text
calibrated_test_positive_macro_f1
= mean(MERGE_F1, PARENT_CHILD_F1)
```

Secondary:

```text
MERGE precision / recall / F1
PARENT_CHILD precision / recall / F1
calibrated thresholds tau_merge / tau_parent
compile success rate
visual QA class: PASS / MINOR / MAJOR / FATAL
```

Accuracy and NONE F1 are diagnostic only.

## Families And Questions

### M: Model Architecture

`M00_full_ce_ohem`

Main system.

`M01_no_message_passing`

Sets `num_layers=0`, so the classifier sees projected node features and edge attributes but no GATv2 message passing.  This answers whether graph propagation is actually helping beyond pairwise edge features.

Expected failure mode: weaker PARENT_CHILD and long-range local-scope relation reasoning.

`M02_no_symmetry_terms`

Uses `edge_feature_mode=simple_concat`, replacing:

```text
concat([Hu, Hv, Hu-Hv, Hu*Hv, Euv])
```

with:

```text
concat([Hu, Hv, Euv])
```

This tests the directional symmetry-breaking terms.  It should especially affect PARENT_CHILD, because `A -> B` is not equivalent to `B -> A`.

`M03_shallow_predictor`

Replaces the deep `[1024,512,128]` edge MLP with one `128` hidden layer.  This tests whether the rare MERGE boundary needs predictor capacity after GAT.

### F: Node Feature Families

`F00_no_scibert`

Zeros the 768-d SciBERT vector.  If performance survives, layout dominates.  If MERGE drops, semantics is useful.

`F01_semantic_only`

Zeros all layout/type/stat/style/flow node features.  This is the control against a pure text-semantic model.

Expected: poor layout hierarchy and weak cross-column robustness.

`F02_no_geometry`

Zeros local geometry and scroll geometry.  This separates geometry from style and type cues.

`F03_no_reading_flow`

Zeros scroll-y, sequence position, column one-hot, layout layer, flow context, and matching edge flow cues.  This is the main test for the v7 reading-flow reconstruction.

Expected: reading-order inversions and parent-child degradation.

`F04_no_style_title`

Zeros PyMuPDF style stats, title probes, and typography edge deltas.  This tests heading hierarchy and font-cluster value.

### E: Edge Feature Families

`E00_no_edge_semantic`

Zeros semantic cosine only.  This should primarily affect MERGE.

`E01_no_punctuation`

Zeros terminal punctuation and hyphen probes.  This should hurt MERGE precision/recall around paragraph endings and hyphenated line breaks.

`E02_no_gutter_overlap`

Zeros `y_overlap_ratio` and `has_x_gutter`.  This tests the cross-column barrier.

Expected failure mode: cross-column false MERGE.

`E03_no_index_bins`

Zeros binned sequence distance.  This verifies whether index bins help without becoming a brittle direct answer key.

### T: Training Objective / Long Tail

`T00_no_ohem`

Plain CE.  This is the long-tail baseline.

`T01_random_none_dropout`

Randomly drops NONE edges during training.  It should raise recall but may damage precision because hard negatives vanish.

`T02_focal_inverse`

Focal loss with inverse class weights.  It is expected to be high-recall and possibly over-predict MERGE.

`T03_weighted_ce`

Hand-weighted CE without OHEM.  This tests whether a simpler weighted objective is enough.

## Running

Generate commands:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v2.json \
  --output-sh data/08_runs/run_ablation_matrix_v2.sh \
  --output-json data/09_eval_reports/ablation_matrix_v2_commands.json
```

Launch:

```bash
bash data/08_runs/run_ablation_matrix_v2.sh
```

Run only a small subset:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v2.json \
  --only M00_full_ce_ohem,F03_no_reading_flow,T00_no_ohem \
  --output-sh data/08_runs/run_ablation_core_subset.sh
```

Summarize after runs:

```bash
python scripts/pipeline/summarize_ablation_results.py \
  --root data/09_eval_reports/ablations_v2 \
  --output-json data/09_eval_reports/ablations_v2_summary.json \
  --output-csv data/09_eval_reports/ablations_v2_summary.csv
```

## Visual QA Protocol

After metric training, select:

```text
M00_full_ce_ohem
best feature ablation competitor
best training-objective competitor
worst expected ablation: F03_no_reading_flow or E02_no_gutter_overlap
```

Run end-to-end inference on the same 10-20 held-out PDFs.  Score generated PDFs with:

```text
PASS: compile-safe and structural reading order is acceptable
MINOR: paragraph split, list indentation, cosmetic table placeholder
MAJOR: reading-order inversion, wrong section hierarchy, broken references
FATAL: compile error, formula corruption, large text loss, section tree collapse
```

The final paper table should combine:

```text
metric table: mean/std over 3 seeds
visual QA table: PASS/MINOR/MAJOR/FATAL counts
qualitative figure: original PDF vs generated PDF for representative failure/success
```

## Interpretation Rules

Strong project evidence:

```text
M00 > F01_semantic_only
M00 > F00_no_scibert
M00 > M01_no_message_passing
M00 > M02_no_symmetry_terms
M00 > T00_no_ohem
F03_no_reading_flow drops visibly on reading-order cases
E02_no_gutter_overlap increases cross-column false MERGE
```

If a feature ablation has no measurable effect across all seeds and no visual-QA difference, demote that feature to optional or remove it from the GNN contract.
