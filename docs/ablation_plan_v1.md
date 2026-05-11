# Ablation Plan v1

This document defines the first reproducible ablation suite for the v7 PDF-to-LaTeX graph pipeline.  The goal is not to prove that every heuristic is perfect; the goal is to isolate which layers actually carry signal:

1. v7 reading-flow and layout features
2. SciBERT semantic vectors after model-side bottleneck normalization
3. PyMuPDF style and heading probes
4. edge-level physical barriers such as gutter, band, and punctuation
5. GATv2 message passing and deep directional edge head
6. long-tail training strategy

## Fixed Dataset Contract

All ablations must consume the same clean labeled `.pt` manifest.

Required input:

```text
MinerU v7 JSON
PyMuPDF style spans
SciBERT features
v7 graph tensors
TeX AST labels
quality-gated manifest
```

The ablation suite never rewrites `.pt` files.  Feature removal is done at model runtime by zeroing selected node feature ranges or edge feature columns.  This keeps the comparison honest: every experiment sees the same documents, candidate edges, labels, train/val/test split, and graph topology.

## Primary Metrics

Accuracy is not a decision metric because NONE dominates.

Use these metrics in order:

1. `val_positive_macro_f1`: mean of MERGE F1 and PARENT_CHILD F1.
2. MERGE precision/recall/F1.
3. PARENT_CHILD precision/recall/F1.
4. visual QA pass/fail on generated PDFs after threshold calibration.
5. compile success rate for generated `.tex`.

NONE F1 is reported but should not drive model selection.

## Experiment Matrix

The executable matrix is stored in:

```text
configs/ablation_matrix_v1.json
```

Current experiments:

| ID | Purpose |
| --- | --- |
| `A00_full_ce_ohem` | Main model: all v7 features + GATv2 + deep directional head + CE/OHEM. |
| `A01_no_scibert` | Remove SciBERT to test whether layout alone can recover structure. |
| `A02_semantic_only` | Keep SciBERT only; remove all non-semantic node features. |
| `A03_no_reading_flow` | Remove scroll-y, sequence, column, band, and flow features. |
| `A04_no_style_title` | Remove PyMuPDF style stats and heading probes. |
| `A05_no_punctuation_probes` | Remove terminal punctuation and hyphen probes. |
| `A06_no_gutter_band_barriers` | Remove cross-column gutter and band-flow edge features. |
| `A07_shallow_predictor` | Replace deep directional edge predictor with a shallow head. |
| `A08_no_ohem` | Full model without online hard negative mining. |
| `A09_focal_inverse` | Focal loss with inverse class weights. |
| `A10_random_none_dropout` | Random NONE dropout instead of OHEM. |

## Command Generation

After the clean manifest is ready, generate the run script:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v1.json \
  --output-sh data/08_runs/run_ablation_matrix_v1.sh \
  --output-json data/09_eval_reports/ablation_matrix_v1_commands.json
```

Run all experiments:

```bash
bash data/08_runs/run_ablation_matrix_v1.sh
```

Run only one or two experiments:

```bash
python scripts/pipeline/prepare_ablation_suite.py \
  --only A00_full_ce_ohem,A03_no_reading_flow \
  --output-sh data/08_runs/run_ablation_smoke.sh
```

## Runtime Feature Masks

The training entrypoint supports these ablation flags:

```text
--ablate-node-groups semantic,type,geometry,scroll,derived,style,sequence,column,title,layout_layer,flow_context,layout_all
--ablate-edge-groups semantic,spatial,typography,overlap_gutter,index_bins,punctuation,layout_flow,all
--ablate-edge-fields field_name_a,field_name_b
```

These masks are stored inside the checkpoint config, so threshold calibration and inference load the same feature-removal behavior automatically.

## Visual QA Layer

Metric ablations are necessary but not sufficient.  After `A00_full_ce_ohem` and the strongest ablated competitors finish:

1. run threshold calibration on validation logits;
2. run batch E2E inference on 10-20 held-out PDFs;
3. compile generated `.tex`;
4. inspect original/generated PDF pairs.

The visual QA table should label failures as:

```text
FATAL: compile error, formula corruption, section tree collapse, large text loss
MAJOR: reading order inversion, wrong section/subsection hierarchy, references broken
MINOR: paragraph split, list spacing, table placeholder, cosmetic mismatch
PASS: structure readable and compile-safe
```

## Expected Interpretations

Strong evidence for this project:

- `A00` beats `A02_semantic_only`, proving geometry/layout is essential.
- `A00` beats `A01_no_scibert`, proving semantics still matters for rare MERGE and ambiguous heading/body boundaries.
- `A03_no_reading_flow` drops on reading-order/parent edges, proving v7 flow features matter.
- `A05_no_punctuation_probes` mainly hurts MERGE.
- `A06_no_gutter_band_barriers` increases cross-column false MERGE.
- `A08_no_ohem` improves NONE but loses MERGE/PARENT_CHILD, proving OHEM is needed for the long tail.

If an ablation does not change metrics, treat that feature family as suspect and either remove it or move it to a generator-only heuristic instead of burdening the GNN.
