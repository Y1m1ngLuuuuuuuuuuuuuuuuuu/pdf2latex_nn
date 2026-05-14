# PDF2LaTeX NN

**Last updated**: 2026-05-14

PDF2LaTeX NN is a structure-aware PDF-to-LaTeX pipeline for born-digital research papers. It does not treat PDF conversion as plain OCR. The current system extracts visual facts from PDF, derives graph relation labels from matching TeX source, trains a GNN to predict document relations, and reconstructs compilable LaTeX through a decoupled IR renderer.

## Current Deployment

The production path is v7-only:

```text
compiled PDF + matching TeX
  -> MinerU content_v2
  -> v7 reading/layout cleanup
  -> PyMuPDF style spans
  -> SciBERT + geometry/style/sequence graph features
  -> TeX AST alignment labels
  -> GATv2/Y-Network edge-relation model
  -> TreeDecoder / RenderTreeIR
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

Old v3/v4/v5 preprocessing variants are no longer production inputs. They are historical experiments only.

## Main Relation Task

The graph model predicts three directed edge classes:

```text
MERGE        = 0  physical continuation / paragraph stitching
PARENT_CHILD = 1  logical hierarchy / attachment
NONE         = 2  no structural relation
```

`SIBLING` is not a learned class. Sibling order is recovered from v7 reading order and renderer sorting.

## Active Interfaces

```text
DocumentIR          PDF-side visual facts
GraphInput.pt       node/edge tensors
GraphLabels         TeX-derived edge labels
PredictedRelations  GNN output probabilities
RenderTreeIR        decoder output
StyleProfile        global/local layout profile
CitationResolution  citation/reference repair state
```

See [docs/frontend_backend_contract_v1.md](docs/frontend_backend_contract_v1.md).

## Key Scripts

```bash
# Continue building new production data from PDF + TeX sources
python scripts/pipeline/build_v7_dataset_staged.py ...

# Rebuild graph tensors and relabel existing v7 content
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh

# Train the current GATv2/Y-Network relation model
python scripts/pipeline/train_edge_gnn_full.py ...

# Generate ablation commands
python scripts/pipeline/prepare_ablation_suite.py

# Batch visual QA / E2E inference with the current IR renderer
python scripts/pipeline/batch_visual_qa_inference.py --renderer ir ...
```

## Current Docs

```text
docs/PROJECT_SOURCE_OF_TRUTH.md      local / GitHub / AutoDL boundary
docs/PROJECT_OVERVIEW.md             architecture and implementation summary
docs/frontend_backend_contract_v1.md decoupled IR contracts
docs/feature_schema_v0.md            graph tensor feature contract
docs/ground_truth_labeling_v0.md     TeX-to-PDF truth-label generation
docs/ablation_plan_v2.md             current ablation protocol
docs/ablation_results_current.md     latest locked ablation results
docs/v7_training_and_monitoring.md   production data/training runbook
docs/LOCAL_CONFIGURATION.md          private local configuration notes
```

## Important Paths

Local project root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Large artifacts stay on AutoDL under:

```text
/root/autodl-tmp/pdf2latex_nn/data
```

Do not commit datasets, checkpoints, generated PDFs, secrets, or AutoDL credentials.
