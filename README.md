# PDF2LaTeX NN

**Last updated**: 2026-05-07

This project builds a structure-aware PDF-to-LaTeX pipeline for born-digital research papers. The current target is not generic OCR. The target is to recover reading order, document hierarchy, formulas, lists, references, and graph relations that can drive LaTeX/IR reconstruction.

## Current Pipeline

```text
PDF + TeX source
  -> MinerU extraction
  -> v7 JSON cleanup and PyMuPDF style injection
  -> SciBERT + geometry/style/sequence graph features
  -> TeX/PDF automatic truth-label generation
  -> GNN edge training and inference
  -> decoder / renderer
  -> generated LaTeX
```

The active PDF-side JSON contract is:

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

v7 keeps MinerU block granularity and raw bbox coordinates. It adds only the metadata needed for robust downstream modeling: list marker recognition, column reading-order repair, reference item preservation, and PyMuPDF style spans. Cross-page or cross-paragraph merging is no longer written back into preprocessing JSON; it belongs to decoder/generator logic.

## Current Graph Target

The GNN edge task is a three-class problem:

```text
MERGE = 0
PARENT_CHILD = 1
NONE = 2
```

`SIBLING` has been removed. Sibling ordering is handled by reading order and renderer sorting.

## Key Documents

- [docs/PROJECT_SOURCE_OF_TRUTH.md](docs/PROJECT_SOURCE_OF_TRUTH.md): local / GitHub / AutoDL boundary and source-of-truth rules.
- [docs/feature_schema_v0.md](docs/feature_schema_v0.md): PDF IR, node features, edge features, and tensor dimensions.
- [docs/ground_truth_labeling_v0.md](docs/ground_truth_labeling_v0.md): TeX AST parsing, fuzzy alignment, label rules, and strict quality gates.
- [docs/LOCAL_CONFIGURATION.md](docs/LOCAL_CONFIGURATION.md): local private configuration notes.

## Important Paths

Local project root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Main source directories:

```text
src/
scripts/pipeline/
tools/
tests/
docs/
```

Runtime data directories:

```text
data/01_raw_pdfs/
data/02_mineru_outputs/
data/03_tex_source_pool/
data/04_ground_truth_ir/
data/06_graph_features_v7/
data/08_output_latex/
data/09_eval_reports/
```

Bulk PDFs, MinerU outputs, graph caches, model checkpoints, and generated reports stay out of Git.

## Common Commands

Run the full test suite on AutoDL:

```bash
cd /root/autodl-tmp/pdf2latex_nn
/root/miniconda3/envs/pdf2latex/bin/python -m pytest tests -q
```

Label one graph with TeX-derived truth:

```bash
python scripts/pipeline/step3_label_graph.py \
  --content-json data/02_mineru_outputs/mineru_output/2501.00050/auto/2501.00050_content_list_v7_styles.json \
  --tex data/03_tex_source_pool/2501.00050/aaai25.tex \
  --graph data/06_graph_features_v7/2501.00050_v7_graph.pt \
  --output data/06_graph_features_v7/2501.00050_v7_truthgen_labeled_graph.pt \
  --mapping-output data/04_ground_truth_ir/2501.00050_v7_alignment_mapping.json \
  --similarity-threshold 65
```

Build a strict 10-document mini dataset:

```bash
python scripts/pipeline/build_mini_dataset.py \
  --target 10 \
  --similarity-threshold 65 \
  --max-orphan-ratio 0.15 \
  --max-unmapped-tex-ratio 0.30 \
  --max-isolated-node-ratio 0.85
```

## Current Status

The source pipeline is synced through GitHub to AutoDL. The latest validated state includes:

```text
v7 JSON input contract
v7-only graph contract (`graph_schema_version=graph_v7`)
818-dimensional node features
15-dimensional edge attributes
3-class graph labels
TexSoup-based automatic truth labeler
strict bad-sample rejection gates
```

Last AutoDL verification:

```text
144 passed
```
