# PDF2LaTeX-NN Project Overview

**Last updated**: 2026-05-14

This document summarizes the current v7 architecture. It is the high-level project explanation; lower-level contracts live in the schema, labeling, and frontend/backend docs.

## 1. Objective

The project builds a structure-aware PDF-to-LaTeX system for research papers:

```text
PDF visual facts + matching TeX source
  -> learned relation model
  -> structured IR
  -> compilable LaTeX
```

The target is not plain OCR and not a single end-to-end language model. The system separates perception, truth generation, relation learning, decoding, and rendering so each layer can be replaced independently.

## 2. Core Modules

```text
PDF Frontend
  MinerU + PyMuPDF + v7 reading/layout cleanup

GNN View Adapter
  full v7 fact layer -> graph-visible node view + reversible v7 mapping

Graph Builder
  SciBERT + geometry + style + layout-flow features

TeX Truth Generator
  LaTeX flattener + TexSoup parser + sliding-window alignment

GNN
  EdgeRelationGAT / Y-Network predicts MERGE / PARENT_CHILD / NONE

Decoder
  merge contraction + structure constraints + tree/IR assembly

Generator
  OriginalLikeIRLatexRenderer + style/citation/float adapters
```

## 3. V7 Frontend Principles

The v7 frontend describes visual facts; it does not bake in final structure decisions.

It keeps MinerU block granularity and raw bboxes, then adds:

```text
reading order metadata
band / column / layout layer hints
toc/header/footer/noise flags
list marker probes
duplicate-continuation detection
PyMuPDF style spans
reference item preservation
table / figure grouping metadata
footnote and margin-note candidates
```

Cross-page and cross-paragraph logical merging belongs in the decoder/generator layer, not in v7 JSON preprocessing.

The complete v7 JSON is never reduced just because a node is not useful for
GNN training. Metadata, floats, annotations, headers/footers, and references
remain in the fact layer. `GNNViewAdapter` builds the narrower graph-visible
view and records the mapping back to full v7 node ids.

## 4. GNN Task

The graph model is intentionally small in label space:

```text
MERGE        physical continuation
PARENT_CHILD logical attachment
NONE         no relation
```

The current best model is a GATv2/Y-Network hybrid. PARENT_CHILD and NONE use propagated GAT states, while MERGE bypasses message passing and uses raw projected node-pair features so paragraph stitching is not over-smoothed by neighboring floats, tables, and unrelated text.

The deep edge heads receive directional node terms:

```text
concat([Hu, Hv, Hu-Hv, Hu*Hv, Euv])
```

This keeps parent-child direction learnable and avoids symmetric false positives.

## 5. Why TeX Labels Exist

For training, the matching TeX source is the source of truth. The labeler:

```text
flattens TeX
parses structural nodes
aligns TeX nodes to PDF blocks
generates edge labels over graph candidate edges
enforces quality gates
```

This is a training-data generator, not an inference dependency. At inference time the model sees only PDF-derived graph features.

## 6. Generator Direction

The current canonical generator is the IR renderer:

```text
OriginalLikeIRLatexRenderer
  -> IRRendererRegistry
    -> FrontMatterRenderer / HeadingRenderer / TextRenderer
    -> MathRenderer / FigureRenderer / TableRenderer
    -> ListRenderer / ReferenceRenderer / NoteRenderer
```

It supports:

```text
page style profiling
single/two/mixed-column approximation
front matter and abstract handling
caption and citation repair
reference rendering fallback
figure/table crop assets
footnote and margin-note rendering
inline-math protection
display equation rendering fallback
```

The generator is still an expandable surface. Journal-template rendering and learned style reproduction should plug in behind the same IR contract.

## 7. Current Data Strategy

Production samples must come from a closed compile loop:

```text
arXiv TeX source -> compiled PDF -> MinerU -> graph -> TeX labels
```

Official PDFs paired only by arXiv id are not production training samples because source/PDF revisions can differ.

## 8. Evaluation Strategy

Use three complementary checks:

```text
1. edge metrics on MERGE and PARENT_CHILD
2. candidate-edge recall and label quality gates
3. visual QA of generated PDFs against originals
```

Accuracy and NONE F1 are not primary metrics because NONE dominates the class distribution.

Latest locked result:

```text
M05_y_network_dual_head
MERGE F1        0.6649
PARENT_CHILD F1 0.9412
Positive Macro  0.8030
```
