# PDF2LaTeX-NN Project Overview

**Last updated**: 2026-05-24

This document summarizes the current v8/default architecture. It is the high-level
project explanation; lower-level contracts live in the schema, labeling, and
frontend/backend docs. For the complete architecture, data-flow, judgment
rules, code map, and metrics taxonomy, see `docs/PROJECT_ARCHITECTURE_FULL.md`.
For a paper-facing long-form description, see
`docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md`.
For the single current production command and output contract, see
`docs/V8_MAINLINE_RECONSTRUCTION_PATH.md`.

## 1. Objective

The project builds a structure-aware PDF-to-LaTeX system for research papers.
The canonical target is:

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction
from rendered scientific PDFs
```

The target is **not** source-level TeX AST recovery.  A PDF is a rendering
result, not a unique encoding of the author's TeX program.  The same PDF can be
produced by many TeX sources, and LaTeX floats can visually move away from their
source location.  The system therefore aims to reconstruct a stable, readable,
editable LaTeX document that preserves page layout and block-level semantic
organization.

```text
PDF visual facts
  -> v8 middle-derived reflow
  -> structured IR
  -> compilable LaTeX
```

The target is not plain OCR, not a single end-to-end language model, and not
author-macro recovery. The system separates perception, truth generation,
relation learning, decoding, and rendering so each layer can be replaced
independently.

The detailed target and evaluation contract lives in
`docs/layout_aware_reconstruction_target.md`.

## 2. Core Modules

```text
PDF Frontend
  MinerU middle/content-list outputs + optional PyMuPDF/style sidecars

V8 Reflow
  raw middle.json line/block evidence -> corrected logical item list

DocumentIR
  complete document fact layer consumed by decoder and renderer

Front Matter / Style / Heading
  deterministic front matter preservation
  document-local heading style registry
  stack skeleton section hierarchy

Generator
  OriginalLikeIRLatexRenderer + style/citation/float adapters

Optional Relation-Learning Branch
  GNNViewAdapter + graph builder + TeX-derived labels + GNN ablations
```

The default reconstruction path does not load a GNN checkpoint.  The GNN branch
is preserved for relation-learning experiments, diagnostics, and ablation
evidence.

## 3. V8 Frontend Principles

The v8 frontend exists because MinerU's `content_list.json` can merge text
before page-local reading order is corrected.  V8 therefore starts from
`middle.json`, reconstructs line/block fragments in their original geometry,
repairs reading order, and only then emits a v7-compatible logical item list.

```text
middle.json
  + content_list.json asset/caption sidecar
  + optional content_list_v7_styles.json style sidecar
  -> content_list_v8.json
  -> DocumentIR
  -> RenderTreeIR
  -> OriginalLikeIRLatexRenderer
```

V8 does not mutate old v7 JSON, does not enter the GNN graph, and does not
change graph schema.

## 4. V7 / GNN Branch Principles

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

The current experimental adapter policy is:

```text
metadata / true page furniture / annotations -> excluded from GNN view
figure / table / algorithm -> included as float proxies
caption text -> used as proxy semantics when available
raw table body / raw figure OCR -> not embedded as normal paragraph text
float -> text message passing -> masked
skip-over-float candidate edges -> added for paragraph continuation
```

## 5. GNN Task

The graph model is intentionally small in label space:

```text
MERGE        physical continuation
PARENT_CHILD logical attachment
NONE         no relation
```

The current locked baseline model is a GATv2/Y-Network hybrid. PARENT_CHILD
and NONE use propagated GAT states, while MERGE bypasses message passing and
uses raw projected node-pair features so paragraph stitching is not
over-smoothed by neighboring floats, tables, and unrelated text.

The float-proxy adapter path is being rebuilt separately and should be compared
against the locked baseline rather than replacing it blindly.

The GNN is no longer the default E2E reconstruction dependency.  Its current
role is to support optional relation-learning studies.  In those studies it
still predicts `MERGE / PARENT_CHILD / NONE` over candidate graph edges, but
production section scope is driven by the heading stack and layout rules.

The deep edge heads receive directional node terms:

```text
concat([Hu, Hv, Hu-Hv, Hu*Hv, Euv])
```

This keeps parent-child direction learnable and avoids symmetric false positives.

## 6. Why TeX Labels Exist

For optional GNN training, the matching TeX source is the source of truth. The labeler:

```text
flattens TeX
parses structural nodes
aligns TeX nodes to PDF blocks
generates edge labels over graph candidate edges
enforces quality gates
```

This is a training-data generator, not an inference dependency for the default
v8 path.

## 7. Generator Direction

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

Current implemented v8 generator-side features include:

```text
source PDF page size selection
body font / line-height / paragraph spacing detection
heading style registry with centered vs left-aligned levels
FrontMatter Phase 0 title/author/affiliation/email/abstract preservation
wide figure* / table* rendering outside multicols
ordered enumerate vs itemize recovery
```

## 7. Current Data Strategy

Production samples must come from a closed compile loop:

```text
arXiv TeX source -> compiled PDF -> MinerU -> graph -> TeX labels
```

Official PDFs paired only by arXiv id are not production training samples because source/PDF revisions can differ.

## 8. Evaluation Strategy

Use layered checks.  The important separation is that visual reconstruction,
text coverage, paragraph boundaries, block-level headings, body section
attachment, float/caption recovery, and references answer different questions.

```text
1. edge metrics on MERGE and PARENT_CHILD
2. candidate-edge recall and label quality gates
3. visual QA of generated PDFs against originals
```

Accuracy and NONE F1 are not primary metrics because NONE dominates the class distribution.

Block-level heading metrics only include:

```text
\section
\subsection
\subsubsection
```

Run-in `\paragraph` / `\subparagraph` structures are normalized into paragraph
inline labels for comparison because they are not reliably distinguishable from
bold paragraph prefixes in rendered PDFs.

`section_attachment_f1` is an auxiliary structure metric.  The fairer primary
variant for document structure is `section_attachment_body_no_float_f1`, which
excludes floats, captions, references, footnotes, page furniture, and run-in
headings.  Float/caption behavior is evaluated separately.

Latest locked baseline result:

```text
M07_y_network_plus_gaussian_edge_feature
MERGE F1        0.6331
PARENT_CHILD F1 0.9620
Positive Macro  0.7976
```

Current paper-facing evaluation track:

```text
active data/model family: v7_floatproxy_adapter_20260516_205926
current ablation matrix: configs/ablation_matrix_current.json
full evaluation suite: scripts/pipeline/run_current_full_eval_suite.py
result rollup: scripts/pipeline/collect_current_eval_results.py
```

This track reuses the latest generator and compares the current model against
Nougat through the neutral comparison structure.  Generator-only edits do not
require relabeling or retraining; they require rerunning E2E and comparison
evaluation.
