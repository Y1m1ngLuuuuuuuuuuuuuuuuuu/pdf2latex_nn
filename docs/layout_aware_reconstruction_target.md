# Layout-Aware Reconstruction Target

**Last updated**: 2026-05-23

This document fixes the project target and evaluation philosophy.  It is the
contract that prevents the system from being optimized toward an impossible
goal: recovering the author's original TeX source tree from a rendered PDF.

## 1. Canonical Target

The project target is:

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction
from rendered scientific PDFs
```

The target is **not**:

```text
source-level TeX AST recovery
```

A PDF is a rendering result, while TeX source is a generation program.  Many
different TeX sources can produce visually equivalent PDFs, and LaTeX float
placement can move figures and tables away from the source location where they
were written.  Therefore, global evaluation against raw author TeX AST is too
strict and sometimes structurally unfair for a PDF-first reconstruction system.

## 2. Core Principle

Separate what can be observed in the PDF from what only exists in the source
program.

```text
Observable from PDF:
  text blocks
  block-level headings
  visual reading regions
  figure/table/caption locations
  references region
  page geometry and styles

Not uniquely observable from PDF:
  exact author macro choices
  run-in \paragraph versus \textbf prefix
  exact source location of floats
  journal-template implementation details
  original file decomposition / macro AST
```

The system should reconstruct a stable, readable, editable LaTeX document that
preserves visual layout and block-level semantic organization.  It should not
pretend that PDF evidence uniquely determines the author's original TeX AST.

## 1.1 Current Default Reconstruction Policy

As of 2026-05-23, the default reconstruction path is layout-aware and
rules-first:

```text
full v7 facts -> DocumentIR/LayoutIR -> heading stack + layout decoder
-> RenderTreeIR -> LaTeX renderer
```

The GNN relation model is retained as an explicit experimental branch, not as
the default reconstruction dependency.  This preserves the project target while
keeping relation-learning evidence available for ablations.

## 3. IR Separation

The reconstruction stack should keep three concepts separate.

### Layout IR

Layout IR describes where visual content appears.

```text
page
bbox
column / band
visual order
reading order
font / style statistics
float visual slot
```

### Semantic IR

Semantic IR describes the recoverable document structure.

```text
block-level heading tree
paragraphs
lists
display equations
references
captions
footnotes / margin notes
cross references
```

### Float IR

Float IR must separate visual placement from semantic anchor.

```text
visual float slot:
  where the figure/table appears in the rendered PDF

caption-float pairing:
  which caption belongs to which figure/table

semantic anchor:
  which section or paragraph the float is most related to
```

These are not the same relation.  A figure can visually appear near one block
while being written elsewhere in TeX source.

## 4. GNN Responsibility

The GNN remains the learned relation core of the project.  It keeps the current
three-class task:

```text
MERGE
PARENT_CHILD
NONE
```

`PARENT_CHILD` is still a learned structural attachment relation derived from
TeX labels over the graph-visible view.  It is not renamed to a local-only label
and it is not split into many sparse heads.

The decoder may add deterministic constraints around the GNN output:

```text
heading evidence / stack priors
section-scope safety gates
float/caption grouping
physical impossibility vetoes
renderer layout policies
```

These constraints are safeguards, not a replacement for the learned relation
model.  They prevent physically impossible structures and make the generated
LaTeX stable, but the supervised GNN task remains the same three-class relation
prediction problem.

This means the project keeps the previous GNN design:

```text
candidate graph -> GNN relation probabilities -> constrained decoder -> IR renderer
```

The evaluation may report body-only and float-separated variants because raw
source AST attachment is not always a fair PDF-first metric.  That evaluation
normalization does not change what the GNN is trained to predict.

## 5. Run-In Headings

Run-in headings are visually ambiguous in rendered PDFs.

Examples:

```text
Summary. The method consists of ...
Linear Relationship in Predictors: The linear predictor ...
```

They may come from:

```tex
\paragraph{Summary.} The method consists of ...
```

or:

```tex
\textbf{Summary.} The method consists of ...
```

The PDF often does not contain enough evidence to distinguish these reliably.

Policy:

```text
run-in headings are paragraph inline labels by default
```

They are excluded from block-level heading evaluation.  The generator may render
them as bold inline prefixes.  Only strong document-wide evidence should promote
them to `\paragraph{}`.

## 6. Evaluation Philosophy

Evaluation must be layered.  A single AST score is not appropriate.

### Visual Reconstruction

```text
compile_success
page_count_match
layout_similarity
block_bbox_iou
float visual slot recovery
```

### Text And Content

```text
paragraph_text_coverage
normalized edit distance
formula recovery
caption text recall
reference item recall
```

### Paragraph And Block Boundaries

```text
paragraph_boundary_f1
paragraph/list/reference/caption split-merge tolerance
block_type_f1
```

### Block-Level Headings

Only block-level headings are scored:

```text
\section
\subsection
\subsubsection
```

The following are excluded from heading tree metrics:

```text
\paragraph
\subparagraph
bold run-in paragraph prefixes
caption labels
reference items
list items
front-matter metadata unless explicitly evaluated
```

### Body Section Attachment

`section_attachment_f1` should not be the global success criterion.  The primary
fair variant is:

```text
body_no_float_section_attachment
```

It evaluates only body-like blocks:

```text
paragraph
list item
display equation
ordinary body text
algorithm/code block when used as body content
```

It excludes:

```text
figure
table
caption
footnote
margin note
reference item
run-in heading
header/footer/page number
front matter
```

### Float Metrics

Float evaluation is separate:

```text
caption detection
caption label accuracy
caption-float pairing
float visual slot recovery
float semantic anchor
```

The semantic anchor should be low-weight because it is inherently affected by
LaTeX float placement.

## 7. Baseline Comparison

Nougat and similar systems should be treated as markup-oriented scientific
document transcription baselines, not as full layout-preserving LaTeX
reconstruction systems.

Shared comparison dimensions:

```text
text coverage
formula/caption/reference recovery
block-level heading tree
body section attachment
reading order
```

Ours-only or reconstruction-specific dimensions:

```text
compilable LaTeX output
page layout similarity
float visual slots
crop-based figure/table preservation
style and column reconstruction
```

The expected claim is not "we recover the original TeX better than Nougat".
The expected claim is:

```text
we reconstruct compilable, layout-aware LaTeX with block-level semantic
structure, while preserving page layout and float/reference structure beyond
plain markup transcription
```

## 8. Practical Consequences

This target definition implies:

```text
1. Do not optimize the whole system for raw TeX AST equivalence.
2. Do not let section_attachment alone decide model quality.
3. Keep run-in headings out of block heading metrics.
4. Separate float visual position from float semantic anchor.
5. Use heading skeleton/state stack as decoder priors and safety constraints,
   not as a replacement for the learned relation model.
6. Keep the GNN as the three-class relation predictor
   (`MERGE / PARENT_CHILD / NONE`); do not narrow it to local-only or split it
   into many sparse heads unless that is an explicitly separate experiment.
7. Report visual, text, heading, body attachment, float, and reference metrics separately.
```

This is the project-level contract for future model, generator, and evaluation
changes.
