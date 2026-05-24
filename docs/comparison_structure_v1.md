# Comparison Structure V1

**Last updated**: 2026-05-24

This schema is the neutral layer used to compare outputs from different
document-to-text systems.  It is intentionally coarser than the production IR:
the goal is to compare recoverable block-level document structure, not OCR
quality and not source-level TeX AST equivalence.

The comparison target follows `docs/layout_aware_reconstruction_target.md`.
Generated outputs are evaluated as layout-aware, block-structure-preserving,
compilable LaTeX reconstructions from rendered PDFs.

As of 2026-05-24, the default system output is produced by the v8/layout-first
path.  Historical v7/GNN and Nougat outputs can still be converted through the
same comparison layer so paper-facing metrics remain comparable.

## Inputs

- LaTeX: `tools/convert_latex_to_comparison.py`
- Markdown / Nougat MMD: `tools/convert_markdown_to_comparison.py`
- Auto-detect wrapper: `tools/convert_to_comparison_structure.py`
- Metrics: `tools/evaluate_comparison_structure.py`
- Rendered output QA: `tools/evaluate_rendered_output.py`

## Output

Each converter writes a JSON document with:

- `schema_version`: always `comparison_structure_v1`.
- `source_format`: `latex` or `markdown`.
- `blocks`: ordered structural blocks.
- `reading_order`: block IDs in output order.
- `heading_tree`: extracted heading nodes with levels and parents.
- `parent_edges`: parent-child structural edges.
- `test_items`: normalized evaluation targets.

## Block Types

- `document_title`
- `author_block`
- `abstract`
- `heading`
- `paragraph`
- `list`
- `list_item`
- `display_math`
- `figure`
- `table`
- `caption`
- `reference_item`
- `algorithm`

## Normalization Rules

The comparison layer normalizes away distinctions that are not reliably
observable from PDF layout alone.

### Block-Level Headings

Only these LaTeX heading commands enter the heading tree and heading attachment
metrics:

```text
\section
\subsection
\subsubsection
```

The following are treated as paragraph inline labels instead of block headings:

```text
\paragraph
\subparagraph
bold run-in prefixes
```

For example:

```tex
\paragraph{Summary.} The method consists of ...
```

is normalized as one paragraph with an inline label:

```text
Summary. The method consists of ...
```

This prevents the metric from asking the model to distinguish `\paragraph{}` from
`\textbf{...}` when the rendered PDF does not contain enough evidence.

### Floats

Figure/table/algorithm floats are not part of body section attachment.  They are
evaluated with float-specific metrics:

```text
caption detection
caption label accuracy
caption-float pairing
float visual slot recovery
float semantic anchor
```

### Body Attachment

The preferred section-attachment score is
`section_attachment_body_no_float_f1`.  It evaluates ordinary body content only:

```text
paragraph
abstract/body paragraph
list item
display math
algorithm/code block when used as body content
```

It excludes:

```text
figure/table/caption
reference item
footnote/margin note
header/footer/page number
front matter
run-in heading labels
```

## Test Items

`test_items` contains the full set of comparison targets:

- `document_titles`
- `author_blocks`
- `text_blocks`
- `headings`
- `paragraphs`
- `lists`
- `list_items`
- `figures`
- `tables`
- `captions`
- `references`
- `display_math`
- `citations`
- `cross_refs`
- `counts`

This lets us compare our generator with Nougat or other baselines on the parts
we claim to improve: reading order, hierarchy, list/paragraph grouping,
float/caption placement, references, citations, and cross references.

The comparison layer intentionally does not score exact font family, exact
line spacing, or journal-template appearance. Those belong to generator visual
QA. The cross-system comparison focuses on structure.

## Metrics

Convert both outputs into this schema first, then run:

```bash
python3 tools/evaluate_comparison_structure.py \
  --gold gold_from_source_tex.json \
  --pred prediction_from_ours_or_nougat.json \
  --output metrics.json
```

The metric JSON contains:

- `heading_tree_accuracy`: heading text, level, and parent-heading agreement.
- `reading_order_accuracy`: pairwise order agreement over matched blocks.
- `strict_block_match`: legacy one-to-one block matching summary.
- `window_matching`: many-to-one / one-to-many sliding-window text-block
  matching summary. This prevents merged or split paragraphs from being
  counted as missing solely because block granularity differs.
- `paragraph_boundary_f1`: strict paragraph/list/reference/caption block
  boundary F1.
- `paragraph_merge_f1`: deprecated backward-compatible alias for
  `paragraph_boundary_f1`. It is retained in low-level metric payloads only
  for old scripts and must not be reported as an independent result.
- `paragraph_text_coverage_f1`: token-level text coverage over sliding-window
  paragraph-like matches. This answers whether the text content is present,
  independent of paragraph split/merge boundaries.
- `section_attachment_f1`: content-to-section attachment F1.  This is kept for
  continuity but should not be treated as the single global success criterion,
  because floats, front matter, references, and run-in headings can make raw
  source-AST attachment unfair for a PDF-first reconstruction system.
- `section_attachment_body_no_float_f1`: body-only section attachment F1 over
  paragraphs, abstract, list items, display math, and algorithms; figure,
  table, and caption are excluded.
- `section_attachment_oracle_heading_flow_f1`: diagnostic upper-bound style
  score that ignores predicted parent edges, walks predicted reading order, and
  uses matched gold headings as active heading identities.
- `section_attachment_breakdown`: body / float / references / appendix
  attachment scores for diagnosing mixed failure modes.
- `reference_section_completeness`: matched bibliography item coverage.
- `float_caption_attachment_accuracy`: caption attachment to figure/table/algorithm.
- `generated_structure_validity`: internal consistency checks for the predicted structure.
- `macro_structure_score`: average of the available structure scores.

These metrics intentionally do not score raw OCR correctness, figure pixels,
table cell recognition, formula OCR quality, or exact journal-template
appearance. Those belong to separate OCR/CV or visual QA tracks. This schema
measures the recoverable structural layer that our system adds on top of MinerU.

## Compile And Layout QA

Compilation success and rendered page geometry are measured by a separate
rendered-output evaluator:

```bash
python3 tools/evaluate_rendered_output.py \
  --gold-pdf original.pdf \
  --pred-tex ours_generated.tex \
  --output rendered_eval.json
```

If a baseline already provides a PDF, pass it directly:

```bash
python3 tools/evaluate_rendered_output.py \
  --gold-pdf original.pdf \
  --pred-pdf baseline.pdf \
  --output rendered_eval.json
```

The report contains:

- `latex_compile_success`: engine, success flag, elapsed time, output PDF, and
  LaTeX error summary.
- `page_layout_similarity`: page-count score plus per-page geometry similarity.

The layout score renders PDFs to grayscale pages and compares ink bounding
boxes plus horizontal/vertical ink-density profiles.  It is a geometry proxy,
not an OCR or formula-recognition metric.
