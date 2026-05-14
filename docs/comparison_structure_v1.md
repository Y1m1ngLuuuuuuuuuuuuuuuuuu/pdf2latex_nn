# Comparison Structure V1

**Last updated**: 2026-05-14

This schema is the neutral layer used to compare outputs from different
document-to-text systems.  It is intentionally coarser than the production IR:
the goal is to compare document structure, not OCR quality.

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
- `paragraph_merge_f1`: paragraph/list/reference/caption block boundary F1.
- `section_attachment_f1`: content-to-section attachment F1.
- `reference_section_completeness`: matched bibliography item coverage.
- `float_caption_attachment_accuracy`: caption attachment to figure/table/algorithm.
- `generated_structure_validity`: internal consistency checks for the predicted structure.
- `macro_structure_score`: average of the available structure scores.

These metrics intentionally do not score raw OCR correctness, figure pixels,
table cell recognition, formula OCR quality, or visual page-layout similarity.
Those belong to separate OCR/CV or visual QA tracks.  This schema measures the
structural layer that our system adds on top of MinerU.

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
