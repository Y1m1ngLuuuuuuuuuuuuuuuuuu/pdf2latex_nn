# Table Engine Contract

**Last updated**: 2026-05-20

This contract defines how current and future table extraction engines plug into
PDF2LaTeX-NN without changing the GNN or generator architecture.

## Owner Modules

```text
src/generation/table_assets.py
src/generation/ir_renderers/tables.py
src/generation/latex_helpers.py
src/adapters/mineru_v7_document_ir.py
src/ir/schema.py
```

The current production fallback is visual: render the table from a PDF crop or
asset when structured reconstruction is unreliable. A future table engine can
replace that fallback by filling structured table payloads in `DocumentIR`.

## Conceptual Model

Tables have two different surfaces:

```text
visual table body    -> image/crop/structured table asset
semantic caption     -> text used for references and float pairing
```

The GNN should not embed raw table-cell OCR as ordinary body text. The graph
sees table/figure/algorithm as float proxies, usually represented by caption or
placeholder text. The renderer sees the full `DocumentIR` and can use visual or
structured table assets.

## Input Contract For A Table Engine

A table engine may attach these fields to a v7 record or `DocumentNode` metadata:

| Field | Meaning |
| --- | --- |
| `table_engine` | Engine name and version. |
| `table_confidence` | Confidence score in `[0, 1]`. |
| `table_bbox` / `bbox` | Table body bbox, preferably excluding caption. |
| `caption_bbox` | Caption bbox if separately known. |
| `caption_text` | Canonical table caption text. |
| `table_html` | HTML table representation. |
| `latex_tabular` | Ready-to-render tabular code, without float wrapper. |
| `cells` | Structured cell grid with row/col spans. |
| `crop_asset_path` | Existing table crop image. |
| `source_pdf_path` | PDF used for crop fallback. |
| `page_idx` | Page containing the table body. |

The table engine should preserve source ids:

```text
source_v7_ids: [ ... ]
```

These ids allow the renderer, diagnostics, and relation bridge to trace the
table back to the full v7 fact layer.

## Renderer Selection Order

The table renderer must choose the safest available representation:

1. Valid `latex_tabular` if it compiles and fits the requested float width.
2. Valid `table_html` converted to LaTeX if conversion is available.
3. Structured `cells` converted to `tabular` / `tabularx` / `resizebox`.
4. `crop_asset_path` or PDF bbox crop fallback.
5. Placeholder float with caption and source-node trace.

When a structured table overflows, the renderer may fall back to crop rather
than emitting broken LaTeX.

## Caption Contract

Captions are structural text and should be emitted exactly once:

```latex
\begin{table}[H]
  ...
  \caption{...}
  \label{tab:...}
\end{table}
```

Rules:

1. Table caption text must not also render as a normal paragraph.
2. If the table body is missing but the caption is detected, emit a placeholder
   table group rather than losing the caption.
3. Duplicate caption suppression may remove only same type + same number + same
   normalized text duplicates.
4. Table IV / Table S1 / Tab. 2 styles should keep their visible numbering if
   LaTeX cannot safely own the counter.

## GNN Contract

For graph learning:

```text
table body OCR       -> not paragraph semantics
table proxy          -> graph-visible float proxy
caption text         -> proxy semantic text
MERGE with table     -> blocked
float -> body message passing -> masked or strongly limited
skip-over-float edge -> allowed for paragraph continuation candidates
```

Changing the table engine does not require GNN retraining if:

- source v7 ids stay stable;
- graph-visible proxy text/type stays compatible;
- node feature dimensions do not change.

Retraining or graph rebuild is required if:

- table proxy construction changes;
- table/caption source-node ids change;
- new table features are added to graph tensors;
- float masking policy changes.

## Layout Contract

The renderer decides float width from bbox geometry:

| Visual width | Render mode |
| --- | --- |
| inside one column | `table` or local column table |
| spanning most page width | `table*` or temporary single-column region |
| narrow paired table/figure | keep in column/minipage if grouping supports it |

Large tables should not be forced into one column. Small tables should not be
expanded to page width unless the original bbox is full-width.

## Integration Checklist

When adding a new table engine:

1. Add adapter mapping from engine output to `DocumentNode.metadata`.
2. Add tests for one structured table, one crop fallback, one missing-body
   caption-only table, and one wide table.
3. Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_ir_renderer_registry.py \
  tests/test_generation_style_citations.py \
  tests/test_postprocess_renderer.py
```

4. Verify no graph schema changes unless intentionally planned.

