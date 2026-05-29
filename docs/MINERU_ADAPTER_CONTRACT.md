# MinerU Adapter Contract

**Last updated**: 2026-05-24

This contract defines the boundary between MinerU-style document extraction and
the maintained PDF2LaTeX-NN pipeline. MinerU is a replaceable frontend. The
rest of the system should depend on `DocumentIR`, not on MinerU internals.

## Owner Modules

```text
src/perception/content_resolver.py
src/adapters/mineru_v7_document_ir.py
src/perception/mineru_v8_reflow.py
src/adapters/mineru_v8_document_ir.py
src/ir/schema.py
src/ir/validators.py
src/pipeline/v7_contract.py
```

`mineru_v8_reflow.py` is the current default frontend normalizer. It starts
from MinerU `middle.json`, repairs reading order, and emits a v7-compatible v8
logical payload. `mineru_v8_document_ir.py` converts that payload to
`DocumentIR`.

`content_resolver.py`, `mineru_v7_document_ir.py`, and `v7_contract.py` remain
for the optional v7/GNN relation branch and historical graph compatibility.

## Input Contract

The current default frontend inputs are:

```text
middle.json                         primary body text, line/block geometry
content_list.json                   asset/caption/table sidecar
content_list_v7_styles.json         optional style sidecar
```

The current default normalized artifact is:

```text
content_list_v8.json
```

For the optional v7/GNN relation branch, the canonical frontend artifact remains:

```text
content_list_v7_styles.json
```

Both v8 and v7 adapters produce the same `DocumentIR` contract. A normalized
record may contain extra fields, but these fields are the stable minimum:

| Field | Meaning | Required |
| --- | --- | --- |
| `_v7_node_id` / `id` / `index` | Stable source id used for graph and render mapping. | yes |
| `type` / `canonical_type` | Raw and normalized block type. | yes |
| `text` / `text_for_embedding` | Visible text or semantic proxy text. | yes for text-like nodes |
| `page_idx` | Zero-based page index. | yes |
| `bbox` | Source bounding box `[x0, y0, x1, y1]`. | yes when available |
| `page_width`, `page_height` | Page geometry in the same coordinate space as bbox. | recommended |
| `global_order` / `layout_flow_order` | Reading-order hints. | recommended |
| `layout_layer` | `main_text_flow`, `metadata_layer`, `float_layer`, `annotation_layer`, `noise_layer`, etc. | recommended |
| `layout_role` | Fine role such as `body_text`, `heading`, `author`, `page_header`, `figure_caption`. | recommended |
| `style_spans` | Font, size, bold, italic, inline math/code spans. | recommended |
| `asset_path` / `crop_bbox` / `table_html` | Visual or structured float/table payload. | optional |
| `metadata` | Engine version, confidence, role hints, provenance. | optional |

Unknown fields must be preserved in node metadata when practical. They must not
be silently dropped if they may help future renderers.

## Output Contract

The adapter outputs:

```text
DocumentIR
  pages: list[PageIR]
  nodes: list[DocumentNode]
  reading_order: list[node_id]
  metadata: dict
```

Required invariants:

1. Every `DocumentNode.node_id` must be stable inside the document.
2. Every node must retain source provenance through `source_refs` or metadata.
3. Page indices and bboxes must stay in the frontend coordinate system.
4. Style spans must be attached to the node that owns the visible text.
5. Metadata, front matter, footnotes, figures, tables, captions, references,
   headers, and page numbers remain in `DocumentIR`.
6. "Excluded from GNN" never means "deleted from `DocumentIR`".

## Full-v7 First Rule

The full normalized frontend payload and `DocumentIR` are the complete fact
layer. In the current default path this is v8. In optional relation experiments
this can still be v7. The GNN view is a filtered or proxied training view built
later by `GNNViewAdapter`.

```text
full normalized fact layer
  -> DocumentIR for generation/evaluation
  -> GNNViewAdapter for graph learning
```

Production rendering must use:

```text
DocumentIR + RenderTreeIR + StyleProfile
```

It must not render directly from GNN records.

## Type Mapping

The adapter maps MinerU/v7 types into `BlockType` conservatively:

| Frontend type/role | `BlockType` | Notes |
| --- | --- | --- |
| title, heading, section title | `TITLE` | Body headings are not the same as document title. |
| text, paragraph | `TEXT` | Main body paragraph-like content. |
| list item | `LIST_ITEM` | Preserve markers if available. |
| equation, display_formula, equation_interline | `EQUATION` | Formula OCR fragments must not become ordinary paragraphs. |
| image, figure | `FIGURE` | Prefer crop/asset metadata over OCR text. |
| table | `TABLE` | Structured cells optional; crop fallback required. |
| algorithm | `ALGORITHM` | Current fallback is crop/image or verbatim-like block. |
| caption | `CAPTION` | Caption text is semantic, float body is visual. |
| reference item | `REFERENCE_ITEM` | Keep boundaries where possible. |
| footnote, margin note | `FOOTNOTE` / `MARGIN_NOTE` | Annotation layer, not body paragraph. |

If MinerU adds new types, extend the adapter mapping first. Do not add ad hoc
type checks in graph builder or renderer before updating this contract.

## Upgrade Procedure

When MinerU changes version or output format:

1. Add a small sample JSON fixture from the new frontend.
2. Update only `content_resolver.py` and `mineru_v7_document_ir.py` if the
   `DocumentIR` contract can remain unchanged.
3. Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_ir_contracts.py \
  tests/test_mineru_v7_document_ir_adapter.py \
  tests/test_v7_contract.py
```

4. If node ids, node granularity, feature fields, or GNN-visible filtering
   change, rebuild graphs and relabel before retraining.
5. If only metadata fields improve and `DocumentIR` stays compatible, no GNN
   change is required.

## Non-Goals

- The adapter does not infer TeX labels.
- The adapter does not run GNN inference.
- The adapter does not decide final section parentage.
- The adapter does not delete content to improve GNN training.
- The adapter does not recover the author's source-level TeX AST.
