# Generator Logic Audit 2026-05-17

This note freezes the current generator architecture after the v7/GNN-view
decoupling fixes.  It is meant to answer one question: which module owns which
part of reconstruction, and which path is allowed to produce final E2E PDFs?

## Current Principle

The production generator is full-v7 first.

```text
content_list_v7_styles.json  (complete facts: text, bbox, styles, floats, notes)
  -> DocumentIR              (stable full-document IR)
  + GNN predicted edges      (only structural relation hints on GNN-view ids)
  -> relation bridge         (gnn_idx -> exact v7 node id sequence)
  -> RenderTreeIR            (decoder tree expressed in full-v7 source ids)
  -> OriginalLikeIRLatexRenderer
  -> compilable LaTeX / PDF
```

The GNN view is not a render source.  It exists only to let the model predict
local relations.  The final renderer must always return to complete v7
records, otherwise title, authors, header/footer, footnotes, floats and full
style spans disappear or get misclassified as noise.

## Production Entrypoints

| Entrypoint | Current role |
| --- | --- |
| `scripts/pipeline/step5_generate_tex.py` | Single-document canonical inference/generation entry. Defaults to `--renderer ir` and `--heading-skeleton-mode stack`. |
| `scripts/pipeline/step5_run_inference.py` | Compatibility wrapper only. It now forwards `--renderer ir` and `--heading-skeleton-mode stack` to `step5_generate_tex.py`. |
| `scripts/pipeline/batch_visual_qa_inference.py` | Batch E2E visual QA path. Defaults to the IR renderer and stack heading skeleton. |
| `scripts/pipeline/run_e2e_inference.py` | Full E2E front-end + inference + compile path. Defaults to the IR renderer and stack heading skeleton. |
| `scripts/pipeline/run_m05_e2e_comparison.py` | Locked-model comparison wrapper. Defaults to the IR renderer and stack heading skeleton. |

`src.generation.latex_renderer.render_latex_document()` is a legacy/debug
surface for historical tests and low-level helpers only.  Current production
scripts no longer accept `--renderer tree`; they expose only `--renderer ir`.

## Graph-to-v7 Bridge Contract

The graph stores the exact GNN node to v7 source id mapping:

```text
data.gnn_to_v7_id
data.gnn_to_v7_ids
data.gnn_to_v7_index
```

`scripts/pipeline/step5_generate_tex.py::select_records_for_graph()` must use
this mapping as source of truth.  Node count equality is insufficient because a
reordered rebuilt GNN view can silently attach logits and `edge_index` to the
wrong v7 blocks.

Current behavior:

1. Rebuild the GNN view from full v7 using the graph's `micro_fusion_applied`
   mode.
2. Compare rebuilt `_v7_source_node_ids` with graph-side `gnn_to_v7_ids`.
3. If exact sequence matches, continue.
4. Try the opposite micro-fusion setting only as a compatibility fallback.
5. If neither exact sequence matches, fail fast.

This protects section/caption/merge decisions from silent mapping drift.

## Generator Module Map

### Public surface

`src/generation/render_surface.py`

- Computes missing `StyleProfile`, `CitationResolution`, and
  `SourceFloatLayout`.
- Delegates to `OriginalLikeIRLatexRenderer`.
- This is the preferred API for scripts.

### Full v7 to IR adapter

`src/adapters/mineru_v7_document_ir.py`

- Converts v7 JSON to `DocumentIR`.
- Preserves raw MinerU metadata in node metadata.
- Maps MinerU types into stable `BlockType`.
- Important mappings:
  - `page_header`, `page_footer`, `page_number`, `noise_layer` ->
    `BlockType.HEADER_FOOTER`
  - `page_footnote`, `image_footnote`, `table_footnote`,
    `chart_footnote` -> `BlockType.FOOTNOTE`
  - `aside_text`, `margin_note`, `sidenote`, `sidebar` ->
    `BlockType.MARGIN_NOTE`
  - image/table/code captions fold into their float block type so the float
    renderer can group them.
  - `ref_text` maps to `BlockType.REFERENCE`.

This is where front-end annotation is translated into generator behavior.  Do
not use GNN filtering to remove data needed by this adapter.

### Global style/profile

`src/generation/style_profile.py`

Owns global and statistical layout decisions:

- paper size and geometry
- body font size and font clusters
- role-specific font/spacing clusters
- single/two/mixed column profile
- references column profile
- header/footer statistics
- title/author/abstract/front-matter style hints

Header/footer rendering only activates when enough repeated
`BlockType.HEADER_FOOTER` evidence exists.  A single accidental header/footer
node is kept out of body style statistics but does not create a global
`fancyhdr` setup.

### Owner renderer

`src/generation/ir_renderer.py::OriginalLikeIRLatexRenderer`

This is still the main owner.  It maintains document-level state that small
renderers share:

- active style profile
- active footnote/margin-note context
- source float layout sidecar
- cross-reference registry
- rendered float group/caption dedup sets
- mixed-column scope
- global preamble and package list

It also owns the mature helper logic that has not yet been split out:

- `_render_children()` ordering and list grouping
- float placement and dedup
- table/figure/algorithm crop fallback
- references rendering and reference label stripping
- citation/cross-reference rewrites
- note anchoring
- mixed-column wrappers
- heading prefix policy
- partial-span safety

This file is large, but it is intentionally still the coordination layer.  The
registry split below should reduce local complexity without duplicating this
document-level state.

### Role/block renderer registry

`src/generation/ir_renderers/`

The registry dispatches by `RenderRole` first, then by source `BlockType`.
Renderers are small and should call owner helpers when document-level state is
needed.

| Renderer | Owns |
| --- | --- |
| `front_matter.py` | document title, author block, abstract, TOC placeholder |
| `headings.py` | section/subsection/subsubsection command emission through the owner heading-prefix policy |
| `text.py` | paragraph, caption, raw LaTeX, fallback text, span/citation-safe text rendering |
| `math.py` | inline/display equation, algorithm/code role dispatch |
| `figures.py` | figure role and figure `BlockType`, delegating crop/grouping to owner |
| `tables.py` | table role and table `BlockType`, delegating crop/grouping to owner |
| `lists.py` | itemize/enumerate role rendering, marker stripping |
| `references.py` | bibliography/reference section role rendering |
| `notes.py` | footnote/margin-note role; actual anchor logic is owner `_NoteContext` |

### Legacy renderer

`src/generation/latex_renderer.py`

This file remains for:

- low-level escaping/math/list helper functions
- legacy TreeDecoder tests
- regression debugging

It is not the production generator.  New output-quality logic should go through
`DocumentIR + RenderTreeIR + OriginalLikeIRLatexRenderer`.

## Preserved Feature Modules

### Header/footer

Path:

```text
v7 layout_layer/type/role
  -> BlockType.HEADER_FOOTER
  -> StyleProfileExtractor._extract_header_footer_style()
  -> renderer_options["header_footer"]
  -> _header_footer_commands()
  -> fancyhdr preamble
```

Tests:

- `test_style_profile_keeps_header_footer_out_of_body_style`
- `test_style_profile_infers_global_header_footer_and_renderer_emits_fancyhdr`

Design note: page headers/footers are global statistical features, not normal
body nodes.  They should not go into GNN body reasoning, but they remain in
full v7/DocumentIR for profile extraction.

### Footnotes and margin notes

Path:

```text
v7 footnote/aside/margin role
  -> BlockType.FOOTNOTE / BlockType.MARGIN_NOTE
  -> _NoteContext.from_document()
  -> explicit marker / superscript marker / nearest-anchor resolution
  -> \footnote{}, \marginpar{}, or \footnotetext{}
```

Tests:

- `test_original_like_ir_renderer_anchors_footnotes_and_margin_notes`
- `test_original_like_ir_renderer_replaces_superscript_marker_with_matching_footnote`
- `test_original_like_ir_renderer_keeps_unanchored_footnote_as_footnotetext`

### Figures/tables/algorithms

Path:

```text
v7 full float records
  -> DocumentIR float nodes
  -> TreeDecoder/owner float grouping and caption consumption
  -> crop fallback from source PDF or existing MinerU asset
  -> [H] figure/table/algorithm slot, with cross-ref label where possible
```

Float/caption duplication protections live in both TreeDecoder and the IR
renderer.  Figures/tables are allowed in the GNN view only as float proxies, not
as noisy body text.

### Citations/references

Path:

```text
DocumentIR reference nodes + optional source TeX
  -> CitationResolver
  -> body citation marker rewrite
  -> bibliography/reference rendering
```

Fallback behavior still emits synthetic `ref_i` keys if source keys are not
recoverable.  Reference labels from OCR are stripped before `\bibitem` text.

### Heading prefix policy

Current rendering policy:

- If the visual heading has no prefix, render starred (`\section*{Introduction}`)
  so LaTeX does not invent a number.
- If the prefix is ordinary decimal numbering compatible with the role, strip
  it and let LaTeX own numbering (`1.1 Method` -> `\subsection{Method}`).
- If the prefix style is non-default for that command (`0.1`, Roman, alpha,
  Chinese, custom template), preserve the visible prefix and render starred.
- Appendix headings keep the appendix transition behavior and strip only the
  appendix marker.

This prevents the earlier failures where unnumbered headings gained fake
numbers, while custom visible prefixes were lost.

## Current Risk Points

1. `ir_renderer.py` is still the coordination center.  The registry split is
   real but not complete; avoid duplicating owner state in role files.
2. Header/footer and footnote support depends on v7 type/layer/role annotation.
   If MinerU or preprocessing labels these as plain text, the renderer cannot
   recover them reliably downstream.
3. `build_document_ir_from_graph_records()` is disabled at runtime.  Production
   rendering must use `build_document_ir_from_full_v7()`.
4. E2E scripts that disable crop rendering to save disk may visually omit
   assets; single-document generation defaults to crop rendering.
5. Generator improvements do not require relabel/retrain unless the feature
   must enter `.pt` node/edge tensors.

## Audit Checklist For Future Changes

Before accepting a generator change, verify:

1. The E2E command uses `--renderer ir`.
2. The decoder uses `--heading-skeleton-mode stack` unless intentionally
   benchmarking legacy mode.
3. Graph records are bridged by exact `gnn_to_v7_ids`, not by count.
4. Full v7 data is converted to `DocumentIR` for rendering.
5. Metadata/noise filtering only affects GNN view, not full v7 render facts.
6. Header/footer, footnote/margin-note, float and references tests still pass.
7. Any new block type is mapped in `mineru_v7_document_ir.py` before renderer
   logic is added.
