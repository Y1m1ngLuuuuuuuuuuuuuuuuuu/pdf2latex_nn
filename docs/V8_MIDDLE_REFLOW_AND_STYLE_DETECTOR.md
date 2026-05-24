# V8 Middle Reflow And Style Detector

**Last updated**: 2026-05-24

This document records the current v8 reconstruction path and its parameters.
V8 is a new input normalization path; it does not mutate v7 JSON, does not build
a GNN view, and does not change graph schema.

## Goal

V8 starts from MinerU raw `middle.json` so that page-local reading order can be
fixed before MinerU/content-list paragraph merging is trusted.  The output is a
v7-compatible logical item list and then the existing `DocumentIR` /
`RenderTreeIR` / original-like renderer path.

## Production Path

```text
MinerU middle.json
  + optional content_list.json asset/caption sidecar
  + optional content_list_v7_styles.json style sidecar
  -> src/perception/mineru_v8_reflow.py
  -> content_list_v8_reflow_v1
  -> src/adapters/mineru_v8_document_ir.py
  -> stable DocumentIR
  -> src/reasoning/front_matter_extractor.py
  -> src/reasoning/v8_heading_style_stack.py
  -> src/reasoning/v8_render_tree.py
  -> src/generation/v8_style_detector.py
  -> StyleProfile
  -> src/generation/render_surface.py
  -> OriginalLikeIRLatexRenderer
  -> generated.tex / generated.pdf
```

The renderer still consumes full `DocumentIR` and `RenderTreeIR`; it never
renders from a GNN view.

## Main Runner

Use:

```bash
python3 scripts/pipeline/run_v8_layout_reconstruction.py \
  --doc-id <doc_id> \
  --middle-json <path/to/*_middle.json> \
  --content-list-json <path/to/*_content_list.json> \
  --style-content-list-json <path/to/*_content_list_v7_styles.json> \
  --source-tex <optional/source.tex> \
  --pdf <path/to/original.pdf> \
  --output-dir <run/output/dir> \
  --middle-block-source preproc_blocks \
  --compile-engine auto \
  --compile-timeout 180
```

### Parameters

| Parameter | Meaning |
| --- | --- |
| `--middle-json` | MinerU raw middle file. This is the primary source for v8 block/line geometry and reflow. |
| `--content-list-json` | Optional MinerU content list used as a sidecar for float/table/caption assets and conservative same-bbox text replacement. |
| `--style-content-list-json` | Optional styled v7 content list used only for font/style enrichment by bbox matching. It must not overwrite v8 text order. |
| `--source-tex` | Optional source sidecar for citation/bibliography and source float layout repair only. It is not used as inference content. |
| `--middle-block-source` | Defaults to `preproc_blocks`; `para_blocks` is diagnostic fallback only. |
| `--debug-page` | Writes page-local ordering diagnostics for one page. |
| `--no-resolve-citations` | Disables the standard citation/bibliography repair path. Default is to resolve citations. |

## Output Files

The v8 runner writes:

| File | Purpose |
| --- | --- |
| `<doc_id>_content_list_v8.json` | v8 logical item payload, v7-compatible but built from middle reflow. |
| `<doc_id>_v8_diagnostics.json` | reading-order and continuation diagnostics. |
| `document_ir.json` | stable backend input. |
| `front_matter_diag.json` | deterministic front matter grouping diagnostics. |
| `render_tree_ir.json` | v8 no-GNN render tree. |
| `style_profile.json` | StyleProfile consumed by renderer. |
| `v8_style_detector_diag.json` | font/line/paragraph/heading style detector evidence. |
| `generated.tex` | compilable LaTeX reconstruction. |
| `generated.pdf` | compiled PDF if compile is enabled. |
| `compile_report.json` | compile status. |
| `v8_layout_reconstruction_record.json` | run record and all major paths. |

## Style Detector

`src/generation/v8_style_detector.py` wraps the generic `StyleProfileExtractor`
and adds v8-specific evidence:

- body font size from style spans or v8 sidecar fallback;
- body line height from preserved middle `source_lines` line bboxes;
- paragraph spacing from same-page/same-column body block gaps;
- paragraph indent from column-local left-edge deltas;
- heading font size/alignment from the v8 heading style registry.

The detector writes normal `StyleProfile.renderer_options`, including:

```json
{
  "body_font_size": 9.0,
  "body_line_height": 10.8,
  "paragraph_spacing": 3.0,
  "paragraph_indent": 0.0,
  "source": "v8_style_detector"
}
```

Renderer-side use is intentionally narrow:

- page geometry now follows the source PDF by default; a Letter original renders
  as Letter and an A4 original renders as A4 instead of forcing all outputs to
  A4;
- `body_font_size` and `body_line_height` control `\AtBeginDocument{\fontsize...}`.
- `paragraph_spacing` controls `\parskip`.
- `paragraph_indent` controls `\parindent`.
- heading style commands still come from `RenderTreeIR.metadata.heading_style_registry`.

## Front Matter Phase 0

V8 uses `src/reasoning/front_matter_extractor.py` before body rendering:

```text
DocumentIR
  -> FrontMatterLineBuilder
  -> RuleBasedFrontMatterSequenceTagger
  -> FrontMatterIR
```

The extractor preserves and separates:

```text
document title
author block
affiliation-like lines
email / correspondence lines
front notes
abstract title
abstract body
```

This is not exact author-affiliation-email linking.  Its current job is:

```text
1. keep visible front matter from being lost;
2. keep title/author/affiliation/email out of the body heading tree;
3. render a stable original-like title/author/abstract surface.
```

Future precise parsing should be a separate FrontMatter Phase 1/2 entity and
linking model, not a GNN graph change.

## Wide Float Rendering

V8 keeps float grouping in the existing renderer/table-assets layer instead of
creating a separate v8-only float grouper:

- table fragments are still grouped by `src/generation/table_assets.py`;
- wide tables use source TeX `table*` hints when `--source-tex` is available,
  otherwise bbox width is the fallback;
- wide figures use visual bbox/group width because source figure layout hints
  are not extracted yet;
- a wide float inside a mixed-column body first exits `multicols`, renders as
  `figure*` / `table*`, then re-enters the two-column flow;
- ordinary column-width floats keep the existing `figure[H]` / `table[H]`
  behavior.

Starred floats intentionally use `[!t]` rather than `[H]`: LaTeX does not pin
double-column floats reliably with the `float` package's `H` placement. For
local environments without PyMuPDF, PDF crops fall back to Poppler
`pdftocairo`/`pdfinfo` so figures and tables do not degrade to TODO comments.

## 00050 Smoke Command

The current local 00050 smoke uses:

```bash
python3 scripts/pipeline/run_v8_layout_reconstruction.py \
  --doc-id 2501.00050 \
  --middle-json data/09_eval_reports/v8_reflow_20260523/inputs/2501.00050/auto/2501.00050_middle.json \
  --content-list-json data/09_eval_reports/v8_reflow_20260523/inputs/2501.00050/auto/2501.00050_content_list.json \
  --style-content-list-json data/02_mineru_outputs/mineru_output/2501.00050/auto/2501.00050_content_list_v7_styles.json \
  --source-tex data/03_tex_source_pool/2501.00050/aaai25.tex \
  --pdf data/09_eval_reports/post_audit_20260519/hardcase_intermediates/2501.00050/01_input/original.pdf \
  --output-dir data/09_eval_reports/v8_reflow_20260523/2501.00050_full_v8 \
  --debug-page 5 \
  --compile-engine auto \
  --compile-timeout 180
```

## Current Policy

- V8 is the default reconstruction direction for middle-based repair experiments.
- GNN remains available as an archived/experimental relation path, but v8 does
  not depend on GNN.
- Merge/GNN analysis tools from the previous branch are under `_obsolete`
  folders so they do not pollute the main path.
- Future MinerU upgrades should implement the same middle/content-list/style
  adapter contract rather than changing renderer internals.
- Current 00050 verification includes source-page-size output, starred wide
  floats, heading style registry rendering, and ordered `enumerate` recovery.
