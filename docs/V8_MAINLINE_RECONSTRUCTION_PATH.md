# V8 Mainline Reconstruction Path

> Historical/current implementation background. The overall project is now
> organized around stable interfaces rather than a single v8 paper story. See
> `docs/INTERFACE_DESIGN_CURRENT_20260601.md` before promoting any v8-specific
> behavior into a paper claim.

**Last updated**: 2026-05-26

This document is the single current production path contract for PDF2LaTeX-NN.
If another document or script appears to imply a second default path, this file
and `docs/PROJECT_SOURCE_OF_TRUTH.md` take precedence.

## Status

The maintained default reconstruction path is:

```text
MinerU middle.json
  + optional content_list.json asset/caption sidecar
  + optional content_list_v7_styles.json style sidecar
  -> v8 middle reflow
  -> DocumentIR
  -> FrontMatterIR
  -> v8 heading style stack
  -> RenderTreeIR
  -> v8 style detector / StyleProfile
  -> OriginalLikeIRLatexRenderer
  -> generated.tex / generated.pdf
```

Default reconstruction does **not** load a GNN checkpoint and does **not** render
from the GNN view.

## Canonical Entrypoint

Use only this script for the current v8 reconstruction path:

```bash
python3 scripts/pipeline/run_v8_layout_reconstruction.py \
  --doc-id <doc_id> \
  --middle-json <path/to/*_middle.json> \
  --content-list-json <path/to/*_content_list.json> \
  --style-content-list-json <path/to/*_content_list_v7_styles.json> \
  --pdf <path/to/original_or_compiled_input.pdf> \
  --output-dir data/09_eval_reports/v8_reflow_<YYYYMMDD>/<doc_id>_<run_tag> \
  --middle-block-source preproc_blocks \
  --compile-engine auto \
  --compile-timeout 180
```

`--source-tex` is optional and is used only as a sidecar for citation,
bibliography, and source-float-layout repair. It is not the default inference
source and must not be used to recover source-level TeX AST.

## Input Ownership

| Input | Role |
| --- | --- |
| `middle.json` | Primary body text, line/block geometry, page-local evidence, and v8 reading-order repair source. |
| `content_list.json` | Sidecar for float/table/image/algorithm assets, captions, HTML/table metadata, and conservative title/caption text. |
| `content_list_v7_styles.json` | Optional sidecar for font/style spans and bbox-matched typography evidence. |
| source PDF | Page size, compile/layout comparison, crop coordinates, and rendered-output reference. |
| source TeX | Optional citation/bibliography/source-float sidecar only. |

Body paragraph text must come from v8/middle canonical text. `content_list`
and PyMuPDF spans must not replace body text unless the text strictly aligns to
the v8 canonical node text.

## Output Contract

Each v8 run directory should contain:

```text
<doc_id>_content_list_v8.json
<doc_id>_v8_diagnostics.json
document_ir.json
front_matter_diag.json
render_tree_ir.json
style_profile.json
v8_style_detector_diag.json
generated.tex
generated.pdf
compile_report.json
v8_layout_reconstruction_record.json
```

Use this naming pattern:

```text
data/09_eval_reports/v8_reflow_<YYYYMMDD>/<doc_id>_<short_run_tag>/
```

Examples:

```text
data/09_eval_reports/v8_reflow_20260523/2501.00050_full_v8_textfix_floatskip_20260524/
```

Do not place current v8 outputs in root-level `e2e_outputs/`, `local_outputs/`,
or a free-form smoke folder.

## Merge And Reading-Order Policy

V8 repairs reading order before paragraph continuation materialization.
Current deterministic continuation reasons are:

```text
same_column_open_sentence
cross_column_open_sentence
cross_page_open_sentence
float_skip_continuation
```

The narrow `float_skip_continuation` rule is allowed only when:

```text
previous body text is open-ended
+ the gap contains only float/table/chart/code-like asset blocks
+ current body text begins with a table/figure/equation reference or citation-like parenthetical
+ no new heading intervenes
+ reading order is forward
```

This is a v8 reflow decision, not a GNN prediction.

## Heading And Style Policy

The production heading tree is built by:

```text
front matter negative mask
+ document-local heading style registry
+ stack skeleton
```

GNN PARENT edges are not the production authority for heading hierarchy.
The heading style registry may render top-level headings centered and lower-level
headings left-aligned when the document evidence supports that mapping.

## Archived GNN Interface

The GNN branch is preserved as an archived research interface, not as a current
production option:

```text
v7 or future v8-derived graph-visible view
  -> graph.pt
  -> MERGE / PARENT_CHILD / NONE logits
  -> constrained decoder or diagnostic report
```

Rules:

1. GNN view is never a renderer source.
2. Any GNN edge must be bridged back to exact source ids before affecting
   `RenderTreeIR`.
3. The default v8 command above must remain runnable without a checkpoint.
4. Future learned-merge experiments require a new explicit research branch and
   must not silently alter this v8 deterministic path.

The old middle-derived continuation JSON route is archived in:

```text
docs/_archive/v8_gnn_merge_experiments_20260526/V8_ATOMIC_MERGE_GNN_ROUTE.md
```

Its tools are archived under:

```text
tools/_archive/v8_gnn_merge_experiments_20260526/v8_atomic/
```

Do not import archived v8 atomic MERGE tools from the default renderer.

## Selected200 Final Control Result

The 2026-05-26 selected200 rerun closed the learned MERGE route as a production
candidate and promoted deterministic v8 as the mainline. Full tables are in:

```text
data/09_eval_reports/v8_mainline_final_20260526/V8_MAINLINE_FINAL_REPORT.md
data/09_eval_reports/v8_mainline_final_20260526/v8_mainline_result_table.csv
```

Summary:

| variant | status | docs | macro | body coverage | ordered body coverage | order inversion | missing merge | wrong merge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contentlist_direct | reference baseline | 200 | 0.846652 | 0.755339 | 0.708756 | 0.065588 | 0.112875 | 0.023076 |
| v8_layout_batch | production mainline | 200 | 0.848726 | 0.753216 | 0.712951 | 0.056164 | 0.114064 | 0.022445 |
| v8_contentlist_merge_hint | deterministic refinement candidate | 200 | 0.849746 | 0.756806 | 0.716568 | 0.056328 | 0.109810 | 0.022324 |
| ranker_recall_cap5 | archived learned experiment | 200 | N/A | 0.753465 | 0.713364 | 0.055885 | 0.114543 | 0.022536 |
| selector_keep_mid_recall | archived learned experiment | 200 | N/A | 0.748780 | 0.709018 | 0.055806 | 0.132749 | 0.021947 |

Conclusion: learned branches can move the output, but they did not provide a
stable improvement under the wrong-merge safety constraint. Production remains
v8 deterministic.

## Paragraph Order Metric Layers

Paragraph/source audits now report three layers side by side. None of these
replace the older metrics.

| Metric family | Purpose |
| --- | --- |
| `body_source_coverage_rate` | Legacy body-level content coverage after removing source-only TeX implementation fragments. This is content recall and is not strongly order-sensitive. |
| `body_ordered_source_coverage_rate` / `body_source_order_inversion_rate` | Body-level order-sensitive coverage using pairwise source/generated paragraph inversions. Useful but can be amplified by one badly displaced paragraph. |
| `visible_prose_*` | Type-aware ordinary-prose order metrics. These exclude front matter, abstract title, captions, floats, references, display math, formula-only blocks, URL/metadata/note blocks, OCR artifacts, and no-render style fragments. |

The visible-prose layer adds:

```text
visible_prose_ordered_coverage_rate
visible_prose_order_inversion_rate
adjacent_prose_inversion_rate
displaced_prose_paragraph_rate_010
displaced_prose_paragraph_rate_015
visible_prose_lis_disorder_rate
```

Interpretation:

- `visible_prose_order_inversion_rate` answers how much ordinary prose remains
  globally out of order.
- `adjacent_prose_inversion_rate` checks only adjacent visible-prose paragraph
  pairs, so a single misplaced paragraph does not explode the score.
- `displaced_prose_paragraph_rate_*` estimates how many prose paragraphs are
  substantially early/late in normalized document order.
- `visible_prose_lis_disorder_rate` estimates the minimum fraction of matched
  prose paragraphs that must move to make the generated sequence monotonic.

Matching is also type-aware in the visible-prose layer: ordinary source prose is
matched only to generated ordinary prose/list text, captions to captions,
reference items to reference items, abstract to abstract, and display/formula
content is excluded from ordinary-prose order.

## Legacy Entrypoints

The following scripts are maintained for historical v7/GNN evaluation,
ablation, and comparison only:

```text
scripts/pipeline/batch_visual_qa_inference.py
scripts/pipeline/run_e2e_inference.py
scripts/pipeline/step5_generate_tex.py
scripts/pipeline/run_current_full_eval_suite.py
```

They are not the default reconstruction path unless an experiment explicitly
states that it is using the optional v7/GNN branch.
