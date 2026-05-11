# Project Source Of Truth

**Last updated**: 2026-05-11

This project should be managed as a source repository plus external runtime artifacts. The source repository is for code and reproducibility metadata. AutoDL is for data, feature caches, training, and generated outputs.

## Canonical Repository

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

The intended code flow is:

```text
local source edits -> GitHub -> AutoDL git pull
```

Avoid blind local-to-server recursive overwrites. If a direct sync is needed during recovery, use `upload_to_server.sh`, which only syncs lightweight source paths.

## Local Machine

Local root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Responsibilities:

```text
code editing
documentation
small syntax checks
manifest inspection
GitHub commits and review
```

Do not run heavy training locally. The local Python/conda environment is only for code sanity checks.

Active source skeleton:

```text
src/
configs/
scripts/pipeline/
tests/
docs/
data/00_manifests/
data/01_raw_pdfs/
data/02_mineru_outputs/
data/03_tex_sources/
data/03_tex_source_pool/
data/04_ground_truth_ir/
data/05_observed_ir/
data/06_graph_features_v7/
data/07_predicted_ir/
data/08_output_latex/
data/09_eval_reports/
```

Current maintained docs:

```text
docs/PROJECT_SOURCE_OF_TRUTH.md      repository / AutoDL boundary
docs/PROJECT_OVERVIEW.md             project framework, methodology, implementation summary
docs/ablation_plan_v2.md             reproducible GNN and feature ablation protocol
docs/frontend_backend_contract_v1.md decoupled PDF/TeX/GNN/generator interfaces
docs/feature_schema_v0.md            PDF IR and tensor feature contract
docs/ground_truth_labeling_v0.md     TeX-to-PDF alignment and GNN label contract
docs/v7_training_and_monitoring.md   v7 batch, strict PDF/TeX pairing, training commands
docs/LOCAL_CONFIGURATION.md          local private configuration notes
```

Production training samples must come from a compile-closed pair:

```text
TeX source -> latexmk compiled PDF in data/01_raw_pdfs -> MinerU/GNN labels
```

The v7 builders should prefer compile `accepted.jsonl` manifests and
`data/03_tex_sources` over implicit same-ID pairing against
`data/03_tex_source_pool`. `data/03_tex_source_pool` is kept as a legacy/source
staging area, not as the authoritative TeX side for new production runs.

Legacy reference material is stored outside the active skeleton:

```text
_legacy_reference/2026_04_29_pre_rebuild/
```

This directory is for lookup and migration only. It is ignored by Git and should not be committed.

## AutoDL Server

Remote project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Responsibilities:

```text
large downloads
PDF/source extraction
MinerU processing
feature generation
training
large QA batches
checkpoints and runtime artifacts
```

AutoDL has Miniconda and the working conda environment is:

```text
pdf2latex
```

## Secrets

Secrets stay outside Git.

Current private files:

```text
local:  /Users/lu/.kaggle/access_token
local:  .env.local
AutoDL: /root/.kaggle/access_token
AutoDL: /root/autodl-tmp/pdf2latex_nn/.env.autodl
```

The committed `.env.example` records variable names only.

## Commit To Git

Commit:

```text
README.md
目标.md
src/
scripts/
tools/
tests/
docs/
requirements.txt
requirements_server.txt
verify_environment.py
config*.yaml
.env.example
.gitignore
small manifests and sample id lists
```

Do not commit:

```text
data/
artifacts/
archive/
server_cache_backup/
paper_artifacts/
reports*/
demo_*/
logs*/
.model_cache/
.venv*/
_legacy_reference/
checkpoints
bulk PDFs
bulk arXiv sources
MinerU outputs
feature caches
private env files
Kaggle tokens
SSH passwords
```

## Data Rebuild Rule

Every dataset or experiment batch should have a manifest:

```text
dataset name
source
download command
sample ids
file counts
hashes where useful
generation timestamp
code commit hash
output root
```

The manifest can be committed. The bulk data should stay on AutoDL or external storage.

## Current Pipeline Contracts

The active architecture is interface-first. Heavy front-end extraction,
truth-label generation, GNN inference, TreeDecoder, and LaTeX rendering must
communicate through stable IR boundaries rather than importing each other's
private implementation details. The canonical interface document is:

```text
docs/frontend_backend_contract_v1.md
```

Code-level IR contracts live in:

```text
src/ir/schema.py
src/ir/serialization.py
src/ir/validators.py
```

The active v7-to-IR adapter lives in:

```text
src/adapters/mineru_v7_document_ir.py
scripts/pipeline/convert_v7_to_document_ir.py
```

v7 content JSON is a frontend-private format. `DocumentIR` is the stable
frontend/backend boundary. Renderers, style extractors, citation repair, and
future journal-template generators should consume `DocumentIR`, not raw
`*_content_list_v7_styles.json`.

The stable cross-module artifacts are:

```text
DocumentIR          PDF frontend output
GraphInput          unlabeled GNN input tensor reference
GraphLabels         TeX-derived truth labels
PredictedRelations  model inference output
RenderTreeIR        TreeDecoder output for generation
StyleProfile        rendering mode/style template
CitationResolution  citation/bibliography repair sidecar
```

Current implementation may continue to write legacy `.pt` graph files while the
IR layer is adopted incrementally. New code should add sidecar JSON manifests
instead of changing running production `.pt` schemas in place.

Generation-side original-layout reconstruction now starts from explicit sidecar
modules instead of embedding style and citation rules inside TreeDecoder:

```text
src/generation/style_profile.py  DocumentIR -> StyleProfile
src/generation/citations.py      DocumentIR -> CitationResolution
src/generation/ir_renderer.py    DocumentIR + RenderTreeIR + StyleProfile -> .tex
src/generation/render_surface.py canonical one-call generation entrypoint
```

`src/generation/latex_renderer.py` is no longer a production document render
surface. It is retained as a low-level helper module for escaping, math,
algorithm, table, and figure block rendering; its legacy `render_latex_document`
function should not be used by new code.

`StyleProfileExtractor` owns global document appearance such as page size,
margins, column count/gap, body font, title font clusters, paragraph spacing,
and bibliography style. `CitationResolver` owns `[1]`/`[1-3]` citation repair
and strips OCR reference labels before emitting `\bibitem`. These two modules
are additive and do not change the active AutoDL extraction process.

The original-like backend now covers the first reconstruction layer:

```text
1. page/layout profile:
   page size, margin ratios, text area, column count/gap, column mode
2. role-level typography:
   body/heading/list/math/bibliography font size and spacing estimates
   header/footer is tracked separately and does not pollute body style
   stable page headers, footers, and page numbers are inferred statistically
   from repeated edge nodes and rendered globally with fancyhdr; OCR edge
   text is never replayed page-by-page
   document-level font clusters are derived from StyleSpan font_size/font_name
   weights and exposed as `font_clusters` / `role_font_clusters`
   PDF font names are canonicalized into font classes and optional fontspec
   fallbacks (`TeX Gyre Termes/Heros/Cursor`, `Latin Modern`). Recognition
   uses embedded PDF span metadata and does not require installed fonts;
   exact font replay is opt-in because it requires XeLaTeX/LuaLaTeX and local
   font availability.
3. local spans:
   bold, italic, inline code, inline math, and citation-aware span rendering
   local font-family changes, font-size changes, and super/subscript rendering
   must be driven by StyleSpan font/bbox/size features, not content keywords
4. citations/references:
   numeric marker repair, real key passthrough from reference_items, OCR
   reference-label stripping, author-year key/optional-label inference,
   author-year body-marker repair, numeric range expansion with `cite`
   package compression, and thebibliography rendering
5. notes:
   explicit `footnote` / `margin_note` nodes are removed from ordinary body
   flow, anchored to the nearest preceding source node, and rendered with
   `\footnote{...}` / `\marginpar{...}`. Unanchored footnotes fall back to
   `\footnotetext{...}`; generic bottom-edge text is not guessed as a footnote.
6. table/figure fallback:
   adjacent MinerU table fragments are grouped; only the primary group node is
   rendered. Batch rendering defaults to a placeholder to avoid large image
   assets; union-bbox PDF crops require an explicit render flag/config. Figure
   nodes use the same bbox PDF crop fallback when crop output is enabled.
7. render-tree safety:
   IR renderer sorts siblings by source reading_index before rendering, so MST
   insertion order cannot scramble the body flow. Consecutive list-like
   siblings are grouped into itemize/enumerate, and display equations or other
   structural blocks between list items stay inside the active list item.
8. type dispatch:
   abstract, table, figure, algorithm, code, toc placeholder, display equation,
   inline math, footnote, margin note, references, and raw LaTeX roles have
   explicit renderer branches instead of falling through to ordinary paragraphs.
9. OCR/math safety:
   lone symbol-font braces, unicode math glyphs, bare inline TeX math commands
   inside text spans, single equation tags, and simple align rows are guarded so
   dirty OCR spans do not produce uncompilable inline/display math. Repeated
   reference render nodes are collapsed into one bibliography block.
```

Structured table-to-tabular reconstruction, exact float placement, and journal
template learning remain later backend phases.
`table_body`/HTML is retained as weak evidence. Default original-like rendering
does not create table/figure PDF crop images unless explicitly requested. Figure
roles first reuse MinerU image assets (`img_path` / `image_path`); when no asset
exists, both table and figure roles can use bbox PDF crop fallback with
`\includegraphics` once crop asset output is enabled. Mixed single/double-column
pages use v7 layout-band metadata, fall back to bbox width/center-crossing
inference when metadata is absent, and render local double-column runs with
`multicols`.

The current PDF-side production JSON format is:

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

v7 keeps MinerU block granularity and raw bbox coordinates. It only adds list-marker metadata, column-reading-order repair, page-object layer metadata, local band/column metadata, reference item preservation, and PyMuPDF style spans. Cross-paragraph and cross-page text merging should happen in decoder/generator logic, not in the v7 preprocessing JSON.

Layer/band metadata is now part of the v7 contract. Main text flow, display/inline math, floats, front-matter metadata, and header/footer noise must be explicitly marked before graph building. Candidate MERGE labels and decoder contraction are only allowed inside compatible layers and local bands; floats and math are attached structurally rather than merged as ordinary paragraph text.

All graph-producing and graph-consuming scripts must use the v7 contract:

```text
graph_schema_version = graph_v7
pipeline_version = v7
source_path = *_content_list_v7_styles.json
```

Old v3/v4/v5 JSONs and old `data/06_graph_features/` graph caches are legacy diagnostics only. They must not be used for feature extraction, training, inference, or generator QA.

The current graph supervision target is a three-class edge label:

```text
MERGE = 0
PARENT_CHILD = 1
NONE = 2
```

`SIBLING` is intentionally removed. Sibling order is recovered from v7 reading order and renderer sorting, not from GNN labels.
`PARENT_CHILD` is strictly directed: `parent -> child` is label `1`, while `child -> parent` is label `2` (`NONE`) even when bidirectional candidate edges exist.

Candidate edge construction is recall-first. The default graph builder uses:

```text
sequential_window = 15
spatial_k = 3
long_sight_window = 40
scope_anchor_window = 160
float_skip_window = 40
```

Before training on a rebuilt graph batch, run `tools/profile_candidate_edge_recall.py` on representative labeled samples. If any MERGE or PARENT_CHILD oracle edge is absent from `edge_index`, training on that sample is invalid because the GNN can never predict a label for an edge it was not given.

The current automatic truth-labeling contract is documented in:

```text
docs/ground_truth_labeling_v0.md
```

Batch labeling must use strict quality gates for training data. Samples with high PDF orphan ratio, high unmapped TeX ratio, missing core section structure, or excessive isolated nodes should be skipped rather than saved as dirty `.pt` files.

Orphan accounting is not raw node accounting. Expected visual-only artifacts such as page headers, footers, page numbers, and very short edge noise are exempt from orphan/isolated ratios. Successfully matched pre-section metadata such as title, authors, abstract, and keywords is treated as scoped under a virtual `DOCUMENT_ROOT` for quality checks and TreeDecoder rendering, while the stored graph node count remains unchanged.

## Current Cleanup Decision

The working tree currently contains many tracked deletions from previous cleanup/deletion attempts. Do not stage them automatically.

Before the first clean GitHub push, choose explicitly:

```text
keep old files as ignored legacy reference material, then remove them from the tracked source tree in a deliberate cleanup commit
```

or:

```text
restore selected old source files from Git into the new src/ skeleton after review
```

The safer default is to keep old material in `_legacy_reference/` and migrate only specific functions after reviewing them.
