# PDF2LaTeX NN

## Post-PRCV Status (2026-05-31)

PRCV 2026 submission is complete. The current project-facing source of truth is
the observable-fact-guided reconstruction pipeline:

```text
PDF / MinerU parser outputs
  -> Observable Fact Layer
  -> DocumentIR
  -> RenderTreeIR
  -> compile-safe role renderers
  -> ComparisonStructure evaluation
```

The PRCV-facing mainline is not GNN-driven and does not target source-level TeX
AST recovery. Final PRCV evidence starts at
`data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/`. The paper-side
evidence index is mirrored under
`paper_assets/00_FINAL_EVIDENCE_20260531/` in the PRCV manuscript workspace.

The submitted evidence hierarchy is:

- selected2000: primary large-scale direct-parser comparison, n=1972 fair
  intersection, Ours / ContentList Direct / MinerU Direct.
- selected200: controlled four-method comparison including Nougat as an
  external MMD/Markdown baseline.
- compile/visual QA: defined only for complete LaTeX outputs.

See `docs/PROJECT_ARCHITECTURE_CURRENT_20260531.md`,
`docs/PRCV_EVIDENCE_REGISTRY_20260531.md`, and
`docs/PATH_CONFIGURATION.md` before using older reports.

**Last updated**: 2026-05-26

PDF2LaTeX NN is a layout-aware PDF-to-LaTeX reconstruction pipeline for
born-digital research papers. It does not treat PDF conversion as plain OCR, and
it does not aim to recover the author's original source-level TeX AST. The
current default path reconstructs compilable, block-structure-preserving LaTeX
from rendered PDF facts through a decoupled `DocumentIR` / `RenderTreeIR`
renderer.

## Current Deployment

The maintained default reconstruction path is now v8 / layout-first:

```text
compiled PDF
  -> MinerU middle.json + content_list.json
  -> v8 middle reflow and page-local reading-order repair
  -> DocumentIR
  -> deterministic front matter extraction
  -> full-document heading style registry + stack skeleton
  -> RenderTreeIR
  -> v8 style detector
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

V8 does not mutate v7 JSON, does not build a GNN view, and does not change graph
schema. It exists because MinerU content-list merging can combine text before
the reading order is corrected; v8 rebuilds logical content from `middle.json`
line/block evidence first, then reuses the existing IR renderer.

The historical GNN relation model is retained only as an archived/optional
research branch:

```text
relation-learning branch:
  content_list_v7_styles.json
  -> GNNViewAdapter
  -> graph.pt
  -> TeX-derived MERGE/PARENT_CHILD/NONE labels
  -> GNN training / ablation / diagnostics

historical locked baseline/results:
  v7_registry_adapteraware_20260515_181724
  edge_attr_dim=22

historical float-proxy branch:
  v7_floatproxy_adapter_20260516_205926
  edge_attr_dim=26
```

Do not use GNN view as a renderer source. Generation must consume full
`DocumentIR` / `RenderTreeIR`. As of 2026-05-26, the v8 atomic MERGE / learned
overlay route is archived and is not part of the production path.

## Current Default Capabilities

The current v8 path focuses on:

```text
middle-derived reading-order repair
source-PDF page size preservation
document-local heading style registry
stack skeleton section hierarchy
deterministic front matter preservation
abstract handling
single/two/mixed-column approximation
wide figure* / table* rendering
table/figure crop fallback from original PDF
ordered/list marker recovery
citation and bibliography repair
```

Author / affiliation / email handling is currently **FrontMatter Phase 0**:
the system preserves and separates front-matter blocks, but it does not yet
train an entity/linking model for exact author-affiliation-email binding.

## Relation-Learning Task

The graph model predicts three directed edge classes:

```text
MERGE        = 0  physical continuation / paragraph stitching
PARENT_CHILD = 1  logical hierarchy / attachment
NONE         = 2  no structural relation
```

`SIBLING` is not a learned class. Sibling order is recovered from v7 reading order and renderer sorting.

## Active Interfaces

```text
middle.json             raw MinerU layout/line/block evidence for v8
content_list.json       MinerU asset/caption/table sidecar
content_v7_styles.json  optional style-span enrichment sidecar
DocumentIR              complete document fact layer consumed by renderer
RenderTreeIR            decoder/render structure
StyleProfile            page/style/template profile
CitationResolution      citation/reference repair state

GNNViewAdapter          archived/optional graph-visible view for relation experiments
GraphInput.pt           optional node/edge tensors
GraphLabels             optional TeX-derived edge labels over the GNN view
PredictedRelations      optional GNN output probabilities
```

See [docs/frontend_backend_contract_v1.md](docs/frontend_backend_contract_v1.md).
The single current production command/output contract is
[docs/V8_MAINLINE_RECONSTRUCTION_PATH.md](docs/V8_MAINLINE_RECONSTRUCTION_PATH.md).

## Key Scripts

```bash
# Run the current v8 layout reconstruction path
python scripts/pipeline/run_v8_layout_reconstruction.py \
  --doc-id <doc_id> \
  --middle-json <path/to/*_middle.json> \
  --content-list-json <path/to/*_content_list.json> \
  --style-content-list-json <path/to/*_content_list_v7_styles.json> \
  --pdf <path/to/original.pdf> \
  --output-dir data/09_eval_reports/v8_reflow_<YYYYMMDD>/<doc_id>_<run_tag> \
  --compile-engine auto

# Archived relation-learning scripts still exist for traceability, but they are
# no longer part of the default reconstruction workflow.
```

The current 00050 v8 smoke command is documented in
[docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md](docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md).

Current paper-facing evaluation and historical GNN ablations are still kept for
traceability, but the default reconstruction claim is now the v8 layout-aware
deterministic path.

Paragraph/source audits report both legacy body coverage and type-aware
visible-prose order metrics. The latter excludes front matter, captions,
references, display math/formula-only blocks, URL/metadata, and OCR/no-render
artifacts before computing prose-order inversion, adjacent inversion,
displacement, and LIS-disorder rates.

## Current Report Layout

```text
data/09_eval_reports/v8_reflow_20260523/                 current v8 smoke outputs
data/09_eval_reports/v8_mainline_final_20260526/         final selected200 v8/GNN-closure summary
data/09_eval_reports/post_audit_20260519/                post-audit diagnostics
data/09_eval_reports/targeted_structure_fix_20260519/    targeted diagnostics
data/09_eval_reports/_archive/                           old preserved runs
data/09_eval_reports/_obsolete/                          invalidated/retired experiments
```

## Current Docs

```text
docs/PROJECT_FILE_LAYOUT.md          local/AutoDL directory and artifact map
docs/PROJECT_ARCHITECTURE_FULL.md     complete architecture, logic, metrics, and code map
docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md paper-facing full project description
docs/PROJECT_SOURCE_OF_TRUTH.md      local / GitHub / AutoDL boundary
docs/PROJECT_OVERVIEW.md             architecture and implementation summary
docs/V8_MAINLINE_RECONSTRUCTION_PATH.md single current v8 production path contract
docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md current v8 path and parameters
docs/_archive/v8_gnn_merge_experiments_20260526/ archived v8 atomic MERGE/GNN route
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md future author/affiliation/email parser plan
docs/ENVIRONMENT_SETUP.md             conda/venv setup and dependency profiles
docs/frontend_backend_contract_v1.md decoupled IR contracts
docs/feature_schema_v0.md            graph tensor feature contract
docs/ground_truth_labeling_v0.md     TeX-to-PDF truth-label generation
docs/ablation_plan_v2.md             current ablation protocol
docs/ablation_results_current.md     latest locked ablation results
docs/v7_training_and_monitoring.md   optional relation-learning runbook
docs/interface_audit_2026_05_14.md   current interface audit and stale-path check
docs/LOCAL_CONFIGURATION.md          private local configuration notes
```

## Important Paths

Local project root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Large artifacts stay on AutoDL under:

```text
/root/autodl-tmp/pdf2latex_nn/data
```

The canonical folder contract is now:

```text
docs/PROJECT_FILE_LAYOUT.md
```

Do not commit datasets, checkpoints, generated PDFs, secrets, or AutoDL credentials.
