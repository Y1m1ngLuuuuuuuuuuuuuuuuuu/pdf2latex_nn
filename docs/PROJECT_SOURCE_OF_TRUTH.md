# Project Source Of Truth

## Post-PRCV Current Source of Truth (2026-05-31)

PRCV submission is complete. The local repository is the source-control home,
AutoDL is the runtime home, and the PRCV paper workspace is the submission home.

Canonical evidence starts here:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Paper-facing canonical evidence starts here:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh/paper_assets/00_FINAL_EVIDENCE_20260531/
```

The current evidence hierarchy is:

- selected2000: primary large-scale direct-parser comparison, n=1972 fair
  intersection, Ours / ContentList Direct / MinerU Direct.
- selected200: controlled four-method comparison including Nougat.
- compile/visual QA: only for complete LaTeX outputs.

The current architecture source of truth is
`docs/PROJECT_ARCHITECTURE_CURRENT_20260531.md`. The current path configuration
source of truth is `docs/PATH_CONFIGURATION.md`.

**Last updated**: 2026-05-24

This repository is the source-control home for the PDF2LaTeX-NN reconstruction
system. AutoDL is the runtime home for datasets, MinerU outputs, graph tensors,
checkpoints, generated PDFs, and long-running jobs.

## Source Flow

```text
local source edits -> GitHub -> AutoDL git pull / targeted sync
```

Avoid broad recursive overwrites from local to AutoDL. If targeted sync is necessary, sync source files only. Runtime artifacts should stay remote.

## Roots

Local:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL:

```text
/root/autodl-tmp/pdf2latex_nn
```

GitHub:

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

Do not treat this document as a static commit pin.  Check the current source
state with:

```bash
git log --oneline -5
```

The canonical local/AutoDL directory map is:

```text
docs/PROJECT_FILE_LAYOUT.md
```

The 2026-05-22 remote recovery, active rebuild run, and current MERGE-label
decisions are recorded in:

```text
docs/RECOVERY_AND_CURRENT_RUNBOOK_2026_05_22.md
```

## Production Pipeline

Canonical target:

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction
from rendered scientific PDFs
```

Non-goal:

```text
recovering the author's original source-level TeX AST
```

All future model, decoder, renderer, and evaluation changes must respect this
target definition.  The detailed rationale and metric contract are documented in
`docs/layout_aware_reconstruction_target.md`.

The production reconstruction pipeline is now **v8 / layout-first**.  As of
2026-05-26, the default E2E reconstruction path does not load learned GNN
relation logits:

```text
compiled PDF
  -> MinerU middle.json + content_list.json
  -> v8 middle reflow and reading-order repair
  -> DocumentIR
  -> deterministic FrontMatterIR
  -> full-document heading style registry + stack skeleton
  -> RenderTreeIR
  -> StyleProfile / v8 style detector
  -> OriginalLikeIRLatexRenderer
```

GNN artifacts are now archived as historical/optional relation-learning
material:

```text
content_v7 + style spans
  -> GNNViewAdapter
  -> graph.pt
  -> TeX-derived relation labels
  -> GNN training / diagnostics / ablations
```

Do not treat GNN outputs as the default reconstruction path. The paper-facing
default compares the v8 layout-aware deterministic reconstruction system. GNN
relation studies are retained only as auxiliary historical ablations.

The single current production path contract is:

```text
docs/V8_MAINLINE_RECONSTRUCTION_PATH.md
```

Canonical command:

```bash
python3 scripts/pipeline/run_v8_layout_reconstruction.py \
  --doc-id <doc_id> \
  --middle-json <path/to/*_middle.json> \
  --content-list-json <path/to/*_content_list.json> \
  --style-content-list-json <path/to/*_content_list_v7_styles.json> \
  --pdf <path/to/input.pdf> \
  --output-dir data/09_eval_reports/v8_reflow_<YYYYMMDD>/<doc_id>_<run_tag>
```

Do not introduce another default reconstruction route.  If a v7/GNN script is
used, its run name and report must say that it is an optional relation-learning
or legacy comparison branch.

The old root-level `e2e_outputs/` directory and superseded generator iteration
outputs have been archived under:

```text
data/09_eval_reports/_archive/
```

Obsolete merge/GNN generator debug experiments are under:

```text
data/09_eval_reports/_obsolete/
```

The selected200 learned-MERGE closure report is:

```text
data/09_eval_reports/v8_mainline_final_20260526/V8_MAINLINE_FINAL_REPORT.md
```

Paragraph/source audits now report legacy body coverage and visible-prose order
metrics side by side. The visible-prose layer is type-aware and excludes front
matter, captions, references, display math/formula-only blocks, URL/metadata,
and no-render artifacts before computing prose-order inversion, adjacent
inversion, displacement, and LIS-disorder rates.

Old v3/v4/v5 JSON variants are historical experiments. Do not feed them into training or evaluation.

Current data rebuild note, 2026-05-22:

```text
Active run:
  arxiv2025_compilable_tex8000_idscan_20260522

Purpose:
  rebuild a fresh 2025 arXiv TeX-source pool after the remote runtime reset.

Policy:
  download arXiv e-print TeX source, compile locally, keep TeX source and the
  locally compiled PDF. Do not download arXiv-hosted original PDFs for this
  dataset.
```

Monitor:

```bash
cd /root/autodl-tmp/pdf2latex_nn
cat data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/progress.json
tail -f logs/arxiv2025_compilable_tex8000_idscan_20260522.log
```

Storage contract for this run:

```text
TeX source trees:
  data/03_tex_sources/{doc_id}/

locally compiled PDFs used by MinerU:
  data/01_raw_pdfs/{doc_id}.pdf

run progress and accepted/rejected logs:
  data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/
```

Do not put accepted sources or input PDFs under `data/09_eval_reports/` or
`local_outputs/`; those directories are for reports, copied inspections, and
temporary experiment outputs.

The full normalized frontend payload is the complete fact layer.  In the current
default path this is `content_list_v8.json`; in optional relation-learning
experiments this can still be `content_list_v7_styles.json`.  Neither form may
delete or rewrite metadata, figures, tables, footnotes, headers, captions, or
references just because the GNN does not consume them directly. The
graph-visible view is built separately by `src/perception/gnn_view_adapter.py`.

GNN predictions are never rendered directly. The model predicts per-edge
MERGE/PARENT_CHILD/NONE logits on the graph-visible view. `TreeDecoder` turns
those probabilities into a constrained structure, then the relation bridge maps
graph indices back to exact v7 source ids before the IR renderer reads the full
v7 fact layer.

Current model/data tracks:

```text
locked baseline/results:
  tag: v7_registry_adapteraware_20260515_181724
  raw edge_attr_dim: 22
  keep all reports/checkpoints/generator outputs

active experimental rebuild:
  tag: v7_floatproxy_adapter_20260516_205926
  raw edge_attr_dim: 26
  float proxy + skip-over-float features
```

Do not delete previous test results or weights while the new path is being
validated.

## Active Entrypoints

Current production reconstruction:

```text
scripts/pipeline/run_v8_layout_reconstruction.py
```

Current v8 helper modules:

```text
tools/mineru_v8/build_v8_from_middle.py
tools/mineru_v8/render_v8_smoke.py
src/perception/mineru_v8_reflow.py
src/adapters/mineru_v8_document_ir.py
src/reasoning/v8_heading_style_stack.py
src/reasoning/v8_render_tree.py
src/generation/v8_style_detector.py
```

Optional v7/GNN data from PDF + TeX:

```text
scripts/pipeline/build_v7_dataset_staged.py
```

Optional rebuild and relabel existing v7 content:

```text
scripts/pipeline/run_current_v7_rebuild_relabel.sh
scripts/pipeline/rebuild_graphs_from_manifest.py
scripts/pipeline/relabel_manifest.py
```

Optional GNN training:

```text
scripts/pipeline/train_edge_gnn_full.py
```

Archived v8 atomic MERGE JSON route:

```text
tools/_archive/v8_gnn_merge_experiments_20260526/v8_atomic/
docs/_archive/v8_gnn_merge_experiments_20260526/V8_ATOMIC_MERGE_GNN_ROUTE.md
```

Optional GNN ablation:

```text
configs/ablation_matrix_v7_adapteraware_20260514_2109.json
scripts/pipeline/prepare_ablation_suite.py
data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh
```

Legacy/optional v7-GNN E2E inference and visual QA:

```text
scripts/pipeline/batch_visual_qa_inference.py --renderer ir
scripts/pipeline/run_e2e_inference.py --renderer ir
scripts/pipeline/step5_generate_tex.py --renderer ir
```

Experimental float-proxy rebuild/relabel:

```bash
TAG=v7_floatproxy_adapter_$(date +%Y%m%d_%H%M%S) \
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

`--renderer ir` is the only supported surface for these legacy/optional scripts.
The standalone TreeDecoder renderer is not a production surface; production
scripts no longer accept `--renderer tree`.  These scripts are not the v8
default path unless a run explicitly says it is a v7/GNN branch.

Generator module ownership and the current full-v7/GNN-view/render-tree bridge
are frozen in:

```text
docs/generator_logic_audit_2026_05_17.md
```

The replaceable frontend/table/style boundaries are frozen in:

```text
docs/MINERU_ADAPTER_CONTRACT.md
docs/TABLE_ENGINE_CONTRACT.md
docs/STYLE_TEMPLATE_CONTRACT.md
```

When generator behavior looks inconsistent, check that document before changing
`ir_renderer.py`; most failures come from crossing the full-v7 render facts with
the filtered GNN view.

Decoder heading mode:

```text
--heading-skeleton-mode stack    canonical mode: layout heading detector supplies
                                 candidates/hints; deterministic stack provides
                                 outline priors and section-scope safety gates;
                                 GNN parent edges remain part of the relation
                                 bridge under physical/heading constraints
```

Use `stack` for all current E2E generation. It does not require MinerU reruns,
graph rebuilds, relabeling, or model retraining. The stack mode explicitly
filters false heading evidence such as front-matter paper titles and long
math/OCR fragments before building the outline. Local code now rejects the old
heading decoder modes instead of silently keeping a second production path.

## Current Manifest Families

Locked baseline trainable set and checkpoint family:

```text
data/00_manifests/v7_registry_adapteraware_20260515_181724_labeled.json
data/06_graph_features/v7_registry_adapteraware_20260515_181724_labeled_graphs
data/09_eval_reports/ablations_v7_registry_adapteraware_20260515_181724/
```

Active experimental float-proxy set:

```text
data/00_manifests/v7_floatproxy_adapter_20260516_205926_rebuilt.json
data/00_manifests/v7_floatproxy_adapter_20260516_205926_labeled.json
data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json
data/06_graph_features/v7_floatproxy_adapter_20260516_205926_graphs
data/06_graph_features/v7_floatproxy_adapter_20260516_205926_labeled_graphs
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_summary.json
```

Current paper-facing evaluation suite:

```text
configs/ablation_matrix_current.json
scripts/pipeline/run_current_full_eval_suite.py
scripts/pipeline/collect_current_eval_results.py

data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current/
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.json
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.csv
data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

`run_current_full_eval_suite.py` is the reproducible top-level command for
collecting the current model/generator results.  It can skip completed stages
and reuse outputs.  `collect_current_eval_results.py` is read-only and can be
run at any time to generate a pending or final rollup.

The ablation matrix filename still contains `20260514` for reproducibility, but
new experiments must pass an explicit manifest and graph root. Do not infer the
current data family from the matrix filename alone.

Current float-proxy status:

```text
rebuild/relabel complete
trainable docs: 1829
edge_attr_dim: 26
best ablation by positive macro F1: M06_y_network_plus_merge_gate
M06 positive macro F1: 0.7532
M06 MERGE F1: 0.5739
M06 PARENT_CHILD F1: 0.9325
M06 E2E smoke: 20 / 20 compiled
```

The float-proxy path is a valid experimental schema and should remain separate
from the locked registry-adapter M07 baseline until heading-tree and
float-caption E2E quality are improved.

Current locked relation model direction:

```text
M05_current_y_network
```

M05 keeps type-aware GAT message passing for PARENT_CHILD while bypassing it
for MERGE, using raw projected edge-pair features for the MERGE logit.  The
hard MERGE gate is part of the current main path, not a separate post-hoc
cleanup.

Current MERGE investigation direction:

```text
PARENT_CHILD:
  stack heading skeleton is the main section-scope authority.
  GNN parent edges are hints unless a specific ablation proves an override path.

MERGE:
  focus of the current GNN contribution investigation.
  Do not lower tau_merge globally.
  Use channel/family-aware audit and small branches first:
    BODY_TEXT/LIST  -> lower threshold only under precision gates
    REFERENCE       -> separate high threshold or reference continuation
    WEAK/MASKED     -> mask or low weight
    LAYOUT_MISMATCH -> hard negative
```

Relevant audit tools:

```text
tools/audit/channel_aware_merge_label_audit.py
tools/audit/audit_missing_below_threshold_merge.py
tools/audit/family_specific_merge_calibration.py
tools/audit/probe_merge_visibility.py
```

## Evaluation Contract

Evaluation is layered.  Do not use raw source-AST section attachment as the sole
definition of success.

Primary structural/reporting families:

```text
compile_success / layout_similarity
paragraph_text_coverage
paragraph_boundary_f1
heading_tree_accuracy over section/subsection/subsubsection only
section_attachment_body_no_float_f1
float_caption_attachment_accuracy
reference_section_completeness
generated_structure_validity
```

Run-in `\paragraph` and `\subparagraph` are normalized as paragraph inline
labels for comparison.  Figure/table/caption/footnote/reference/front-matter
nodes are evaluated in their own tracks and should not dominate body section
attachment.

## Runtime Boundaries

Commit:

```text
source code
configs
docs
tests
lightweight manifests when useful
```

Do not commit:

```text
PDF corpora
TeX corpora
MinerU outputs
graph .pt caches
model checkpoints
generated PDFs
secrets
AutoDL passwords
Kaggle tokens
```

## Current Maintained Docs

```text
README.md
docs/PROJECT_ARCHITECTURE_FULL.md
docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md
docs/PROJECT_SOURCE_OF_TRUTH.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_FILE_LAYOUT.md
docs/V8_MAINLINE_RECONSTRUCTION_PATH.md
docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md
docs/_archive/v8_gnn_merge_experiments_20260526/V8_ATOMIC_MERGE_GNN_ROUTE.md
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md
docs/ENVIRONMENT_SETUP.md
docs/frontend_backend_contract_v1.md
docs/feature_schema_v0.md
docs/ground_truth_labeling_v0.md
docs/ablation_plan_v2.md
docs/ablation_results_current.md
docs/v7_training_and_monitoring.md
docs/RECOVERY_AND_CURRENT_RUNBOOK_2026_05_22.md
docs/interface_audit_2026_05_14.md
docs/LOCAL_CONFIGURATION.md
```

Anything outside this list is either source-code comments, generated reports, or historical reference material.
