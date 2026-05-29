# Project File Layout

**Last updated**: 2026-05-26

This document is the current directory contract for both the local checkout and
the AutoDL runtime checkout.  When a path exists on both machines, it should
carry the same meaning.  AutoDL may contain large runtime artifacts that are not
present locally, but the folder semantics should remain identical.

## Roots

```text
local:  /Users/lu/Code/Project/pdf2latex_nn/test_4_19
AutoDL: /root/autodl-tmp/pdf2latex_nn
```

Code changes should flow through GitHub or a targeted source-file sync.  Runtime
artifacts should not be recursively copied from local to AutoDL.

## Top-Level Source Folders

| Path | Meaning | Keep in Git | Notes |
| --- | --- | ---: | --- |
| `src/` | Production Python package. | Yes | Core pipeline code. |
| `src/perception/` | PDF/v7 perception layer, graph-visible adapter, reading order, style/title probes. | Yes | Does not own final rendering. |
| `src/adapters/` | Frontend adapters, especially MinerU v7 -> `DocumentIR`. | Yes | Replaceable frontend boundary. |
| `src/reasoning/` | Decoder/postprocess, heading/float logic, and archived relation-learning modules. | Yes | Production v8 uses deterministic reasoning; GNN modules are historical/optional. |
| `src/ir/` | Intermediate representation schemas, serialization, validators. | Yes | Shared semantic/layout IR contract. |
| `src/generation/` | Render surface, IR renderer, style profile, table/figure/citation rendering helpers. | Yes | Production rendering is full-v7-first through IR. |
| `src/evaluation/` | Comparison structure and metric implementations. | Yes | Paper-facing evaluation logic. |
| `src/datasets/` | Dataset wrappers/loaders for graph training. | Yes | Should not embed machine-specific paths. |
| `src/pipeline/` | Shared pipeline contracts and validation helpers. | Yes | Includes v7 contract checks. |
| `scripts/pipeline/` | Long-running data, training, E2E, ablation, and collection entrypoints. | Yes | AutoDL jobs should start here. |
| `scripts/debug/` | Manual visualization/debug scripts. | Yes | Useful for bbox/reading-order inspection. |
| `tools/` | One-off tools and reusable CLIs for evaluation, conversion, audit, and external bridges. | Yes | Tools should write outputs under `data/09_eval_reports/` unless otherwise documented. |
| `tools/audit/` | Diagnostic/audit tools for labels, MERGE/PARENT usage, heading/float/layout issues. | Yes | Read-only or report-generating by default. |
| `tools/_archive/` | Archived experimental tools that should not be imported by the main path. | Yes | Includes closed v8 atomic MERGE / GNN tools. |
| `tools/api_baselines/` | API/VLM baseline pipeline scaffolding. | Yes | Real API calls require explicit environment opt-in. |
| `tools/comphrdoc/` | External CompHRDoc/HRDH adapter smoke tooling. | Yes | Not a main training target. |
| `configs/` | Ablation, prompt, external-eval, and API-baseline configuration. | Yes | No secrets. |
| `tests/` | Unit and regression tests. | Yes | Prefer synthetic tests for logic changes. |
| `docs/` | Current contracts, architecture, runbooks, and historical result summaries. | Yes | This file is the directory source of truth. |
| `environment.yml` | Conda wrapper for the base Python environment. | Yes | Uses `requirements.txt` from project root. |
| `requirements.txt` | Base v8/GNN dependency list. | Yes | Laptop/default install target. |
| `requirements_server.txt` | Server/AutoDL extras. | Yes | Heavier optional OCR/API/data packages. |
| `third_party/` | Third-party code/data placeholders. | Partial | Large third-party datasets stay on AutoDL and should not be committed. |
| `_legacy_reference/` | Archived pre-current code/reference snapshots. | Yes, small only | Do not use as production path. |

## Top-Level Runtime / Output Folders

| Path | Meaning | Git Policy | Cleanup Policy |
| --- | --- | --- | --- |
| `data/` | Canonical project data/artifact root. | Mostly ignored | Preserve active and historical run families. |
| `logs/` | AutoDL long-run logs and PID files. | Ignored | Preserve active run logs; archive or delete only after review. |
| `local_outputs/` | Local inspection copies and visual QA outputs. | Ignored | Local-only; keep as manual visual reference, not a production path. |
| `data/09_eval_reports/_archive/` | Archived historical run outputs that are not on the current path. | Ignored | Preserve for traceability; do not use as default input. |
| `data/09_eval_reports/_obsolete/` | Obsolete experiments or invalidated debug branches. | Ignored | Kept only so old reasoning can be audited. |
| `.venv*/` | Local Python environments. | Ignored | Local-only. |
| `audit_bundle_*.zip` | External audit bundles. | Usually ignored | Store under `data/09_eval_reports/_archive/<date>_audit_bundle/`, not the project root. |

## `data/` Contract

The same `data/` layout is used locally and on AutoDL.  Local may contain only a
small subset; AutoDL is expected to hold the full runtime artifacts.

| Path | Meaning | Produced By | Consumed By |
| --- | --- | --- | --- |
| `data/00_manifests/` | Dataset/run manifests. | Download/build/relabel scripts. | Every later stage. |
| `data/01_raw_pdfs/` | Canonical input PDFs for MinerU. For the 2026-05-22 rebuild, these are locally compiled PDFs from arXiv TeX sources, not downloaded arXiv-hosted PDFs. | `step0_build_compilable_arxiv_dataset.py` or external ingestion. | MinerU / v7 builder / rendered-output evaluation. |
| `data/02_mineru_outputs/` | Raw MinerU outputs and associated frontend extraction artifacts. | MinerU/v7 staged builder. | v7 normalization and `DocumentIR` conversion. |
| `data/03_tex_sources/` | Accepted TeX source trees, one directory per `doc_id`. | arXiv source downloader/compiler. | TeX AST flattener, labeler, reproducibility. |
| `data/03_tex_source_pool/` | Source-pool staging/archive area. | Older ingestion/repass workflows. | Only if referenced by a manifest. |
| `data/04_ground_truth_ir/` | TeX-derived alignment mappings, gold/comparison IR, label reports. | Label generation / evaluation conversion. | Relabeling, audit, evaluation. |
| `data/05_observed_ir/` | Observed document IR or frontend-derived IR sidecars. | v7/MinerU adapters and diagnostic exporters. | Generator/evaluation diagnostics. |
| `data/06_graph_features/` | Current graph and labeled graph families. | Graph builder / relabel scripts. | GNN training and inference. |
| `data/06_graph_features_v7/` | Historical v7 graph-feature families. | Older v7 rebuilds. | Historical comparison only unless a manifest explicitly points here. |
| `data/06_graph_features_v5/` | Historical v5 graph-feature families. | Old experiments. | Not production. |
| `data/06_graph_features_oracle/` | Oracle/debug graph families. | Diagnostics. | Not production training unless explicitly documented. |
| `data/07_predicted_ir/` | Model prediction sidecars and predicted relation IR. | Inference/decoder scripts. | Decoder/generator audits. |
| `data/08_output_latex/` | Generated LaTeX/PDF outputs from controlled runs. | E2E/generator scripts. | Evaluation and visual QA. |
| `data/08_runs/` | Generated shell scripts / command bundles for batch runs. | `prepare_ablation_suite.py` and runbook tools. | AutoDL execution. |
| `data/09_eval_reports/` | Reports, summaries, eval outputs, audit outputs, run-specific logs, many checkpoints. | Training/eval/audit scripts. | Papers, dashboards, follow-up analysis. |
| `data/10_checkpoints/` | Optional checkpoint storage/mirrors. | Training or manual curation. | Inference/training resume. |
| `data/external/` | External benchmark material and predictions. | External bridge tools. | External benchmark evaluation only. |
| `data/_tmp_*` | Temporary work directories. | Long-running builders. | Safe to clean only after confirming no active job uses them. |

## `data/09_eval_reports/` Organization

The top level of `data/09_eval_reports/` should stay readable.  Current or
paper-facing reports remain directly under `data/09_eval_reports/<run_tag>/`.
Old exploratory output should be moved under `_archive/` or `_obsolete/` instead
of staying mixed with the current run list.

Current local organization:

```text
data/09_eval_reports/
  v8_reflow_20260523/                 current v8 middle-reflow / style path smoke outputs
  v8_mainline_final_20260526/         final selected200 deterministic-v8 / learned-branch closure
  post_audit_20260519/                post-audit diagnostic artifacts
  targeted_structure_fix_20260519/    targeted heading/float/layout diagnostic artifacts
  pre_expansion_wait_20260519/        wait-state reports and runbooks
  virtual_heading_nodes_20260519/     virtual heading implementation report
  current_eval_rollup_local_pending_20260517/
  layout_default_pivot_20260523/
  api_baselines/
  comphrdoc_test500/
  _archive/
    20260509_legacy_e2e_outputs/      old root-level `e2e_outputs/`
    20260515_generator_iterations/    superseded generator hardcase iterations
    20260519_audit_bundle/            external audit bundle zip
  _obsolete/
    legacy_merge_gnn_generator_debug_20260523/
```

Current v8 runs must use:

```text
data/09_eval_reports/v8_reflow_<YYYYMMDD>/<doc_id>_<short_run_tag>/
```

Each run directory should keep the full v8 trace:

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

Do not create a second current output family for the same v8 pipeline.  If a run
uses GNN logits, v7 content, or a relation-source comparison, its directory name
must say so explicitly and it belongs to the optional/legacy relation branch.

Archived v8 atomic MERGE JSON experiments used:

```text
data/09_eval_reports/_archive/v8_gnn_closed_20260526/
tools/_archive/v8_gnn_merge_experiments_20260526/v8_atomic/
```

Do not create new v8 atomic MERGE output as a current report family unless a
new research branch is explicitly reopened.

Do not put new experiment families at the project root.  For example:

```text
bad:  ./e2e_outputs/new_run/
good: data/09_eval_reports/<run_tag>/
```

The root-level `e2e_outputs/` folder was archived on 2026-05-24 and is no
longer a current output location.

## Current Fresh Data Rebuild Layout

The active 2026-05-22 fresh source rebuild uses:

```text
run_name: arxiv2025_compilable_tex8000_idscan_20260522

candidate ids:
  data/00_manifests/arxiv_2025_idscan_candidates_360000.jsonl

accepted TeX source trees:
  data/03_tex_sources/{doc_id}/

locally compiled PDFs:
  data/01_raw_pdfs/{doc_id}.pdf

run state:
  data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/
  logs/arxiv2025_compilable_tex8000_idscan_20260522.log
```

After completion, freeze `accepted.jsonl` into a manifest such as:

```text
data/00_manifests/arxiv2025_compilable_tex_pdf_clean8000_20260522.jsonl
```

Each row should preserve:

```json
{
  "doc_id": "2501.12345",
  "source_dir": "data/03_tex_sources/2501.12345",
  "main_tex": "main.tex",
  "pdf_path": "data/01_raw_pdfs/2501.12345.pdf"
}
```

## Model / Experiment Family Naming

Every full data/model experiment should use a stable tag:

```text
<purpose>_<schema_or_adapter>_<YYYYMMDD_HHMMSS>
```

Typical outputs for tag `${TAG}`:

```text
data/00_manifests/${TAG}_rebuilt.json
data/00_manifests/${TAG}_labeled.json
data/00_manifests/${TAG}_trainable_recall98.json
data/06_graph_features/${TAG}_graphs/
data/06_graph_features/${TAG}_labeled_graphs/
data/04_ground_truth_ir/${TAG}_mappings/
data/09_eval_reports/${TAG}/
logs/${TAG}_run.log
```

Never mix graph/checkpoint schema families.  In particular, the historical
`edge_attr_dim=22` registry-adapter family and the `edge_attr_dim=26`
float-proxy family are incompatible.

## What Belongs Where

| Artifact | Correct Place | Do Not Put In |
| --- | --- | --- |
| Accepted source tree | `data/03_tex_sources/{doc_id}/` | `data/09_eval_reports/`, `local_outputs/` |
| Compiled input PDF | `data/01_raw_pdfs/{doc_id}.pdf` | `data/09_eval_reports/` |
| Raw MinerU output | `data/02_mineru_outputs/<tag>/...` | `src/`, `docs/` |
| v7 content JSON | Manifest-referenced frontend/output directory under the run tag | Ad hoc root folders |
| Graph tensor | `data/06_graph_features/${TAG}_graphs/` | `data/09_eval_reports/` |
| Labeled graph tensor | `data/06_graph_features/${TAG}_labeled_graphs/` | overwrite of unlabeled graph folder |
| Training checkpoint | `data/09_eval_reports/<run>/<model>/seed_*/best_model.pth` or `data/10_checkpoints/` | `src/` |
| Generated TeX/PDF for evaluation | `data/08_output_latex/` or run-specific `data/09_eval_reports/<run>/` | `data/01_raw_pdfs/` |
| Summary report | `data/09_eval_reports/<run>/` | random top-level folders |
| Temporary extraction/build cache | `data/_tmp_*` | long-term manifests |

## Cleanup Rules

Before deleting anything on AutoDL, classify it:

```text
ACTIVE_KEEP                  current datasets, current run outputs, current checkpoints
HISTORICAL_KEEP              paper/audit reproducibility artifacts
ARCHIVE_CANDIDATE            old large run outputs not on the current path
DELETE_CANDIDATE_AFTER_CONFIRM tmp/stage/cache/duplicate rendered outputs
DO_NOT_TOUCH_RUNNING         active run logs/tmp/output directories
```

Never delete:

```text
data/03_tex_sources/
data/01_raw_pdfs/
active manifests under data/00_manifests/
active graph/checkpoint/eval run families
logs for active or recently completed long runs
```

Safe cleanup usually targets only:

```text
data/_tmp_*
stale smoke outputs
duplicate local rendered PDFs
old cache/stage folders not referenced by any manifest
```

## Documentation Roles

Use these files for current questions:

| Question | Canonical Doc |
| --- | --- |
| What does each folder/file family mean? | `docs/PROJECT_FILE_LAYOUT.md` |
| What is the architecture and module ownership? | `docs/PROJECT_ARCHITECTURE_FULL.md` |
| What is the project target and metric philosophy? | `docs/layout_aware_reconstruction_target.md` |
| What is the local/GitHub/AutoDL boundary? | `docs/PROJECT_SOURCE_OF_TRUTH.md` |
| What is the single current production path? | `docs/V8_MAINLINE_RECONSTRUCTION_PATH.md` |
| What is the current v8 layout reconstruction path? | `docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md` |
| How do we refresh paragraph order metrics? | `tools/audit/refresh_paragraph_order_audits.py` |
| Where is the archived middle-derived atomic MERGE route? | `docs/_archive/v8_gnn_merge_experiments_20260526/V8_ATOMIC_MERGE_GNN_ROUTE.md` |
| How will precise author/affiliation/email parsing be added? | `docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md` |
| How do we run data/training/eval? | `docs/v7_training_and_monitoring.md` |
| How are labels generated? | `docs/ground_truth_labeling_v0.md` |
| How does the generator consume v7/GNN/IR? | `docs/generator_logic_audit_2026_05_17.md` |
| What are frontend/table/style plugin contracts? | `docs/MINERU_ADAPTER_CONTRACT.md`, `docs/TABLE_ENGINE_CONTRACT.md`, `docs/STYLE_TEMPLATE_CONTRACT.md` |
