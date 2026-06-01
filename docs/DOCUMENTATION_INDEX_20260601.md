# Documentation Index

Last updated: 2026-06-01.

Use this index to decide which documents are current source-of-truth documents
and which documents are historical process records.

## Current Source of Truth

Read these first:

- `README.md`
- `docs/PROJECT_SOURCE_OF_TRUTH.md`
- `docs/PROJECT_ORGANIZATION_AND_MODULE_FLOW_20260601.md`
- `docs/PROJECT_SCOPE_AND_PAPER_MODULES_20260601.md`
- `docs/INTERFACE_DESIGN_CURRENT_20260601.md`
- `docs/PROJECT_ARCHITECTURE_CURRENT_20260531.md`
- `docs/MAIN_PATH_LAYOUT_AFTER_SUBMISSION.md`
- `docs/CANONICAL_PROJECT_PATHS_POST_SUBMISSION.md`
- `docs/PATH_CONFIGURATION.md`
- `docs/SECRET_HANDLING.md`

## Current PRCV Paper Module

- `docs/PRCV_EVIDENCE_REGISTRY_20260531.md`
- `data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/README.md`
- `data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/EVIDENCE_INDEX.md`
- `data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/CLAIM_BOUNDARY.md`
- `data/09_eval_reports/README.md`
- `data/09_eval_reports/EVAL_REPORTS_MAP_20260531.md`

## Interface and Contract References

These are still useful contract documents, but check the current interface index
first:

- `docs/frontend_backend_contract_v1.md`
- `docs/MINERU_ADAPTER_CONTRACT.md`
- `docs/TABLE_ENGINE_CONTRACT.md`
- `docs/STYLE_TEMPLATE_CONTRACT.md`
- `docs/comparison_structure_v1.md`
- `docs/layout_aware_reconstruction_target.md`

## Historical Implementation Background

These documents are retained for engineering memory and future-paper mining.
They are not current paper claim registries:

- `docs/PROJECT_ARCHITECTURE_FULL.md`
- `docs/PROJECT_FILE_LAYOUT.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md`
- `docs/V8_MAINLINE_RECONSTRUCTION_PATH.md`
- `docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md`
- `docs/RECOVERY_AND_CURRENT_RUNBOOK_2026_05_22.md`
- `docs/v7_training_and_monitoring.md`
- ablation result documents
- translated historical documents under `docs/translate_doc/`

## Rule for Future Papers

Create a new paper registry instead of modifying the PRCV registry. A paper
registry should name the dataset denominator, baselines, locked tables, claim
boundary, and backup location. It should also identify which project interfaces
it exercises or extends.
