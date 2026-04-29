# Environment Cleanup Status

**Date**: 2026-04-29

## Completed

- Old implementation files, one-off scripts, old docs, old data outputs, reports, caches, and artifacts were moved into `_legacy_reference/2026_04_29_pre_rebuild/`.
- `_legacy_reference/` is ignored by Git.
- Local Git remote now points to the new GitHub repository.
- Local Kaggle access token is installed outside the repo.
- AutoDL Kaggle access token is installed outside the repo.
- Local private `.env.local` is ignored by Git.
- AutoDL private `.env.autodl` is outside Git control.
- `.gitignore` now excludes secrets, local env files, heavy generated artifacts, model binaries, reports, caches, and runtime data.
- `upload_to_server.sh` now reads AutoDL settings from `.env.local` and syncs only lightweight source paths. It does not delete remote-only files.
- `deploy_to_server.sh` is now an environment verification helper instead of an unconditional dependency installer.
- `docs/PROJECT_SOURCE_OF_TRUTH.md` records the local/GitHub/AutoDL boundary.

## Verified

Kaggle authentication smoke test succeeded on both machines:

```bash
kaggle datasets list -s arxiv -p 1
```

AutoDL environment observed:

```text
Python: 3.12.3
Conda env: pdf2latex
Torch: 2.8.0+cu128
PyG: 2.6.1
GPU: NVIDIA GeForce RTX 4090 D
```

## Still Dirty

The current local worktree is not clean. It contains:

```text
modified: .gitignore
modified: upload_to_server.sh
modified: deploy_to_server.sh
new: .env.example
new: src/ skeleton
new: configs/ skeleton
new: scripts/pipeline/ skeleton
new: tests/ skeleton
new: data/00-09 .gitkeep directories
new: docs/LOCAL_CONFIGURATION.md
new: docs/PROJECT_SOURCE_OF_TRUTH.md
new: docs/ENVIRONMENT_CLEANUP_STATUS_2026_04_29.md
many tracked deletions from older source/artifact paths
```

The tracked deletions are expected after moving old material into ignored legacy reference storage. They are intentionally not staged yet.

Deleted tracked source-like files currently need a decision:

```text
config_native_blocks_enhanced.yaml
docs/gnn_task_selection_report.md
docs/mineru_first_gnn_design.md
source_code/consolidate_native_blocks.py
source_code/mineru_gnn_dataset.py
source_code/mineru_gnn_model.py
source_code/mineru_gnn_refiner.py
source_code/native_block_candidates.py
source_code/preprocess_gnn_cache.py
source_code/reference_gnn_dataset.py
source_code/reference_gnn_model.py
source_code/reference_structurer.py
source_code/spatial_gnn.py
source_code/train_gnn.py
scripts/apply_mineru_gnn_refinement.py
scripts/apply_reference_structurer.py
scripts/audit_bibliography_references.py
scripts/audit_mineru_gnn_v0.py
scripts/audit_reference_label_dataset.py
scripts/build_reference_label_dataset.py
scripts/compare_frontends_qa.py
scripts/evaluate_native_v2_acceptance.py
scripts/export_native_block_candidates.py
scripts/fuse_reference_labels_v2.py
scripts/llm_adjudicate_reference_labels.py
scripts/merge_reference_llm_labels.py
scripts/render_frontend_compare_overlay.py
scripts/review_mineru_gnn_parent_refinement.py
scripts/run_native_block_consolidation.sh
scripts/smoke_check_figure_ir.py
scripts/summarize_gnn_task_selection.py
scripts/tensorize_reference_labels.py
scripts/train_float_env_policy_smoke.py
scripts/train_mineru_parent_smoke.py
scripts/train_reference_continuation_smoke.py
scripts/train_reference_start_smoke.py
scripts/verify_mineru_gnn_v1_clean.py
```

Deleted tracked generated/historical paths include `archive/`, `demo_frontend_compare_overlay_phase_c2/`, `reports_native_v2_evidence_gated_qa/`, and `reports_qualitative_native_blocks_full/`.

Current active top-level layout:

```text
README.md
configs/
data/
deploy_to_server.sh
docs/
requirements.txt
requirements_server.txt
scripts/
src/
tests/
upload_to_server.sh
verify_environment.py
```

Ignored local/reference top-level paths:

```text
.env.local
.venv_kaggle/
_legacy_reference/
```

## Important Caution

Many generated artifacts and caches are already tracked in the old Git history, including model cache files, reports, server cache backup files, and checkpoints. `.gitignore` prevents future additions but does not remove historical tracked files.

The first clean GitHub commit should be prepared carefully:

1. Decide whether to restore deleted source files such as old GNN/reference modules.
2. Decide whether generated artifacts should be removed from Git in a separate cleanup commit.
3. Do not include private credentials.
4. Do not include bulk data.

## Recommended Next Move

Create a clean repository baseline in two commits:

```text
commit 1: active skeleton + environment/docs/sync safety changes
commit 2: deliberate removal of old tracked generated artifacts and obsolete source paths
```

This keeps the recovery story readable and makes future rollback less painful.
