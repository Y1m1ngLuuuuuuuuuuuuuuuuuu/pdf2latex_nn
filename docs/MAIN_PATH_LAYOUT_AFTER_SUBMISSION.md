# Main Path Layout After Submission

This document is the short map for what should remain in each main project
area after the PRCV submission.

## Local Code Root

Keep:

- source code under `src/`
- entrypoint scripts under `scripts/`
- audit and utility tools under `tools/`
- tests under `tests/`
- project documentation under `docs/`
- configuration examples under `config/` and `configs/`
- final summary evidence under `data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/`
- path configuration utilities

Do not let the local code root accumulate long-term:

- old generated outputs
- old compile logs
- failed figure variants
- scratch directories
- raw PDFs
- checkpoints
- per-document runtime outputs

## Paper Workspace

Keep:

- final submitted paper package:
  `/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh/submission_package_20260531/`
- final source, PDF, bibliography, figures, and template files
- paper-facing canonical evidence:
  `paper_assets/00_FINAL_EVIDENCE_20260531/`
- final table sources and locked numbers

Do not let the paper workspace accumulate long-term:

- unused drafts
- old figure attempts
- old table variants
- patch previews
- LaTeX auxiliary files

## AutoDL Runtime Root

Keep in place:

- raw PDFs
- MinerU outputs
- TeX sources used as gold targets
- graph tensors
- checkpoints
- selected200/selected2000 per-document outputs
- generated TeX/PDF artifacts
- compile logs
- Nougat outputs

Do not let the AutoDL root accumulate long-term:

- obsolete overlays
- root fragments
- upload scratch directories
- temporary staging folders

## Evidence Rule

Final PRCV evidence starts at
`data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/`. Older dated folders are
process history unless explicitly listed in the final evidence index.

