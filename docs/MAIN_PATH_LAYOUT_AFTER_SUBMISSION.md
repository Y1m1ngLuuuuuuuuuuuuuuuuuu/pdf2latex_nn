# Main Path Layout After Submission

Last updated: 2026-06-01.

This is the short map for what belongs in each area after the PRCV submission
and local source-folder reorganization.

## Source Repository

Path:

```text
/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction
```

Keep:

- source code under `src/`
- entrypoint scripts under `scripts/`
- audit and utility tools under `tools/`
- tests under `tests/`
- project documentation under `docs/`
- configuration examples under `config/` and `configs/`
- final summary evidence under `data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/`
- path configuration utilities

Do not let this repo accumulate:

- raw PDFs
- MinerU output corpora
- generated PDFs or full compile logs
- checkpoints and graph tensors
- paper drafts or submission packages
- AutoDL/netdisk tar packages
- private environment files

## Process History

Path:

```text
/Users/lu/Code/Project/pdf2latex_nn/project_process_history
```

Contains old reports, diagnostic runs, retired eval folders, and local
reorganization reports. These are useful for audit trails but should not be the
starting point for current paper claims.

## Legacy Runtime Materials

Path:

```text
/Users/lu/Code/Project/pdf2latex_nn/legacy_runtime_materials
```

Contains local generated/sample outputs and other runtime material that should
not live in the source repo.

## Private Configuration

Path:

```text
/Users/lu/Code/Project/pdf2latex_nn/private_config_do_not_upload
```

Contains ignored private files such as `.env.local`. Do not commit or package
this folder.

## Paper Workspace

Path:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

Keep:

- final submitted paper package
- final source, PDF, bibliography, figures, and LNCS template files
- paper-facing canonical evidence
- final table sources and locked numbers

The paper workspace is separate from the project source repo.

## AutoDL Runtime Root

Path:

```text
/root/autodl-tmp/pdf2latex_nn
```

Keep in place until backup/reset decisions are explicit:

- raw PDFs
- MinerU outputs
- TeX sources used as gold targets
- graph tensors and checkpoints
- selected200/selected2000 per-document outputs
- generated TeX/PDF artifacts
- compile logs
- Nougat outputs and runtime material
- export tar/checksum folders

## Evidence Rule

Final PRCV evidence starts at:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Older dated folders are process history unless explicitly listed in the final
evidence index. Future papers should create their own evidence registry instead
of overloading the PRCV registry.
