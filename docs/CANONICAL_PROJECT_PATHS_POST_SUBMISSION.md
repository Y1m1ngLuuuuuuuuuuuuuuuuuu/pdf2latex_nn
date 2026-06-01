# Canonical Project Paths After Local Reorganization

Last updated: 2026-06-01.

## Local Source

Canonical source repository:

```text
/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction
```

Compatibility symlink for older scripts, reports, and Codex contexts:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Use the canonical path in new documentation and configuration. Keep the symlink
only as a compatibility entrypoint.

## Local Sibling Areas

```text
/Users/lu/Code/Project/pdf2latex_nn/project_process_history/
/Users/lu/Code/Project/pdf2latex_nn/legacy_runtime_materials/
/Users/lu/Code/Project/pdf2latex_nn/legacy_reference/
/Users/lu/Code/Project/pdf2latex_nn/local_envs/
/Users/lu/Code/Project/pdf2latex_nn/private_config_do_not_upload/
```

These folders are intentionally outside the GitHub source repository.

## Paper Workspace

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

The paper workspace owns manuscript sources, final PDFs, figures, tables,
submission packages, and paper-facing evidence. It should not be folded into
the GitHub source repo.

## AutoDL Runtime

```text
/root/autodl-tmp/pdf2latex_nn
```

AutoDL owns heavy runtime material: raw PDFs, MinerU outputs, TeX sources,
per-document generated outputs, compile logs, checkpoints, graph tensors,
Nougat runtime material, and export tar packages.

## Canonical Evidence

Source-repo final PRCV evidence:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531
```

Paper-facing PRCV evidence:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh/paper_assets/00_FINAL_EVIDENCE_20260531
```

AutoDL summary evidence:

```text
/root/autodl-tmp/pdf2latex_nn/data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531
```

## Future WSL Suggestions

Suggested source root:

```text
/home/<user>/projects/pdf2latex-observable-reconstruction
```

Suggested data and output roots:

```text
/mnt/d/pdf2latex_data
/mnt/d/pdf2latex_outputs
```

Use `PDF2LATEX_CONFIG` or environment variables rather than editing source
files for machine-specific paths.
