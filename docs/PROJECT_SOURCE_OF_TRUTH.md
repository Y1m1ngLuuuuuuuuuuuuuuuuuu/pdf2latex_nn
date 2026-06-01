# Project Source of Truth

Last updated: 2026-06-01.

This document defines the current operational source of truth for the whole
PDF2LaTeX project. PRCV 2026 is a submitted paper module inside the broader
project; it is not the whole project.

## Three-Layer Ownership

### 1. GitHub Source Repository

Role: source-control home.

Canonical local path:

```text
/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction
```

Compatibility symlink:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

GitHub remote:

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex-observable-reconstruction.git
```

This repository keeps source code, tests, tools, configuration examples,
project documentation, and small evidence summaries. It must not become a raw
data or paper-draft archive.

### 2. AutoDL Runtime

Role: runtime and heavy-material home.

Canonical remote path:

```text
/root/autodl-tmp/pdf2latex_nn
```

AutoDL stores heavy runtime material: raw PDFs, MinerU outputs, TeX sources,
generated per-document outputs, full logs, checkpoints, graph tensors, Nougat
runtime material, and tar packages prepared for netdisk backup.

### 3. Local Paper Workspace

Role: paper-authoring home.

Canonical PRCV workspace:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

This workspace owns LaTeX manuscripts, final PDFs, figures, bibliography,
submission packages, table sources, paper assets, and paper-facing locked
numbers. It is separate from the GitHub source repository.

## Sibling Local Areas

The parent directory now separates project roles:

```text
/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction/  source repo
/Users/lu/Code/Project/pdf2latex_nn/project_process_history/              old reports and process history
/Users/lu/Code/Project/pdf2latex_nn/legacy_runtime_materials/             local generated/sample runtime material
/Users/lu/Code/Project/pdf2latex_nn/legacy_reference/                     legacy reference material
/Users/lu/Code/Project/pdf2latex_nn/local_envs/                           local virtual environments
/Users/lu/Code/Project/pdf2latex_nn/private_config_do_not_upload/          private ignored config
```

## Current Project Mainline

Reusable platform pipeline:

```text
PDF / parser outputs
  -> Observable Fact Layer
  -> DocumentIR
  -> RenderTreeIR
  -> compile-safe role renderers
  -> generated LaTeX
  -> ComparisonStructure evaluation
```

Current non-goals:

- deterministic recovery of the author's original source-level TeX AST
- broad semantic table-cell reconstruction as a solved claim
- broad Algorithm renderer as a current paper claim
- rendering directly from a GNN relation graph

GNN and relation-learning components remain useful research modules and future
paper material, but they are not the current PRCV-facing production path.

## Paper Module Policy

Every paper should be treated as a module over the project core. A paper module
must define:

- dataset slice and denominator
- methods and baselines
- claim boundary
- evidence registry
- table sources and locked numbers
- backup/package policy
- what not to claim

The PRCV 2026 module is documented in:

```text
docs/PRCV_EVIDENCE_REGISTRY_20260531.md
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Future paper modules should create their own registry while reusing the same
core interfaces in `docs/INTERFACE_DESIGN_CURRENT_20260601.md`.

## Final PRCV Evidence

Canonical source-repo evidence:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Paper-facing evidence:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh/paper_assets/00_FINAL_EVIDENCE_20260531/
```

AutoDL summary evidence:

```text
/root/autodl-tmp/pdf2latex_nn/data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Selected2000 and selected200 have different roles:

- selected2000: primary large-scale direct-parser comparison, n=1972 fair
  intersection, Ours / ContentList Direct / MinerU Direct.
- selected200: controlled four-method comparison including Nougat.
- compile and visual QA: defined only for complete LaTeX outputs.

## Backup State

AutoDL material has been separated into netdisk-oriented backup packages:

- PRCV clean export package/folder
- full 8000 raw PDFs and MinerU outputs
- extra Nougat runtime/checkpoint/environment package

These packages are runtime/material backups, not GitHub source artifacts.

## Configuration

Path resolution is centralized in:

```text
src/config/project_paths.py
docs/PATH_CONFIGURATION.md
```

Preferred precedence:

1. explicit CLI path
2. environment variable
3. local config
4. example config
5. repository-relative default

Do not hard-code Mac, AutoDL, WSL, or paper-workspace absolute paths in new
active code.

## Commit Policy

Commit:

- source code
- tests
- tools
- docs
- configuration examples
- small final evidence summaries

Do not commit:

- raw PDFs
- MinerU outputs
- TeX source corpora
- generated TeX/PDF directories
- full compile logs
- checkpoints
- graph tensors
- paper drafts and submission packages
- tarballs
- `.env.local`, credentials, tokens, keys, or passwords

## Current Reading Order

Start with these documents:

```text
README.md
docs/PROJECT_SCOPE_AND_PAPER_MODULES_20260601.md
docs/INTERFACE_DESIGN_CURRENT_20260601.md
docs/PROJECT_ARCHITECTURE_CURRENT_20260531.md
docs/PRCV_EVIDENCE_REGISTRY_20260531.md
docs/PATH_CONFIGURATION.md
```

Older v7, v8, GNN, ablation, and sprint docs are process history unless a
current registry explicitly promotes them.
