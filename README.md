# PDF2LaTeX Observable Reconstruction

PDF2LaTeX is a long-running research and engineering project for reconstructing
LaTeX-oriented document structure from scientific PDFs. This repository is the
source-code home for the project. It is not a single-paper artifact dump, and it
does not contain heavy runtime data by design.

The current project identity is:

```text
scientific PDF / parser outputs
  -> Observable Fact Layer
  -> DocumentIR
  -> RenderTreeIR
  -> compile-safe role renderers
  -> generated LaTeX and neutral structural evaluation
```

The PRCV 2026 submission is one paper module built on top of these interfaces.
Future papers should reuse the same project interfaces and add their own paper
module registries, evidence folders, and claim boundaries.

## Start Here

- Current project scope and paper-module boundary:
  `docs/PROJECT_SCOPE_AND_PAPER_MODULES_20260601.md`
- Current interface design:
  `docs/INTERFACE_DESIGN_CURRENT_20260601.md`
- Documentation index:
  `docs/DOCUMENTATION_INDEX_20260601.md`
- Current source of truth:
  `docs/PROJECT_SOURCE_OF_TRUTH.md`
- Current architecture:
  `docs/PROJECT_ARCHITECTURE_CURRENT_20260531.md`
- Path configuration:
  `docs/PATH_CONFIGURATION.md`
- PRCV evidence registry:
  `docs/PRCV_EVIDENCE_REGISTRY_20260531.md`

Older v7, v8, GNN, ablation, and sprint documents remain in `docs/` as process
history. Use the documents above before relying on older reports or claims.

## Repository Role

This Git repository should contain:

- source code under `src/`
- entrypoint scripts under `scripts/`
- audit, conversion, and baseline tools under `tools/`
- tests under `tests/`
- project documentation under `docs/`
- configuration examples under `config/` and `configs/`
- small final evidence summaries under `data/09_eval_reports/`

This Git repository should not contain:

- raw PDF corpora
- MinerU parser-output corpora
- per-document generated TeX/PDF outputs
- full compile logs
- checkpoints or graph tensors
- paper draft workspaces
- AutoDL tarballs or netdisk backup packages
- secrets such as `.env.local`, tokens, passwords, keys, or credentials

## Current Local Layout

Canonical source repository:

```text
/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction
```

Compatibility symlink for older commands and reports:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Sibling folders separate non-source material from the code repository:

```text
/Users/lu/Code/Project/pdf2latex_nn/project_process_history/
/Users/lu/Code/Project/pdf2latex_nn/legacy_runtime_materials/
/Users/lu/Code/Project/pdf2latex_nn/legacy_reference/
/Users/lu/Code/Project/pdf2latex_nn/local_envs/
/Users/lu/Code/Project/pdf2latex_nn/private_config_do_not_upload/
```

AutoDL remains the runtime home:

```text
/root/autodl-tmp/pdf2latex_nn
```

The PRCV manuscript workspace remains local and separate from this source repo:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

## PRCV 2026 Paper Module

The submitted PRCV module evaluates observable-fact-guided PDF-to-LaTeX
reconstruction with two complementary evidence tracks:

- selected2000: primary large-scale direct-parser comparison on a 1972-document
  fair intersection for Ours, ContentList Direct, and MinerU Direct.
- selected200: controlled four-method comparison including Nougat as an
  external MMD/Markdown baseline.

Large-scale usability for Ours on selected2000 is recorded as:

```text
generated.tex: 2000/2000
compile success: 1852/2000
comparison conversion: 1999/2000
structure metrics: 1980/2000
```

Do not claim Nougat selected2000, selected2000 four-method completion,
selected2000 metrics 2000/2000, source-level TeX AST recovery, solved semantic
table-cell reconstruction, or compile/visual QA for parser-output baselines.

Final PRCV evidence starts at:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

## Core Interfaces

The project should evolve through stable interfaces rather than paper-specific
shortcuts:

- path/runtime configuration: `src/config/project_paths.py`
- parser adapters: `src/adapters/`
- observable parsing and reading-order facts: `src/perception/`
- document and render IR schemas: `src/ir/`
- reasoning modules: `src/reasoning/`
- role renderers: `src/generation/ir_renderers/`
- structural and compile evaluation: `src/evaluation/`
- paper/evidence registries: `data/09_eval_reports/` and `docs/`

Each future paper should define its paper module separately from the reusable
project core: its dataset slice, evidence denominator, baselines, output tables,
claim boundary, and package/backup policy.

## Quick Checks

Resolve configured paths:

```bash
python3 scripts/setup/print_project_paths.py
```

Compile lightweight Python files after source edits:

```bash
python3 -m py_compile src/config/project_paths.py scripts/setup/print_project_paths.py
```

Check Git status:

```bash
git status --short
git log --oneline -5
```
