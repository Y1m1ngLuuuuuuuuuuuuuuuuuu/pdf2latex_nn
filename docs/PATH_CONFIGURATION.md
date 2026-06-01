# Path Configuration

Path resolution is centralized in:

```text
src/config/project_paths.py
```

The project now supports multiple homes: local macOS source, AutoDL runtime,
future WSL source/runtime, and paper workspaces. New active code should use this
path layer instead of hard-coded absolute paths.

## Environment Variables

- `PDF2LATEX_PROJECT_ROOT`
- `PDF2LATEX_DATA_ROOT`
- `PDF2LATEX_OUTPUT_ROOT`
- `PDF2LATEX_REPORT_ROOT`
- `PDF2LATEX_RUNTIME_ROOT`
- `PDF2LATEX_PAPER_ROOT`
- `PDF2LATEX_AUTODL_ROOT`
- `PDF2LATEX_CONFIG`

## Precedence

1. Explicit CLI/path argument.
2. Environment variable.
3. `PDF2LATEX_CONFIG` or `config/paths.local.yaml`.
4. `config/paths.example.yaml`.
5. Repository-relative default.

## macOS Local Example

Canonical source path:

```bash
export PDF2LATEX_PROJECT_ROOT=/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction
export PDF2LATEX_DATA_ROOT=/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction/data
export PDF2LATEX_REPORT_ROOT=/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction/data/09_eval_reports
export PDF2LATEX_PAPER_ROOT=/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

Compatibility path for old commands:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

This is a symlink to the canonical source path. Do not use it in new docs unless
you are explaining legacy compatibility.

## AutoDL Example

```bash
export PDF2LATEX_PROJECT_ROOT=/root/autodl-tmp/pdf2latex_nn
export PDF2LATEX_DATA_ROOT=/root/autodl-tmp/pdf2latex_nn/data
export PDF2LATEX_OUTPUT_ROOT=/root/autodl-tmp/pdf2latex_nn/data/09_eval_reports
export PDF2LATEX_AUTODL_ROOT=/root/autodl-tmp/pdf2latex_nn
```

## Future WSL Example

```bash
export PDF2LATEX_PROJECT_ROOT=/home/<user>/projects/pdf2latex-observable-reconstruction
export PDF2LATEX_DATA_ROOT=/mnt/d/pdf2latex_data
export PDF2LATEX_OUTPUT_ROOT=/mnt/d/pdf2latex_outputs
export PDF2LATEX_CONFIG=/home/<user>/projects/pdf2latex-observable-reconstruction/config/paths.local.yaml
```

## Check Resolved Paths

```bash
python3 scripts/setup/print_project_paths.py
```

## Rule for Future Paper Modules

Paper modules should receive paths through CLI arguments, environment
variables, or the central path config. A paper module must not bake a local
paper workspace, AutoDL root, or WSL path into reusable source code.
