# Path Configuration

Path resolution is centralized in `src/config/project_paths.py`.

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

## Examples

macOS local:

```bash
export PDF2LATEX_PROJECT_ROOT=/Users/lu/Code/Project/pdf2latex_nn/test_4_19
export PDF2LATEX_DATA_ROOT=/Users/lu/Code/Project/pdf2latex_nn/test_4_19/data
```

WSL:

```bash
export PDF2LATEX_PROJECT_ROOT=/home/<user>/projects/pdf2latex_nn
export PDF2LATEX_DATA_ROOT=/mnt/d/pdf2latex_nn_data
export PDF2LATEX_OUTPUT_ROOT=/mnt/d/pdf2latex_nn_outputs
```

AutoDL:

```bash
export PDF2LATEX_PROJECT_ROOT=/root/autodl-tmp/pdf2latex_nn
export PDF2LATEX_DATA_ROOT=/root/autodl-tmp/pdf2latex_nn/data
```

