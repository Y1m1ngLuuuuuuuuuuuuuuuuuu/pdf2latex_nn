# Environment Setup

**Last updated**: 2026-05-24

This file records the reproducible Python environment for moving the project to
a laptop or a fresh AutoDL machine.

## Recommended Choice

Use conda when possible:

```bash
cd /path/to/pdf2latex_nn
bash scripts/setup/create_conda_env.sh
conda activate pdf2latex
```

Use venv when you do not want conda:

```bash
cd /path/to/pdf2latex_nn
PYTHON_BIN=python3.11 bash scripts/setup/create_venv.sh
source .venv/bin/activate
```

## Profiles

Base profile:

```bash
bash scripts/setup/create_conda_env.sh
```

Installs:

```text
v8 layout reconstruction dependencies
PDF/image processing
TeX parsing
GNN relation branch dependencies
tests and formatting tools
```

Server profile:

```bash
PROFILE=server USE_NETWORK_TURBO=1 bash scripts/setup/create_conda_env.sh
```

Adds:

```text
arXiv/S3 helpers
optional PaddleOCR / detector / formula packages
OpenAI SDK for API-baseline experiments
```

The server profile is intentionally heavier.  It is not needed just to inspect
generated outputs or run the current v8 renderer over existing MinerU outputs.

## Dependency Files

```text
requirements.txt          base project environment
requirements_server.txt   server / AutoDL extras
environment.yml           conda environment wrapper
verify_environment.py     environment smoke checker
```

## System Dependencies

Python packages are not enough for full reconstruction.  Install these outside
pip if you need compilation/rendered-output evaluation:

```text
TeX Live / MacTeX
latexmk
pdflatex
xelatex
poppler tools: pdfinfo, pdftoppm
MinerU runtime if generating new MinerU outputs
```

On macOS:

```bash
brew install poppler
```

Install MacTeX or BasicTeX separately.

On Ubuntu/AutoDL:

```bash
apt-get update
apt-get install -y poppler-utils texlive-latex-base texlive-latex-extra texlive-fonts-recommended latexmk
```

## Validation

After installation:

```bash
python verify_environment.py --profile base
python -m pytest tests/test_v8_render_tree_lists.py tests/test_generation_style_citations.py
```

If only the v8 renderer is needed, `verify_environment.py --profile base` is the
right smoke check.  If you are rebuilding data on AutoDL, use
`--profile server`.

## Notes

- Do not commit `.venv`, conda env folders, pip caches, or model caches.
- API keys belong in local environment variables or `.env.local`, never in Git.
- Large data stays under `data/` and should be moved separately from the Python
  environment.
