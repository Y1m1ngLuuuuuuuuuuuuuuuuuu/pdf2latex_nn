#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pdf2latex}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROFILE="${PROFILE:-base}"  # base | server
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${USE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
  # AutoDL network accelerator; harmlessly skipped elsewhere.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found. Install Miniconda/Mambaforge first, or use scripts/setup/create_venv.sh." >&2
    exit 2
  fi
fi

echo "[setup] project: ${PROJECT_ROOT}"
echo "[setup] env: ${ENV_NAME}"
echo "[setup] python: ${PYTHON_VERSION}"
echo "[setup] profile: ${PROFILE}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env '${ENV_NAME}' already exists; reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
if [[ "${INSTALL_CONDA_TOOLS:-1}" == "1" ]]; then
  conda install -y -c conda-forge poppler
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ "${PROFILE}" == "server" ]]; then
  python -m pip install -r "${PROJECT_ROOT}/requirements_server.txt"
fi

python "${PROJECT_ROOT}/verify_environment.py" --profile "${PROFILE}"

cat <<'EOF'

[setup] done.

System tools still needed outside pip/conda for full PDF/LaTeX work:
  - TeX Live / MacTeX / latexmk / xelatex / pdflatex
  - poppler tools if not installed through conda
  - MinerU runtime if you want to generate new MinerU outputs locally

Activate later with:
  conda activate pdf2latex
EOF
