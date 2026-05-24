#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${PROFILE:-base}"  # base | server
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${USE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
  # AutoDL network accelerator; harmlessly skipped elsewhere.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

cd "${PROJECT_ROOT}"
echo "[setup] project: ${PROJECT_ROOT}"
echo "[setup] venv: ${VENV_DIR}"
echo "[setup] python: ${PYTHON_BIN}"
echo "[setup] profile: ${PROFILE}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ "${PROFILE}" == "server" ]]; then
  python -m pip install -r requirements_server.txt
fi

python verify_environment.py --profile "${PROFILE}"

cat <<EOF

[setup] done.

Activate later with:
  source ${VENV_DIR}/bin/activate

For full PDF/LaTeX reconstruction, install system tools separately:
  - TeX Live / MacTeX / latexmk / xelatex / pdflatex
  - poppler tools
  - MinerU runtime if you need fresh MinerU outputs
EOF
