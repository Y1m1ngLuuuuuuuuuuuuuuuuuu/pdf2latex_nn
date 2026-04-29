#!/usr/bin/env bash
# Sync source files to the AutoDL project directory.
#
# Prefer GitHub for normal code synchronization. This script is only for quick
# emergency syncs before the remote Git workflow is fully rebuilt.

set -euo pipefail

if [ -f .env.local ]; then
  # shellcheck disable=SC1091
  source .env.local
fi

: "${AUTODL_HOST:?Set AUTODL_HOST in .env.local}"
: "${AUTODL_PORT:?Set AUTODL_PORT in .env.local}"
: "${AUTODL_USER:=root}"
: "${AUTODL_PROJECT:=/root/autodl-tmp/pdf2latex_nn}"

SERVER="${AUTODL_USER}@${AUTODL_HOST}"
SSH_OPTS=(-p "${AUTODL_PORT}")
SCP_OPTS=(-P "${AUTODL_PORT}")

echo "=== Syncing lightweight source files to AutoDL ==="
echo "remote: ${SERVER}:${AUTODL_PROJECT}"

ssh "${SSH_OPTS[@]}" "${SERVER}" "mkdir -p '${AUTODL_PROJECT}'"

scp "${SCP_OPTS[@]}" \
  README.md \
  requirements.txt \
  requirements_server.txt \
  verify_environment.py \
  "${SERVER}:${AUTODL_PROJECT}/"

for path in source_code scripts docs; do
  echo "sync ${path}/"
  rsync -az \
    -e "ssh -p ${AUTODL_PORT}" \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "${path}/" "${SERVER}:${AUTODL_PROJECT}/${path}/"
done

echo
echo "Sync complete."
echo "SSH: ssh -p ${AUTODL_PORT} ${SERVER}"
echo "Remote project: cd ${AUTODL_PROJECT}"
