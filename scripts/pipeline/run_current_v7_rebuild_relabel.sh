#!/usr/bin/env bash
# Rebuild graph tensors and relabel them with the current v7 contract.
#
# This script is the stable production entrypoint after MinerU has already
# produced styled content_v7 JSON files. It intentionally does not run MinerU.

set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT_MANIFEST="${INPUT_MANIFEST:-data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json}"
TAG="${TAG:-v7_current_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-4}"
MAX_DOCS="${MAX_DOCS:-0}"
MODEL_PATH="${MODEL_PATH:-models/huggingface/allenai/scibert_scivocab_uncased}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

REBUILT_MANIFEST="data/00_manifests/${TAG}_rebuilt.json"
LABELED_MANIFEST="data/00_manifests/${TAG}_labeled.json"
GRAPH_DIR="data/06_graph_features/${TAG}_graphs"
LABELED_GRAPH_DIR="data/06_graph_features/${TAG}_labeled_graphs"
CONTENT_DIR="data/02_mineru_outputs/${TAG}_content"
MAPPING_DIR="data/04_ground_truth_ir/${TAG}_mappings"
RUN_LOG="logs/${TAG}_run.log"
REBUILD_ERRORS="logs/${TAG}_rebuild_errors.jsonl"
LABEL_ERRORS="logs/${TAG}_label_errors.jsonl"
DELTA_REPORT="logs/${TAG}_delta.json"

mkdir -p logs

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

max_docs_args=()
if [[ "$MAX_DOCS" != "0" ]]; then
  max_docs_args=(--max-docs "$MAX_DOCS")
fi

{
  echo "[v7-pipeline] start $(date)"
  echo "[v7-pipeline] input=$INPUT_MANIFEST"
  echo "[v7-pipeline] tag=$TAG workers=$WORKERS max_docs=$MAX_DOCS"

  "$PYTHON_BIN" scripts/pipeline/rebuild_graphs_from_manifest.py \
    --input-manifest "$INPUT_MANIFEST" \
    --output-manifest "$REBUILT_MANIFEST" \
    --graph-output-dir "$GRAPH_DIR" \
    --content-output-dir "$CONTENT_DIR" \
    --error-log "$REBUILD_ERRORS" \
    --model-path "$MODEL_PATH" \
    --embedding-device "$EMBEDDING_DEVICE" \
    --workers "$WORKERS" \
    "${max_docs_args[@]}" \
    --force

  "$PYTHON_BIN" scripts/pipeline/relabel_manifest.py \
    --input-manifest "$REBUILT_MANIFEST" \
    --output-manifest "$LABELED_MANIFEST" \
    --graph-output-dir "$LABELED_GRAPH_DIR" \
    --mapping-output-dir "$MAPPING_DIR" \
    --delta-report "$DELTA_REPORT" \
    --error-log "$LABEL_ERRORS" \
    --workers "$WORKERS" \
    --profile-candidate-recall \
    --force

  echo "[v7-pipeline] rebuilt_manifest=$REBUILT_MANIFEST"
  echo "[v7-pipeline] labeled_manifest=$LABELED_MANIFEST"
  echo "[v7-pipeline] done $(date)"
} 2>&1 | tee "$RUN_LOG"
