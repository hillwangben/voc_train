#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_TYPE="${TASK_TYPE:-asr}"
DATA_DIR="${DATA_DIR:-/workspace/input}"
MODEL_DIR="${MODEL_DIR:-/workspace/model}"
SAVE_DIR="${SAVE_DIR:-/workspace/output/train}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND="${BACKEND:-cpu_reference}"

mkdir -p "${SAVE_DIR}"

echo "[speech-train] task_type=${TASK_TYPE}"
echo "[speech-train] data_dir=${DATA_DIR}"
echo "[speech-train] model_dir=${MODEL_DIR}"
echo "[speech-train] save_dir=${SAVE_DIR}"

if [ -f "src/train.py" ]; then
  "${PYTHON_BIN}" src/train.py \
    --task-type "${TASK_TYPE}" \
    --data-dir "${DATA_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --save-dir "${SAVE_DIR}" \
    --backend "${BACKEND}"
else
  echo "src/train.py not found. Please place real training code under main/src/."
fi
