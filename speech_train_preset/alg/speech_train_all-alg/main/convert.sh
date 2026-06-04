#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_TYPE="${TASK_TYPE:-asr}"
TARGET_MODEL="${TARGET_MODEL:-onnx}"
MODEL_DIR="${MODEL_DIR:-/workspace/model}"
SAVE_DIR="${SAVE_DIR:-/workspace/output/adapt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${SAVE_DIR}"

echo "[speech-convert] task_type=${TASK_TYPE}"
echo "[speech-convert] target_model=${TARGET_MODEL}"
echo "[speech-convert] model_dir=${MODEL_DIR}"
echo "[speech-convert] save_dir=${SAVE_DIR}"

if [ -f "src/convert.py" ]; then
  "${PYTHON_BIN}" src/convert.py \
    --task-type "${TASK_TYPE}" \
    --target-model "${TARGET_MODEL}" \
    --model-dir "${MODEL_DIR}" \
    --save-dir "${SAVE_DIR}"
else
  echo "src/convert.py not found. Please place real conversion code under main/src/."
fi
