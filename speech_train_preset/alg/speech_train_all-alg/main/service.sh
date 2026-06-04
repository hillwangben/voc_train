#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK_TYPE="${TASK_TYPE:-asr}"
MODEL_DIR="${MODEL_DIR:-/workspace/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[speech-service] task_type=${TASK_TYPE}"
echo "[speech-service] model_dir=${MODEL_DIR}"
echo "[speech-service] listen=${HOST}:${PORT}"

if [ -f "src/service.py" ]; then
  "${PYTHON_BIN}" src/service.py \
    --task-type "${TASK_TYPE}" \
    --model-dir "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
else
  echo "src/service.py not found. Please place real service code under main/src/."
fi
