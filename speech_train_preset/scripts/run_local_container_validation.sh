#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-speech-train-all:local-cpu}"
CONTAINER_NAME="${CONTAINER_NAME:-speech-train-asr-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRESET_DIR}/runs/container_validation/asr_vosk_small}"
ALGORITHM_DIR="${PRESET_DIR}/alg/speech_train_all-alg"
DATASET_DIR="${PRESET_DIR}/dataset/speech_train_sample-dataset"
MODEL_DIR="${PRESET_DIR}/model/asr_vosk_small-cpu-model/model/asr_vosk_small"
CATALOG_FILE="${PRESET_DIR}/docs/open_source_cpu_models.json"

mkdir -p "${OUTPUT_DIR}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "[run] image=${IMAGE_NAME}"
echo "[run] container=${CONTAINER_NAME}"
echo "[run] task_type=asr backend=vosk_small"
echo "[run] dataset=${DATASET_DIR} -> /workspace/input (read-only)"
echo "[run] model=${MODEL_DIR} -> /workspace/model (read-only)"
echo "[run] output=${OUTPUT_DIR} -> /workspace/output"

docker create \
  --name "${CONTAINER_NAME}" \
  --workdir /workspace/preset/alg/speech_train_all-alg/main \
  --env TASK_TYPE=asr \
  --env BACKEND=vosk_small \
  --env DATA_DIR=/workspace/input \
  --env MODEL_DIR=/workspace/model \
  --env SAVE_DIR=/workspace/output \
  --mount "type=bind,src=${ALGORITHM_DIR},dst=/workspace/preset/alg/speech_train_all-alg,readonly" \
  --mount "type=bind,src=${DATASET_DIR},dst=/workspace/input,readonly" \
  --mount "type=bind,src=${MODEL_DIR},dst=/workspace/model,readonly" \
  --mount "type=bind,src=${MODEL_DIR},dst=/workspace/preset/model/real_cpu_models/vosk-model-small-en-us-0.15,readonly" \
  --mount "type=bind,src=${CATALOG_FILE},dst=/workspace/preset/docs/open_source_cpu_models.json,readonly" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/workspace/output" \
  "${IMAGE_NAME}" \
  bash -lc './train.sh && SAVE_DIR=/workspace/output/evaluate ./eval.sh' \
  >/dev/null

docker start --attach "${CONTAINER_NAME}"

echo "[run] container retained in state: $(docker inspect --format '{{.State.Status}} (exit={{.State.ExitCode}})' "${CONTAINER_NAME}")"
echo "[run] output files:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f -printf '  %P\n' | sort
