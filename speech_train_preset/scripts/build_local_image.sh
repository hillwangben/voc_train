#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${PRESET_DIR}/env/speech_train_all-env"
IMAGE_NAME="${IMAGE_NAME:-speech-train-all:local-cpu}"

echo "[build] context=${ENV_DIR}"
echo "[build] image=${IMAGE_NAME}"

docker build \
  --file "${ENV_DIR}/Dockerfile.cpu" \
  --tag "${IMAGE_NAME}" \
  "${ENV_DIR}"

docker image inspect "${IMAGE_NAME}" \
  --format '[build] image_id={{.Id}} size={{.Size}} created={{.Created}}'
