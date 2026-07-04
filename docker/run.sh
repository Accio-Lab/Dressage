#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/image_tag.sh"

HOST_CHECKPOINT_DIR="${HOST_CHECKPOINT_DIR:-${BASE_FOLDER:-${HOME}/models}}"
CONTAINER_CHECKPOINT_DIR="${CONTAINER_CHECKPOINT_DIR:-${HOST_CHECKPOINT_DIR}}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"

gpu_args=()
if [[ "${DOCKER_GPUS}" != "none" ]]; then
  gpu_args=(--gpus "${DOCKER_GPUS}")
fi

mount_args=(
  -v "${REPO_ROOT}:/root/Dressage"
)
env_args=(
  -e "DRESSAGE_BLACKBOX_RUNNER_MODE=bwrap"
  -e "DRESSAGE_BLACKBOX_BWRAP_BIN=bwrap"
  -e "OPENCODE_BIN=/usr/local/bin/opencode"
  -e "OPENCLAW_BIN=/usr/local/bin/openclaw"
  -e "CLAUDE_CODE_BIN=/usr/local/bin/claude"
  -e "CODEX_BIN=/usr/local/bin/codex"
)

if [[ -d "${HOST_CHECKPOINT_DIR}" ]]; then
  mount_args+=(-v "${HOST_CHECKPOINT_DIR}:${CONTAINER_CHECKPOINT_DIR}")
  env_args+=(-e "BASE_FOLDER=${CONTAINER_CHECKPOINT_DIR}")
else
  echo "checkpoint directory not mounted because it does not exist: ${HOST_CHECKPOINT_DIR}" >&2
  echo "set HOST_CHECKPOINT_DIR or BASE_FOLDER to mount model/checkpoint files" >&2
fi

docker run --rm -it \
  "${gpu_args[@]}" \
  --network host \
  --ipc host \
  --privileged \
  "${mount_args[@]}" \
  "${env_args[@]}" \
  -w /root/Dressage \
  "${DRESSAGE_IMAGE_NAME}" \
  "$@"
