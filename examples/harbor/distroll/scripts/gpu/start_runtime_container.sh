#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_command docker
: "${GPU_RUNTIME_IMAGE:?set GPU_RUNTIME_IMAGE to the pre-pulled derived image RepoDigest}"
if [[ "${ALLOW_LOCAL_GPU_IMAGE:-0}" != "1" && \
      ! "${GPU_RUNTIME_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "GPU_RUNTIME_IMAGE must be an immutable RepoDigest" >&2
  exit 1
fi

GPU_CONTAINER_NAME="${GPU_CONTAINER_NAME:-dressage-distroll-gpu}"
GPU_MODEL_DIR="${GPU_MODEL_DIR:-/models}"
GPU_DATA_DIR="${GPU_DATA_DIR:-${PROJECT_ROOT}/data}"
GPU_OUTPUT_DIR="${GPU_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
GPU_CONFIG_DIR="${GPU_CONFIG_DIR:-/etc/dressage-distroll}"
DRESSAGE_ROOT="${DRESSAGE_ROOT:-$(cd -- "${DISTROLL_ROOT}/../../.." && pwd)}"

for path in "${GPU_MODEL_DIR}" "${GPU_DATA_DIR}" "${GPU_OUTPUT_DIR}" "${GPU_CONFIG_DIR}"; do
  [[ -e "${path}" ]] || {
    echo "required GPU container mount is missing: ${path}" >&2
    exit 1
  }
done
docker image inspect "${GPU_RUNTIME_IMAGE}" >/dev/null 2>&1 || {
  echo "GPU runtime is not pre-pulled: ${GPU_RUNTIME_IMAGE}" >&2
  exit 1
}
if docker container inspect "${GPU_CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "GPU container already exists: ${GPU_CONTAINER_NAME}" >&2
  exit 1
fi

host_project_root="${PROJECT_ROOT}"
container_dressage_root=/workspace/Dressage
container_project_root=${container_dressage_root}/examples/harbor/distroll
for path in \
  "${host_project_root}/contract/src/agentic_rollout_contract" \
  "${host_project_root}/gpu_adapter/src/dressage_distroll_gpu_adapter" \
  "${DRESSAGE_ROOT}/dressage" \
  "${host_project_root}/runtime-src/slime/slime" \
  "${host_project_root}/runtime-src/Megatron-LM/megatron"; do
  [[ -d "${path}" ]] || {
    echo "required host GPU source is missing: ${path}" >&2
    exit 1
  }
done
source_pythonpath="${container_project_root}/contract/src:${container_project_root}/gpu_adapter/src:${container_dressage_root}:${container_project_root}/runtime-src/slime:${container_project_root}/runtime-src/Megatron-LM"

docker run -d \
  --name "${GPU_CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --mount "type=bind,src=${GPU_MODEL_DIR},dst=/models,readonly" \
  --mount "type=bind,src=${DRESSAGE_ROOT},dst=${container_dressage_root},readonly" \
  --mount "type=bind,src=${host_project_root},dst=${container_project_root},readonly" \
  --mount "type=bind,src=${GPU_DATA_DIR},dst=${container_project_root}/data,readonly" \
  --mount "type=bind,src=${GPU_OUTPUT_DIR},dst=${container_project_root}/outputs" \
  --mount "type=bind,src=${GPU_CONFIG_DIR},dst=/etc/dressage-distroll,readonly" \
  --env "GPU_SOURCE_ROOT=${container_project_root}" \
  --env "PROJECT_ROOT=${container_project_root}" \
  --env "GPU_RUNTIME_LOCK=${container_project_root}/configs/gpu-runtime-lock.json" \
  --env "DRESSAGE_ROOT=${container_dressage_root}" \
  --env "SLIME_ROOT=${container_project_root}/runtime-src/slime" \
  --env "MEGATRON_ROOT=${container_project_root}/runtime-src/Megatron-LM" \
  --env "PYTHONPATH=${source_pythonpath}" \
  "${GPU_RUNTIME_IMAGE}" sleep infinity

docker exec "${GPU_CONTAINER_NAME}" \
  /opt/dressage-distroll/bin/python \
  "${container_project_root}/tools/check_gpu_runtime.py" --local >/dev/null
echo "GPU runtime container started: ${GPU_CONTAINER_NAME}"
