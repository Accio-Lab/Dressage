#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
set -a
source "${GPU_ENV_FILE:-${PROJECT_ROOT}/configs/gpu.training.env.example}"
set +a

RUN_NAME="${RUN_NAME:-tmax-grpo-sync}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/dressage-ray}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
ROLLOUT_GATEWAY_RESOURCE_NAME="${ROLLOUT_GATEWAY_RESOURCE_NAME:-rollout_gateway}"
: "${GPU_NODE_IP:?set GPU_NODE_IP to the Ray Head internal address}"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "${GPU_COUNT}" -eq "${NUM_GPUS_PER_NODE}" ]] || {
  echo "Ray head expects ${NUM_GPUS_PER_NODE} visible GPUs; found ${GPU_COUNT}" >&2
  exit 1
}
if "${GPU_RAY}" status >/dev/null 2>&1; then
  echo "Ray is already running on ${GPU_NODE_IP}" >&2
  exit 1
fi

mkdir -p "${RAY_TEMP_DIR}"
"${GPU_RAY}" start --head \
  --node-ip-address "${GPU_NODE_IP}" \
  --port "${RAY_PORT:-6379}" \
  --num-gpus "${NUM_GPUS_PER_NODE}" \
  --resources "{\"${ROLLOUT_GATEWAY_RESOURCE_NAME}\":1}" \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "${RAY_DASHBOARD_PORT:-8265}" \
  --temp-dir "${RAY_TEMP_DIR}" \
  --disable-usage-stats

echo "Ray head ${GPU_NODE_IP} started with gateway resource ${ROLLOUT_GATEWAY_RESOURCE_NAME}"
