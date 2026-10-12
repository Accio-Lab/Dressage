#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
: "${RAY_HEAD_INTERNAL_IP:?set the Ray Head internal address}"
: "${GPU_NODE_IP:?set this GPU node internal address}"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "${GPU_COUNT}" -gt 0 ]] || {
  echo "no visible GPUs were found" >&2
  exit 1
}

if "${GPU_RAY}" status >/dev/null 2>&1; then
  echo "Ray is already running on ${GPU_NODE_IP}" >&2
  exit 1
fi

"${GPU_RAY}" start \
  --address "${RAY_HEAD_INTERNAL_IP}:${RAY_PORT:-6379}" \
  --node-ip-address "${GPU_NODE_IP}" \
  --num-gpus "${GPU_COUNT}" \
  --disable-usage-stats

echo "Ray GPU worker ${GPU_NODE_IP} joined ${RAY_HEAD_INTERNAL_IP}:${RAY_PORT:-6379}"
