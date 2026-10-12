#!/usr/bin/env bash
set -euo pipefail

# Optional source/package fingerprint diagnostic. Run after adding a Ray node,
# changing mounted runtime sources, or upgrading the GPU environment image.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
set -a
source "${GPU_ENV_FILE:-${PROJECT_ROOT}/configs/gpu.training.env.example}"
set +a

"${GPU_PYTHON}" "${PROJECT_ROOT}/tools/check_gpu_runtime.py" \
  --ray --expected-nodes "${GPU_CLUSTER_NUM_NODES:-1}"
