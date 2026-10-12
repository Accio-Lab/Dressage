#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
: "${RUN_NAME:?set the exact active run ID}"
: "${STEP:?set the exact uncommitted rollout step}"
"${GPU_PYTHON}" "${PROJECT_ROOT}/tools/request_stop.py" \
  --run-id "${RUN_NAME}" --step "${STEP}" \
  --diagnostics-dir "${ROLLOUT_DIAGNOSTICS_DIR:-${PROJECT_ROOT}/outputs/rl-state/${RUN_NAME}}"
echo "Cooperative stop requested; wait for cleanup and generation barriers."
