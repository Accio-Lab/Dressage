#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
"${GPU_RAY}" status || true
if [[ -f "${PROJECT_ROOT}/outputs/pids/ray-submission-id" ]]; then
  submission_id="$(cat "${PROJECT_ROOT}/outputs/pids/ray-submission-id")"
  "${GPU_RAY}" job status "${submission_id}" || true
fi
