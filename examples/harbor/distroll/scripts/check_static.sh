#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
CHECK_PYTHON="${CHECK_PYTHON:-python3}"
RUFF_BIN="${RUFF_BIN:-ruff}"
require_command "${CHECK_PYTHON}"
require_command "${RUFF_BIN}"
CHECK_TMP=$(mktemp -d)
trap 'rm -rf -- "$CHECK_TMP"' EXIT
export PYTHONPYCACHEPREFIX="${CHECK_TMP}/pycache"
"${CHECK_PYTHON}" -m compileall -q contract/src rollout_service/src gpu_adapter/src tools tests
"${RUFF_BIN}" check --no-cache contract/src rollout_service/src gpu_adapter/src tools tests
"${RUFF_BIN}" format --check --no-cache contract/src rollout_service/src gpu_adapter/src tools tests
while IFS= read -r -d '' script; do bash -n "$script"; done < <(find scripts -type f -name '*.sh' -print0)
git diff --check
echo "Static checks passed; no services or remote commands were executed."
