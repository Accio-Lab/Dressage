#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
set -a
source "${WORKER_ENV_FILE:-/etc/dressage-distroll/worker.env}"
set +a

CPU_PYTHON="${CPU_PYTHON:-${PROJECT_ROOT}/.venv-cpu/bin/python}"
test -x "${CPU_PYTHON}"
policy_url="$(
  "${CPU_PYTHON}" -c \
    'import sys; from rollout_service.config import load_inventory; print(load_inventory(sys.argv[1]).policy.url)' \
    "${ROLLOUT_INVENTORY}"
)"

curl -fsS --noproxy '*' --connect-timeout 15 --max-time 30 \
  "${ROLLOUT_CONTROL_URL}/healthz" >/dev/null
curl -fsS --noproxy '*' --connect-timeout 15 --max-time 30 "${policy_url}/healthz" >/dev/null

echo "Worker Control and Policy connectivity checks passed"
