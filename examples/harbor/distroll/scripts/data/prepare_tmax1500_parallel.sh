#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

[[ "$(id -u)" -eq 0 ]] || {
  echo "run on CPU Head as root" >&2
  exit 1
}
[[ "${ACK_TRUSTED_SECURITY_TASKS:-0}" == "1" ]] || {
  echo "set ACK_TRUSTED_SECURITY_TASKS=1 for the requested Claude-only publication" >&2
  exit 1
}

TMAX1500_RELEASE_ID="${TMAX1500_RELEASE_ID:-tmax1500-claude21207}"
TMAX1500_ROOT="${TMAX1500_ROOT:?set the dataset preparation directory}"
TMAX1500_TASK_LIST="${TMAX1500_TASK_LIST:-${TMAX1500_ROOT}/task-ids-1500.txt}"
TMAX1500_SELECTION_MANIFEST="${TMAX1500_SELECTION_MANIFEST:-${TMAX1500_ROOT}/source/selection.json}"
TMAX1500_RAW_ROOT="${TMAX1500_RAW_ROOT:-${TMAX1500_ROOT}/raw/tmax-15k}"
TMAX1500_OUTPUT_ROOT="${TMAX1500_OUTPUT_ROOT:-${TMAX1500_ROOT}/release}"
TMAX1500_WORK_ROOT="${TMAX1500_WORK_ROOT:-${TMAX1500_ROOT}/work}"
TMAX1500_JOBS="${TMAX1500_JOBS:-4}"
TMAX1500_TASK_ATTEMPTS="${TMAX1500_TASK_ATTEMPTS:-2}"
: "${CPU_HEAD_INTERNAL_IP:?CPU Head内网IP待确定，请先设置 CPU_HEAD_INTERNAL_IP}"
TMAX1500_IMAGE_REPOSITORY="${TMAX1500_IMAGE_REPOSITORY:-${CPU_HEAD_INTERNAL_IP}:5000/dressage-distroll/tmax-claude-only}"
TMAX1500_CLAUDE_RUNTIME_REPOSITORY="${TMAX1500_CLAUDE_RUNTIME_REPOSITORY:-${CPU_HEAD_INTERNAL_IP}:5000/dressage-distroll/tmax-claude-runtime}"
TMAX1500_SOURCE_MIRROR_REPOSITORY="${TMAX1500_SOURCE_MIRROR_REPOSITORY:-${CPU_HEAD_INTERNAL_IP}:5000/dressage-distroll/tmax-source}"
TMAX1500_NODE_RUNTIME_IMAGE="${TMAX1500_NODE_RUNTIME_IMAGE:?请设置预先发布的不可变Node runtime镜像}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.207}"
CPU_PYTHON="${CPU_PYTHON:-${PROJECT_ROOT}/.venv-cpu/bin/python}"

for command in curl docker jq; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "missing required command: ${command}" >&2
    exit 1
  }
done
[[ -x "${CPU_PYTHON}" ]] || {
  echo "missing installed CPU Python: ${CPU_PYTHON}" >&2
  exit 1
}
docker buildx version >/dev/null
curl --fail --silent --show-error --noproxy '*' --max-time 5 \
  "http://${CPU_HEAD_INTERNAL_IP}:5000/v2/" >/dev/null
[[ -f "${TMAX1500_TASK_LIST}" ]] || {
  echo "missing frozen task list: ${TMAX1500_TASK_LIST}" >&2
  exit 1
}
[[ -f "${TMAX1500_SELECTION_MANIFEST}" ]] || {
  echo "missing frozen selection manifest: ${TMAX1500_SELECTION_MANIFEST}" >&2
  exit 1
}
[[ "$(grep -Evc '^[[:space:]]*(#|$)' "${TMAX1500_TASK_LIST}")" -eq 1500 ]] || {
  echo "frozen task list must contain exactly 1500 task IDs" >&2
  exit 1
}
[[ -d "${TMAX1500_RAW_ROOT}" ]] || {
  echo "missing raw TMAX catalog: ${TMAX1500_RAW_ROOT}" >&2
  exit 1
}

capacity="$(curl --fail --silent --show-error --noproxy '*' --max-time 5 \
  "http://${CPU_HEAD_INTERNAL_IP}:19443/v4/capacity")"
if ! jq -e '
  .reserved_executions == 0 and .queued_executions == 0 and
  all(.workers[]; .allocated_charge.cpu_units == 0 and
                    .allocated_charge.memory_mb == 0)
' <<<"${capacity}" >/dev/null; then
  echo "CPU rollout cluster is busy; image publication refuses to compete with rollout" >&2
  exit 1
fi

resume_args=()
if [[ "${TMAX1500_RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi
mode_args=()
if [[ "${TMAX1500_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  mode_args+=(--check-inputs-only)
fi

"${CPU_PYTHON}" "${PROJECT_ROOT}/tools/prepare_tmax_parallel.py" \
  --project-root "${PROJECT_ROOT}" \
  --validation-profile trusted-security \
  --ack-trusted-security \
  --task-list "${TMAX1500_TASK_LIST}" \
  --selection-manifest "${TMAX1500_SELECTION_MANIFEST}" \
  --task-root "${TMAX1500_RAW_ROOT}" \
  --output-root "${TMAX1500_OUTPUT_ROOT}" \
  --work-root "${TMAX1500_WORK_ROOT}" \
  --release-id "${TMAX1500_RELEASE_ID}" \
  --expected-task-count 1500 \
  --jobs "${TMAX1500_JOBS}" \
  --task-attempts "${TMAX1500_TASK_ATTEMPTS}" \
  --task-retry-backoff-sec "${TMAX1500_TASK_RETRY_BACKOFF_SEC:-5}" \
  --output-repository "${TMAX1500_IMAGE_REPOSITORY}" \
  --claude-runtime-repository "${TMAX1500_CLAUDE_RUNTIME_REPOSITORY}" \
  --source-mirror-repository "${TMAX1500_SOURCE_MIRROR_REPOSITORY}" \
  --source-mirror-interval-sec "${TMAX1500_SOURCE_MIRROR_INTERVAL_SEC:-45}" \
  --source-rate-limit-cooldown-sec \
    "${TMAX1500_SOURCE_RATE_LIMIT_COOLDOWN_SEC:-3600}" \
  --source-mirror-attempts "${TMAX1500_SOURCE_MIRROR_ATTEMPTS:-24}" \
  --node-runtime-image "${TMAX1500_NODE_RUNTIME_IMAGE}" \
  --claude-code-version "${CLAUDE_CODE_VERSION}" \
  --docker-root /var/lib/docker \
  --min-docker-free-gib "${TMAX1500_MIN_DOCKER_FREE_GIB:-50}" \
  --min-work-free-gib "${TMAX1500_MIN_WORK_FREE_GIB:-500}" \
  --min-memory-available-gib "${TMAX1500_MIN_MEMORY_AVAILABLE_GIB:-16}" \
  "${resume_args[@]}" \
  "${mode_args[@]}"

if [[ "${TMAX1500_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "TMAX-1500 input preflight complete; no builder was created"
  exit 0
fi

if docker buildx ls --format '{{.Name}}' 2>/dev/null |
  grep -F \
    -e "tmax-${TMAX1500_RELEASE_ID}-" \
    -e "tmax-runtime-${TMAX1500_RELEASE_ID}" \
    -e "tmax-source-mirror-${TMAX1500_RELEASE_ID}" >/dev/null; then
  echo "dedicated TMAX builder remained after publication" >&2
  exit 1
fi

echo "TMAX-1500 publication ready: ${TMAX1500_OUTPUT_ROOT}"
