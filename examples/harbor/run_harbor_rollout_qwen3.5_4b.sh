#!/usr/bin/env bash

# Harbor rollout/evaluation served by a local Qwen3.5-4B SGLang worker.
# Activate the Python 3.12 Harbor environment before running this script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${DRESSAGE_HARBOR_JOB_CONFIG:?Set DRESSAGE_HARBOR_JOB_CONFIG to a Harbor JobConfig path}"

export DRESSAGE_HARBOR_INTEGRATION_CONFIG="${DRESSAGE_HARBOR_INTEGRATION_CONFIG:-${REPO_ROOT}/examples/harbor/dressage_profiles/rollout-native-local.yaml}"
export DRESSAGE_HARBOR_JOB_CONFIG
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen3.5-4B}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-127.0.0.1}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-30000}"
SGLANG_WORKER_HOST="${SGLANG_WORKER_HOST:-127.0.0.1}"
SGLANG_WORKER_PORT="${SGLANG_WORKER_PORT:-30001}"
DRESSAGE_PROXY_HOST="${DRESSAGE_PROXY_HOST:-127.0.0.1}"
DRESSAGE_PROXY_PORT="${DRESSAGE_PROXY_PORT:-8800}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
LOG_DIR="${HARBOR_LOG_DIR:-/tmp/dressage-harbor/logs/rollout}"

# Never reuse a caller-provided static service credential.
set +x
export DRESSAGE_PROXY_API_KEY="$(openssl rand -hex 48)"

ROUTER_URL="http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}"
WORKER_URL="http://${SGLANG_WORKER_HOST}:${SGLANG_WORKER_PORT}"
PROXY_URL="http://${DRESSAGE_PROXY_HOST}:${DRESSAGE_PROXY_PORT}"

mkdir -p "${LOG_DIR}"
PIDS=()

cleanup() {
  local status=$?
  if ((${#PIDS[@]})); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_url() {
  local url="$1" retries="${2:-120}"
  for ((attempt=1; attempt<=retries; attempt++)); do
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

"${PYTHON_BIN}" -m sglang_router.launch_router \
  --host "${SGLANG_ROUTER_HOST}" \
  --port "${SGLANG_ROUTER_PORT}" \
  --log-level warn \
  >"${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("${ROUTER_PID}")
wait_for_url "${ROUTER_URL}/workers" 60

"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${HF_CHECKPOINT}" \
  --served-model-name Qwen/Qwen3.5-4B \
  --host "${SGLANG_WORKER_HOST}" \
  --port "${SGLANG_WORKER_PORT}" \
  --tp-size "${SGLANG_TP_SIZE:-4}" \
  --context-length "${CONTEXT_WINDOW}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  >"${LOG_DIR}/worker.log" 2>&1 &
WORKER_PID=$!
PIDS+=("${WORKER_PID}")
wait_for_url "${WORKER_URL}/health_generate" "${SGLANG_START_TIMEOUT_SEC:-900}"

curl -fsS --max-time 15 \
  -H 'Content-Type: application/json' \
  -d "{\"url\":\"${WORKER_URL}\",\"worker_type\":\"regular\"}" \
  "${ROUTER_URL}/workers" >/dev/null

"${PYTHON_BIN}" -m dressage.proxy.server \
  --sglang-router-url "${ROUTER_URL}" \
  --tokenizer-path "${HF_CHECKPOINT}" \
  --host "${DRESSAGE_PROXY_HOST}" \
  --port "${DRESSAGE_PROXY_PORT}" \
  --api-key "${DRESSAGE_PROXY_API_KEY}" \
  --token-build-mode tito \
  --token-build-model qwen3_5 \
  --tito-model qwen3_5 \
  --model-mask-type qwen3_5 \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3 \
  --context-window "${CONTEXT_WINDOW}" \
  >"${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
PIDS+=("${PROXY_PID}")
wait_for_url "${PROXY_URL}/health" 120

harbor run \
  --config "${DRESSAGE_HARBOR_JOB_CONFIG}" \
  --plugin dressage \
  --yes
