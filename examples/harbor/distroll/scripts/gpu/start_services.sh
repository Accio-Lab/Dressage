#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
set -a
source "${GPU_ENV_FILE:-${PROJECT_ROOT}/configs/gpu.training.env.example}"
set +a
validate_gpu_sequence_budget
MODEL_ROOT="${MODEL_ROOT:-/models}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/Qwen3-30B-A3B}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/services}"
PID_DIR="${PID_DIR:-${PROJECT_ROOT}/outputs/pids}"
DRESSAGE_TRANSFER_QUEUE_CONFIG="${DRESSAGE_TRANSFER_QUEUE_CONFIG:-${DRESSAGE_ROOT}/examples/scripts/default/dressage_transfer_queue.yaml}"
DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS="${DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS:-86400}"
DRESSAGE_TRANSFER_PARAMS="${DRESSAGE_TRANSFER_PARAMS:-logprobs,routed_experts}"
DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY="${DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY:-1}"
DRESSAGE_ENABLE_TRANSFER_QUEUE="${DRESSAGE_ENABLE_TRANSFER_QUEUE:-1}"
DRESSAGE_TOKEN_BUILD_MODE="${DRESSAGE_TOKEN_BUILD_MODE:-snapshot}"
DRESSAGE_TOKEN_BUILD_MODEL="${DRESSAGE_TOKEN_BUILD_MODEL:-qwen3_5}"
[[ "${DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY}" =~ ^[01]$ ]] || {
  echo "DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY must be 0 or 1" >&2
  exit 1
}
[[ "${DRESSAGE_ENABLE_TRANSFER_QUEUE}" =~ ^[01]$ ]] || {
  echo "DRESSAGE_ENABLE_TRANSFER_QUEUE must be 0 or 1" >&2
  exit 1
}
case "${DRESSAGE_TOKEN_BUILD_MODE}" in
  snapshot | tito) ;;
  *)
    echo "DRESSAGE_TOKEN_BUILD_MODE must be snapshot or tito" >&2
    exit 1
    ;;
esac
case "${DRESSAGE_TOKEN_BUILD_MODEL}" in
  qwen3_5 | qwen3_6) ;;
  *)
    echo "DRESSAGE_TOKEN_BUILD_MODEL must be qwen3_5 or qwen3_6" >&2
    exit 1
    ;;
esac
normalized_transfer_params="${DRESSAGE_TRANSFER_PARAMS//,/ }"
if [[ "${DRESSAGE_TOKEN_BUILD_MODE}" == "tito" ]]; then
  export DRESSAGE_PROXY_SOURCE="${DRESSAGE_PROXY_SOURCE:-${PROJECT_ROOT}/outputs/runtime-deps/tito-production/server.py}"
  [[ -f "${DRESSAGE_PROXY_SOURCE}" ]] || {
    echo "TITO requires the reviewed private Proxy source; run prepare_tito_proxy_overlay.py first" >&2
    exit 1
  }
fi
read -r -a transfer_params <<<"${normalized_transfer_params}"
export DRESSAGE_TRANSFER_QUEUE_CONFIG \
  DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS DRESSAGE_TRANSFER_PARAMS \
  DRESSAGE_ENABLE_TRANSFER_QUEUE DRESSAGE_TOKEN_BUILD_MODE \
  DRESSAGE_TOKEN_BUILD_MODEL
mkdir -p "${LOG_DIR}" "${PID_DIR}"
export POLICY_DEBUG_LOG="${POLICY_DEBUG_LOG:-${LOG_DIR}/policy-debug.jsonl}"

[[ -f "${TOKENIZER_PATH}/config.json" ]] || {
  echo "missing tokenizer/model at ${TOKENIZER_PATH}" >&2
  exit 1
}
[[ -f "${DRESSAGE_TRANSFER_QUEUE_CONFIG}" && "${#transfer_params[@]}" -gt 0 ]] || {
  echo "invalid TransferQueue configuration" >&2
  exit 1
}

start_one() {
  local name="$1"
  shift
  local pid_file="${PID_DIR}/${name}.pid"
  local pid state command expected
  if [[ -f "${pid_file}" ]]; then
    read -r pid <"${pid_file}"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      state=$(awk '/^State:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)
      if [[ "${state}" != Z ]]; then
        [[ -r "/proc/${pid}/cmdline" ]] || {
          echo "${name} PID state is uncertain; refusing to start another process" >&2
          return 1
        }
        command=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
        case "${name}" in
          policy-service) expected=dressage_distroll_gpu_adapter.cli ;;
          dressage-proxy)
            if [[ "${command}" == *"-m dressage.proxy.server"* ]]; then
              echo "legacy project Proxy still running; drain/reconcile it before upgrading" >&2
              return 1
            fi
            expected=dressage_distroll_gpu_adapter.proxy_server ;;
          *) echo "unknown GPU service: ${name}" >&2; return 1 ;;
        esac
        [[ "${command}" == *"${expected}"* ]] || {
          echo "${name} PID belongs to another process; refusing to replace it" >&2
          return 1
        }
        echo "${name} already running"
        return
      fi
    fi
  fi
  "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
  echo $! >"${pid_file}"
}

proxy_args=(
  -m dressage_distroll_gpu_adapter.proxy_server \
  --sglang-router-url "${SGLANG_ROUTER_URL:-http://${GPU_NODE_IP:-127.0.0.1}:8000}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --host 127.0.0.1 --port 8800 \
  --context-window "${ROLLOUT_MAX_CONTEXT_LEN:-32768}" \
  --max-output-tokens "${ROLLOUT_MAX_RESPONSE_LEN:-8192}" \
  --token-build-mode "${DRESSAGE_TOKEN_BUILD_MODE}" \
  --token-build-model "${DRESSAGE_TOKEN_BUILD_MODEL}" \
  --record-token-versions \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3 \
  --tool-call-parse-backend sglang_api \
  --reasoning-parse-backend sglang_api \
  --session-timeout 14400 \
  --group-timeout 14400
)
if [[ "${DRESSAGE_TOKEN_BUILD_MODE}" == "tito" ]]; then
  proxy_args+=(--tito-model "${DRESSAGE_TOKEN_BUILD_MODEL}")
fi
if [[ "${DRESSAGE_ENABLE_TRANSFER_QUEUE}" == "1" ]]; then
  proxy_args+=(
    --enable-transfer-queue
    --transfer-queue-config "${DRESSAGE_TRANSFER_QUEUE_CONFIG}"
    --transfer-queue-retention-seconds "${DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS}"
    --transfer-params "${transfer_params[@]}"
  )
fi
if [[ "${DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
  proxy_args+=(--use-rollout-routing-replay --routed-experts-dtype uint8)
elif [[ " ${transfer_params[*]} " == *" routed_experts "* ]]; then
  echo "routed_experts transfer requires rollout routing replay" >&2
  exit 1
fi
DRESSAGE_DEBUG_LOG="${LOG_DIR}/proxy-debug.jsonl" \
  start_one dressage-proxy "${GPU_PYTHON}" "${proxy_args[@]}"

for _ in $(seq 1 60); do
  curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:8800/health >/dev/null && break
  sleep 1
done
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:8800/health >/dev/null || {
  echo "Dressage Proxy did not become healthy; inspect ${LOG_DIR}/dressage-proxy.log" >&2
  exit 1
}

curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS http://127.0.0.1:8800/health | "${GPU_PYTHON}" -c '
import json, os, sys
health = json.load(sys.stdin)
config = health.get("config", {})
expected_token_mode = os.environ["DRESSAGE_TOKEN_BUILD_MODE"]
actual_token_mode = config.get("token_build_mode")
if actual_token_mode != expected_token_mode:
    raise SystemExit(
        f"Dressage Proxy token mode mismatch: expected={expected_token_mode} "
        f"actual={actual_token_mode}"
    )
expected_token_model = os.environ["DRESSAGE_TOKEN_BUILD_MODEL"]
actual_token_model = config.get("token_build_model")
if actual_token_model != expected_token_model:
    raise SystemExit(
        f"Dressage Proxy token model mismatch: expected={expected_token_model} "
        f"actual={actual_token_model}"
    )
expected_enabled = os.environ["DRESSAGE_ENABLE_TRANSFER_QUEUE"] == "1"
actual_enabled = config.get("enable_transfer_queue", False) is True
if actual_enabled != expected_enabled:
    raise SystemExit(
        "Dressage Proxy TransferQueue mode disagrees with the requested mode"
    )
if expected_enabled:
    expected = set(os.environ["DRESSAGE_TRANSFER_PARAMS"].replace(",", " ").split())
    actual = set(config.get("transfer_params", []))
    if actual != expected:
        raise SystemExit(f"unexpected TransferQueue fields: {sorted(actual)}")
expected_context = int(os.environ.get("ROLLOUT_MAX_CONTEXT_LEN", "32768"))
actual_context = config.get("context_window")
if actual_context != expected_context:
    raise SystemExit(
        f"Dressage Proxy context mismatch: expected={expected_context} "
        f"actual={actual_context}"
    )
expected_output = int(os.environ.get("ROLLOUT_MAX_RESPONSE_LEN", "8192"))
actual_output = config.get("max_output_tokens")
if actual_output != expected_output:
    raise SystemExit(
        f"Dressage Proxy output-token mismatch: expected={expected_output} "
        f"actual={actual_output}"
    )
default_router_host = os.environ.get("GPU_NODE_IP", "127.0.0.1")
expected_router = os.environ.get(
    "SGLANG_ROUTER_URL", f"http://{default_router_host}:8000"
).rstrip("/")
actual_router = str(config.get("sglang_router_url") or "").rstrip("/")
if actual_router != expected_router:
    raise SystemExit(
        f"Dressage Proxy router mismatch: expected={expected_router} "
        f"actual={actual_router}"
    )
'

start_one policy-service \
  "${GPU_PYTHON}" -c \
  'from dressage_distroll_gpu_adapter.cli import policy_main; policy_main()'
for _ in $(seq 1 60); do
  curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${POLICY_SERVICE_URL}/healthz" >/dev/null && break
  sleep 1
done
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${POLICY_SERVICE_URL}/healthz" >/dev/null || {
  echo "Policy Service did not become healthy; inspect ${LOG_DIR}/policy-service.log" >&2
  exit 1
}

echo "Dressage Proxy and GPU Policy Service started"
