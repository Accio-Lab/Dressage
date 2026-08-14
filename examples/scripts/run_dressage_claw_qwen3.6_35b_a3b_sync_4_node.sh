#!/bin/bash

# dressage_claw 35B (Qwen3.6-35B-A3B) Sync — Multi-Node E2B (4 nodes)
#
# Topology (CP=8, colocate mode):
#   All 4 nodes: Actor training (8 GPU each) + Rollout inference (colocate) + Ray
#   Sync mode: rollout and training alternate on the same GPUs (--colocate).
#
# One-click launch: auto-detects RANK from hostname (master-N / worker-N)
# or env vars (RANK / NODE_RANK / GROUP_RANK / OMPI_COMM_WORLD_RANK).
# Just run the same script on all 4 nodes — no manual RANK needed.
#
# Required env:
#   MODEL_ROOT            model/checkpoint dir (contains ${MODEL_NAME} and ${MODEL_NAME}_torch_dist)
#   DRESSAGE_E2B_API_KEY  E2B API key (sandbox provider is e2b)
# Optional:
#   PROMPT_DATA           defaults to examples/data/dressage_claw_e2b.jsonl
#   BOOTSTRAP_DIR         dir for bootstrap logs + MASTER_ADDR discovery file;
#                         must be a SHARED filesystem in multi-node setups
#                         (or just set MASTER_ADDR on workers instead)
#   USE_WANDB             uncomment the WANDB_ARGS block below to enable W&B logging

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
rm -rf /tmp/ray 2>/dev/null || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../../dressage" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  else
    echo "REPO_ROOT is required when this script is not inside examples/scripts/." >&2
    exit 1
  fi
fi
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MODEL_ROOT="${MODEL_ROOT:-${BASE_FOLDER:-}}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-35B-A3B}"
MODEL_CONFIG="${MODEL_CONFIG:-qwen3.6-35B-A3B.sh}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}}"
DEFAULT_ENV_FILE="${DEFAULT_ENV_FILE:-${REPO_ROOT}/examples/scripts/default/dressage_env_defaults.sh}"

[[ -f "${MODEL_CONFIG_PATH}" ]] || {
  echo "Cannot find slime model config: ${MODEL_CONFIG_PATH}" >&2
  exit 1
}
[[ -f "${DEFAULT_ENV_FILE}" ]] || {
  echo "Cannot find Dressage default env file: ${DEFAULT_ENV_FILE}" >&2
  exit 1
}
: "${PROMPT_DATA:=${REPO_ROOT}/examples/data/dressage_claw_e2b.jsonl}"
: "${MODEL_ROOT:?MODEL_ROOT must point to model/checkpoint files, e.g. /path/to/models}"

BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/tmp/dressage_bootstrap}"
mkdir -p "${BOOTSTRAP_DIR}"

BOOTSTRAP_HOST="$(hostname)"
BOOTSTRAP_LOG="${BOOTSTRAP_DIR}/${BOOTSTRAP_HOST}-$$_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${BOOTSTRAP_LOG}") 2>&1
echo "bootstrap_log=${BOOTSTRAP_LOG}"
echo "hostname=${BOOTSTRAP_HOST}"
echo "hostname_ips=$(hostname -I 2>/dev/null || true)"
env | grep -E '^(RANK|NODE_RANK|GROUP_RANK|OMPI_COMM_WORLD_RANK|WORLD_SIZE|NNODES|MLFLOW|POD|HOSTNAME)=' | sort || true

WORLD_SIZE="${WORLD_SIZE:-${NNODES:-4}}"
RANK="${RANK:-${NODE_RANK:-${GROUP_RANK:-${OMPI_COMM_WORLD_RANK:-}}}}"
MASTER_ADDR="${MASTER_ADDR:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE}}"
CP_SIZE="${CP_SIZE:-8}"

JOB_KEY="${BOOTSTRAP_HOST}"
JOB_KEY="${JOB_KEY%-master-*}"
JOB_KEY="${JOB_KEY%-worker-*}"
MASTER_ADDR_FILE="${BOOTSTRAP_DIR}/master_addr_${JOB_KEY}_${WORLD_SIZE}nodes.txt"

if [[ -z "${RANK}" ]]; then
  HOSTNAME_VALUE="$(hostname)"
  if [[ "${WORLD_SIZE}" == "1" ]]; then
    RANK=0
  elif [[ "${HOSTNAME_VALUE}" =~ master-([0-9]+)$ ]]; then
    RANK="${BASH_REMATCH[1]}"
  elif [[ "${HOSTNAME_VALUE}" =~ worker-([0-9]+)$ ]]; then
    RANK="$((BASH_REMATCH[1] + 1))"
  else
    echo "RANK is required for multi-node launch. Set RANK=0 on master and RANK=1..$((WORLD_SIZE - 1)) on workers." >&2
    echo "hostname=${HOSTNAME_VALUE}" >&2
    exit 1
  fi
fi

if [[ -z "${MASTER_ADDR}" ]]; then
  HOSTNAME_VALUE="${HOSTNAME_VALUE:-$(hostname)}"
  if [[ "${HOSTNAME_VALUE}" =~ worker-[0-9]+$ ]]; then
    MASTER_ADDR="${HOSTNAME_VALUE/%worker-*/master-0}"
  elif [[ "${HOSTNAME_VALUE}" =~ master-[1-9][0-9]*$ ]]; then
    MASTER_ADDR="${HOSTNAME_VALUE/%master-*/master-0}"
  fi
fi

export PYTHONBUFFERED=16
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MEGATRON_ROOT:+${MEGATRON_ROOT}:}${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export SOCKET_IFNAME="${SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-3600000}"
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-3600000}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NNODES="${NNODES:-${WORLD_SIZE}}"
export NODE_RANK="${NODE_RANK:-${RANK}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${MASTER_ADDR}" && "${RANK}" == "0" ]]; then
  MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
CURRENT_NODE_IP="${CURRENT_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ "${RANK}" == "0" ]]; then
  printf '%s\n' "${CURRENT_NODE_IP:-${MASTER_ADDR}}" >"${MASTER_ADDR_FILE}"
else
  for i in $(seq 1 120); do
    [[ -s "${MASTER_ADDR_FILE}" ]] && {
      MASTER_ADDR="$(cat "${MASTER_ADDR_FILE}")"
      break
    }
    sleep 1
  done
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is required. On workers set it to the rank0 node IP." >&2
  exit 1
fi
export MASTER_ADDR

echo "============================================================"
echo "  Multi-node sync: RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}"
echo "  MASTER_ADDR=${MASTER_ADDR}, CURRENT_NODE_IP=${CURRENT_NODE_IP}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}, GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "  CP_SIZE=${CP_SIZE}"
echo "============================================================"

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS="${RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS:-10}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Set before sourcing the env file, whose :- default would otherwise pin qwen3_5.
TOKEN_BUILD_MODEL="${TOKEN_BUILD_MODEL:-qwen3_6}"

source "${MODEL_CONFIG_PATH}"
source "${DEFAULT_ENV_FILE}"

RUN_NAME="${RUN_NAME:-dressage-claw-${MODEL_NAME,,}-sync-4node}"
DRESSAGE_SANDBOX_PROVIDER="${DRESSAGE_SANDBOX_PROVIDER:-e2b}"
# Pre-set so apply_common_defaults does not fall back to a possibly local-only
# hostname -i result (E2B sandboxes must be able to reach the proxy).
PROXY_PUBLIC_HOST="${PROXY_PUBLIC_HOST:-${CURRENT_NODE_IP:-${MASTER_ADDR}}}"
dressage_apply_common_defaults "${RUN_NAME}" blackbox "${DRESSAGE_SANDBOX_PROVIDER}"
DRESSAGE_PROXY_PUBLIC_URL="${DRESSAGE_PROXY_PUBLIC_URL:-${DRESSAGE_PROXY_URL}}"

if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "e2b" ]]; then
  : "${DRESSAGE_E2B_API_KEY:?DRESSAGE_E2B_API_KEY is required for DRESSAGE_SANDBOX_PROVIDER=e2b}"
fi
if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "custom" ]]; then
  : "${DRESSAGE_SANDBOX_PROVIDER_CLASS:?DRESSAGE_SANDBOX_PROVIDER_CLASS is required for DRESSAGE_SANDBOX_PROVIDER=custom}"
fi

dressage_validate_proxy_defaults
if [[ "${DRESSAGE_CLEAR_TRAJECTORY_LOGS:-0}" == "1" ]]; then
  dressage_clear_trajectory_logs
fi

if [[ "${TOKEN_BUILD_MODE}" != "tito" && "${TOKEN_BUILD_MODE}" != "snapshot" ]]; then
  echo "TOKEN_BUILD_MODE must be tito or snapshot, got: ${TOKEN_BUILD_MODE}" >&2
  exit 1
fi

dressage_export_common_env
dressage_compute_context_window 8192 "${CP_SIZE}"

ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
COMPACT_RESERVE_TOKENS="${COMPACT_RESERVE_TOKENS:-8192}"
DRESSAGE_BLACKBOX_COMPACT_THRESHOLD="${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD:-$((CONTEXT_WINDOW - COMPACT_RESERVE_TOKENS))}"
if [[ "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" -le 0 ]]; then
  echo "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD must be positive; got ${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" >&2
  exit 1
fi
echo "effective_compact: CONTEXT_WINDOW=${CONTEXT_WINDOW} ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN} COMPACT_RESERVE=${COMPACT_RESERVE_TOKENS} COMPACT_THRESHOLD=${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}"
export DRESSAGE_BLACKBOX_COMPACT_THRESHOLD

export DRESSAGE_PROXY_PUBLIC_URL
export DRESSAGE_SANDBOX_PROVIDER
export DRESSAGE_SANDBOX_PROVIDER_CLASS="${DRESSAGE_SANDBOX_PROVIDER_CLASS:-}"
export DRESSAGE_E2B_API_KEY="${DRESSAGE_E2B_API_KEY:-}"
export DRESSAGE_SANDBOX_DEFAULT_IMAGE="${DRESSAGE_SANDBOX_DEFAULT_IMAGE:-e2b-dressage-claw-blackbox}"
export DRESSAGE_BLACKBOX_BACKEND_TIMEOUT="${DRESSAGE_BLACKBOX_BACKEND_TIMEOUT:-900}"
export DRESSAGE_BLACKBOX_MAX_STEPS="${DRESSAGE_BLACKBOX_MAX_STEPS:-100}"
export DRESSAGE_PROXY_MAX_STEPS_PER_SESSION="${DRESSAGE_PROXY_MAX_STEPS_PER_SESSION:-100}"
export DRESSAGE_ROLLOUT_MAX_RETRIES="${DRESSAGE_ROLLOUT_MAX_RETRIES:-0}"
export DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH="${DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH:-0}"
export DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER="${DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER:-2}"
export DRESSAGE_REWARD_MODULES="${DRESSAGE_REWARD_MODULES:-dressage.recipes.dressage_claw.reward}"

# Sync colocate mode: all nodes are actor nodes, rollout shares GPUs with training.
# CP=8 → TP=2 × CP=8 × PP=1 = 16 GPUs per replica → 2 replicas on 4 nodes.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-${WORLD_SIZE}}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${NPROC_PER_NODE}}
RAY_NUM_GPUS_PER_NODE=${NPROC_PER_NODE}
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-${CURRENT_NODE_IP}}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"

COMM_ARGS=(
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
)
TQ_PROXY_ARGS=()
TRAIN_ENTRY=(python3 train.py)
if dressage_transfer_queue_enabled; then
    NORMALIZED_TRANSFER_PARAMS="${DRESSAGE_TRANSFER_PARAMS//,/ }"
    read -r -a TRANSFER_PARAMS <<< "${NORMALIZED_TRANSFER_PARAMS}"
    TQ_PROXY_ARGS=(
        --enable-transfer-queue
        --transfer-queue-config "${DRESSAGE_TRANSFER_QUEUE_CONFIG}"
        --transfer-queue-retention-seconds "${DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS}"
        --transfer-params "${TRANSFER_PARAMS[@]}"
    )
    if [[ " ${NORMALIZED_TRANSFER_PARAMS} " == *" routed_experts "* ]]; then
        COMM_ARGS+=(--use-rollout-routing-replay)
    fi
    TRAIN_ENTRY=(python3 -m dressage.training.tq_train)
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
PROXY_ARGS=(
   --tokenizer-path "${TOKENIZER_PATH}"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --token-build-mode "${TOKEN_BUILD_MODE}"
   --token-build-model "${TOKEN_BUILD_MODEL}"
   "${COMM_ARGS[@]}"
   "${TQ_PROXY_ARGS[@]}"
   --context-window "${CONTEXT_WINDOW}"
   --default-max-tokens "${ROLLOUT_MAX_RESPONSE_LEN}"
)

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}"
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_ROOT:-${MODEL_ROOT}/${MODEL_NAME}_slime}/"
   --save-interval "${SAVE_INTERVAL:-20}"
   --no-save-optim
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync
   --custom-generate-function-path dressage.recipes.dressage_claw.dispatch.generate
   --custom-rm-path dressage.reward.custom_rm.custom_rm
   --data-source-path dressage.rollout.data_source.DressageDataSource
   --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
   --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
   --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-10000}"
   # 4-node sync colocate: 4 actor nodes (32 GPU) → all shared for training + rollout.
   # ROLLOUT_BATCH_SIZE=8 → 8 prompt × 8 samples = 64 trajectories per step.
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --balance-data
   --rollout-global-dataset
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

   --log-probs-chunk-size 512
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${ADAM_BETA1:-0.9}"
   --adam-beta2 "${ADAM_BETA2:-0.98}"
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project dressage-claw
   # --wandb-group dressage-claw-qwen3.6-35b-a3b-sync-4node
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.5
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
   --sglang-log-level warning
   --sglang-chunked-prefill-size 4096
   --sglang-max-prefill-tokens 8192
   --sglang-max-running-requests 64
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
   --router-policy consistent_hashing
)

MISC_ARGS=(
   --custom-config-path "${SCRIPT_DIR}/default/dressage_staleness.yaml"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [[ -f "${PROXY_PID_FILE}" ]]; then
  OLD_PROXY_PID="$(cat "${PROXY_PID_FILE}")"
  if ! kill -0 "${OLD_PROXY_PID}" 2>/dev/null; then
    rm -f "${PROXY_PID_FILE}"
  fi
fi

cleanup() {
  status=$?
  set +e
  ray stop --force 2>/dev/null || true
  rm -rf /tmp/ray 2>/dev/null || true
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    PROXY_PID="$(cat "${PROXY_PID_FILE}")"
    kill "${PROXY_PID}" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
  exit "${status}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Branch execution by RANK (after common setup)
# ---------------------------------------------------------------------------
if [[ "${RANK}" != "0" ]]; then
  export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR}"
  export NO_PROXY="${no_proxy}"
  echo "[WORKER RANK=${RANK}] Joining Ray cluster at ${MASTER_ADDR}:${RAY_PORT}"
  until ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${CURRENT_NODE_IP}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats; do
    echo "[WORKER RANK=${RANK}] Ray head is not ready yet; retrying in 5s..."
    sleep 5
  done
  echo "[WORKER RANK=${RANK}] Joined Ray cluster. Sleeping forever."
  tail -f /dev/null
fi

echo "[MASTER RANK=0] Starting Dressage proxy..."
cd "${REPO_ROOT}"
python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
echo $! > "${PROXY_PID_FILE}"
echo "[MASTER RANK=0] Started proxy: pid=$(cat "${PROXY_PID_FILE}") log=${PROXY_LOG_FILE}"

for i in $(seq 1 60); do
  if curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1; then
    echo "[MASTER RANK=0] Proxy is healthy"
    break
  fi
  if [[ "${i}" -eq 60 ]]; then
    echo "[MASTER RANK=0] Proxy failed health check; see ${PROXY_LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}"
export NO_PROXY="${no_proxy}"

echo "[MASTER RANK=0] Starting Ray head node..."
cd "${SLIME_ROOT}"
ray start --head --block \
  --port="${RAY_PORT}" \
  --node-ip-address "${RAY_NODE_IP_ADDRESS}" \
  --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host="${RAY_DASHBOARD_HOST}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" &
sleep 10

EXPECTED_NODES="${WORLD_SIZE}"
JOIN_DEADLINE=$((SECONDS + ${RAY_JOIN_TIMEOUT_SEC:-1800}))
while true; do
  node_count=$(ray status 2>/dev/null | grep -c "node_" || true)
  echo "[MASTER RANK=0] Ray node count: ${node_count}/${EXPECTED_NODES}"
  [[ "${node_count}" -ge "${EXPECTED_NODES}" ]] && break
  if [[ "${SECONDS}" -ge "${JOIN_DEADLINE}" ]]; then
    echo "[MASTER RANK=0] Timeout waiting for workers (got ${node_count}/${EXPECTED_NODES})" >&2
    ray status || true
    exit 1
  fi
  sleep 30
done

RUNTIME_ENV_JSON="$(
  python3 -c 'import json, os
keys = [
  "no_proxy", "NO_PROXY", "MASTER_ADDR", "PYTHONPATH",
  "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "NCCL_DEBUG",
  "NCCL_IB_DISABLE", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME",
  "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NCCL_SOCKET_TIMEOUT_MS",
  "NCCL_TIMEOUT_MS", "NCCL_CUMEM_ENABLE", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
  "WANDB_KEY", "WANDB_API_KEY", "WANDB_MODE",
  "DRESSAGE_PROXY_URL", "DRESSAGE_PROXY_PUBLIC_URL",
  "DRESSAGE_PADDOCK_MODE", "DRESSAGE_SANDBOX_PROVIDER",
  "DRESSAGE_SANDBOX_PROVIDER_CLASS", "DRESSAGE_E2B_API_KEY",
  "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR",
  "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", "DRESSAGE_REWARD_MODULES",
  "DRESSAGE_BLACKBOX_MAX_STEPS", "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD",
  "DRESSAGE_PROXY_MAX_STEPS_PER_SESSION", "DRESSAGE_BLACKBOX_BACKEND_TIMEOUT",
  "DRESSAGE_SANDBOX_DEFAULT_IMAGE",
  "DRESSAGE_ROLLOUT_MAX_RETRIES",
  "DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH",
  "DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER",
  "DRESSAGE_ENABLE_TRANSFER_QUEUE", "DRESSAGE_TRANSFER_QUEUE_CONFIG",
  "DRESSAGE_TRANSFER_PARAMS", "DRESSAGE_TRANSFER_QUEUE_STORE_ID",
]
for raw_key in os.environ.get("DRESSAGE_EXTRA_RUNTIME_ENV_KEYS", "").split(","):
    key = raw_key.strip()
    if key and key not in keys:
        keys.append(key)
print(json.dumps({"env_vars": {k: os.environ.get(k, "") for k in keys if os.environ.get(k, "") != ""}}))'
)"

echo "[MASTER RANK=0] Submitting Ray job..."
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_ENTRY[@]}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${COMM_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
