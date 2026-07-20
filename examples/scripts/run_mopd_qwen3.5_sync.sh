#!/bin/bash
# Same-base multi-teacher OPD with mixed blackbox/whitebox rollouts.
#
# Dataset routes, teacher checkpoints, reward modules, and task worker env keys
# come from DRESSAGE_MOPD_TEACHER_CONFIG. Platform setup and credentials remain
# caller-owned.

set -euo pipefail

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
DEFAULT_ENV_FILE="${DEFAULT_ENV_FILE:-${REPO_ROOT}/examples/scripts/default/dressage_env_defaults.sh}"

: "${DRESSAGE_MOPD_TEACHER_CONFIG:?Set DRESSAGE_MOPD_TEACHER_CONFIG to the MOPD JSON}"
[[ -f "${DRESSAGE_MOPD_TEACHER_CONFIG}" ]] || {
  echo "MOPD config does not exist: ${DRESSAGE_MOPD_TEACHER_CONFIG}" >&2
  exit 1
}
mapfile -t MOPD_LAUNCH_CONFIG < <(
  PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m dressage.training.mopd_launch_config "${DRESSAGE_MOPD_TEACHER_CONFIG}"
)

PROMPT_DATA="${PROMPT_DATA:-${MOPD_LAUNCH_CONFIG[0]:-}}"
: "${PROMPT_DATA:?MOPD config has no datasets; set PROMPT_DATA explicitly}"
MOPD_MODES=",${MOPD_LAUNCH_CONFIG[1]:-},"
if [[ "${MOPD_MODES}" == *,blackbox,* ]]; then
  DRESSAGE_PADDOCK_MODE="${DRESSAGE_PADDOCK_MODE:-blackbox}"
else
  DRESSAGE_PADDOCK_MODE="${DRESSAGE_PADDOCK_MODE:-whitebox}"
  DRESSAGE_SANDBOX_PROVIDER="${DRESSAGE_SANDBOX_PROVIDER:-local_bwrap}"
fi
if [[ -n "${MOPD_LAUNCH_CONFIG[2]:-}" ]]; then
  DRESSAGE_EXTRA_RUNTIME_ENV_KEYS="${DRESSAGE_EXTRA_RUNTIME_ENV_KEYS:+${DRESSAGE_EXTRA_RUNTIME_ENV_KEYS},}${MOPD_LAUNCH_CONFIG[2]}"
fi
DRESSAGE_REWARD_MODULES="${DRESSAGE_REWARD_MODULES:-${MOPD_LAUNCH_CONFIG[3]:-}}"

BASE_MODEL="${MOPD_LAUNCH_CONFIG[4]:-}"
if [[ -n "${BASE_MODEL}" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-$(dirname -- "${BASE_MODEL}")}"
  MODEL_NAME="${MODEL_NAME:-$(basename -- "${BASE_MODEL}")}"
fi
MODEL_ROOT="${MODEL_ROOT:-}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
if [[ -z "${MODEL_CONFIG:-}" && "${MODEL_NAME}" == Qwen* ]]; then
  MODEL_CONFIG="q${MODEL_NAME#Q}.sh"
fi
: "${MODEL_CONFIG:?Set MODEL_CONFIG for the student architecture}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}}"

[[ -f "${MODEL_CONFIG_PATH}" ]] || {
  echo "Missing Slime model config: ${MODEL_CONFIG_PATH}" >&2
  exit 1
}
[[ -f "${DEFAULT_ENV_FILE}" ]] || {
  echo "Missing Dressage default env file: ${DEFAULT_ENV_FILE}" >&2
  exit 1
}
: "${MODEL_ROOT:?MODEL_ROOT must point to model/checkpoint files, e.g. /path/to/models}"
if [[ -n "${BASE_MODEL}" && "$(realpath -m -- "${MODEL_ROOT}/${MODEL_NAME}")" != "$(realpath -m -- "${BASE_MODEL}")" ]]; then
  echo "MOPD base_model mismatch: config=${BASE_MODEL} student=${MODEL_ROOT}/${MODEL_NAME}" >&2
  exit 1
fi
export PROMPT_DATA DRESSAGE_MOPD_TEACHER_CONFIG DRESSAGE_PADDOCK_MODE
export DRESSAGE_SANDBOX_PROVIDER DRESSAGE_REWARD_MODULES DRESSAGE_EXTRA_RUNTIME_ENV_KEYS

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
CURRENT_NODE_IP="${CURRENT_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is required and could not be inferred." >&2
  exit 1
fi
export MASTER_ADDR CURRENT_NODE_IP

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONUNBUFFERED=1
PYTHONPATH_ENTRIES=("${REPO_ROOT}" "${SLIME_ROOT}")
if [[ -n "${MEGATRON_ROOT:-}" ]]; then
  PYTHONPATH_ENTRIES=("${MEGATRON_ROOT}" "${PYTHONPATH_ENTRIES[@]}")
fi
PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${WORLD_SIZE:-1}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE}}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-${CURRENT_NODE_IP}}"
RAY_JOIN_TIMEOUT_SEC="${RAY_JOIN_TIMEOUT_SEC:-600}"
SOCKET_IFNAME="${SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
HOSTFILE="${HOSTFILE:-}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-3600000}"
export NCCL_TIMEOUT_MS="${NCCL_TIMEOUT_MS:-3600000}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

source "${MODEL_CONFIG_PATH}"
source "${DEFAULT_ENV_FILE}"

TP_SIZE="${TP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
dressage_compute_context_window "${MAX_TOKENS_PER_GPU}" "${CP_SIZE}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-${CONTEXT_WINDOW}}"

TOTAL_ACTOR_GPUS="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
MODEL_PARALLEL_SIZE="$((TP_SIZE * CP_SIZE))"
if (( TOTAL_ACTOR_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
  echo "total actor GPUs (${TOTAL_ACTOR_GPUS}) must be divisible by TP_SIZE*CP_SIZE (${MODEL_PARALLEL_SIZE})" >&2
  exit 1
fi
DP_SIZE="$((TOTAL_ACTOR_GPUS / MODEL_PARALLEL_SIZE))"

COMPACT_RESERVE_TOKENS="${COMPACT_RESERVE_TOKENS:-$((ROLLOUT_MAX_RESPONSE_LEN / 2))}"
DRESSAGE_BLACKBOX_COMPACT_THRESHOLD="${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD:-$((CONTEXT_WINDOW - COMPACT_RESERVE_TOKENS))}"
if [[ "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" -le 0 ]]; then
  echo "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD must be positive; got ${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-mopd-${MODEL_NAME,,}-sync}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs/${RUN_NAME}}"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

PROXY_PUBLIC_HOST="${PROXY_PUBLIC_HOST:-${CURRENT_NODE_IP:-${MASTER_ADDR}}}"
DRESSAGE_SANDBOX_PROVIDER="${DRESSAGE_SANDBOX_PROVIDER:-e2b}"
dressage_apply_common_defaults "${RUN_NAME}" "${DRESSAGE_PADDOCK_MODE}" "${DRESSAGE_SANDBOX_PROVIDER}"
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
dressage_export_common_env

export DRESSAGE_PROXY_PUBLIC_URL
export DRESSAGE_SANDBOX_DEFAULT_IMAGE="${DRESSAGE_SANDBOX_DEFAULT_IMAGE:-}"
export DRESSAGE_E2B_API_KEY="${DRESSAGE_E2B_API_KEY:-}"
export DRESSAGE_SANDBOX_PROVIDER_CLASS="${DRESSAGE_SANDBOX_PROVIDER_CLASS:-}"

export DRESSAGE_BLACKBOX_BACKEND_TIMEOUT="${DRESSAGE_BLACKBOX_BACKEND_TIMEOUT:-900}"
export DRESSAGE_BLACKBOX_MAX_STEPS="${DRESSAGE_BLACKBOX_MAX_STEPS:-80}"
export DRESSAGE_PROXY_MAX_STEPS_PER_SESSION="${DRESSAGE_PROXY_MAX_STEPS_PER_SESSION:-100}"
export DRESSAGE_ROLLOUT_MAX_RETRIES="${DRESSAGE_ROLLOUT_MAX_RETRIES:-0}"
export DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH="${DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH:-0}"
export DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER="${DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER:-2}"
: "${DRESSAGE_REWARD_MODULES:?MOPD config must declare reward_modules or set DRESSAGE_REWARD_MODULES}"

MOPD_ARGS=(
  --use-opd
  --opd-type megatron
  --opd-kl-coef "${OPD_KL_COEF:-0.1}"
  --opd-teacher-load "${MOPD_LAUNCH_CONFIG[5]}"
)
if [[ -n "${MOPD_LAUNCH_CONFIG[6]:-}" ]]; then
  MOPD_ARGS+=(--opd-teacher-ckpt-step "${MOPD_LAUNCH_CONFIG[6]}")
fi

echo "effective_parallelism: total_actor_gpus=${TOTAL_ACTOR_GPUS} TP_SIZE=${TP_SIZE} CP_SIZE=${CP_SIZE} DP_SIZE=${DP_SIZE} CONTEXT_WINDOW=${CONTEXT_WINDOW} ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "effective_blackbox: provider=${DRESSAGE_SANDBOX_PROVIDER} proxy=${DRESSAGE_PROXY_URL} backend_timeout=${DRESSAGE_BLACKBOX_BACKEND_TIMEOUT} max_steps=${DRESSAGE_BLACKBOX_MAX_STEPS} compact_threshold=${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}"
echo "effective_mopd: teacher_config=${DRESSAGE_MOPD_TEACHER_CONFIG} modes=${MOPD_LAUNCH_CONFIG[1]:-legacy} opd_kl_coef=${OPD_KL_COEF:-0.1}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
PROXY_ARGS=(
  --tokenizer-path "${TOKENIZER_PATH}"
  --host "${PROXY_HOST}"
  --port "${PROXY_PORT}"
  --trajectory-build-mode "${TRAJECTORY_BUILD_MODE}"
  --trajectory-build-model "${TRAJECTORY_BUILD_MODEL}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --context-window "${CONTEXT_WINDOW}"
)

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist/}"
CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_ROOT}/"
  --save-interval "${SAVE_INTERVAL:-10}"
)
if [[ "${RESUME_FROM_CKPT:-0}" == "1" ]]; then
  : "${RESUME_CKPT_ROOT:?RESUME_CKPT_ROOT is required when RESUME_FROM_CKPT=1}"
  CKPT_ARGS+=(--load "${RESUME_CKPT_ROOT%/}/")
  if [[ -n "${RESUME_CKPT_STEP:-}" ]]; then
    CKPT_ARGS+=(--ckpt-step "${RESUME_CKPT_STEP}")
  fi
  if [[ "${RESUME_LOAD_OPTIM:-1}" != "1" ]]; then
    CKPT_ARGS+=(--no-load-optim)
  fi
else
  CKPT_ARGS+=(--no-load-optim)
fi

ROLLOUT_ARGS=(
  --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync
  --custom-rm-path dressage.reward.custom_rm.custom_rm
  --data-source-path dressage.rollout.data_source.DressageDataSource
  --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT:-500}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
  --balance-data
  --rollout-global-dataset
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef "${ENTROPY_COEF:-0.0}"
  --eps-clip "${EPS_CLIP:-0.2}"
  --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)
if [[ "${NORMALIZE_ADVANTAGES:-0}" == "1" ]]; then
  GRPO_ARGS+=(--normalize-advantages)
fi
if [[ "${USE_TIS:-0}" == "1" ]]; then
  GRPO_ARGS+=(--use-tis)
fi
if [[ "${USE_KL_LOSS:-0}" == "1" ]]; then
  GRPO_ARGS+=(
    --use-kl-loss
    --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
    --kl-loss-type "${KL_LOSS_TYPE:-low_var_kl}"
  )
fi

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

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
  : "${WANDB_PROJECT:?Set WANDB_PROJECT when USE_WANDB=1}"
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP:-${RUN_NAME}}"
  )
  if [[ -n "${WANDB_TEAM:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  # On a single-node launcher the outer wrapper can authenticate once and let
  # workers use /root/.netrc.  In that mode, do not put the credential in the
  # long-lived trainer command line (and consequently in Ray job logs / ps).
  # Credentials normally travel in Ray's job-scoped runtime environment.
  # Putting them in train.py argv exposes them in `ps` and Ray job logs, so
  # retain that legacy behavior only behind an explicit opt-in.
  if [[ "${WANDB_USE_NETRC:-0}" != "1" && "${WANDB_KEY_IN_COMMAND_LINE:-0}" == "1" && -n "${WANDB_KEY:-${WANDB_API_KEY:-}}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY:-${WANDB_API_KEY}}")
  fi
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
  fi
  if [[ "${WANDB_DISABLE_RANDOM_SUFFIX:-0}" == "1" || -n "${WANDB_RUN_ID:-}" ]]; then
    WANDB_ARGS+=(--disable-wandb-random-suffix)
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-4}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.5}"
  --sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder
  --sglang-log-level warning
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

if [[ "${MOPD_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "effective_prompt_data=${PROMPT_DATA}"
  echo "effective_model=${MODEL_ROOT}/${MODEL_NAME} model_config=${MODEL_CONFIG}"
  echo "effective_runtime_env_keys=${DRESSAGE_EXTRA_RUNTIME_ENV_KEYS:-none}"
  echo "effective_reward_modules=${DRESSAGE_REWARD_MODULES}"
  exit 0
fi

DEBUG_ARGS=()
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
  DEBUG_ROLLOUT_DATA_PATH="${SAVE_DEBUG_ROLLOUT_DATA:-}"
  if [[ -z "${DEBUG_ROLLOUT_DATA_PATH}" ]]; then
    DEBUG_ROLLOUT_DATA_PATH="${SAVE_ROOT}/rollout_data/{rollout_id}.pt"
  fi
  DEBUG_ARGS=(
    --debug-rollout-only
    --save-debug-rollout-data "${DEBUG_ROLLOUT_DATA_PATH}"
  )
fi

PROXY_LOG_FILE="${PROXY_LOG_FILE:-${LOG_DIR}/dressage-proxy.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-${LOG_DIR}/dressage-proxy.pid}"
python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
echo $! >"${PROXY_PID_FILE}"

cleanup() {
  status=$?
  set +e
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    kill "$(cat "${PROXY_PID_FILE}")" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
  if [[ "${DRESSAGE_RAY_STOP_ON_EXIT:-0}" == "1" ]]; then
    ray stop --force || true
  fi
  exit "${status}"
}
trap cleanup EXIT

if [[ "${DRESSAGE_SKIP_PROXY_HEALTHCHECK:-0}" != "1" ]]; then
  for i in $(seq 1 60); do
    curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1 && break
    if [[ "${i}" -eq 60 ]]; then
      echo "Dressage proxy failed; see ${PROXY_LOG_FILE}" >&2
      exit 1
    fi
    sleep 1
  done
fi

PUBLIC_HOST="${PROXY_PUBLIC_HOST:-${CURRENT_NODE_IP:-${MASTER_ADDR}}}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${PUBLIC_HOST},${SGLANG_ROUTER_HOST:-}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_ROOT}"
ray start --head --block --port="${RAY_PORT}" --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
  --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats \
  --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port="${RAY_DASHBOARD_PORT}" &
sleep 10

if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
  : "${RAY_SSH_USER:?RAY_SSH_USER is required when HOSTFILE is set}"
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh "${RAY_SSH_USER}"@"${WORKER_IP}" \
      "ray stop --force || true; ray start --address=${MASTER_ADDR}:${RAY_PORT} --num-gpus ${RAY_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

JOIN_DEADLINE=$((SECONDS + RAY_JOIN_TIMEOUT_SEC))
while true; do
  NODE_COUNT="$(ray status 2>/dev/null | grep -c node_ || true)"
  echo "Current Ray node count: ${NODE_COUNT}; expected ACTOR_NUM_NODES: ${ACTOR_NUM_NODES}"
  [[ "${NODE_COUNT}" -eq "${ACTOR_NUM_NODES}" ]] && break
  if [[ "${SECONDS}" -ge "${JOIN_DEADLINE}" ]]; then
    echo "Timed out waiting for Ray workers to join." >&2
    exit 1
  fi
  sleep 10
done

RUNTIME_ENV_JSON="$(
  python3 - <<'PY'
import json
import os

keys = [
    "no_proxy",
    "NO_PROXY",
    "MASTER_ADDR",
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "WANDB_MODE",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_DIR",
    "WANDB_DATA_DIR",
    "WANDB_ARTIFACT_DIR",
    "DRESSAGE_PROXY_URL",
    "DRESSAGE_PROXY_PUBLIC_URL",
    "DRESSAGE_PADDOCK_MODE",
    "DRESSAGE_SANDBOX_PROVIDER",
    "DRESSAGE_SANDBOX_PROVIDER_CLASS",
    "DRESSAGE_SANDBOX_DEFAULT_IMAGE",
    "DRESSAGE_E2B_API_KEY",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR",
    "DRESSAGE_REWARD_MODULES",
    "DRESSAGE_BLACKBOX_BACKEND_TIMEOUT",
    "DRESSAGE_BLACKBOX_MAX_STEPS",
    "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD",
    "DRESSAGE_PROXY_MAX_STEPS_PER_SESSION",
    "DRESSAGE_ROLLOUT_MAX_RETRIES",
    "DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH",
    "DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER",
    "DRESSAGE_MOPD_TEACHER_CONFIG",
]
if os.environ.get("WANDB_USE_NETRC", "0") != "1":
    keys.extend(["WANDB_KEY", "WANDB_API_KEY"])
for raw_key in os.environ.get("DRESSAGE_EXTRA_RUNTIME_ENV_KEYS", "").split(","):
    key = raw_key.strip()
    if key and key not in keys:
        keys.append(key)
env = {}
for key in keys:
    value = os.environ.get(key)
    if value not in (None, ""):
        env[key] = value
print(json.dumps({"env_vars": env}))
PY
)"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -m dressage.training.mopd_train \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
  "${DEBUG_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MOPD_ARGS[@]}" \
  "${MISC_ARGS[@]}"
