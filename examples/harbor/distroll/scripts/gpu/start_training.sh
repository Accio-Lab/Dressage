#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
: "${RUN_NAME:?set a unique RUN_NAME for this training run}"
# Preserve explicit per-run knobs across the deployment profile source;
# otherwise stale profile values silently win over `docker exec -e ...`.
RUN_OVERRIDE_NAMES=(
  RUN_NAME
  PROMPT_DATA
  NUM_ROLLOUT
  TRAIN_RESUME_FROM
  TRAIN_RESUME_CURSOR
  REUSE_RAY
  ROLLOUT_BATCH_SIZE
  ROLLOUT_SEED
  N_SAMPLES_PER_PROMPT
  GLOBAL_BATCH_SIZE
  ROLLOUT_MAX_CONCURRENT_EXECUTIONS
  ROLLOUT_ACCEPT_MAX_TURNS
  ROLLOUT_ADMISSION_TIMEOUT_SEC
  ROLLOUT_GROUP_MAX_RETRIES
  ROLLOUT_FAILED_GROUP_POLICY
  ROLLOUT_MIN_LIVE_GROUP_RATIO
  ROLLOUT_CONTROL_RETRY_ATTEMPTS
  ROLLOUT_RETRY_BASE_SEC
  ROLLOUT_RETRY_MAX_SEC
  CLAUDE_CODE_MAX_TURNS
  CLAUDE_CODE_AGENT_TIMEOUT_SEC
  CLAUDE_AUTOCOMPACT_PCT_OVERRIDE
  CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS
  USE_KL_LOSS
  KL_LOSS_COEF
  ENTROPY_COEF
  DYNAMIC_SAMPLING_FILTER_PATH
  OVER_SAMPLING_BATCH_SIZE
  DYNAMIC_SAMPLING_MAX_GROUPS
  DYNAMIC_SAMPLING_EARLY_STOP
  ROLLOUT_GROUP_MIN_SUCCESS_RATIO
  ROLLOUT_SPECULATIVE_CANCEL_TIMEOUT_SEC
  ROLLOUT_FUNCTION_PATH
  USE_TENSORBOARD
  TB_PROJECT_NAME
  TB_EXPERIMENT_NAME
  TENSORBOARD_DIR
  LOG_PASSRATE
  DRESSAGE_TOKEN_BUILD_MODE
  DRESSAGE_TOKEN_BUILD_MODEL
)
declare -A RUN_OVERRIDES=()
for name in "${RUN_OVERRIDE_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    RUN_OVERRIDES["${name}"]="${!name}"
  fi
done
set -a
source "${GPU_ENV_FILE:-${PROJECT_ROOT}/configs/gpu.training.env.example}"
set +a
for name in "${!RUN_OVERRIDES[@]}"; do
  printf -v "${name}" '%s' "${RUN_OVERRIDES[${name}]}"
  export "${name}"
done
unset RUN_OVERRIDES RUN_OVERRIDE_NAMES name
export DRESSAGE_DISTROLL_RUN_ID="${RUN_NAME}"
export TB_EXPERIMENT_NAME="${TB_EXPERIMENT_NAME:-${RUN_NAME}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${PROJECT_ROOT}/outputs/tensorboard/${RUN_NAME}}"

MODEL_ROOT="${MODEL_ROOT:-/models}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B}"
MODEL_CONFIG="${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/data/tmax1500.jsonl}"
RUN_NAME="${RUN_NAME:-tmax-grpo-sync}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/outputs/checkpoints/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/${RUN_NAME}}"
PID_DIR="${PID_DIR:-${PROJECT_ROOT}/outputs/pids}"
# Ray appends a long session/sockets suffix. Keep its ephemeral runtime root
# short and on the container-local Docker filesystem so AF_UNIX stays below
# Linux's 107-byte pathname limit. Durable logs/results/checkpoints remain under
# PROJECT_ROOT.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/dressage-ray}"
SUBMISSION_ID="${RAY_SUBMISSION_ID:-${RUN_NAME}}"
export DRESSAGE_TRANSFER_QUEUE_CONFIG="${DRESSAGE_TRANSFER_QUEUE_CONFIG:-${DRESSAGE_ROOT}/examples/scripts/default/dressage_transfer_queue.yaml}"
export DRESSAGE_TRANSFER_PARAMS="${DRESSAGE_TRANSFER_PARAMS:-logprobs,routed_experts}"
export DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS="${DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS:-86400}"
export DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY="${DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY:-1}"
SLIME_ENTRY_ARGS=(-m dressage_distroll_gpu_adapter.persistent_train)
ROLLOUT_DIAGNOSTICS_DIR="${ROLLOUT_DIAGNOSTICS_DIR:-${PROJECT_ROOT}/outputs/rl-state/${RUN_NAME}}"
export ROLLOUT_ACCEPT_MAX_TURNS=${ROLLOUT_ACCEPT_MAX_TURNS:-0}
[[ "${ROLLOUT_ACCEPT_MAX_TURNS}" =~ ^[01]$ ]] || {
  echo "ROLLOUT_ACCEPT_MAX_TURNS must be 0 or 1" >&2
  exit 1
}
if [[ "${TRAIN_DATA_CONVERTER:-dressage.rollout.convert_samples.convert_samples_to_train_data}" != \
  dressage.rollout.convert_samples.convert_samples_to_train_data ]]; then
  echo "V4 native delivery requires the original Dressage converter" >&2
  exit 1
fi
if [[ -n "${ROLLOUT_DIAGNOSTICS_DIR:-}" ]]; then
  export ROLLOUT_DIAGNOSTICS_DIR
fi

required_paths=(
  "${GPU_PYTHON}"
  "${MODEL_CONFIG}"
  "${DRESSAGE_TRANSFER_QUEUE_CONFIG}"
  "${HF_CHECKPOINT}/config.json"
  "${PROMPT_DATA}"
)
required_paths+=("${REF_LOAD}")
for path in "${required_paths[@]}"; do
  [[ -e "${path}" ]] || {
    echo "required training input is missing: ${path}" >&2
    exit 1
  }
done
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
GPU_CLUSTER_NUM_NODES="${GPU_CLUSTER_NUM_NODES:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
validate_gpu_sequence_budget
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "${GPU_COUNT}" -eq "${NUM_GPUS_PER_NODE}" ]] || {
  echo "this GPU node expects ${NUM_GPUS_PER_NODE} visible GPUs; found ${GPU_COUNT}" >&2
  exit 1
}

"${SCRIPT_DIR}/start_services.sh"
"${SCRIPT_DIR}/check_cross_cluster.sh"
source "${MODEL_CONFIG}"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${RAY_TEMP_DIR}"
mkdir -p "${SAVE_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export no_proxy="${ROLLOUT_NO_PROXY:-127.0.0.1,localhost}"
export NO_PROXY="${no_proxy}"
ROLLOUT_GATEWAY_RESOURCE_NAME="${ROLLOUT_GATEWAY_RESOURCE_NAME:-rollout_gateway}"
ROLLOUT_GATEWAY_RESOURCE_AMOUNT="${ROLLOUT_GATEWAY_RESOURCE_AMOUNT:-0.01}"
export ROLLOUT_GATEWAY_RESOURCE_NAME ROLLOUT_GATEWAY_RESOURCE_AMOUNT
export NUM_GPUS_PER_NODE GPU_CLUSTER_NUM_NODES

if "${GPU_RAY}" status >/dev/null 2>&1; then
  if [[ "${REUSE_RAY:-0}" != "1" ]]; then
    echo "an existing Ray cluster is running; set REUSE_RAY=1 or stop it explicitly" >&2
    exit 1
  fi
else
  RUN_NAME="${RUN_NAME}" RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
    NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE}" \
    "${SCRIPT_DIR}/start_ray_head.sh"
fi

"${GPU_PYTHON}" - <<'PY'
import os
import math

import ray

ray.init(address="auto", logging_level="ERROR")
name = os.environ["ROLLOUT_GATEWAY_RESOURCE_NAME"]
required = float(os.environ["ROLLOUT_GATEWAY_RESOURCE_AMOUNT"])
if not math.isfinite(required) or required <= 0:
    raise SystemExit("ROLLOUT_GATEWAY_RESOURCE_AMOUNT must be finite and positive")
available = float(ray.cluster_resources().get(name, 0.0))
expected_nodes = int(os.environ["GPU_CLUSTER_NUM_NODES"])
expected_gpus = expected_nodes * int(os.environ["NUM_GPUS_PER_NODE"])
nodes = [node for node in ray.nodes() if node.get("Alive")]
live_nodes = len(nodes)
available_gpus = int(ray.cluster_resources().get("GPU", 0))
gateway_nodes = [
    node
    for node in nodes
    if float(node.get("Resources", {}).get(name, 0.0)) >= required
]
expected_gateway_ip = os.environ["GPU_NODE_IP"]
ray.shutdown()
if available < required:
    raise SystemExit(
        f"Ray cluster is missing gateway resource {name!r}: "
        f"required={required} available={available}"
    )
if len(gateway_nodes) != 1:
    addresses = [node.get("NodeManagerAddress") for node in gateway_nodes]
    raise SystemExit(
        f"Ray requires exactly one alive {name!r} node; found {addresses}"
    )
actual_gateway_ip = gateway_nodes[0].get("NodeManagerAddress")
if actual_gateway_ip != expected_gateway_ip:
    raise SystemExit(
        f"Ray gateway resource is on {actual_gateway_ip}, expected {expected_gateway_ip}"
    )
if live_nodes < expected_nodes or available_gpus < expected_gpus:
    raise SystemExit(
        "Ray GPU cluster is incomplete: "
        f"expected_nodes={expected_nodes} live_nodes={live_nodes} "
        f"expected_gpus={expected_gpus} available_gpus={available_gpus}"
    )
if os.environ.get("DRESSAGE_ACTOR_NODE_IPS"):
    expected_ips = set(os.environ["DRESSAGE_ACTOR_NODE_IPS"].split(","))
    expected_ips.add(os.environ["DRESSAGE_ROLLOUT_NODE_IP"])
    gpu_nodes = {
        n["NodeManagerAddress"]: n["Resources"].get("GPU", 0)
        for n in nodes if n.get("Resources", {}).get("GPU", 0)
    }
    if gpu_nodes != {ip: 8 for ip in expected_ips}:
        raise SystemExit(f"Ray GPU roles require {sorted(expected_ips)} x8; got {gpu_nodes}")
PY

RUNTIME_ENV_JSON="$("${GPU_PYTHON}" - <<'PY'
import json, os
keys = [
    "PYTHONPATH", "PYTHONUNBUFFERED", "GPU_SOURCE_ROOT",
    "PROJECT_ROOT", "GPU_RUNTIME_LOCK", "DRESSAGE_ROOT", "SLIME_ROOT", "MEGATRON_ROOT",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE", "NCCL_DEBUG", "no_proxy", "NO_PROXY",
    "ROLLOUT_CONTROL_URL", "POLICY_SERVICE_URL",
    "ROLLOUT_MAX_CONTEXT_LEN",
    "DRESSAGE_PROXY_URL",
    "DRESSAGE_DISTROLL_RUN_ID", "ROLLOUT_MODEL_NAME",
    "ROLLOUT_FUNCTION_PATH",
    "ROLLOUT_CONTROL_OUTAGE_TIMEOUT_SEC", "ROLLOUT_WAIT_LONG_POLL_SEC",
    "ROLLOUT_MAX_CONCURRENT_EXECUTIONS", "ROLLOUT_GROUP_MAX_RETRIES",
    "ROLLOUT_SEED",
    "ROLLOUT_ADMISSION_TIMEOUT_SEC",
    "ROLLOUT_FAILED_GROUP_POLICY", "ROLLOUT_MIN_LIVE_GROUP_RATIO",
    "ROLLOUT_DIAGNOSTICS_DIR",
    "CLAUDE_CODE_MAX_TURNS", "CLAUDE_CODE_VERSION",
    "ROLLOUT_ACCEPT_MAX_TURNS",
    "CLAUDE_CODE_AGENT_TIMEOUT_SEC", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS",
    "ROLLOUT_CONTROL_RETRY_ATTEMPTS", "ROLLOUT_PROXY_RETRY_ATTEMPTS",
    "ROLLOUT_RETRY_BASE_SEC", "ROLLOUT_RETRY_MAX_SEC",
    "DYNAMIC_SAMPLING_FILTER_PATH", "OVER_SAMPLING_BATCH_SIZE",
    "DYNAMIC_SAMPLING_MAX_GROUPS", "DYNAMIC_SAMPLING_EARLY_STOP",
    "ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "ROLLOUT_SPECULATIVE_CANCEL_TIMEOUT_SEC",
    "USE_TENSORBOARD", "TB_PROJECT_NAME",
    "TB_EXPERIMENT_NAME", "TENSORBOARD_DIR", "USE_KL_LOSS", "KL_LOSS_COEF",
    "ENTROPY_COEF",
    "LOG_PASSRATE",
    "DRESSAGE_TOKEN_BUILD_MODE", "DRESSAGE_TOKEN_BUILD_MODEL",
    "ROLLOUT_GATEWAY_RESOURCE_NAME", "ROLLOUT_GATEWAY_RESOURCE_AMOUNT",
    "WANDB_API_KEY", "WANDB_MODE",
    "DRESSAGE_TRANSFER_QUEUE_CONFIG", "DRESSAGE_TRANSFER_PARAMS",
    "DRESSAGE_TRANSFER_QUEUE_RETENTION_SECONDS", "DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY",
    "DRESSAGE_ACTOR_NODE_IPS", "DRESSAGE_ROLLOUT_NODE_IP",
    "TRAIN_RESUME_FROM", "TRAIN_RESUME_CURSOR",
]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))
PY
)"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_ROOT}"
  --save-interval "${SAVE_INTERVAL:-10}"
)
DATA_SOURCE_PATH=dressage_distroll_gpu_adapter.training_resume.TrainingDataSource
if [[ -n "${TRAIN_RESUME_FROM:-}" ]]; then
  resume_iteration=$("${GPU_PYTHON}" -m dressage_distroll_gpu_adapter.training_resume)
  CKPT_ARGS+=(--load "${TRAIN_RESUME_FROM}" --ckpt-step "${resume_iteration}")
else
  CKPT_ARGS+=(--no-load-optim)
fi

ROLLOUT_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH:-dressage_distroll_gpu_adapter.rollout.generate_remote_harbor_sync}"
  --data-source-path "${DATA_SOURCE_PATH}"
  --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
  --custom-convert-samples-to-train-data-path "${TRAIN_DATA_CONVERTER:-dressage.rollout.convert_samples.convert_samples_to_train_data}"
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data
  # Harbor rewards are mappings so verifier auxiliary signals remain available.
  # Tell Slime which scalar to use in its built-in metrics and training paths.
  --reward-key reward
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --rollout-seed "${ROLLOUT_SEED:-42}"
  --num-rollout "${NUM_ROLLOUT:-10}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-1}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"
  --balance-data
)
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]]; then
  : "${OVER_SAMPLING_BATCH_SIZE:?set OVER_SAMPLING_BATCH_SIZE when dynamic sampling is enabled}"
  : "${DYNAMIC_SAMPLING_MAX_GROUPS:?set DYNAMIC_SAMPLING_MAX_GROUPS when dynamic sampling is enabled}"
  [[ "${OVER_SAMPLING_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "OVER_SAMPLING_BATCH_SIZE must be a positive integer" >&2
    exit 1
  }
  if (( OVER_SAMPLING_BATCH_SIZE < ROLLOUT_BATCH_SIZE )); then
    echo "Slime requires OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE" >&2
    exit 1
  fi
  ROLLOUT_ARGS+=(
    --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  )
fi
if [[ "${LOG_PASSRATE:-0}" == "1" ]]; then
  ROLLOUT_ARGS+=(--log-passrate)
elif [[ "${LOG_PASSRATE:-0}" != "0" ]]; then
  echo "LOG_PASSRATE must be 0 or 1" >&2
  exit 1
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-4}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
  --log-probs-chunk-size 1024
)
if [[ "${SEQUENCE_PARALLEL:-0}" == "1" ]]; then
  PERF_ARGS+=(--sequence-parallel)
fi
if [[ "${CALCULATE_PER_TOKEN_LOSS:-0}" == "1" ]]; then
  PERF_ARGS+=(--calculate-per-token-loss)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)
if [[ "${OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-tis
  --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF:-0.0}"
  --eps-clip 0.2
  --eps-clip-high 0.28
)
if [[ "${USE_KL_LOSS:-1}" == "1" ]]; then
  GRPO_ARGS+=(--use-kl-loss)
elif [[ "${USE_KL_LOSS:-1}" != "0" ]]; then
  echo "USE_KL_LOSS must be 0 or 1" >&2
  exit 1
fi

TRACKING_ARGS=()
if [[ "${USE_TENSORBOARD:-0}" == "1" ]]; then
  TRACKING_ARGS+=(--use-tensorboard)
  [[ -n "${TB_PROJECT_NAME:-}" || -n "${TENSORBOARD_DIR:-}" ]] || {
    echo "USE_TENSORBOARD=1 requires TB_PROJECT_NAME or TENSORBOARD_DIR" >&2
    exit 1
  }
  [[ -z "${TB_PROJECT_NAME:-}" ]] || TRACKING_ARGS+=(--tb-project-name "${TB_PROJECT_NAME}")
  [[ -z "${TB_EXPERIMENT_NAME:-}" ]] || TRACKING_ARGS+=(--tb-experiment-name "${TB_EXPERIMENT_NAME}")
elif [[ "${USE_TENSORBOARD:-0}" != "0" ]]; then
  echo "USE_TENSORBOARD must be 0 or 1" >&2
  exit 1
fi

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-server-concurrency 16
  --sglang-served-model-name "${ROLLOUT_MODEL_NAME}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}"
  --sglang-reasoning-parser "${SGLANG_REASONING_PARSER:-qwen3}"
  --sglang-tool-call-parser "${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
  --sglang-log-level warning
  --sglang-router-port 8000
  --router-policy consistent_hashing
)
if [[ -n "${SGLANG_WEIGHT_VERSION:-}" ]]; then
  SGLANG_ARGS+=(--sglang-weight-version "${SGLANG_WEIGHT_VERSION}")
fi
case "${SGLANG_DISABLE_CUDA_GRAPH:-0}" in
  0) ;;
  1) SGLANG_ARGS+=(--sglang-disable-cuda-graph) ;;
  *) echo "SGLANG_DISABLE_CUDA_GRAPH must be 0 or 1" >&2; exit 1 ;;
esac

TQ_ARGS=()
if [[ "${DRESSAGE_USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
  TQ_ARGS+=(--use-rollout-routing-replay)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --update-weight-mode full
  --update-weight-transport nccl
  --no-offload-train
  --no-offload-rollout
)

cd "${PROJECT_ROOT}"
"${GPU_RAY}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
  --submission-id "${SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${GPU_PYTHON}" "${SLIME_ENTRY_ARGS[@]}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${TRACKING_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${TQ_ARGS[@]}" \
  "${MISC_ARGS[@]}"

printf '%s\n' "${SUBMISSION_ID}" >"${PROJECT_ROOT}/outputs/pids/ray-submission-id"
submission_key="$(printf '%s' "${SUBMISSION_ID}" | sha256sum | awk '{print $1}')"
log_follower_pid_file="${PID_DIR}/ray-job-log-${submission_key}.pid"
if [[ -f "${log_follower_pid_file}" ]] \
  && kill -0 "$(cat "${log_follower_pid_file}")" 2>/dev/null; then
  echo "Ray log follower already active for ${SUBMISSION_ID}" >&2
  exit 1
fi
nohup "${SCRIPT_DIR}/follow_ray_job_logs.sh" \
  "${GPU_RAY}" \
  "${SUBMISSION_ID}" \
  "${LOG_DIR}/ray-job.log" \
  "${LOG_DIR}/ray-status-final.txt" \
  >/dev/null 2>&1 &
echo $! >"${log_follower_pid_file}"
echo "Slime synchronous training job submitted as ${SUBMISSION_ID}"
echo "logs: ${GPU_RAY} job logs --follow ${SUBMISSION_ID}"
echo "durable logs: ${LOG_DIR}/ray-job.log"
