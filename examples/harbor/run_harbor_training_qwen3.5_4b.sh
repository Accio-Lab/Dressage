#!/usr/bin/env bash

# Synchronous Harbor/slime training for Qwen/Qwen3.5-4B.
# Activate the Python 3.12 Harbor/slime environment before running this script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${DRESSAGE_HARBOR_JOB_CONFIG:?Set DRESSAGE_HARBOR_JOB_CONFIG to a Harbor JobConfig path}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-/root/Qwen3.5-4B_torch_dist}"
CKPT_LOAD="${CKPT_LOAD:-/root/Qwen3.5-4B_slime/}"
CKPT_SAVE="${HARBOR_TRAINING_CHECKPOINT_DIR:-/root/dressage-harbor/checkpoints}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
SLIME_HOST_IP="${SLIME_HOST_IP:-${MASTER_ADDR}}"
DRESSAGE_PROXY_HOST="${DRESSAGE_PROXY_HOST:-127.0.0.1}"
DRESSAGE_PROXY_PORT="${DRESSAGE_PROXY_PORT:-8800}"
SGLANG_ROUTER_PORT="${TRAINING_SGLANG_ROUTER_PORT:-8000}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-49152}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
LOG_DIR="${HARBOR_LOG_DIR:-/root/dressage-harbor/logs/training}"
DEBUG_DIR="${HARBOR_DEBUG_DIR:-/root/dressage-harbor/debug}"
PROXY_URL="http://${DRESSAGE_PROXY_HOST}:${DRESSAGE_PROXY_PORT}"
MODEL_ARGS_SCRIPT="${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

export DRESSAGE_HARBOR_INTEGRATION_CONFIG="${DRESSAGE_HARBOR_INTEGRATION_CONFIG:-${REPO_ROOT}/examples/harbor/dressage_profiles/training-bwrap.yaml}"
export DRESSAGE_HARBOR_JOB_CONFIG
export DRESSAGE_HARBOR_RUN_ID="${DRESSAGE_HARBOR_RUN_ID:-harbor-training-$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="${MEGATRON_ROOT}:${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"

# A fresh backend credential is generated for every run.
set +x
export DRESSAGE_PROXY_API_KEY="$(openssl rand -hex 48)"

export MASTER_ADDR SLIME_HOST_IP
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${SLIME_HOST_IP}"
export NO_PROXY="${no_proxy}"

mkdir -p "${LOG_DIR}" "${DEBUG_DIR}" "${CKPT_SAVE}"
source "${MODEL_ARGS_SCRIPT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD}"
  --load "${CKPT_LOAD}"
  --save "${CKPT_SAVE}"
  --save-interval 20
)
ROLLOUT_ARGS=(
  --data-source-path dressage.integrations.harbor.data_source.HarborDataSource
  --rollout-function-path dressage.integrations.harbor.rollout.generate_rollout_harbor_sync
  --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
  --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data
  --reward-key reward
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${CONTEXT_WINDOW}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id}.pt"
  --save-debug-train-data "${DEBUG_DIR}/train_{rollout_id}_{rank}.pt"
)
COMM_ARGS=(--rollout-temperature 1.0)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
)
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)
PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-context-length "${CONTEXT_WINDOW}"
  --sglang-mem-fraction-static "${TRAINING_SGLANG_MEM_FRACTION_STATIC:-0.5}"
  --sglang-router-port "${SGLANG_ROUTER_PORT}"
  --router-policy consistent_hashing
  --sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder
  --sglang-log-level warning
  --sglang-cuda-graph-max-bs 64
  --sglang-max-running-requests 64
)
MISC_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --num-gpus-per-node "${RAY_NUM_GPUS_PER_NODE}"
  --colocate
  --loss-mask-type qwen3_5
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
WANDB_ARGS=(
  # --use-wandb
  # --wandb-project dressage-opensource
  # --wandb-group qwen3.5-4B-dressage-harbor
  # --wandb-key "${WANDB_KEY}"
)

wait_for_url() {
  local url="$1" retries="${2:-120}"
  for ((attempt=1; attempt<=retries; attempt++)); do
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

PROXY_PID=""
RAY_STARTED=0
cleanup() {
  local status=$?
  if [[ "${RAY_STARTED}" == "1" ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
  if [[ -n "${PROXY_PID}" ]]; then
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" -m dressage.proxy.server \
  --sglang-router-url "http://${SLIME_HOST_IP}:${SGLANG_ROUTER_PORT}" \
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
  --record-token-versions \
  --context-window "${CONTEXT_WINDOW}" \
  --max-output-tokens "${ROLLOUT_MAX_RESPONSE_LEN}" \
  >"${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_url "${PROXY_URL}/health" 120

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"
RAY_STARTED=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "DRESSAGE_HARBOR_INTEGRATION_CONFIG": "${DRESSAGE_HARBOR_INTEGRATION_CONFIG}",
    "DRESSAGE_HARBOR_JOB_CONFIG": "${DRESSAGE_HARBOR_JOB_CONFIG}",
    "DRESSAGE_HARBOR_RUN_ID": "${DRESSAGE_HARBOR_RUN_ID}",
    "DRESSAGE_PROXY_API_KEY": "${DRESSAGE_PROXY_API_KEY}",
    "E2B_API_KEY": "${E2B_API_KEY:-}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "${CUDA_DEVICE_MAX_CONNECTIONS}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "SLIME_HOST_IP": "${SLIME_HOST_IP}",
    "PYTHONPATH": "${PYTHONPATH}",
    "no_proxy": "${no_proxy}",
    "NO_PROXY": "${NO_PROXY}"
  }
}
EOF_JSON
)

cd "${SLIME_ROOT}"
ray job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  "${PYTHON_BIN}" "${SLIME_ROOT}/train.py" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${COMM_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
