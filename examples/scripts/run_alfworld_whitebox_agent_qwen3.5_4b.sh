#!/bin/bash

# ALFWorld whitebox training on Qwen3.5-4B.
#
# Layout follows run_hotpotqa_whitebox_agent_qwen3.5_4b.sh but swaps in the ALFWorld agent:
#   - rollout: dressage.recipes.alfworld.agent_whitebox.generate (WhiteboxAgent)
#   - reward : dressage.recipes.alfworld.reward (success/format/0 from metadata)
#   - data   : repo-local jsonl prepared with examples/data/alfworld/prepare_alfworld.py
#
# Quickstart prerequisites (one-time):
#   place Qwen3.5-4B and Qwen3.5-4B_torch_dist under /root/
#   pip install alfworld[full]
#   ALFWORLD_DATA="$(pwd)/examples/data/alfworld/alfworld_data" alfworld-download
#   python examples/data/alfworld/prepare_alfworld.py

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
BASE_FOLDER="${BASE_FOLDER:-/root}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/examples/data}"

if [[ ! -f "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" ]]; then
  echo "Cannot find slime model config: ${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" >&2
  echo "Set REPO_ROOT or SLIME_ROOT to match the current checkout layout." >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set." >&2
  exit 1
fi

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-24576}
# CONTEXT_WINDOW=${CONTEXT_WINDOW:-$((MAX_TOKENS_PER_GPU * CP_SIZE))}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
HOSTFILE=${HOSTFILE:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

# --- Proxy configuration ---
PROXY_HOST=${PROXY_HOST:-0.0.0.0}
PROXY_PORT=${PROXY_PORT:-8800}
PROXY_PUBLIC_HOST=${PROXY_PUBLIC_HOST:-$(hostname -i)}
DRESSAGE_PROXY_URL=${DRESSAGE_PROXY_URL:-http://${PROXY_PUBLIC_HOST}:${PROXY_PORT}}
SGLANG_ROUTER_HOST=${SGLANG_ROUTER_HOST:-$(hostname -i)}
SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT:-8000}
SGLANG_ROUTER_URL=${SGLANG_ROUTER_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}

# --- Checkpoint configuration ---
HF_CHECKPOINT=${HF_CHECKPOINT:-${BASE_FOLDER}/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-${BASE_FOLDER}/Qwen3.5-4B_torch_dist/}
CKPT_LOAD=${CKPT_LOAD:-${BASE_FOLDER}/Qwen3.5-4B_slime/}
CKPT_SAVE=${CKPT_SAVE:-${BASE_FOLDER}/Qwen3.5-4B_slime/}

# --- Training data ---
PROMPT_DATA=${PROMPT_DATA:-${DATA_ROOT}/alfworld/train.jsonl}

# --- ALFWorld agent loop knobs ---
# Concat-mode rollout: each trajectory is one append-only conversation. 30 steps
# matches Embodied-Planner-R1 (arXiv 2506.23127) and keeps the worst-case sample
# total_lengths under the 12288 max-tokens-per-gpu cap (measured: 50 steps at
# rollout 0 averaged 11394 total tokens with peak ~24k; 30 steps caps the tail
# at ~18k and still leaves room as the policy learns to explore more).
ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-30}
ALFWORLD_MAX_EPISODE_STEPS=${ALFWORLD_MAX_EPISODE_STEPS:-30}
ALFWORLD_HISTORY_WINDOW=${ALFWORLD_HISTORY_WINDOW:-10}

# --- Model format configuration (proxy + parser) ---
# ALFWorld now uses tool_call schema (env_step) to mirror the StepPO recipe;
# the proxy parses tool_call output via the configured backend.
MODEL_MASK_TYPE=${MODEL_MASK_TYPE:-qwen3_5}
MODEL_TOOL_CALL_TYPE=${MODEL_TOOL_CALL_TYPE:-qwen3_5}
TOOL_CALL_PARSE_BACKEND=${TOOL_CALL_PARSE_BACKEND:-sglang_api}
MODEL_REASONING_TYPE=${MODEL_REASONING_TYPE:-qwen3}
REASONING_PARSE_BACKEND=${REASONING_PARSE_BACKEND:-sglang_api}
TRAJECTORY_BUILD_MODE=${TRAJECTORY_BUILD_MODE:-concat}
TITO_MODEL=${TITO_MODEL:-qwen3_5}
TOKENIZER_PATH=${TOKENIZER_PATH:-${HF_CHECKPOINT}}

LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/log/alfworld-qwen3.5-4B}
DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR=${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR:-${LOG_DIR}/trajectory_payload}
DRESSAGE_TRAJECTORY_ERROR_LOG_DIR=${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR:-${LOG_DIR}/trajectory_err}
PROXY_LOG_FILE=${PROXY_LOG_FILE:-${LOG_DIR}/dressage-proxy.log}
PROXY_PID_FILE=${PROXY_PID_FILE:-${LOG_DIR}/dressage-proxy.pid}
mkdir -p "${LOG_DIR}" "$(dirname "${PROXY_PID_FILE}")" "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}" "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}"

for TRAJECTORY_LOG_DIR_VAR in DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR DRESSAGE_TRAJECTORY_ERROR_LOG_DIR; do
  TRAJECTORY_LOG_DIR="${!TRAJECTORY_LOG_DIR_VAR}"
  if [[ -z "${TRAJECTORY_LOG_DIR}" || "${TRAJECTORY_LOG_DIR}" == "/" ]]; then
    echo "Refusing to clear unsafe ${TRAJECTORY_LOG_DIR_VAR}: ${TRAJECTORY_LOG_DIR}" >&2
    exit 1
  fi
  echo "Clearing ${TRAJECTORY_LOG_DIR_VAR}: ${TRAJECTORY_LOG_DIR}"
  find "${TRAJECTORY_LOG_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
done

if [[ "${TRAJECTORY_BUILD_MODE}" != "last_step" && "${TRAJECTORY_BUILD_MODE}" != "concat" ]]; then
  echo "TRAJECTORY_BUILD_MODE must be last_step or concat, got: ${TRAJECTORY_BUILD_MODE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
export DRESSAGE_PROXY_URL
export DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR DRESSAGE_TRAJECTORY_ERROR_LOG_DIR
export ALFWORLD_MAX_STEPS ALFWORLD_MAX_EPISODE_STEPS ALFWORLD_HISTORY_WINDOW

COMM_ARGS=(
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
)

PROXY_ARGS=(
   --sglang-router-url "${SGLANG_ROUTER_URL}"
   --tokenizer-path "${TOKENIZER_PATH}"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --model-mask-type "${MODEL_MASK_TYPE}"
   --model-tool-call-type "${MODEL_TOOL_CALL_TYPE}"
   --tool-call-parse-backend "${TOOL_CALL_PARSE_BACKEND}"
   --model-reasoning-type "${MODEL_REASONING_TYPE}"
   --reasoning-parse-backend "${REASONING_PARSE_BACKEND}"
   --trajectory-build-mode "${TRAJECTORY_BUILD_MODE}"
   "${COMM_ARGS[@]}"
  #  --context-window "${CONTEXT_WINDOW}"
   --tito-model "${TITO_MODEL}"
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${CKPT_LOAD}"
   --save "${CKPT_SAVE}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync
   --custom-generate-function-path "dressage.recipes.alfworld.agent_whitebox.generate"
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

   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
   --balance-data
)

EVAL_ARGS=(
   # Multi-segment whitebox does not support evaluation yet.
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
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
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group alfworld-qwen3.5-4B-whitebox
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
   --router-policy consistent_hashing
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# --- Start Dressage proxy ---
if [[ -f "${PROXY_PID_FILE}" ]]; then
  OLD_PROXY_PID="$(cat "${PROXY_PID_FILE}")"
  if ! kill -0 "${OLD_PROXY_PID}" 2>/dev/null; then
    rm -f "${PROXY_PID_FILE}"
  fi
fi

if [[ ! -f "${PROXY_PID_FILE}" ]]; then
  cd "${REPO_ROOT}"
  python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
  echo $! > "${PROXY_PID_FILE}"
  echo "Started Dressage proxy: pid=$(cat "${PROXY_PID_FILE}") log=${PROXY_LOG_FILE}"
fi

cleanup() {
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    PROXY_PID="$(cat "${PROXY_PID_FILE}")"
    kill "${PROXY_PID}" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
}
trap cleanup EXIT

for i in $(seq 1 60); do
  if curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1; then
    echo "Dressage proxy is healthy"
    break
  fi
  if [[ "${i}" -eq 60 ]]; then
    echo "Dressage proxy failed health check; see ${PROXY_LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

# --- Start Ray and submit training job ---
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}"
cd "${SLIME_ROOT}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [ -n "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus ${RAY_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:${REPO_ROOT}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "DRESSAGE_PROXY_URL": "${DRESSAGE_PROXY_URL}",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR": "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR": "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}",
    "DRESSAGE_REWARD_MODULES": "dressage.recipes.alfworld.reward",
    "ALFWORLD_MAX_STEPS": "${ALFWORLD_MAX_STEPS}",
    "ALFWORLD_MAX_EPISODE_STEPS": "${ALFWORLD_MAX_EPISODE_STEPS}",
    "ALFWORLD_HISTORY_WINDOW": "${ALFWORLD_HISTORY_WINDOW}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m train \
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
