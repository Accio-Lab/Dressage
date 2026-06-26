#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# unset proxy to avoid distributed startup issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${REPO_ROOT}/slime}"
BASE_FOLDER="${BASE_FOLDER:-/root}"

if [[ ! -f "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh" ]]; then
  echo "Cannot find slime model config: ${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh" >&2
  echo "Set REPO_ROOT or SLIME_ROOT to match the current checkout layout." >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"

if [ -z "${BASE_FOLDER:-}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-}
if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-4}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
HOSTFILE=${HOSTFILE:-}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"
source "${SCRIPT_DIR}/default/dressage_env_defaults.sh"

dressage_apply_common_defaults "qwen3.5-35B-A3B-sync-remote" "blackbox" "e2b"
DRESSAGE_SANDBOX_DEFAULT_IMAGE=${DRESSAGE_SANDBOX_DEFAULT_IMAGE:-dressage-blackbox}
: "${DRESSAGE_SANDBOX_DEFAULT_IMAGE:?DRESSAGE_SANDBOX_DEFAULT_IMAGE is required}"
: "${DRESSAGE_E2B_API_KEY:?DRESSAGE_E2B_API_KEY is required}"

dressage_validate_proxy_defaults
dressage_clear_trajectory_logs

if [[ "${TRAJECTORY_BUILD_MODE}" != "last_step" && "${TRAJECTORY_BUILD_MODE}" != "concat" ]]; then
  echo "TRAJECTORY_BUILD_MODE must be last_step or concat, got: ${TRAJECTORY_BUILD_MODE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
dressage_export_common_env
dressage_compute_context_window 8192 "${CP_SIZE}"
export DRESSAGE_SANDBOX_DEFAULT_IMAGE DRESSAGE_E2B_API_KEY

COMM_ARGS=(
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
   # --use-rollout-routing-replay
)

PROXY_ARGS=(
   --tokenizer-path "${BASE_FOLDER}/Qwen3.5-35B-A3B"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --trajectory-build-mode "${TRAJECTORY_BUILD_MODE}"
   --trajectory-build-model "${TRAJECTORY_BUILD_MODEL}"
   "${COMM_ARGS[@]}"
   --context-window "${CONTEXT_WINDOW}"
)

CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/Qwen3.5-35B-A3B"
   --ref-load "${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist"
   --load "${BASE_FOLDER}/Qwen3.5-35B-A3B_slime/"
   --save "${BASE_FOLDER}/Qwen3.5-35B-A3B_slime/"
   --save-interval 20
   --no-save-optim
   --no-load-optim
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync
   --custom-generate-function-path dressage.rollout.generate.blackbox_dispatch.generate
   --custom-rm-path dressage.reward.custom_rm.custom_rm
   --data-source-path dressage.rollout.data_source.DressageDataSource
   --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
   --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
   --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data

   --prompt-data "${PROMPT_DATA:-${REPO_ROOT}/examples/data/dressage_dapo_prompts.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 128
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
   --balance-data
   --rollout-global-dataset
)

EVAL_ARGS=(
   # Sync blackbox rollout does not support evaluation yet (parity with async).
   # --eval-interval 20
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

   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-A3B-dressage
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
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
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

#    --moe-token-dispatcher-type flex
#    --moe-enable-deepep
)

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
    "DRESSAGE_PADDOCK_MODE": "${DRESSAGE_PADDOCK_MODE}",
    "DRESSAGE_SANDBOX_PROVIDER": "${DRESSAGE_SANDBOX_PROVIDER}",
    "DRESSAGE_BLACKBOX_MAX_STEPS": "${DRESSAGE_BLACKBOX_MAX_STEPS}",
    "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD": "${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD}",
    "DRESSAGE_SANDBOX_DEFAULT_IMAGE": "${DRESSAGE_SANDBOX_DEFAULT_IMAGE}",
    "DRESSAGE_E2B_API_KEY": "${DRESSAGE_E2B_API_KEY}",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR": "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR": "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}",
    "DRESSAGE_REWARD_MODULES": "${DRESSAGE_REWARD_MODULES:-}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
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
