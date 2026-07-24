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

if [[ ! -f "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" ]]; then
  echo "Cannot find slime model config: ${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh" >&2
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
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
RAY_NUM_GPUS_PER_NODE=${RAY_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-2}
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
source "${SCRIPT_DIR}/default/dressage_env_defaults.sh"

dressage_apply_common_defaults "qwen3.5-4B-partial-rollout-async-local" "blackbox" "local_bwrap"
dressage_apply_local_bwrap_defaults 16

if [[ "${DRESSAGE_BLACKBOX_RUNNER_MODE}" == "bwrap" || "${DRESSAGE_BLACKBOX_RUNNER_MODE}" == "bubblewrap" ]]; then
  command -v "${DRESSAGE_BLACKBOX_BWRAP_BIN}" >/dev/null || {
    echo "missing bubblewrap binary: ${DRESSAGE_BLACKBOX_BWRAP_BIN}" >&2
    exit 1
  }
fi

WORKER_COUNT=0
if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
  WORKER_COUNT=$(awk -v master="${MASTER_ADDR}" '$1 != master { count += 1 } END { print count + 0 }' "${HOSTFILE}")
fi
dressage_compute_local_bwrap_resources "${WORKER_COUNT}"
dressage_validate_proxy_defaults
dressage_clear_trajectory_logs

if [[ "${TOKEN_BUILD_MODE}" != "snapshot" && "${TOKEN_BUILD_MODE}" != "tito" ]]; then
  echo "TOKEN_BUILD_MODE must be snapshot or tito, got: ${TOKEN_BUILD_MODE}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
dressage_export_common_env
dressage_export_local_bwrap_env
dressage_compute_context_window 24576 "${CP_SIZE}"

COMM_ARGS=(
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
)

PROXY_ARGS=(
   --tokenizer-path "${BASE_FOLDER}/Qwen3.5-4B"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --token-build-mode "${TOKEN_BUILD_MODE}"
   --token-build-model "${TOKEN_BUILD_MODEL}"
   "${COMM_ARGS[@]}"
   --dressage-partial-rollout
   --context-window "${CONTEXT_WINDOW}"
   --record-token-versions
   --mask-nonlast-version-tokens
)

CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/Qwen3.5-4B"
   --ref-load "${BASE_FOLDER}/Qwen3.5-4B_torch_dist/"
   --load "${BASE_FOLDER}/Qwen3.5-4B_slime/"
   --save "${BASE_FOLDER}/Qwen3.5-4B_slime/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.partial_async_rollout.generate_rollout_partial_async
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
   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --balance-data
   --rollout-global-dataset
)

EVAL_ARGS=(
   # Partial async blackbox rollout does not support evaluation yet.
   # --eval-interval 20
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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
   # --wandb-group qwen3.5-4B-dressage
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
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

_stop_local_bwrap_pool_on_exit() {
  if [[ "${DRESSAGE_LOCAL_BWRAP_CLEANUP_ON_EXIT:-1}" == "1" && "${DRESSAGE_SANDBOX_PROVIDER}" == "local_bwrap" ]]; then
    python -m dressage.sandbox.scripts.stop_local_bwrap || true
  fi
}

_stop_ray_cluster_on_exit() {
  if [[ "${DRESSAGE_RAY_STOP_ON_EXIT:-1}" != "1" ]]; then
    return
  fi
  if [[ -n "${HOSTFILE:-}" && -f "${HOSTFILE}" ]]; then
    for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
      if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
        continue
      fi
      ssh root@"${WORKER_IP}" "ray stop --force || true" &
    done
    wait || true
  fi
  ray stop --force || true
}

_stop_proxy_on_exit() {
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    PROXY_PID="$(cat "${PROXY_PID_FILE}")"
    kill "${PROXY_PID}" 2>/dev/null || true
    rm -f "${PROXY_PID_FILE}"
  fi
}

cleanup() {
  status=$?
  set +e
  _stop_local_bwrap_pool_on_exit
  _stop_ray_cluster_on_exit
  _stop_proxy_on_exit
  exit "${status}"
}
trap cleanup EXIT

_stop_local_bwrap_pool_on_exit
_stop_ray_cluster_on_exit

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
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 "${DRESSAGE_BLACKBOX_HEAD_RESOURCE_ARGS[@]}"

if [ -n "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "ray stop --force || true ; ray start --address=${MASTER_ADDR}:6379 --num-gpus ${RAY_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats ${DRESSAGE_BLACKBOX_WORKER_RESOURCE_ARGS}" &
  done
  wait
fi

if [[ "${DRESSAGE_SANDBOX_PROVIDER}" == "local_bwrap" && "${DRESSAGE_LOCAL_BWRAP_AUTO_START}" == "1" ]]; then
  python -m dressage.sandbox.scripts.start_local_bwrap
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
    "DRESSAGE_LOCAL_BWRAP_POOL_MODE": "${DRESSAGE_LOCAL_BWRAP_POOL_MODE}",
    "DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE": "${DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE}",
    "DRESSAGE_LOCAL_BWRAP_MANAGER_NAME": "${DRESSAGE_LOCAL_BWRAP_MANAGER_NAME}",
    "DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS": "${DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS}",
    "DRESSAGE_LOCAL_BWRAP_BASE_PORT": "${DRESSAGE_LOCAL_BWRAP_BASE_PORT}",
    "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC": "${DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC}",
    "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR": "${DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR}",
    "DRESSAGE_HARBOR_ATIF_EXPORT": "${DRESSAGE_HARBOR_ATIF_EXPORT:-0}",
    "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR": "${DRESSAGE_TRAJECTORY_ERROR_LOG_DIR}",
    "DRESSAGE_REWARD_MODULES": "${DRESSAGE_REWARD_MODULES:-}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m dressage.training.train_async_with_rollout_pause \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
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
