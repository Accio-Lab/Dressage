#!/usr/bin/env bash
set -euo pipefail

GPU_RUNTIME_BIN="${GPU_RUNTIME_BIN:-/opt/dressage-distroll/bin}"
GPU_PYTHON="${GPU_PYTHON:-${GPU_RUNTIME_BIN}/python}"
GPU_RAY="${GPU_RAY:-${GPU_RUNTIME_BIN}/ray}"
GPU_TORCHRUN="${GPU_TORCHRUN:-${GPU_RUNTIME_BIN}/torchrun}"
GPU_HF="${GPU_HF:-${GPU_RUNTIME_BIN}/hf}"
GPU_RUNTIME_LOCK="${GPU_RUNTIME_LOCK:-${PROJECT_ROOT}/configs/gpu-runtime-lock.json}"

GPU_SOURCE_ROOT="${GPU_SOURCE_ROOT:-${PROJECT_ROOT}}"
DRESSAGE_ROOT="${DRESSAGE_ROOT:-$(cd -- "${DISTROLL_ROOT}/../../.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-${GPU_SOURCE_ROOT}/runtime-src/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${GPU_SOURCE_ROOT}/runtime-src/Megatron-LM}"
expected_pythonpath="${GPU_SOURCE_ROOT}/contract/src:${GPU_SOURCE_ROOT}/gpu_adapter/src:${DRESSAGE_ROOT}:${SLIME_ROOT}:${MEGATRON_ROOT}"
base_pythonpath="${expected_pythonpath}"
if [[ -n "${TQ_SOURCE_ROOT:-}" ]]; then
  [[ -f "${TQ_SOURCE_ROOT}/linger-overlay.json" ]] || {
    echo "TQ overlay manifest missing; prepare the isolated dependency first" >&2
    exit 1
  }
  "${GPU_PYTHON}" "${GPU_SOURCE_ROOT}/tools/prepare_tq_linger_overlay.py" \
    --target "${TQ_SOURCE_ROOT}" --check
  expected_pythonpath="${TQ_SOURCE_ROOT}:${expected_pythonpath}"
fi

for path in \
  "${GPU_SOURCE_ROOT}/contract/src/agentic_rollout_contract" \
  "${GPU_SOURCE_ROOT}/gpu_adapter/src/dressage_distroll_gpu_adapter" \
  "${DRESSAGE_ROOT}/dressage" "${SLIME_ROOT}/slime" \
  "${MEGATRON_ROOT}/megatron"; do
  [[ -d "${path}" ]] || {
    echo "missing required host-mounted GPU source: ${path}" >&2
    exit 1
  }
done

if [[ -n "${PYTHONPATH:-}" && "${PYTHONPATH}" != "${expected_pythonpath}" && \
      "${PYTHONPATH}" != "${base_pythonpath}" ]]; then
  echo "unexpected GPU PYTHONPATH: ${PYTHONPATH}" >&2
  exit 1
fi
export GPU_SOURCE_ROOT DRESSAGE_ROOT SLIME_ROOT MEGATRON_ROOT
export PYTHONPATH="${expected_pythonpath}"

validate_gpu_sequence_budget() {
  local context_len="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
  local max_tokens_per_gpu="${MAX_TOKENS_PER_GPU:-32768}"
  local context_parallel_size="${CONTEXT_PARALLEL_SIZE:-1}"
  local name value
  for name in context_len max_tokens_per_gpu context_parallel_size; do
    value="${!name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
      echo "${name} must be a positive integer; got ${value}" >&2
      return 1
    }
  done
  local sample_token_cap=$((max_tokens_per_gpu * context_parallel_size))
  if (( context_len > sample_token_cap )); then
    echo "invalid GPU sequence budget: ROLLOUT_MAX_CONTEXT_LEN=${context_len} exceeds" >&2
    echo "MAX_TOKENS_PER_GPU=${max_tokens_per_gpu} * CONTEXT_PARALLEL_SIZE=${context_parallel_size} = ${sample_token_cap}" >&2
    echo "Dressage would truncate the trajectory tail and may remove every trainable response token." >&2
    return 1
  fi
  export ROLLOUT_MAX_CONTEXT_LEN="${context_len}"
  export MAX_TOKENS_PER_GPU="${max_tokens_per_gpu}"
  export CONTEXT_PARALLEL_SIZE="${context_parallel_size}"
}

for executable in "${GPU_PYTHON}" "${GPU_RAY}"; do
  [[ -x "${executable}" ]] || {
    echo "missing immutable GPU runtime executable: ${executable}" >&2
    echo "run inside the admitted dressage-distroll derived image" >&2
    exit 1
  }
done
[[ -f "${GPU_RUNTIME_LOCK}" ]] || {
  echo "missing GPU runtime lock: ${GPU_RUNTIME_LOCK}" >&2
  exit 1
}
[[ -d "${DRESSAGE_ROOT}" && -d "${SLIME_ROOT}" && -d "${MEGATRON_ROOT}" ]] || {
  echo "official Dressage/Slime/Megatron source roots are incomplete" >&2
  exit 1
}
