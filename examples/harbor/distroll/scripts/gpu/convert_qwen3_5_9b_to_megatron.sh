#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"

HF_DIR=${HF_CHECKPOINT:?set HF_CHECKPOINT to the read-only Qwen3.5-9B HF directory}
DIST_DIR=${REF_LOAD:-${PROJECT_ROOT}/outputs/models/Qwen3.5-9B_torch_dist}
MODEL_CONFIG=${MODEL_CONFIG:-${SLIME_ROOT}/scripts/models/qwen3.5-9B.sh}
CONVERTER=${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py
CONVERSION_GPUS=${CONVERSION_GPUS:-8}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/outputs/logs/model-conversion/Qwen3.5-9B}

[[ -f "${HF_DIR}/config.json" ]] || { echo "missing HF config: ${HF_DIR}/config.json" >&2; exit 1; }
[[ -f "${MODEL_CONFIG}" && -f "${CONVERTER}" ]] || { echo "locked Slime lacks Qwen3.5-9B conversion inputs" >&2; exit 1; }
[[ ! -e "${DIST_DIR}" ]] || { echo "refusing to overwrite Megatron checkpoint: ${DIST_DIR}" >&2; exit 1; }
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "${CONVERSION_GPUS}" =~ ^[1-9][0-9]*$ && "${CONVERSION_GPUS}" -le "${GPU_COUNT}" ]] || { echo "invalid CONVERSION_GPUS=${CONVERSION_GPUS}" >&2; exit 1; }

"${GPU_PYTHON}" - "${HF_DIR}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
text = config.get("text_config") or config
expected = {"model_type":"qwen3_5", "hidden_size":4096, "intermediate_size":12288,
            "num_hidden_layers":32, "num_attention_heads":16, "num_key_value_heads":4,
            "vocab_size":248320}
for key, value in expected.items():
    if text.get(key) != value and config.get(key) != value:
        raise SystemExit(f"unexpected Qwen3.5-9B config field {key}: {text.get(key)!r}")
print("Qwen3.5-9B config matches the locked recipe")
PY

source "${MODEL_CONFIG}"
mkdir -p "$(dirname -- "${DIST_DIR}")" "${LOG_DIR}"
"${GPU_TORCHRUN}" --nproc-per-node "${CONVERSION_GPUS}" \
  "${CONVERTER}" --hf-checkpoint "${HF_DIR}" --save "${DIST_DIR}" \
  "${MODEL_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/Qwen3.5-9B-to-torch-dist.log"

[[ -f "${DIST_DIR}/latest_checkpointed_iteration.txt" ]]
grep -Fx release "${DIST_DIR}/latest_checkpointed_iteration.txt"
[[ -d "${DIST_DIR}/release" ]]
echo "Megatron distributed checkpoint ready: ${DIST_DIR}"
