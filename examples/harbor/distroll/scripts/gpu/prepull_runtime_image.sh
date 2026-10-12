#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_command docker
: "${GPU_RUNTIME_IMAGE:?set GPU_RUNTIME_IMAGE to the derived image RepoDigest}"
[[ "${GPU_RUNTIME_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "GPU_RUNTIME_IMAGE must be an immutable RepoDigest" >&2
  exit 1
}

docker pull "${GPU_RUNTIME_IMAGE}"
docker run --rm \
  --network none \
  --entrypoint /opt/dressage-distroll/bin/python \
  "${GPU_RUNTIME_IMAGE}" \
  /opt/dressage-distroll/tools/check_gpu_image_environment.py \
  --lock /opt/dressage-distroll/configs/gpu-runtime-lock.json >/dev/null
echo "pre-pulled and admitted ${GPU_RUNTIME_IMAGE}"
