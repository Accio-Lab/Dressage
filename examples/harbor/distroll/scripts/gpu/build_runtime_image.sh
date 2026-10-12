#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

LOCK_FILE="${GPU_RUNTIME_LOCK:-${PROJECT_ROOT}/configs/gpu-runtime-lock.json}"
IMAGE="${GPU_RUNTIME_IMAGE:-dressage-distroll-gpu:official-20260827b}"
require_command docker

mapfile -t runtime_lock < <(
  python3 - "${LOCK_FILE}" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
print(lock["official_image"]["reference"])
print(lock["official_image"]["digest"])
print(lock["sources"]["dressage"]["revision"])
print(lock["sources"]["slime"]["revision"])
print(lock["sources"]["megatron"]["revision"])
PY
)
[[ "${#runtime_lock[@]}" -eq 5 ]] || {
  echo "invalid GPU runtime lock: ${LOCK_FILE}" >&2
  exit 1
}
DRESSAGE_BASE_IMAGE="${runtime_lock[0]}"
DRESSAGE_BASE_DIGEST="${runtime_lock[1]}"
DRESSAGE_EXPECTED_REVISION="${runtime_lock[2]}"
SLIME_EXPECTED_REVISION="${runtime_lock[3]}"
MEGATRON_EXPECTED_REVISION="${runtime_lock[4]}"

docker build --network none \
  --pull \
  --build-arg "DRESSAGE_BASE_IMAGE=${DRESSAGE_BASE_IMAGE}" \
  --build-arg "DRESSAGE_BASE_DIGEST=${DRESSAGE_BASE_DIGEST}" \
  --build-arg "DRESSAGE_EXPECTED_REVISION=${DRESSAGE_EXPECTED_REVISION}" \
  --build-arg "SLIME_EXPECTED_REVISION=${SLIME_EXPECTED_REVISION}" \
  --build-arg "MEGATRON_EXPECTED_REVISION=${MEGATRON_EXPECTED_REVISION}" \
  -t "${IMAGE}" \
  -f "${PROJECT_ROOT}/deploy/gpu/Dockerfile" \
  "${PROJECT_ROOT}"

docker run --rm \
  --network none \
  --entrypoint /opt/dressage-distroll/bin/python \
  "${IMAGE}" \
  /opt/dressage-distroll/tools/check_gpu_image_environment.py \
  --lock /opt/dressage-distroll/configs/gpu-runtime-lock.json >/dev/null

image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
echo "built and locally admitted ${IMAGE} (${image_id})"
echo "base: ${DRESSAGE_BASE_IMAGE}"
echo "push the derived image to an OCI registry and deploy its RepoDigest, not this mutable tag"
