#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

require_command docker
LOCAL_IMAGE="${GPU_RUNTIME_IMAGE:-dressage-distroll-gpu:official-20260827b}"
: "${GPU_RUNTIME_PUSH_REF:?set GPU_RUNTIME_PUSH_REF to an OCI registry tag}"
[[ "${GPU_RUNTIME_PUSH_REF}" == *:* && "${GPU_RUNTIME_PUSH_REF}" != *@* ]] || {
  echo "GPU_RUNTIME_PUSH_REF must be a mutable publication tag, not a digest" >&2
  exit 1
}

docker image inspect "${LOCAL_IMAGE}" >/dev/null
docker tag "${LOCAL_IMAGE}" "${GPU_RUNTIME_PUSH_REF}"
push_log="$(mktemp)"
trap 'rm -f "${push_log}"' EXIT
docker push "${GPU_RUNTIME_PUSH_REF}" | tee "${push_log}"
digest="$(sed -n 's/^.*digest: \(sha256:[0-9a-f]\{64\}\).*$/\1/p' "${push_log}" | tail -1)"
[[ -n "${digest}" ]] || {
  echo "registry push did not return an immutable digest" >&2
  exit 1
}
repository="${GPU_RUNTIME_PUSH_REF%:*}"
immutable_ref="${repository}@${digest}"
docker buildx imagetools inspect "${immutable_ref}" >/dev/null

echo "published immutable GPU runtime: ${immutable_ref}"
echo "pre-pull this exact reference on every GPU node before starting Ray"
