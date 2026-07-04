#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
source "${SCRIPT_DIR}/image_tag.sh"

build_args=(
  "$@"
  --build-arg "DRESSAGE_BASE_PLATFORM=${DOCKER_PLATFORM}"
  --platform "${DOCKER_PLATFORM}"
  -f "${SCRIPT_DIR}/Dockerfile"
  -t "${DRESSAGE_IMAGE_NAME}"
)

if [[ "${TAG_LATEST:-0}" == "1" ]]; then
  build_args+=(-t "${DRESSAGE_LATEST_IMAGE_NAME}")
fi

if [[ "${PUSH:-0}" == "1" ]]; then
  docker buildx build --push "${build_args[@]}" "${REPO_ROOT}"
else
  docker build "${build_args[@]}" "${REPO_ROOT}"
fi
