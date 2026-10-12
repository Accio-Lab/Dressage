#!/usr/bin/env bash
set -euo pipefail

GPU_CONTAINER_NAME="${GPU_CONTAINER_NAME:-dressage-distroll-gpu}"
[[ "$#" -gt 0 ]] || set -- bash
exec docker exec -it "${GPU_CONTAINER_NAME}" "$@"
