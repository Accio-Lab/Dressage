#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
[[ "$(id -u)" -eq 0 ]] || {
  echo "run as root" >&2
  exit 1
}
command -v docker >/dev/null 2>&1 || {
  echo "Docker Engine with compose/buildx plugins is required" >&2
  exit 1
}
docker compose version >/dev/null
docker buildx version >/dev/null
if [[ ! -r "${PROJECT_ROOT}/configs/cluster.inventory.yaml" ]]; then
  echo "cannot read ${PROJECT_ROOT}/configs/cluster.inventory.yaml" >&2
  exit 1
fi

require_command uv
require_command python3

uv python install 3.12
if [[ -x "${PROJECT_ROOT}/.venv-cpu/bin/python" ]]; then
  echo "reusing existing CPU environment: ${PROJECT_ROOT}/.venv-cpu"
else
  uv venv "${PROJECT_ROOT}/.venv-cpu" --python 3.12
fi
uv pip install --python "${PROJECT_ROOT}/.venv-cpu/bin/python" \
  -e "${PROJECT_ROOT}/contract" \
  -e "${PROJECT_ROOT}/rollout_service"

mkdir -p /etc/dressage-distroll /var/lib/dressage-distroll-worker/image-cache \
    "${PROJECT_ROOT}/tasks"
python3 - "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = {
    "cluster.inventory.yaml": "inventory.yaml",
    "control.env.example": "control.env",
    "worker.env.example": "worker.env",
    "network-gateway.yaml": "network-gateway.yaml",
    "rollout-dashboard.yaml.example": "dashboard.yaml",
}
for source, destination in names.items():
    target = Path("/etc/dressage-distroll") / destination
    if not target.exists():
        text = (root / "configs" / source).read_text()
        with target.open("x") as output:
            output.write(text.replace("CPU_PROJECT_ROOT_PLACEHOLDER", str(root)))
        target.chmod(0o600)
PY
cp "${PROJECT_ROOT}/deploy/systemd/dressage-distroll-rollout.slice" /etc/systemd/system/dressage-distroll-rollout.slice
for unit in dressage-distroll-control.service dressage-distroll-worker.service \
  dressage-distroll-network-gateway.service \
  dressage-distroll-image-cleaner.service \
  dressage-distroll-node-observer.service \
  dressage-distroll-dashboard.service; do
  sed "s|CPU_PROJECT_ROOT_PLACEHOLDER|${PROJECT_ROOT}|g" \
    "${PROJECT_ROOT}/deploy/systemd/${unit}" \
    >"/etc/systemd/system/${unit}"
done
systemctl daemon-reload

echo "CPU rollout packages and systemd units installed"
echo "Fill in /etc/dressage-distroll/*.env and inventory.yaml before starting services."
