#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
INVENTORY="${ROLLOUT_INVENTORY:-/etc/dressage-distroll/inventory.yaml}"
WORKER_ID="${ROLLOUT_WORKER_ID:-$(hostname)}"

"${CPU_PYTHON:-${PROJECT_ROOT}/.venv-cpu/bin/python}" - "${INVENTORY}" "${WORKER_ID}" <<'PY'
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

inventory = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
worker = next(item for item in inventory["workers"] if item["worker_id"] == sys.argv[2])
settings = worker.get("image_cache") or {}
root = Path(settings.get("cache_root", "/var/lib/dressage-distroll-worker/image-cache"))
docker_root = Path(settings.get("docker_root", "/var/lib/docker"))
states = Counter()
origins = Counter()
evictable = 0
for path in (root / "records").glob("*.json"):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        states["INVALID"] += 1
        continue
    states[str(record.get("state") or "UNKNOWN")] += 1
    origins[str(record.get("managed_origin") or "UNKNOWN")] += 1
    evictable += int(record.get("evictable") is True)
usage = shutil.disk_usage(docker_root)
print(json.dumps({
    "worker_id": sys.argv[2],
    "enabled": bool(settings.get("enabled")),
    "cache_root": str(root),
    "docker_root": str(docker_root),
    "docker_total_bytes": usage.total,
    "docker_used_bytes": usage.used,
    "docker_free_bytes": usage.free,
    "docker_used_percent": round(usage.used / usage.total * 100, 3),
    "record_states": states,
    "managed_origins": origins,
    "evictable_records": evictable,
}, indent=2, default=dict))
PY
