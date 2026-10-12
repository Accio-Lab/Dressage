#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../common.sh"

for command in git python3 tar patch; do require_command "$command"; done

# Read from the image's Git repositories; never patch those checkouts.
SLIME_SOURCE_ROOT="${SLIME_SOURCE_ROOT:-/root/slime}"
MEGATRON_SOURCE_ROOT="${MEGATRON_SOURCE_ROOT:-/root/Megatron-LM}"
LOCK="${GPU_RUNTIME_LOCK:-${PROJECT_ROOT}/configs/gpu-runtime-lock.json}"
mapfile -t pins < <(python3 - "$LOCK" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for name in ("slime", "megatron"):
    print(lock["sources"][name]["revision"])
print(lock["sources"]["megatron"]["patch_sha256"])
PY
)
[[ ${#pins[@]} -eq 3 ]] || { echo "invalid runtime lock" >&2; exit 1; }

mkdir -p "${PROJECT_ROOT}/runtime-src"
STAGE=$(mktemp -d "${PROJECT_ROOT}/runtime-src/.prepare.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT

verify_source() {
  python3 - "$LOCK" "$1" "$2" <<'PY'
import hashlib, json, sys
from pathlib import Path
lock, name, root = json.load(open(sys.argv[1])), sys.argv[2], Path(sys.argv[3])
pin = lock["sources"][name]
if (root / ".dressage-source-revision").read_text().strip() != pin["revision"]:
    raise SystemExit(f"{name}: source revision mismatch")
if "patch_sha256" in pin:
    if (root / ".dressage-patch-sha256").read_text().strip() != pin["patch_sha256"]:
        raise SystemExit(f"{name}: patch marker mismatch")
for relative, expected in lock["source_hashes"][name].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{name}: source hash mismatch: {relative}")
PY
}

slime_target="${PROJECT_ROOT}/runtime-src/slime"
if [[ ! -e "$slime_target" ]]; then
  mkdir "$STAGE/slime"
  git -c "safe.directory=${SLIME_SOURCE_ROOT}" -C "$SLIME_SOURCE_ROOT" archive "${pins[0]}" |
    tar -C "$STAGE/slime" -xf -
  printf '%s\n' "${pins[0]}" >"$STAGE/slime/.dressage-source-revision"
  verify_source slime "$STAGE/slime"
  mv -T "$STAGE/slime" "$slime_target"
fi
verify_source slime "$slime_target"

megatron_target="${PROJECT_ROOT}/runtime-src/Megatron-LM"
if [[ ! -e "$megatron_target" ]]; then
  mkdir "$STAGE/Megatron-LM"
  git -c "safe.directory=${MEGATRON_SOURCE_ROOT}" -C "$MEGATRON_SOURCE_ROOT" archive "${pins[1]}" |
    tar -C "$STAGE/Megatron-LM" -xf -
  reviewed_patch="$slime_target/docker/patch/latest/megatron.patch"
  echo "${pins[2]}  $reviewed_patch" | sha256sum -c -
  patch -d "$STAGE/Megatron-LM" -p1 --batch --forward <"$reviewed_patch"
  printf '%s\n' "${pins[1]}" >"$STAGE/Megatron-LM/.dressage-source-revision"
  printf '%s\n' "${pins[2]}" >"$STAGE/Megatron-LM/.dressage-patch-sha256"
  verify_source megatron "$STAGE/Megatron-LM"
  mv -T "$STAGE/Megatron-LM" "$megatron_target"
fi
verify_source megatron "$megatron_target"
echo "Locked Slime and patched Megatron sources ready under ${PROJECT_ROOT}/runtime-src"
