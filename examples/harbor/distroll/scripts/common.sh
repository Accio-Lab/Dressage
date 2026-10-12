#!/usr/bin/env bash
set -euo pipefail
DISTROLL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DISTROLL_ROOT}}"
export DISTROLL_ROOT PROJECT_ROOT
require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command is missing: $1" >&2
    exit 1
  }
}
