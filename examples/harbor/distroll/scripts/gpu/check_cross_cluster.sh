#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
source "${SCRIPT_DIR}/runtime.sh"
set -a
source "${GPU_ENV_FILE:-${PROJECT_ROOT}/configs/gpu.training.env.example}"
set +a

control_curl_args=(--fail --silent --show-error --noproxy '*' --connect-timeout 15 --max-time 30)
curl "${control_curl_args[@]}" "${ROLLOUT_CONTROL_URL}/healthz" >/dev/null
curl "${control_curl_args[@]}" "${ROLLOUT_CONTROL_URL}/v4/capabilities" |
  "${GPU_PYTHON}" -c 'import json,sys; h=json.load(sys.stdin); assert h["contract_version"] == "agentic-rollout/v4" and "fixed-execution-deadline-v1" in h["capabilities"]'

curl -fsS --noproxy '*' --connect-timeout 15 --max-time 30 \
  "${POLICY_SERVICE_URL}/healthz" |
  "${GPU_PYTHON}" -c 'import json,sys; h=json.load(sys.stdin); assert h["policy_instance_id"] and h["contract_version"] == "agentic-rollout/v4"'

echo "GPU Policy and CPU Control connectivity checks passed"
