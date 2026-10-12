#!/usr/bin/env bash
set -u -o pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 RAY_BIN SUBMISSION_ID LOG_PATH FINAL_STATUS_PATH" >&2
  exit 2
fi

RAY_BIN="$1"
SUBMISSION_ID="$2"
LOG_PATH="$3"
FINAL_STATUS_PATH="$4"

mkdir -p "$(dirname -- "${LOG_PATH}")" "$(dirname -- "${FINAL_STATUS_PATH}")"

found=0
for _ in $(seq 1 30); do
  if "${RAY_BIN}" job status "${SUBMISSION_ID}" >/dev/null 2>&1; then
    found=1
    break
  fi
  sleep 1
done

if [[ "${found}" -ne 1 ]]; then
  printf 'Ray job %s was not visible within 30 seconds\n' "${SUBMISSION_ID}" \
    >>"${LOG_PATH}"
fi

"${RAY_BIN}" job logs --follow "${SUBMISSION_ID}" >>"${LOG_PATH}" 2>&1
logs_rc=$?
"${RAY_BIN}" job status "${SUBMISSION_ID}" >"${FINAL_STATUS_PATH}" 2>&1
status_rc=$?
{
  printf '\nlog_follower_finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'ray_job_logs_exit_code=%s\n' "${logs_rc}"
  printf 'ray_job_status_exit_code=%s\n' "${status_rc}"
} >>"${LOG_PATH}"

# This helper records the target job's terminal state but is not itself a
# restartable service. The final status file is the source of truth.
exit 0
