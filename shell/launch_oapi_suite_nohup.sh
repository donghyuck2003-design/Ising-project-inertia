#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p logs

RESULTS_TZ="${RESULTS_TZ:-Asia/Seoul}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ="${RESULTS_TZ}" date +%Y%m%d_%H%M%S)}"
export RUN_TIMESTAMP RESULTS_TZ
LOG="${LOG:-logs/oapi_${RUN_TIMESTAMP}.log}"

if [[ -n "${SUITE_ROOT:-}" ]]; then
  EXPECTED_OUT="${SUITE_ROOT}"
elif [[ -n "${REUSE_FOLDER:-}" ]]; then
  if [[ "${REUSE_FOLDER}" = /* ]]; then
    EXPECTED_OUT="${REUSE_FOLDER}"
  else
    EXPECTED_OUT="${RESULTS_ROOT:-${ROOT}/results}/${REUSE_FOLDER}"
  fi
else
  EXPECTED_OUT="${RESULTS_ROOT:-${ROOT}/results}/${RUN_TIMESTAMP}"
fi

nohup bash shell/run_oapi_benchmark_suite.sh > "${LOG}" 2>&1 &
PID=$!
echo "Started PID=${PID}"
echo "Result folder=${EXPECTED_OUT}"
echo "Log=${ROOT}/${LOG}"
echo "Follow: tail -f ${LOG}"
