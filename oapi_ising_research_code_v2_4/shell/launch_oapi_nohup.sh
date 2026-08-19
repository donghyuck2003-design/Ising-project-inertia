#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs
RESULTS_TZ="${RESULTS_TZ:-Asia/Seoul}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ="${RESULTS_TZ}" date +%Y%m%d_%H%M%S)}"
export RUN_TIMESTAMP RESULTS_TZ
LOG="${LOG:-logs/oapi_${RUN_TIMESTAMP}.log}"
PIDFILE="${PIDFILE:-logs/oapi_${RUN_TIMESTAMP}.pid}"

if [[ -n "${OUT:-}" ]]; then
  EXPECTED_OUT="${OUT}"
elif [[ -n "${REUSE_FOLDER:-}" ]]; then
  if [[ "${REUSE_FOLDER}" = /* ]]; then EXPECTED_OUT="${REUSE_FOLDER}"; else EXPECTED_OUT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}/${REUSE_FOLDER}"; fi
else
  EXPECTED_OUT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}/${RUN_TIMESTAMP}"
fi

nohup bash shell/run_oapi_paper.sh "$@" > "${LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${PIDFILE}"
echo "Started OAPI in background."
echo "PID=${PID}"
echo "RESULT=${EXPECTED_OUT}"
echo "LOG=${LOG}"
echo "PIDFILE=${PIDFILE}"
echo "Monitor: tail -f ${LOG}"
