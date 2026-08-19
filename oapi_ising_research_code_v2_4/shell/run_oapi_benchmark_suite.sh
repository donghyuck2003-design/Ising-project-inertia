#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV_ACTIVATE="${VENV_ACTIVATE:-${HOME}/openai-env/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
N="${N:-128}"
EXACT_N="${EXACT_N:-64}"
P="${P:-0.30}"
BATCH="${BATCH:-25}"
TEST_RUNS_PER_INSTANCE="${TEST_RUNS_PER_INSTANCE:-100}"
REFERENCE_RUNS_PER_METHOD="${REFERENCE_RUNS_PER_METHOD:-256}"
TUNE_TRIALS="${TUNE_TRIALS:-40}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
RECIPIENT="${RECIPIENT:-donghyuck200@naver.com}"
SEND_MAIL_SCRIPT="${SEND_MAIL_SCRIPT:-/home/onion120/mail/send_mail.sh}"

# Result-directory policy (v2.4):
#   1) SUITE_ROOT=/absolute/or/relative/path  -> use exactly that path (backward compatible)
#   2) REUSE_FOLDER=<name-or-path>            -> reuse that folder; relative names live under RESULTS_ROOT
#   3) neither supplied                       -> create RESULTS_ROOT/YYYYMMDD_HHMMSS
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
RESULTS_TZ="${RESULTS_TZ:-Asia/Seoul}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ="${RESULTS_TZ}" date +%Y%m%d_%H%M%S)}"
REUSE_FOLDER="${REUSE_FOLDER:-}"

if [[ -n "${SUITE_ROOT:-}" ]]; then
  SUITE_ROOT="${SUITE_ROOT}"
  RUN_FOLDER_MODE="explicit-suite-root"
elif [[ -n "${REUSE_FOLDER}" ]]; then
  if [[ "${REUSE_FOLDER}" = /* ]]; then
    SUITE_ROOT="${REUSE_FOLDER}"
  else
    SUITE_ROOT="${RESULTS_ROOT}/${REUSE_FOLDER}"
  fi
  RUN_FOLDER_MODE="reuse-folder"
else
  SUITE_ROOT="${RESULTS_ROOT}/${RUN_TIMESTAMP}"
  RUN_FOLDER_MODE="new-timestamp-folder"
fi

RUN_PLANTED="${RUN_PLANTED:-1}"
RUN_ER_BEST_KNOWN="${RUN_ER_BEST_KNOWN:-1}"
RUN_SK_BEST_KNOWN="${RUN_SK_BEST_KNOWN:-1}"
RUN_ER_MILP_EXACT="${RUN_ER_MILP_EXACT:-0}"
RUN_SK_MILP_EXACT="${RUN_SK_MILP_EXACT:-0}"
EXACT_BACKEND="${EXACT_BACKEND:-auto}"
EXACT_TIME_LIMIT_S="${EXACT_TIME_LIMIT_S:-0}"
EXACT_THREADS="${EXACT_THREADS:-0}"

mkdir -p "${SUITE_ROOT}" "${ROOT}/logs"

# Append-only provenance so a reused directory records every resume invocation.
{
  echo "StartedAt=$(TZ="${RESULTS_TZ}" date --iso-8601=seconds)"
  echo "RunFolderMode=${RUN_FOLDER_MODE}"
  echo "SuiteRoot=${SUITE_ROOT}"
  echo "RunTimestamp=${RUN_TIMESTAMP}"
  echo "N=${N} ExactN=${EXACT_N} p=${P} Batch=${BATCH}"
  echo "ReferenceRunsPerMethod=${REFERENCE_RUNS_PER_METHOD} TestRunsPerInstance=${TEST_RUNS_PER_INSTANCE}"
  echo "ReuseExisting=${REUSE_EXISTING}"
  echo "---"
} >> "${SUITE_ROOT}/run_history.log"

printf 'Result folder: %s\n' "${SUITE_ROOT}"
printf 'Folder mode:   %s\n' "${RUN_FOLDER_MODE}"

notify () {
  local code="$?" status="SUCCESS"
  trap - EXIT
  if (( code != 0 )); then status="FAILED"; fi
  if [[ -x "${SEND_MAIL_SCRIPT}" ]]; then
    "${SEND_MAIL_SCRIPT}" \
      --to "${RECIPIENT}" \
      --subject "[${status}] OAPI v2.3 resume benchmark suite finished" \
      --body "Status=${status}
ExitCode=${code}
N=${N}, ExactN=${EXACT_N}, p=${P}
Batch=${BATCH}
ReferenceRunsPerMethod=${REFERENCE_RUNS_PER_METHOD}
TestRunsPerInstance=${TEST_RUNS_PER_INSTANCE}
ReuseExisting=${REUSE_EXISTING}
Planted=${RUN_PLANTED}, ER-best-known=${RUN_ER_BEST_KNOWN}, SK-best-known=${RUN_SK_BEST_KNOWN}
ER-exact=${RUN_ER_MILP_EXACT}, SK-exact=${RUN_SK_MILP_EXACT}
ExactBackend=${EXACT_BACKEND}
GPU=${CUDA_VISIBLE_DEVICES}
Output=${SUITE_ROOT}
FolderMode=${RUN_FOLDER_MODE}
FinishedAt=$(TZ="${RESULTS_TZ}" date --iso-8601=seconds)" \
      --script-info "shell/run_oapi_benchmark_suite.sh" || true
  fi
  exit "${code}"
}
trap notify EXIT

run_case () {
  local problem="$1" n="$2" target="$3" out="$4"
  echo "===== ${problem} n=${n} target=${target} batch=${BATCH} test-runs=${TEST_RUNS_PER_INSTANCE} reuse=${REUSE_EXISTING} ====="
  PROBLEM="${problem}" N="${n}" P="${P}" TARGET_MODE="${target}" \
  BATCH="${BATCH}" TEST_RUNS_PER_INSTANCE="${TEST_RUNS_PER_INSTANCE}" \
  REFERENCE_RUNS_PER_METHOD="${REFERENCE_RUNS_PER_METHOD}" \
  TUNE_TRIALS="${TUNE_TRIALS}" BOOTSTRAP="${BOOTSTRAP}" REUSE_EXISTING="${REUSE_EXISTING}" \
  EXACT_BACKEND="${EXACT_BACKEND}" EXACT_TIME_LIMIT_S="${EXACT_TIME_LIMIT_S}" EXACT_THREADS="${EXACT_THREADS}" \
  RECIPIENT="${RECIPIENT}" SEND_MAIL_SCRIPT="${SEND_MAIL_SCRIPT}" SEND_EMAIL=0 \
  OUT="${out}" bash shell/run_oapi_paper.sh
}

if [[ "${RUN_PLANTED}" == "1" ]]; then run_case planted "${N}" auto "${SUITE_ROOT}/01_planted_exact"; fi
if [[ "${RUN_ER_BEST_KNOWN}" == "1" ]]; then run_case er "${N}" best_known "${SUITE_ROOT}/02_er_best_known"; fi
if [[ "${RUN_SK_BEST_KNOWN}" == "1" ]]; then run_case sk "${N}" best_known "${SUITE_ROOT}/03_sk_best_known"; fi
if [[ "${RUN_ER_MILP_EXACT}" == "1" ]]; then run_case er "${EXACT_N}" milp_exact "${SUITE_ROOT}/04_er_milp_exact_n${EXACT_N}"; fi
if [[ "${RUN_SK_MILP_EXACT}" == "1" ]]; then run_case sk "${EXACT_N}" milp_exact "${SUITE_ROOT}/05_sk_milp_exact_n${EXACT_N}"; fi

python scripts/summarize_benchmark_suite.py --suite-root "${SUITE_ROOT}"
