#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

VENV_ACTIVATE="${VENV_ACTIVATE:-${HOME}/openai-env/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RECIPIENT="${RECIPIENT:-donghyuck200@naver.com}"
SEND_MAIL_SCRIPT="${SEND_MAIL_SCRIPT:-/home/onion120/mail/send_mail.sh}"

PROBLEM="${PROBLEM:-planted}"
N="${N:-128}"
P="${P:-0.30}"
TARGET_MODE="${TARGET_MODE:-auto}"
STEPS="${STEPS:-5000}"
BATCH="${BATCH:-25}"
DEVICE="${DEVICE:-cuda}"
# v2.3 defaults: the requested best-known reference uses 256 runs per method.
# Final paper testing stays at the previous 100 runs per method/instance unless
# TEST_RUNS_PER_INSTANCE is explicitly overridden.
TEST_RUNS_PER_INSTANCE="${TEST_RUNS_PER_INSTANCE:-100}"
REFERENCE_RUNS_PER_METHOD="${REFERENCE_RUNS_PER_METHOD:-256}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
TUNE_TRIALS="${TUNE_TRIALS:-40}"
FIXED_RUNS_PER_VALUE="${FIXED_RUNS_PER_VALUE:-64}"
TUNE_RUNS_PER_INSTANCE="${TUNE_RUNS_PER_INSTANCE:-64}"
VALIDATION_RUNS_PER_INSTANCE="${VALIDATION_RUNS_PER_INSTANCE:-100}"
TOP_K_VALIDATION="${TOP_K_VALIDATION:-5}"
EXACT_BACKEND="${EXACT_BACKEND:-auto}"
EXACT_TIME_LIMIT_S="${EXACT_TIME_LIMIT_S:-0}"
EXACT_THREADS="${EXACT_THREADS:-0}"
EXACT_SOLVER_LOG="${EXACT_SOLVER_LOG:-0}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
SEND_EMAIL="${SEND_EMAIL:-1}"
TUNING_SEEDS="${TUNING_SEEDS:-1,2,3,4,5}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-50,51,52,53,54}"
TEST_SEEDS="${TEST_SEEDS:-200,201,202,203,204,205,206,207,208,209}"

# Direct-run result directory policy. The suite normally supplies OUT explicitly.
# When this script is launched by itself:
#   OUT=<path>              -> use exactly that path
#   REUSE_FOLDER=<name/path>-> reuse it (relative names under RESULTS_ROOT)
#   neither                 -> create RESULTS_ROOT/YYYYMMDD_HHMMSS
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"
RESULTS_TZ="${RESULTS_TZ:-Asia/Seoul}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(TZ="${RESULTS_TZ}" date +%Y%m%d_%H%M%S)}"
REUSE_FOLDER="${REUSE_FOLDER:-}"
if [[ -n "${OUT:-}" ]]; then
  OUT="${OUT}"
  OUT_MODE="explicit-out"
elif [[ -n "${REUSE_FOLDER}" ]]; then
  if [[ "${REUSE_FOLDER}" = /* ]]; then OUT="${REUSE_FOLDER}"; else OUT="${RESULTS_ROOT}/${REUSE_FOLDER}"; fi
  OUT_MODE="reuse-folder"
else
  OUT="${RESULTS_ROOT}/${RUN_TIMESTAMP}"
  OUT_MODE="new-timestamp-folder"
fi

mkdir -p "${OUT}" "${SCRIPT_DIR}/logs"
printf 'Workflow output: %s\n' "${OUT}"
printf 'Output mode:     %s\n' "${OUT_MODE}"

send_completion_email () {
  local EXIT_CODE="$?" STATUS="SUCCESS"
  trap - EXIT
  if (( EXIT_CODE != 0 )); then STATUS="FAILED"; fi
  if [[ -x "${SEND_MAIL_SCRIPT}" ]]; then
    "${SEND_MAIL_SCRIPT}" \
      --to "${RECIPIENT}" \
      --subject "[${STATUS}] OAPI v2.3 paper workflow finished" \
      --body "OAPI paper experiment has finished.
Status=${STATUS}
ExitCode=${EXIT_CODE}
Problem=${PROBLEM}, N=${N}, p=${P}
TargetMode=${TARGET_MODE}
Batch=${BATCH}
ReferenceRunsPerMethod=${REFERENCE_RUNS_PER_METHOD}
TestRunsPerInstance=${TEST_RUNS_PER_INSTANCE}
ReuseExisting=${REUSE_EXISTING}
GPU=${CUDA_VISIBLE_DEVICES}
Output=${OUT}
OutputMode=${OUT_MODE}
FinishedAt=$(TZ="${RESULTS_TZ}" date --iso-8601=seconds)" \
      --script-info "shell/run_oapi_paper.sh" || echo "Warning: completion email could not be sent." >&2
  fi
  exit "${EXIT_CODE}"
}
if [[ "${SEND_EMAIL}" == "1" ]]; then trap send_completion_email EXIT; fi

EXTRA_ARGS=()
if [[ "${EXACT_SOLVER_LOG}" == "1" ]]; then EXTRA_ARGS+=(--exact-solver-log); fi
if [[ "${REUSE_EXISTING}" == "1" ]]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi

python scripts/full_paper_workflow.py \
  --problem "${PROBLEM}" \
  --n "${N}" \
  --p "${P}" \
  --target-mode "${TARGET_MODE}" \
  --tuning-seeds "${TUNING_SEEDS}" \
  --validation-seeds "${VALIDATION_SEEDS}" \
  --test-seeds "${TEST_SEEDS}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --test-runs-per-instance "${TEST_RUNS_PER_INSTANCE}" \
  --reference-runs-per-method "${REFERENCE_RUNS_PER_METHOD}" \
  --tune-trials "${TUNE_TRIALS}" \
  --fixed-runs-per-value "${FIXED_RUNS_PER_VALUE}" \
  --tune-runs-per-instance "${TUNE_RUNS_PER_INSTANCE}" \
  --validation-runs-per-instance "${VALIDATION_RUNS_PER_INSTANCE}" \
  --top-k-validation "${TOP_K_VALIDATION}" \
  --bootstrap "${BOOTSTRAP}" \
  --exact-backend "${EXACT_BACKEND}" \
  --exact-time-limit-s "${EXACT_TIME_LIMIT_S}" \
  --exact-threads "${EXACT_THREADS}" \
  --out "${OUT}" \
  --recipient "${RECIPIENT}" \
  --send-mail-script "${SEND_MAIL_SCRIPT}" \
  --no-email \
  "${EXTRA_ARGS[@]}"
