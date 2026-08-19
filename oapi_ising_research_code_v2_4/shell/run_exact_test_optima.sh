#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV_ACTIVATE="${VENV_ACTIVATE:-${HOME}/openai-env/bin/activate}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

PROBLEM="${PROBLEM:-er}"
N="${N:-64}"
P="${P:-0.30}"
TEST_SEEDS="${TEST_SEEDS:-200,201,202,203,204,205,206,207,208,209}"
EXACT_BACKEND="${EXACT_BACKEND:-auto}"
EXACT_TIME_LIMIT_S="${EXACT_TIME_LIMIT_S:-0}"
EXACT_THREADS="${EXACT_THREADS:-0}"
EXACT_SOLVER_LOG="${EXACT_SOLVER_LOG:-0}"
REQUIRE_OPTIMAL="${REQUIRE_OPTIMAL:-1}"
OUT="${OUT:-results/paper_v2/${PROBLEM}_n${N}/test_global_optima.csv}"
RECIPIENT="${RECIPIENT:-donghyuck200@naver.com}"
SEND_MAIL_SCRIPT="${SEND_MAIL_SCRIPT:-/home/onion120/mail/send_mail.sh}"
SEND_EMAIL="${SEND_EMAIL:-1}"

mkdir -p "$(dirname "${OUT}")" logs

notify () {
  local code="$?" status="SUCCESS"
  trap - EXIT
  if (( code != 0 )); then status="FAILED"; fi
  if [[ "${SEND_EMAIL}" == "1" && -x "${SEND_MAIL_SCRIPT}" ]]; then
    "${SEND_MAIL_SCRIPT}" \
      --to "${RECIPIENT}" \
      --subject "[${status}] OAPI exact MILP global-optimum solve finished" \
      --body "Status=${status}
ExitCode=${code}
Problem=${PROBLEM}, N=${N}, p=${P}
Backend=${EXACT_BACKEND}
TimeLimitPerInstance=${EXACT_TIME_LIMIT_S}
Output=${ROOT}/${OUT}
FinishedAt=$(date --iso-8601=seconds)" \
      --script-info "shell/run_exact_test_optima.sh" || true
  fi
  exit "${code}"
}
trap notify EXIT

ARGS=()
if [[ "${EXACT_SOLVER_LOG}" == "1" ]]; then ARGS+=(--solver-log); fi
if [[ "${REQUIRE_OPTIMAL}" == "0" ]]; then ARGS+=(--no-require-optimal); fi

python scripts/solve_test_global_optima.py \
  --problem "${PROBLEM}" \
  --n "${N}" \
  --p "${P}" \
  --instance-seeds "${TEST_SEEDS}" \
  --backend "${EXACT_BACKEND}" \
  --time-limit-s "${EXACT_TIME_LIMIT_S}" \
  --threads "${EXACT_THREADS}" \
  --out "${OUT}" \
  "${ARGS[@]}"
