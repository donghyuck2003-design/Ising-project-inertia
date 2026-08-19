#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
export RESULTS_ROOT
# Leave REUSE_FOLDER empty for a fresh timestamped directory under RESULTS_ROOT.
# To resume a previous run, set either:
#   REUSE_FOLDER=20260818_144500
# or an absolute path:
#   REUSE_FOLDER=/home/onion120/Ising-project/oapi_ising_research_code_v2_2/results/20260818_144500
export REUSE_FOLDER="${REUSE_FOLDER:-}"

export N=128
export EXACT_N=64
export P=0.30
export BATCH=25
export RUN_ER_MILP_EXACT=1
export RUN_SK_MILP_EXACT=1
export EXACT_BACKEND=scip
export EXACT_TIME_LIMIT_S=0
export CUDA_VISIBLE_DEVICES=1
export REFERENCE_RUNS_PER_METHOD=256
export TEST_RUNS_PER_INSTANCE="${TEST_RUNS_PER_INSTANCE:-100}"
export TUNE_TRIALS=40
export BOOTSTRAP=5000
export REUSE_EXISTING=1
export RECIPIENT="${RECIPIENT:-donghyuck200@naver.com}"
# SUITE_ROOT remains supported for backward compatibility. If it is unset,
# run_oapi_benchmark_suite.sh chooses REUSE_FOLDER or creates a timestamped folder.
if [[ -n "${SUITE_ROOT:-}" ]]; then export SUITE_ROOT; fi

bash shell/run_oapi_benchmark_suite.sh
