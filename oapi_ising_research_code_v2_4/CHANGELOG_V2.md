
## v2.4.0 — Timestamped result folders + explicit folder reuse

- Fresh benchmark-suite and direct paper-workflow runs now create `results/YYYYMMDD_HHMMSS/` automatically (timestamp generated in `Asia/Seoul` by default).
- Added `RESULTS_ROOT`, `RESULTS_TZ`, `RUN_TIMESTAMP`, and `REUSE_FOLDER` controls.
- `REUSE_FOLDER=<folder>` resumes/reuses a selected run folder; relative values resolve under `RESULTS_ROOT`, absolute paths are used directly.
- Existing `SUITE_ROOT` remains supported and takes precedence for backward compatibility.
- Added append-only `run_history.log` inside each run folder for resume provenance.
- `launch_oapi_suite_nohup.sh` and `launch_oapi_nohup.sh` now use the same timestamp for the result folder and log filename and print the selected result directory immediately.
# Changelog

## v2.1.0 — exact global-optimum + nohup + email update

### 1. Exact global-optimum comparison

- Added `planted` Ising instances whose exact global optimum is created **before** solver execution.
  - First sample a planted state `s*`.
  - Construct attractive gauge-transformed couplings `J_ij = w_ij s*_i s*_j`.
  - Align nonzero local fields with `s*` so every different spin state has strictly higher energy.
- Added `planted_tsp` instances.
  - The optimal Hamiltonian cycle is created first.
  - Its edges receive the minimum edge cost.
  - Every non-cycle edge is strictly more expensive.
  - The one-hot penalty is chosen so infeasible QUBO states cannot undercut the planted feasible optimum.
- Added `scripts/prepare_global_optima.py` to create the frozen exact-global-optimum manifest before benchmark routes run.
- `estimate_targets.py` now supports `--mode planted` and stores:
  - `exact_global_optimum_energy`
  - optimum state hash
  - planted TSP optimum route/distance when applicable
  - reference provenance
- `paper_benchmark.py` now records per trajectory:
  - `energy_gap_to_global_optimum = E_best - E*`
  - relative gap (%)
  - `log10_energy_gap`
  - exact-optimum hit rate
  - TSP feasible route distance/gap when applicable
- publication summary includes hierarchical-bootstrap CIs for exact-optimum gap.

### 2. Requested plots

- `best_energy.png/.pdf`
  - when exact optima exist, plots `E_best - E*` across methods;
  - zero is the exact global optimum reference.
- `best_energy_log_gap.png/.pdf`
  - log-scale distribution of exact-optimum gap.
- `best_energy_trajectory.png/.pdf`
  - raw ensemble best-energy trajectory with the exact optimum line.
- `best_energy_trajectory_log_gap.png/.pdf`
  - log-scale `max(E_best-E*, epsilon)` trajectory.
  - raw Ising energy is **not** log-scaled because it can be negative.

### 3. Shell / nohup execution

Added:

- `shell/run_oapi_paper.sh` — full paper workflow shell wrapper.
- `shell/launch_oapi_nohup.sh` — launches the wrapper under `nohup`, writes PID and log files.

The shell wrapper is environment-variable configurable and preserves the Python workflow exit code.

### 4. Completion email

- Added `oapi/notify.py`.
- Direct execution of `scripts/full_paper_workflow.py` sends SUCCESS/FAILED mail by default through:
  - `/home/onion120/mail/send_mail.sh`
- Added options:
  - `--recipient`
  - `--send-mail-script`
  - `--no-email`
- `shell/run_oapi_paper.sh` uses an `EXIT` trap and sends SUCCESS/FAILED mail after completion while preserving the original exit status.
- The shell wrapper passes `--no-email` to Python to prevent duplicate notifications.

### 5. Verification

Validated with:

- Python compile checks.
- Existing exact-enumeration/first-hit smoke test.
- Exhaustive verification that a 12-spin planted Ising metadata optimum equals the enumerated ground state.
- Exhaustive verification that a 4-city / 16-spin planted-TSP metadata optimum equals the enumerated ground-state energy.
- End-to-end planted benchmark → bootstrap summary → `best_energy.png` → log-gap trajectory generation.
- Tiny full tuning/validation/test workflow.
- Shell wrapper execution with completion trap.

---

## v2.0.0 — paper experiment suite

- paper-scale independent trajectory benchmark
- success probability / TTS
- hierarchical bootstrap confidence intervals
- frozen exact/best-known targets
- automatic controller tuning
- tuning/validation/test separation
- publication figures and LaTeX tables

## v2.2.0 — External exact MILP references + multi-family suite

- Added `oapi/exact_solvers.py` with a common exact binary MILP formulation for Gurobi, CPLEX/DOcplex, and SCIP/PySCIPOpt.
- Added strict optimality certification: only backend `optimal` results populate `exact_global_optimum_energy` and `target_energy`.
- Added `scripts/solve_test_global_optima.py` to create the frozen `test_global_optima.csv` before final OAPI test routes.
- Added `TARGET_MODE=milp_exact`: tuning/validation stay on disjoint best-known targets; external exact solving is reserved for the untouched final test set.
- Added solver diagnostics/bounds/runtime/model-size fields and compressed exact-state NPZ output.
- Added `scripts/check_exact_solvers.py` and `scripts/exact_solver_smoke_test.py`.
- Added `shell/run_exact_test_optima.sh` with completion/failure email notification.
- Added `shell/run_oapi_benchmark_suite.sh` and `shell/launch_oapi_suite_nohup.sh` for planted exact + ER/SK best-known, with optional ER/SK external-exact runs.
- Added combined suite summary generation.

## v2.3.0 — Resume-safe batch-25 / 256-run workflow
- Default GPU batch reduced from 50 to 25 for long SK/reference jobs.
- Best-known reference generation uses 256 runs per method; final test remains 100 runs per method/instance by default unless explicitly overridden.
- Best-known reference generation remains 256 runs per method.
- `estimate_targets.py --reuse-existing` checkpoints every completed instance and resumes missing seeds after crashes.
- `paper_benchmark.py --reuse-existing` preserves prior rows and appends only missing trajectories (e.g. existing 100 -> add 156 -> total 256).
- `full_paper_workflow.py --reuse-existing` skips completed fixed tuning, controller tuning, validation targets/configs, and resumes target/test stages in place.
- External exact-optimum generation can reuse already proven SCIP/Gurobi/CPLEX rows.
- Ensemble trajectory plotting now accepts `--batch`; 100 visualization trajectories are internally split into batch<=25 instead of one large GPU batch.
- Suite default result root is the existing `results/paper_v2/main_plus_exact` tree.
- Added `shell/rerun_main_plus_exact_batch25_runs256.sh` for the requested resume configuration.
- Clarification: `REFERENCE_RUNS_PER_METHOD=256` is the requested best-known target-generation budget. `TEST_RUNS_PER_INSTANCE` remains 100 by default (matching the previous suite) unless explicitly overridden.
- Resume safety for runtime metrics: if an existing final benchmark was produced with batch=50 and the new request is batch=25, the old `runs.csv` is backed up as `runs.pre_batch50.csv` and the final benchmark is rerun at batch=25 instead of mixing wall-clock/TTS measurements across batch sizes. Earlier tuning/validation/target folders are still reused.
