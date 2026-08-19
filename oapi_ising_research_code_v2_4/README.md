# OAPI Ising Research Code v2.1 — Exact Global Optimum Experiment Suite

This update keeps the v2 paper workflow and adds **pre-planted exact global optima**, exact-optimum gap metrics/plots, `nohup` shell launchers, and SUCCESS/FAILED completion email.

## Quick start: exact global optimum experiment

The recommended exact-reference experiment does **not** ask the test solver to compute its own ground truth. Use a planted problem whose exact optimum is created first by construction.

```bash
python scripts/prepare_global_optima.py \
  --problem planted --n 128 --p 0.30 \
  --instance-seeds 200,201,202,203,204,205,206,207,208,209 \
  --out results/paper_v2/global_optima.csv
```

For TSP, `--problem planted_tsp --n 20` means 20 cities and therefore 400 Ising spins. The optimal cycle is planted first, non-cycle edges are strictly more expensive, and the QUBO penalty is chosen so infeasible states cannot beat the planted feasible optimum.

### Background execution with nohup

Use the supplied launcher:

```bash
nohup bash shell/run_oapi_paper.sh > logs/oapi_exact.log 2>&1 &
echo $!
```

Or simply:

```bash
bash shell/launch_oapi_nohup.sh
```

Useful environment overrides:

```bash
PROBLEM=planted \
N=128 P=0.30 \
CUDA_VISIBLE_DEVICES=1 \
TEST_RUNS_PER_INSTANCE=100 \
OUT=results/paper_v2/exact_optimum_workflow \
nohup bash shell/run_oapi_paper.sh > logs/oapi_exact.log 2>&1 &
```

Monitor with:

```bash
tail -f logs/oapi_exact.log
```

### Completion email

The shell wrapper follows the supplied `EXIT`-trap pattern and uses `/home/onion120/mail/send_mail.sh`. Defaults can be overridden:

```bash
RECIPIENT=donghyuck200@naver.com \
SEND_MAIL_SCRIPT=/home/onion120/mail/send_mail.sh \
nohup bash shell/run_oapi_paper.sh > logs/oapi_exact.log 2>&1 &
```

Direct Python execution also supports `--recipient`, `--send-mail-script`, and `--no-email`. The shell wrapper passes `--no-email` to Python so only one completion email is sent.

### New exact-optimum figures

When the target CSV contains exact global optima, figure generation creates:

```text
best_energy.png
best_energy.pdf
best_energy_log_gap.png
best_energy_log_gap.pdf
trajectory/best_energy_trajectory.png
trajectory/best_energy_trajectory.pdf
trajectory/best_energy_trajectory_log_gap.png
trajectory/best_energy_trajectory_log_gap.pdf
```

`best_energy.png` compares methods through `E_best-E*`; zero is the exact global optimum. The log trajectory uses the **non-negative gap** `max(E_best-E*, epsilon)` rather than raw Ising energy, because raw Ising energy can be negative and therefore cannot be meaningfully displayed on an ordinary log y-axis.

---

# OAPI Ising Research Code v2 — Paper Experiment Suite

Research implementation for **Optimizer-Inspired Oscillation-Aware Adaptive Parallel Ising (OAPI)**.

This v2 expands the original mechanism code into a paper-scale experiment workflow:

- 100–1000+ independent stochastic trajectories per method/condition
- frozen target energies and exact/best-known reference generation
- exact first-hit tick and first-hit spin-update opportunity tracking
- success probability and TTS on wall-clock / tick / spin-update budgets
- hierarchical bootstrap 95% confidence intervals
- paired instance-level bootstrap deltas vs a reference method
- random automatic hyperparameter search
- disjoint tuning / validation / test split support
- paper-scale fixed-PIMI and fixed-partial baseline tuning
- publication figures in PDF + 300-dpi PNG
- CSV and LaTeX result tables
- one-command full paper workflow

The original research ordering is preserved: **mechanism → strong fixed baselines → inertia optimizer → adaptive q → restart → joint value-add → transfer → TSP**.

---

## 1. Installation

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

Use `--device cuda` on a CUDA PyTorch environment, or `--device auto` to choose CUDA when available.

---

## 2. Important statistical design

### Independent trajectory count

`paper_benchmark.py` uses `--runs-per-instance` independently for every method and test instance.
For example:

```text
10 test instances × 100 trajectories = 1000 trajectories / method
```

The default paper benchmark therefore already supports the requested 1000-run scale.

Numerical target comparison uses `SolverConfig.target_atol` (default `1e-6`) so float32 solver energies are not falsely marked as failures against float64 references that differ only by roundoff.

### Do not define the target from final test outputs

Use `estimate_targets.py` before the final benchmark.

Two modes are supported:

1. `exact`: exhaustive enumeration for small Ising problems only.
2. `best_known`: use a separate, disjoint reference-run budget and freeze the best observed energy before evaluating test success probability.

For large ER/SK experiments, label this explicitly as **best-known target**, not exact ground-state energy.

### Bootstrap unit

Hundreds of stochastic trajectories from one graph are not treated as hundreds of independent graph instances. `oapi/statistics.py` uses a **hierarchical bootstrap**:

1. resample problem instances;
2. resample stochastic trajectories inside each sampled instance.

This is used for energy, effective parallelism, oscillation, success probability, and TTS confidence intervals.

### Three TTS axes

The benchmark reports:

- `tts_wallclock_s`: throughput-normalized wall-clock TTS (`batch runtime / batch size` as the per-trajectory runtime estimate)
- `tts_ticks`: global-tick TTS
- `tts_update_opportunities`: spin-update-opportunity TTS

This matters because adaptive `q` changes how many spins are given update opportunities per global tick.

---

## 3. v2 recommended workflow

### Stage A — strong fixed baselines on tuning instances

```bash
python scripts/tune_fixed_baselines_v2.py \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 1,2,3,4,5 \
  --runs-per-value 64 --batch 32 --steps 5000 \
  --device cuda
```

Outputs:

```text
results/paper_v2/fixed_tuning/
  runs.csv
  aggregate.csv
  recommended_fixed_baselines.json
```

Use the selected fixed `xi` and fixed `q` unchanged on validation/test instances.

### Stage B — automatic proposed-controller tuning

```bash
python scripts/auto_tune.py \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 1,2,3,4,5 \
  --trials 40 \
  --runs-per-instance 64 \
  --batch 32 --steps 5000 \
  --device cuda
```

The random search covers controller-specific values such as:

- `rho_o`
- field-conflict weight `b`
- `alpha_xi`
- Adam first/second moment constants
- decoupled inertia decay `lambda_xi`
- `dxi_max`, `xi_max`
- `q_min`, `q_step`
- `O_low`, `O_high`
- slow-loop interval/dwell counts
- stall detector and inertia release

Outputs:

```text
results/paper_v2/tuning/
  tuning_trials.csv
  best_config.json
```

### Stage C — top-K validation on disjoint instances

First freeze validation targets, then compare only the top tuning candidates:

```bash
python scripts/estimate_targets.py \
  --mode best_known \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 50,51,52,53,54 \
  --runs-per-method 256 \
  --config-json results/paper_v2/tuning/best_config.json \
  --device cuda \
  --out results/paper_v2/validation_targets.csv

python scripts/validate_tuning.py \
  --tuning-trials results/paper_v2/tuning/tuning_trials.csv \
  --top-k 5 \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 50,51,52,53,54 \
  --targets results/paper_v2/validation_targets.csv \
  --runs-per-instance 100 \
  --device cuda
```

Use `best_config_validated.json` for the final test.

### Stage D — freeze final test targets

```bash
python scripts/estimate_targets.py \
  --mode best_known \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 200,201,202,203,204,205,206,207,208,209 \
  --runs-per-method 256 \
  --batch 64 --steps 5000 \
  --device cuda \
  --config-json results/paper_v2/validation/best_config_validated.json \
  --fixed-xi 0.30 --fixed-q 0.50 \
  --out results/paper_v2/test_targets.csv
```

The values `0.30` and `0.50` above are examples only. Replace them with the values selected by `recommended_fixed_baselines.json`.

### Stage E — 1000 trajectories per method

```bash
python scripts/paper_benchmark.py \
  --problem er --n 128 --p 0.30 \
  --instance-seeds 200,201,202,203,204,205,206,207,208,209 \
  --runs-per-instance 100 \
  --batch 50 --steps 5000 \
  --targets results/paper_v2/test_targets.csv \
  --config-json results/paper_v2/validation/best_config_validated.json \
  --fixed-xi 0.30 --fixed-q 0.50 \
  --bootstrap 5000 \
  --device cuda
```

Default benchmark methods:

```text
PAR0
Fixed PIMI
Fixed Partial
Heuristic xi
Momentum xi
Adam xi
AdamW + clipping
AdamW + controller RMS normalization
Adaptive q
Joint
Joint + event-triggered restart
```

Outputs:

```text
results/paper_v2/benchmark/
  runs.csv
  summary.csv
  paired_energy_deltas.csv
  benchmark_manifest.json
```

`runs.csv` contains one row per independent stochastic trajectory.

---

## 4. Publication figures and tables

```bash
python scripts/make_paper_figures.py
```

Creates PDF and 300-dpi PNG versions of:

- best-energy distribution
- success probability + bootstrap CI
- wall-clock TTS + bootstrap CI
- effective parallelism vs success
- oscillation vs optimization quality

It also creates:

```text
table_main_metrics.csv
table_main_metrics.tex
```

For the controller trajectory figure:

```bash
python scripts/paper_trajectory_ensemble.py \
  --problem er --n 128 --p 0.30 \
  --instance-seed 200 \
  --methods par0,fixed_pimi,joint_restart \
  --runs 100 --steps 5000 \
  --config-json results/paper_v2/validation/best_config_validated.json \
  --device cuda
```

It produces ensemble trajectories for:

- best energy
- oscillation `O(t)`
- `q(t)`
- mean `xi(t)`
- `beta(t)`
- `eta(t)`

The shaded range is the 10–90% trajectory envelope; inferential 95% CIs are reported in the benchmark summary rather than being conflated with trajectory spread.

---

## 5. One-command paper workflow

The entire tuning → validation → frozen test target → benchmark → figures workflow can be launched with:

```bash
python scripts/full_paper_workflow.py \
  --problem er --n 128 --p 0.30 \
  --tuning-seeds 1,2,3,4,5 \
  --validation-seeds 50,51,52,53,54 \
  --test-seeds 200,201,202,203,204,205,206,207,208,209 \
  --test-runs-per-instance 100 \
  --device cuda
```

Paper-scale defaults are intentionally computationally expensive. For debugging, reduce `--n`, `--steps`, `--tune-trials`, and run counts first.

---

## 6. Exact small-instance target generation

For small N:

```bash
python scripts/estimate_targets.py \
  --mode exact \
  --problem er --n 20 --p 0.30 \
  --instance-seeds 200,201,202 \
  --max-exact-spins 24
```

The exhaustive enumerator lives in `oapi/references.py`. It intentionally refuses large N by default.

---

## 7. Core v2 modules

```text
oapi/
  solver.py       OAPI stochastic solver + exact first-hit tracking
  config.py       controller / annealing / target settings
  benchmark.py    batched independent trajectory runner
  statistics.py   hierarchical bootstrap, success probability, TTS, paired deltas
  references.py   exact small-N ground-state enumeration
  config_io.py    JSON configuration locking/reuse
  reporting.py    publication figure/table generation
  problems.py     ER Max-Cut, signed ER, SK, TSP
```

Important scripts:

```text
scripts/
  tune_fixed_baselines_v2.py
  auto_tune.py
  validate_tuning.py
  estimate_targets.py
  paper_benchmark.py
  make_paper_figures.py
  paper_trajectory_ensemble.py
  full_paper_workflow.py
```

The original phase scripts remain available for mechanism and ablation debugging.

---

## 8. Reproducibility rules

1. Never use test instances to choose hyperparameters.
2. Keep instance seeds and stochastic solver seeds separate.
3. Freeze target energies before calculating final success probability/TTS.
4. Report whether targets are exact or best-known.
5. Tune fixed baselines strongly; do not compare OAPI only against arbitrary default `xi` or `q`.
6. Report tick, spin-update opportunity, and measured runtime together.
7. Include controller computation in runtime.
8. Save failed trajectories, not only successes.
9. Report `q(t)`, `xi(t)`, `O(t)`, `beta(t)`, `eta(t)`, restart counts, and clipping activation.
10. Prefer instance-aware/hierarchical uncertainty estimates when multiple trajectories share a problem instance.

---

## 9. TSP

The original explicit `N^2 × N^2` Ising matrix implementation remains for small TSP demonstrations:

```bash
python scripts/tsp_demo.py --cities 8 --steps 8000 --batch 32 --device cuda
```

The explicit matrix scales as `N^4` in storage for an `N`-city TSP. Do not use the explicit scaffold for `N≈70`; implement the proposal's structured/implicit local-field kernel first.

---

## 10. Interpretation warning

The Adam/AdamW terminology refers to an **optimizer-inspired controller for the inertia state `xi_i`**, not direct gradient descent on the Ising Hamiltonian. The code keeps that distinction: the stochastic Ising proposal remains the solver dynamics, while moment estimation/decay/clipping control the inertia actuator.

## v2.2: external exact global optima for ER / signed-ER / SK

`TARGET_MODE=milp_exact` is intentionally different from `exact`:

- `exact`: internal exhaustive enumeration, only practical for very small spin counts.
- `milp_exact`: tuning/validation use disjoint best-known selection targets, but the untouched **test set** is solved first by an external exact MILP backend. The resulting frozen file is `test_global_optima.csv`.
- A test row is treated as exact only when the external solver reports/proves `optimal`. A time-limit incumbent is retained only as `best_incumbent_energy` and is never promoted to `exact_global_optimum_energy`.

The common MILP uses `s_i = 2 x_i - 1`. For each nonzero Ising coupling, a binary `y_ij=x_i x_j` is introduced with three exact binary-product constraints. This keeps Gurobi, CPLEX, and SCIP on the same mathematical formulation.

### Check installed exact backends

```bash
python scripts/check_exact_solvers.py
```

### Solve only the test global optima

```bash
PROBLEM=er \
N=64 \
P=0.30 \
EXACT_BACKEND=scip \
EXACT_TIME_LIMIT_S=0 \
OUT=results/paper_v2/er_n64/test_global_optima.csv \
nohup bash shell/run_exact_test_optima.sh \
> logs/er_n64_exact.log 2>&1 &
```

For a commercial backend use `EXACT_BACKEND=gurobi` or `EXACT_BACKEND=cplex` after its Python interface, runtime, and license are configured.

### Full OAPI workflow with exact final ER test set

```bash
PROBLEM=er \
N=64 \
P=0.30 \
TARGET_MODE=milp_exact \
EXACT_BACKEND=scip \
EXACT_TIME_LIMIT_S=0 \
TEST_RUNS_PER_INSTANCE=100 \
OUT=results/paper_v2/er_milp_exact_n64 \
nohup bash shell/run_oapi_paper.sh \
> logs/er_milp_exact_n64.log 2>&1 &
```

This produces `test_global_optima.csv` before any final test trajectories and then uses the exact energy for success probability, TTS, `E_best-E*`, relative gap, hit rate, and log-gap figures.

### Recommended paper suite: planted exact + ER/SK best-known

By default the suite runs all three main benchmarks:

1. planted Ising with exact optimum known by construction,
2. ER with a frozen disjoint best-known target,
3. SK with a frozen disjoint best-known target.

```bash
nohup bash shell/run_oapi_benchmark_suite.sh \
> logs/oapi_full_suite.log 2>&1 &
```

or simply:

```bash
bash shell/launch_oapi_suite_nohup.sh
```

Optional external exact ER/SK studies can be added to the same suite. Because proving dense N=128 SK optimality can be expensive, these are disabled by default and use a separate `EXACT_N`:

```bash
RUN_ER_MILP_EXACT=1 \
RUN_SK_MILP_EXACT=1 \
EXACT_N=64 \
EXACT_BACKEND=scip \
nohup bash shell/run_oapi_benchmark_suite.sh \
> logs/oapi_suite_with_exact.log 2>&1 &
```

At completion, `combined_summary.csv` and `combined_main_metrics.csv` aggregate all suite cases.


## v2.4: timestamped result folders and reuse

A fresh suite run creates a new directory directly under `results/` using Korean local time (`Asia/Seoul`) in filesystem-safe `YYYYMMDD_HHMMSS` format. Example:

```bash
nohup bash shell/run_oapi_benchmark_suite.sh > logs/oapi_new.log 2>&1 &
# -> results/20260818_144500/
```

To resume/reuse a specific existing folder, set `REUSE_FOLDER`. A relative value is resolved under `RESULTS_ROOT`:

```bash
REUSE_FOLDER=20260818_144500 \
nohup bash shell/run_oapi_benchmark_suite.sh > logs/oapi_resume.log 2>&1 &
```

An absolute folder is also accepted:

```bash
REUSE_FOLDER=/home/onion120/Ising-project/oapi_ising_research_code_v2_2/results/20260818_144500 \
nohup bash shell/run_oapi_benchmark_suite.sh > logs/oapi_resume.log 2>&1 &
```

For the older existing tree, use:

```bash
REUSE_FOLDER=/home/onion120/Ising-project/oapi_ising_research_code_v2_2/results/paper_v2/main_plus_exact \
TEST_RUNS_PER_INSTANCE=256 \
nohup bash shell/rerun_main_plus_exact_batch25_runs256.sh \
  > logs/oapi_resume_existing_b25_r256.log 2>&1 &
```

Resolution precedence is `SUITE_ROOT` (legacy explicit path) > `REUSE_FOLDER` > a newly generated timestamp folder. Resume logic within the selected folder remains controlled by `REUSE_EXISTING` (default `1`). Each invocation appends its settings to `<run-folder>/run_history.log`. When `run_oapi_paper.sh` is executed directly, its precedence is `OUT` > `REUSE_FOLDER` > a new timestamp folder.
