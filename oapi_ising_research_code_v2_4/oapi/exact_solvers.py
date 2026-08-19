from __future__ import annotations

"""Optional exact MILP backends for Ising global optima.

The same linearized binary model is used for every backend.  For
s_i = 2 x_i - 1 and H(s) = -1/2 s^T J s - h^T s, each product x_i x_j
is replaced by a binary y_ij with the exact McCormick/Fortet constraints

    y_ij <= x_i
    y_ij <= x_j
    y_ij >= x_i + x_j - 1.

Because x_i and x_j are binary, these constraints are exactly equivalent to
y_ij = x_i x_j.  This makes the formulation a pure MILP and keeps the
mathematical model identical across Gurobi, CPLEX, and SCIP.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import importlib.util
import math
import time
import numpy as np

from .problems import IsingProblem
from .references import exact_ground_state


@dataclass
class BinaryMILPFormulation:
    linear: np.ndarray
    pairs: List[Tuple[int, int, float]]
    constant: float
    n_spin_vars: int
    n_product_vars: int
    n_constraints: int


@dataclass
class ExactSolveResult:
    backend: str
    status: str
    optimality_proven: bool
    energy: float
    state: Optional[np.ndarray]
    runtime_s: float
    best_bound_energy: float = math.nan
    relative_gap: float = math.nan
    nodes: float = math.nan
    backend_version: str = ""
    n_spin_vars: int = 0
    n_product_vars: int = 0
    n_constraints: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("state", None)
        return d


class ExactBackendUnavailable(RuntimeError):
    pass


def ising_energy(problem: IsingProblem, state: Sequence[int]) -> float:
    s = np.asarray(state, dtype=np.float64)
    J = problem.J.detach().cpu().numpy().astype(np.float64)
    h = problem.h.detach().cpu().numpy().astype(np.float64)
    return float(-0.5 * s @ J @ s - h @ s)


def build_binary_milp(problem: IsingProblem, coupling_tol: float = 1e-14) -> BinaryMILPFormulation:
    """Return an exact linearized binary representation of the Ising energy."""
    J = problem.J.detach().cpu().numpy().astype(np.float64)
    h = problem.h.detach().cpu().numpy().astype(np.float64)
    J = 0.5 * (J + J.T)
    n = len(h)

    # Diagonal Ising terms are constants because s_i^2 = 1.
    constant = float(-0.5 * np.trace(J) + np.sum(h))
    linear = -2.0 * h.copy()
    pairs: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            Jij = float(J[i, j])
            if abs(Jij) <= coupling_tol:
                continue
            # -Jij (2xi-1)(2xj-1)
            constant += -Jij
            linear[i] += 2.0 * Jij
            linear[j] += 2.0 * Jij
            pairs.append((i, j, -4.0 * Jij))

    return BinaryMILPFormulation(
        linear=linear,
        pairs=pairs,
        constant=float(constant),
        n_spin_vars=n,
        n_product_vars=len(pairs),
        n_constraints=3 * len(pairs),
    )


def evaluate_binary_milp(form: BinaryMILPFormulation, state: Sequence[int]) -> float:
    s = np.asarray(state, dtype=np.int8)
    x = ((s.astype(np.int64) + 1) // 2).astype(np.float64)
    val = float(form.constant + form.linear @ x)
    for i, j, q in form.pairs:
        val += float(q * x[i] * x[j])
    return val


def verify_formulation(problem: IsingProblem, samples: int = 32, seed: int = 1234, atol: float = 1e-9) -> None:
    form = build_binary_milp(problem)
    rng = np.random.default_rng(seed)
    for _ in range(max(1, samples)):
        s = rng.choice([-1, 1], size=problem.n).astype(np.int8)
        a = ising_energy(problem, s)
        b = evaluate_binary_milp(form, s)
        if not np.isclose(a, b, atol=atol, rtol=1e-10):
            raise AssertionError(f"MILP formulation mismatch: Ising={a}, MILP={b}, delta={b-a}")


def available_backends() -> Dict[str, bool]:
    return {
        "gurobi": importlib.util.find_spec("gurobipy") is not None,
        "cplex": importlib.util.find_spec("docplex") is not None,
        "scip": importlib.util.find_spec("pyscipopt") is not None,
        "enumeration": True,
    }


def _safe_gap(energy: float, bound: float) -> float:
    if not (np.isfinite(energy) and np.isfinite(bound)):
        return math.nan
    return float(max(0.0, energy - bound) / max(abs(energy), 1e-12))


def _solve_gurobi(problem: IsingProblem, form: BinaryMILPFormulation, *, time_limit_s: float,
                  threads: int, log: bool) -> ExactSolveResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ExactBackendUnavailable(f"gurobipy unavailable: {e}") from e

    t0 = time.perf_counter()
    try:
        model = gp.Model("oapi_exact_ising")
        model.Params.OutputFlag = 1 if log else 0
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.0
        if threads > 0:
            model.Params.Threads = int(threads)
        if time_limit_s > 0:
            model.Params.TimeLimit = float(time_limit_s)

        x = model.addVars(form.n_spin_vars, vtype=GRB.BINARY, name="x")
        y = {}
        for k, (i, j, _) in enumerate(form.pairs):
            yy = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
            y[k] = yy
            model.addConstr(yy <= x[i])
            model.addConstr(yy <= x[j])
            model.addConstr(yy >= x[i] + x[j] - 1)

        obj = gp.LinExpr()
        obj.addConstant(float(form.constant))
        obj += gp.quicksum(float(form.linear[i]) * x[i] for i in range(form.n_spin_vars))
        obj += gp.quicksum(float(q) * y[k] for k, (_, _, q) in enumerate(form.pairs))
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        status_code = int(model.Status)
        status = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INTERRUPTED: "interrupted",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "inf_or_unbd",
            GRB.UNBOUNDED: "unbounded",
        }.get(status_code, f"status_{status_code}")
        optimal = status_code == GRB.OPTIMAL
        state = None
        energy = math.nan
        if int(model.SolCount) > 0:
            xx = np.array([float(x[i].X) for i in range(form.n_spin_vars)])
            state = np.where(xx >= 0.5, 1, -1).astype(np.int8)
            energy = ising_energy(problem, state)
        bound = float(model.ObjBound) if hasattr(model, "ObjBound") else math.nan
        # ObjBound contains the same objective including the Ising constant.
        runtime = float(model.Runtime) if hasattr(model, "Runtime") else time.perf_counter() - t0
        nodes = float(model.NodeCount) if hasattr(model, "NodeCount") else math.nan
        version = ".".join(map(str, gp.gurobi.version()))
        return ExactSolveResult(
            backend="gurobi", status=status, optimality_proven=optimal,
            energy=energy, state=state, runtime_s=runtime,
            best_bound_energy=bound, relative_gap=_safe_gap(energy, bound), nodes=nodes,
            backend_version=version, n_spin_vars=form.n_spin_vars,
            n_product_vars=form.n_product_vars, n_constraints=form.n_constraints,
        )
    except ExactBackendUnavailable:
        raise
    except Exception as e:
        raise ExactBackendUnavailable(f"Gurobi could not solve/start (installation or license issue possible): {e}") from e


def _solve_cplex(problem: IsingProblem, form: BinaryMILPFormulation, *, time_limit_s: float,
                 threads: int, log: bool) -> ExactSolveResult:
    try:
        import docplex
        from docplex.mp.model import Model
    except Exception as e:
        raise ExactBackendUnavailable(f"DOcplex unavailable: {e}") from e

    t0 = time.perf_counter()
    try:
        mdl = Model(name="oapi_exact_ising")
        # A local CPLEX runtime/license must be available to DOcplex.
        x = mdl.binary_var_list(form.n_spin_vars, name="x")
        y = []
        for i, j, _ in form.pairs:
            yy = mdl.binary_var(name=f"y_{i}_{j}")
            y.append(yy)
            mdl.add_constraint(yy <= x[i])
            mdl.add_constraint(yy <= x[j])
            mdl.add_constraint(yy >= x[i] + x[j] - 1)

        obj = float(form.constant)
        obj += mdl.sum(float(form.linear[i]) * x[i] for i in range(form.n_spin_vars))
        obj += mdl.sum(float(q) * y[k] for k, (_, _, q) in enumerate(form.pairs))
        mdl.minimize(obj)

        mdl.parameters.mip.tolerances.mipgap = 0.0
        try:
            mdl.parameters.mip.tolerances.absmipgap = 0.0
        except Exception:
            pass
        if time_limit_s > 0:
            mdl.parameters.timelimit = float(time_limit_s)
        if threads > 0:
            mdl.parameters.threads = int(threads)

        sol = mdl.solve(log_output=bool(log))
        details = mdl.solve_details
        status = str(getattr(details, "status", "unknown"))
        low = status.lower()
        # CPLEX may report "integer optimal solution". Do not accept tolerance/limit statuses.
        optimal = ("optimal" in low) and ("tolerance" not in low) and ("limit" not in low)
        state = None
        energy = math.nan
        if sol is not None:
            xx = np.array([float(sol.get_value(v)) for v in x])
            state = np.where(xx >= 0.5, 1, -1).astype(np.int8)
            energy = ising_energy(problem, state)
        try:
            bound = float(getattr(details, "best_bound", math.nan))
        except (TypeError, ValueError):
            bound = math.nan
        try:
            runtime = float(getattr(details, "time", time.perf_counter() - t0))
        except (TypeError, ValueError):
            runtime = time.perf_counter() - t0
        try:
            nodes = float(getattr(details, "nb_nodes_processed", math.nan))
        except (TypeError, ValueError):
            nodes = math.nan
        version = str(getattr(docplex, "__version__", ""))
        return ExactSolveResult(
            backend="cplex", status=status, optimality_proven=bool(optimal),
            energy=energy, state=state, runtime_s=runtime,
            best_bound_energy=bound, relative_gap=_safe_gap(energy, bound), nodes=nodes,
            backend_version=version, n_spin_vars=form.n_spin_vars,
            n_product_vars=form.n_product_vars, n_constraints=form.n_constraints,
        )
    except Exception as e:
        raise ExactBackendUnavailable(f"CPLEX/DOcplex could not solve/start (local CPLEX runtime/license required): {e}") from e


def _solve_scip(problem: IsingProblem, form: BinaryMILPFormulation, *, time_limit_s: float,
                threads: int, log: bool) -> ExactSolveResult:
    try:
        import pyscipopt
        from pyscipopt import Model, quicksum
    except Exception as e:
        raise ExactBackendUnavailable(f"PySCIPOpt unavailable: {e}") from e

    t0 = time.perf_counter()
    try:
        model = Model("oapi_exact_ising")
        if not log:
            model.hideOutput()
        if time_limit_s > 0:
            model.setRealParam("limits/time", float(time_limit_s))
        # Zero gap is required before we label the result exact.
        try:
            model.setRealParam("limits/gap", 0.0)
            model.setRealParam("limits/absgap", 0.0)
        except Exception:
            pass
        if threads > 0:
            try:
                model.setIntParam("parallel/maxnthreads", int(threads))
            except Exception:
                pass

        x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(form.n_spin_vars)]
        y = []
        for i, j, _ in form.pairs:
            yy = model.addVar(vtype="B", name=f"y_{i}_{j}")
            y.append(yy)
            model.addCons(yy <= x[i])
            model.addCons(yy <= x[j])
            model.addCons(yy >= x[i] + x[j] - 1)

        obj = float(form.constant)
        obj += quicksum(float(form.linear[i]) * x[i] for i in range(form.n_spin_vars))
        obj += quicksum(float(q) * y[k] for k, (_, _, q) in enumerate(form.pairs))
        model.setObjective(obj, "minimize")
        model.optimize()

        status = str(model.getStatus())
        optimal = status.lower() == "optimal"
        state = None
        energy = math.nan
        if model.getNSols() > 0:
            xx = np.array([float(model.getVal(v)) for v in x])
            state = np.where(xx >= 0.5, 1, -1).astype(np.int8)
            energy = ising_energy(problem, state)
        try:
            bound = float(model.getDualbound())
        except Exception:
            bound = math.nan
        runtime = float(model.getSolvingTime())
        try:
            nodes = float(model.getNTotalNodes())
        except Exception:
            nodes = math.nan
        version = str(getattr(pyscipopt, "__version__", ""))
        return ExactSolveResult(
            backend="scip", status=status, optimality_proven=optimal,
            energy=energy, state=state, runtime_s=runtime,
            best_bound_energy=bound, relative_gap=_safe_gap(energy, bound), nodes=nodes,
            backend_version=version, n_spin_vars=form.n_spin_vars,
            n_product_vars=form.n_product_vars, n_constraints=form.n_constraints,
        )
    except Exception as e:
        raise ExactBackendUnavailable(f"SCIP/PySCIPOpt could not solve/start: {e}") from e


def _solve_enumeration(problem: IsingProblem, form: BinaryMILPFormulation, *, max_spins: int) -> ExactSolveResult:
    t0 = time.perf_counter()
    ref = exact_ground_state(problem, max_spins=max_spins)
    runtime = time.perf_counter() - t0
    return ExactSolveResult(
        backend="enumeration", status="optimal", optimality_proven=True,
        energy=float(ref["energy"]), state=np.asarray(ref["state"], dtype=np.int8),
        runtime_s=float(runtime), best_bound_energy=float(ref["energy"]), relative_gap=0.0,
        backend_version="internal", n_spin_vars=form.n_spin_vars,
        n_product_vars=form.n_product_vars, n_constraints=form.n_constraints,
        message=f"Enumerated {ref['n_states']} states",
    )


def solve_exact_ising(
    problem: IsingProblem,
    backend: str = "auto",
    *,
    time_limit_s: float = 0.0,
    threads: int = 0,
    log: bool = False,
    verify_model: bool = True,
    verify_samples: int = 16,
    verification_atol: float = 1e-7,
    enumeration_max_spins: int = 24,
) -> ExactSolveResult:
    """Solve an Ising instance and only mark it exact if optimality is proven.

    ``backend='auto'`` tries Gurobi, then CPLEX, then SCIP. Enumeration is not
    selected automatically because it is only appropriate for very small N;
    choose it explicitly for tests.
    """
    if verify_model:
        verify_formulation(problem, samples=verify_samples, atol=verification_atol)
    form = build_binary_milp(problem)

    if backend == "enumeration":
        result = _solve_enumeration(problem, form, max_spins=enumeration_max_spins)
    else:
        order = [backend] if backend != "auto" else ["gurobi", "cplex", "scip"]
        errors = []
        result = None
        for b in order:
            try:
                if b == "gurobi":
                    result = _solve_gurobi(problem, form, time_limit_s=time_limit_s, threads=threads, log=log)
                elif b == "cplex":
                    result = _solve_cplex(problem, form, time_limit_s=time_limit_s, threads=threads, log=log)
                elif b == "scip":
                    result = _solve_scip(problem, form, time_limit_s=time_limit_s, threads=threads, log=log)
                else:
                    raise ValueError(f"Unknown exact backend: {b}")
                break
            except ExactBackendUnavailable as e:
                errors.append(f"{b}: {e}")
                if backend != "auto":
                    raise
        if result is None:
            raise ExactBackendUnavailable("No exact backend could be used. " + " | ".join(errors))

    if result.state is not None and np.isfinite(result.energy):
        recomputed = ising_energy(problem, result.state)
        if not np.isclose(recomputed, result.energy, atol=verification_atol, rtol=1e-9):
            raise AssertionError(f"Returned-state energy mismatch: stored={result.energy}, recomputed={recomputed}")
        # If the solver says optimal, its primal objective and bound should agree
        # within numerical tolerance. We keep the solver status authoritative but
        # reject obviously inconsistent certificates.
        if result.optimality_proven and np.isfinite(result.best_bound_energy):
            if abs(result.energy - result.best_bound_energy) > max(verification_atol, 1e-8 * max(1.0, abs(result.energy))):
                raise AssertionError(
                    f"Backend reported optimal but primal/bound disagree: E={result.energy}, bound={result.best_bound_energy}"
                )
    return result
