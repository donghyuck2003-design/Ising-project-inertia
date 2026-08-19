from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from .config import SolverConfig
from .experiment_utils import method_config
from .solver import IsingSolver
from .problems import (
    make_er_maxcut,
    make_sk,
    make_signed_er,
    make_planted_ising,
    make_planted_tsp_ising,
    tsp_decode,
)


def make_problem(family: str, n: int, p: float, seed: int):
    """Create one deterministic problem instance.

    Notes
    -----
    * ``n`` is the number of spins for ER/SK/planted Ising.
    * ``n`` is the number of cities for ``planted_tsp``; the resulting Ising
      problem therefore has n^2 spins.
    """
    if family == "er":
        return make_er_maxcut(n=n, p=p, seed=seed)
    if family == "signed_er":
        return make_signed_er(n=n, p=p, seed=seed)
    if family == "sk":
        return make_sk(n=n, seed=seed)
    if family == "planted":
        return make_planted_ising(n=n, p=p, seed=seed)
    if family == "planted_tsp":
        problem, _ = make_planted_tsp_ising(n_cities=n, seed=seed)
        return problem
    raise ValueError(f"Unknown family: {family}")


def load_target_map(path: Optional[str | Path]) -> Dict[int, float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"instance_seed", "target_energy"}.issubset(df.columns):
        raise ValueError("target CSV must contain instance_seed,target_energy")
    return {int(r.instance_seed): float(r.target_energy) for r in df.itertuples()}


def load_reference_table(path: Optional[str | Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "instance_seed" not in df.columns:
        raise ValueError("reference CSV must contain instance_seed")
    return df.copy()


def reference_row_map(path: Optional[str | Path]) -> Dict[int, Dict[str, Any]]:
    df = load_reference_table(path)
    if df.empty:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        out[int(d["instance_seed"])] = d
    return out


def _finite_or_nan(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def run_method_trajectories(
    problem,
    method: str,
    base_cfg: SolverConfig,
    n_runs: int,
    batch_size: int,
    solver_seed_base: int,
    instance_seed: int,
    target_energy: Optional[float] = None,
    exact_optimum_energy: Optional[float] = None,
    trajectory_id_start: int = 0,
    batch_id_start: int = 0,
) -> pd.DataFrame:
    """Run n_runs independent stochastic trajectories in reproducible batches.

    When ``exact_optimum_energy`` is supplied, every run records absolute and
    relative optimality gaps. This value is a frozen external/reference value;
    it is never inferred from the test trajectories themselves.
    """
    rows = []
    done = 0
    batch_id = int(batch_id_start)
    exact_route_distance = _finite_or_nan((problem.metadata or {}).get("exact_optimum_route_distance", np.nan))
    is_tsp = "n_cities" in (problem.metadata or {}) and "D" in (problem.metadata or {})

    while done < n_runs:
        b = min(batch_size, n_runs - done)
        cfg = method_config(base_cfg, method)
        cfg.batch_size = b
        cfg.seed = int(solver_seed_base + 1000003 * batch_id)
        cfg.target_energy = target_energy
        result = IsingSolver(problem, cfg).run()
        per_traj_runtime = float(result.runtime_s / b)
        for j in range(b):
            be = float(result.best_energy[j])
            exact_gap = float("nan")
            rel_gap = float("nan")
            log10_gap = float("nan")
            if exact_optimum_energy is not None:
                exact_gap = max(0.0, be - float(exact_optimum_energy))
                rel_gap = 100.0 * exact_gap / max(abs(float(exact_optimum_energy)), cfg.target_atol)
                # Useful for log-gap plots; exact hits are placed at the numeric
                # tolerance floor instead of attempting log10(0).
                log10_gap = float(np.log10(max(exact_gap, cfg.target_atol)))

            row = {
                "method": method,
                "instance_seed": int(instance_seed),
                "trajectory_id": int(trajectory_id_start + done + j),
                "solver_seed_batch": int(cfg.seed),
                "batch_id": int(batch_id),
                "best_energy": be,
                "final_energy": float(result.final_energy[j]),
                "target_energy": float(target_energy) if target_energy is not None else np.nan,
                "target_atol": float(cfg.target_atol),
                "success": int(target_energy is not None and be <= target_energy + cfg.target_atol),
                "exact_global_optimum_energy": (
                    float(exact_optimum_energy) if exact_optimum_energy is not None else np.nan
                ),
                "energy_gap_to_global_optimum": exact_gap,
                "relative_energy_gap_percent": rel_gap,
                "log10_energy_gap": log10_gap,
                "first_hit_tick": int(result.first_hit_tick[j]),
                "first_hit_update_ops": int(result.first_hit_update_ops[j]),
                "update_opportunities": int(result.update_opportunities[j]),
                "restarts": int(result.restarts[j]),
                "runtime_s_batch": float(result.runtime_s),
                "runtime_s_per_trajectory": per_traj_runtime,
                "batch_size_actual": int(b),
                "steps": int(cfg.steps),
                "n_spins": int(problem.n),
                "mean_q": float(result.history["q"][:, j].mean()),
                "mean_O": float(result.history["O"][:, j].mean()),
                "mean_xi": float(result.history["xi_mean"][:, j].mean()),
                "clip_rate": float(result.history["clip_rate"][:, j].mean()),
            }

            if is_tsp:
                dec = tsp_decode(
                    result.best_state[j],
                    int(problem.metadata["n_cities"]),
                    np.asarray(problem.metadata["D"], dtype=float),
                )
                row.update({
                    "best_tsp_feasible": int(bool(dec["feasible"])),
                    "best_route_distance": float(dec["tour_distance"]) if dec["feasible"] else np.nan,
                    "exact_optimum_route_distance": exact_route_distance,
                    "route_gap_to_global_optimum": (
                        float(dec["tour_distance"] - exact_route_distance)
                        if dec["feasible"] and np.isfinite(exact_route_distance)
                        else np.nan
                    ),
                })
            rows.append(row)
        done += b
        batch_id += 1
    return pd.DataFrame(rows)
