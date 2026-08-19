from __future__ import annotations
from typing import Callable, Optional, Sequence, Dict, Any
import math
import numpy as np
import pandas as pd


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for an i.i.d. one-dimensional sample."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(x))
    if x.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = x.size
    for i in range(n_boot):
        boots[i] = statistic(x[rng.integers(0, n, size=n)])
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def hierarchical_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "instance_seed",
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-level bootstrap: sample instances, then trajectories inside each instance.

    This avoids pretending that hundreds of stochastic trajectories from the same
    problem instance are fully independent problem instances.
    """
    work = df[[cluster_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return float("nan"), float("nan"), float("nan")
    clusters = list(work[cluster_col].unique())
    arrays = {c: work.loc[work[cluster_col] == c, value_col].to_numpy(float) for c in clusters}
    point = float(statistic(work[value_col].to_numpy(float)))
    if len(clusters) == 1:
        return bootstrap_ci(work[value_col].to_numpy(float), statistic, confidence, n_boot, seed)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    nc = len(clusters)
    for b in range(n_boot):
        chosen = rng.integers(0, nc, size=nc)
        pieces = []
        for idx in chosen:
            arr = arrays[clusters[idx]]
            pieces.append(arr[rng.integers(0, len(arr), size=len(arr))])
        boots[b] = statistic(np.concatenate(pieces))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def _tts_from_p(run_time: float, p_success: float, confidence: float) -> float:
    if p_success <= 0.0:
        return float("inf")
    if p_success >= 1.0:
        return float(run_time)
    return float(run_time * math.log(1.0 - confidence) / math.log(1.0 - p_success))


def bootstrap_success_tts(
    success: Sequence[int | bool],
    run_times: Sequence[float],
    confidence_tts: float = 0.99,
    confidence_ci: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap p_success and throughput-normalized wall-clock TTS.

    run_times should be a per-trajectory wall-clock estimate (for batched GPU
    experiments this code records batch_runtime / batch_size).
    """
    y = np.asarray(success, dtype=float)
    rt = np.asarray(run_times, dtype=float)
    ok = np.isfinite(y) & np.isfinite(rt)
    y, rt = y[ok], rt[ok]
    if y.size == 0:
        return {k: float("nan") for k in ["p_success", "p_success_ci_low", "p_success_ci_high", "tts", "tts_ci_low", "tts_ci_high"]}
    p = float(y.mean())
    run_time = float(np.mean(rt))
    tts_point = _tts_from_p(run_time, p, confidence_tts)
    rng = np.random.default_rng(seed)
    ps = np.empty(n_boot)
    tt = np.empty(n_boot)
    n = y.size
    for b in range(n_boot):
        ix = rng.integers(0, n, size=n)
        pb = float(y[ix].mean())
        rtb = float(rt[ix].mean())
        ps[b] = pb
        tt[b] = _tts_from_p(rtb, pb, confidence_tts)
    alpha = 1 - confidence_ci
    p_lo, p_hi = np.quantile(ps, [alpha / 2, 1 - alpha / 2])
    finite_tts = tt[np.isfinite(tt)]
    if finite_tts.size == 0:
        t_lo = t_hi = float("inf")
    else:
        t_lo, t_hi = np.quantile(finite_tts, [alpha / 2, 1 - alpha / 2])
        # If a substantial bootstrap fraction has p=0, the upper bound is unbounded.
        if np.mean(~np.isfinite(tt)) >= alpha / 2:
            t_hi = float("inf")
    return {
        "p_success": p,
        "p_success_ci_low": float(p_lo),
        "p_success_ci_high": float(p_hi),
        "mean_run_time_s": run_time,
        "tts": tts_point,
        "tts_ci_low": float(t_lo),
        "tts_ci_high": float(t_hi),
    }



def hierarchical_success_tts(
    df: pd.DataFrame,
    confidence_tts: float = 0.99,
    confidence_ci: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """Hierarchical bootstrap for success probability and three TTS budgets.

    Resamples problem instances first and trajectories second. Reported TTS axes:
    throughput-normalized wall clock, global ticks, and spin-update opportunities.
    """
    need = ["instance_seed", "success", "runtime_s_per_trajectory", "steps", "update_opportunities"]
    work = df[need].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return {}
    clusters = list(work.instance_seed.unique())
    arr = {c: work.loc[work.instance_seed == c, need[1:]].to_numpy(float) for c in clusters}

    def calc(x):
        p = float(x[:,0].mean())
        return (
            p,
            _tts_from_p(float(x[:,1].mean()), p, confidence_tts),
            _tts_from_p(float(x[:,2].mean()), p, confidence_tts),
            _tts_from_p(float(x[:,3].mean()), p, confidence_tts),
        )

    point = calc(work[need[1:]].to_numpy(float))
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot,4), dtype=float)
    nc = len(clusters)
    for b in range(n_boot):
        pieces=[]
        for ci in rng.integers(0,nc,size=nc):
            a=arr[clusters[ci]]
            pieces.append(a[rng.integers(0,len(a),size=len(a))])
        boots[b]=calc(np.concatenate(pieces,axis=0))
    alpha=1-confidence_ci
    out={
        "p_success":point[0],
        "tts_wallclock_s":point[1],
        "tts_ticks":point[2],
        "tts_update_opportunities":point[3],
    }
    names=["p_success","tts_wallclock_s","tts_ticks","tts_update_opportunities"]
    for j,name in enumerate(names):
        finite=boots[:,j][np.isfinite(boots[:,j])]
        if finite.size==0:
            lo=hi=float("inf")
        else:
            lo,hi=np.quantile(finite,[alpha/2,1-alpha/2])
            if name.startswith("tts") and np.mean(~np.isfinite(boots[:,j])) >= alpha/2:
                hi=float("inf")
        out[name+"_ci_low"]=float(lo); out[name+"_ci_high"]=float(hi)
    out["mean_run_time_s"]=float(work.runtime_s_per_trajectory.mean())
    return out


def aggregate_paper_metrics(
    runs: pd.DataFrame,
    confidence_tts: float = 0.99,
    confidence_ci: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate run-level benchmark rows into publication-ready method metrics."""
    rows = []
    for mi, (method, dfm) in enumerate(runs.groupby("method", sort=False)):
        e, elo, ehi = hierarchical_bootstrap_ci(
            dfm, "best_energy", "instance_seed", np.mean, confidence_ci, n_boot, seed + 17 * mi
        )
        q, qlo, qhi = hierarchical_bootstrap_ci(
            dfm, "mean_q", "instance_seed", np.mean, confidence_ci, n_boot, seed + 17 * mi + 1
        )
        o, olo, ohi = hierarchical_bootstrap_ci(
            dfm, "mean_O", "instance_seed", np.mean, confidence_ci, n_boot, seed + 17 * mi + 2
        )
        st = hierarchical_success_tts(dfm, confidence_tts, confidence_ci, n_boot, seed + 17 * mi + 3)
        row = {
            "method": method,
            "n_trajectories": int(len(dfm)),
            "n_instances": int(dfm["instance_seed"].nunique()),
            "best_energy_mean": e,
            "best_energy_ci_low": elo,
            "best_energy_ci_high": ehi,
            "mean_q": q,
            "mean_q_ci_low": qlo,
            "mean_q_ci_high": qhi,
            "mean_O": o,
            "mean_O_ci_low": olo,
            "mean_O_ci_high": ohi,
            "mean_update_opportunities": float(dfm["update_opportunities"].mean()),
            "mean_restarts": float(dfm["restarts"].mean()),
            **st,
        }
        if "energy_gap_to_global_optimum" in dfm and dfm["energy_gap_to_global_optimum"].notna().any():
            g, glo, ghi = hierarchical_bootstrap_ci(
                dfm, "energy_gap_to_global_optimum", "instance_seed", np.mean,
                confidence_ci, n_boot, seed + 17 * mi + 4
            )
            row["global_optimum_gap_mean"] = g
            row["global_optimum_gap_ci_low"] = glo
            row["global_optimum_gap_ci_high"] = ghi
            row["exact_optimum_hit_rate"] = float(
                np.mean(dfm["energy_gap_to_global_optimum"].to_numpy(float) <= dfm["target_atol"].to_numpy(float))
            )
        if "relative_energy_gap_percent" in dfm and dfm["relative_energy_gap_percent"].notna().any():
            rg, rglo, rghi = hierarchical_bootstrap_ci(
                dfm, "relative_energy_gap_percent", "instance_seed", np.mean,
                confidence_ci, n_boot, seed + 17 * mi + 5
            )
            row["relative_global_optimum_gap_percent_mean"] = rg
            row["relative_global_optimum_gap_percent_ci_low"] = rglo
            row["relative_global_optimum_gap_percent_ci_high"] = rghi
        if "route_gap_to_global_optimum" in dfm and dfm["route_gap_to_global_optimum"].notna().any():
            rgap, rgaplo, rgaphi = hierarchical_bootstrap_ci(
                dfm, "route_gap_to_global_optimum", "instance_seed", np.mean,
                confidence_ci, n_boot, seed + 17 * mi + 6
            )
            row["route_gap_to_global_optimum_mean"] = rgap
            row["route_gap_to_global_optimum_ci_low"] = rgaplo
            row["route_gap_to_global_optimum_ci_high"] = rgaphi
        if "first_hit_tick" in dfm:
            hit = dfm.loc[dfm["first_hit_tick"] >= 0, "first_hit_tick"]
            row["median_first_hit_tick_successes"] = float(hit.median()) if len(hit) else float("nan")
        if "first_hit_update_ops" in dfm:
            hit = dfm.loc[dfm["first_hit_update_ops"] >= 0, "first_hit_update_ops"]
            row["median_first_hit_update_ops_successes"] = float(hit.median()) if len(hit) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_instance_delta_bootstrap(
    runs: pd.DataFrame,
    reference_method: str,
    value_col: str = "best_energy",
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap differences of instance-level means vs a reference method."""
    means = runs.groupby(["instance_seed", "method"], as_index=False)[value_col].mean()
    pivot = means.pivot(index="instance_seed", columns="method", values=value_col)
    if reference_method not in pivot.columns:
        raise ValueError(f"reference_method={reference_method!r} not found")
    out = []
    rng = np.random.default_rng(seed)
    for method in pivot.columns:
        if method == reference_method:
            continue
        pair = pivot[[method, reference_method]].dropna()
        d = pair[method].to_numpy(float) - pair[reference_method].to_numpy(float)
        if len(d) == 0:
            continue
        boots = np.empty(n_boot)
        for b in range(n_boot):
            boots[b] = np.mean(d[rng.integers(0, len(d), size=len(d))])
        alpha = 1 - confidence
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        out.append({
            "method": method,
            "reference_method": reference_method,
            "metric": value_col,
            "delta_method_minus_reference": float(d.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_paired_instances": int(len(d)),
            "win_rate_vs_reference": float(np.mean(d < 0.0)) if "energy" in value_col else float(np.mean(d > 0.0)),
        })
    return pd.DataFrame(out)
