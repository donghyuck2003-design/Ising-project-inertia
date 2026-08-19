from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from typing import Dict
import numpy as np
import pandas as pd

from .config import SolverConfig
from .solver import IsingSolver


def clone_cfg(cfg: SolverConfig) -> SolverConfig:
    return deepcopy(cfg)


def method_config(base: SolverConfig, method: str) -> SolverConfig:
    c = clone_cfg(base)
    cc, ac = c.controller, c.anneal
    if method == "par0":
        cc.xi_mode = "none"; cc.adaptive_q = False; cc.q_fixed = 1.0; ac.mode = "monotonic"
    elif method == "fixed_pimi":
        cc.xi_mode = "fixed"; cc.adaptive_q = False; cc.q_fixed = 1.0; ac.mode = "monotonic"
    elif method == "fixed_partial":
        cc.xi_mode = "none"; cc.adaptive_q = False; ac.mode = "monotonic"
    elif method == "heuristic_xi":
        cc.xi_mode = "heuristic"; cc.adaptive_q = False; cc.q_fixed = 1.0; ac.mode = "monotonic"
    elif method == "momentum_xi":
        cc.xi_mode = "momentum"; cc.adaptive_q = False; cc.q_fixed = 1.0; ac.mode = "monotonic"
    elif method == "adam_xi":
        cc.xi_mode = "adam"; cc.adaptive_q = False; cc.q_fixed = 1.0; ac.mode = "monotonic"
    elif method == "adamw_xi":
        cc.xi_mode = "adamw"; cc.adaptive_q = False; cc.q_fixed = 1.0; cc.clip_dxi = False; ac.mode = "monotonic"
    elif method == "adamw_clip":
        cc.xi_mode = "adamw"; cc.adaptive_q = False; cc.q_fixed = 1.0; cc.clip_dxi = True; ac.mode = "monotonic"
    elif method == "adamw_rms":
        cc.xi_mode = "adamw"; cc.adaptive_q = False; cc.q_fixed = 1.0; cc.clip_dxi = True; cc.controller_rms_norm = True; ac.mode = "monotonic"
    elif method == "adaptive_q":
        cc.xi_mode = "none"; cc.adaptive_q = True; ac.mode = "monotonic"
    elif method == "joint":
        cc.xi_mode = "adamw"; cc.adaptive_q = True; ac.mode = "monotonic"
    elif method == "joint_restart":
        cc.xi_mode = "adamw"; cc.adaptive_q = True; ac.mode = "event_restart"
    else:
        raise ValueError(f"Unknown method: {method}")
    return c


def result_rows(result, method: str, instance_seed: int, solver_seed: int):
    rows = []
    for b in range(len(result.best_energy)):
        rows.append({
            "method": method,
            "instance_seed": instance_seed,
            "solver_seed": solver_seed,
            "batch": b,
            "best_energy": float(result.best_energy[b]),
            "final_energy": float(result.final_energy[b]),
            "update_opportunities": int(result.update_opportunities[b]),
            "restarts": int(result.restarts[b]),
            "first_hit_tick": int(result.first_hit_tick[b]),
            "first_hit_update_ops": int(result.first_hit_update_ops[b]),
            "runtime_s_total_batch": float(result.runtime_s),
            "mean_q": float(result.history["q"][:, b].mean()),
            "mean_O": float(result.history["O"][:, b].mean()),
            "mean_xi": float(result.history["xi_mean"][:, b].mean()),
            "clip_rate": float(result.history["clip_rate"][:, b].mean()),
        })
    return rows
