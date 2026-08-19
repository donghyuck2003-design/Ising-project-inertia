from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from .problems import IsingProblem


def exact_ground_state(problem: IsingProblem, max_spins: int = 24, chunk_size: int = 1 << 18) -> Dict[str, Any]:
    """Exhaustively enumerate all 2^N states for small Ising instances.

    Intended only for small validation instances. For larger N use a clearly
    labeled best-known reference generated on a disjoint reference budget.
    """
    n = problem.n
    if n > max_spins:
        raise ValueError(f"Exact enumeration disabled for n={n}; max_spins={max_spins}")
    J = problem.J.detach().cpu().numpy().astype(np.float64)
    h = problem.h.detach().cpu().numpy().astype(np.float64)
    total = 1 << n
    best_e = float("inf")
    best_s = None
    bitpos = np.arange(n, dtype=np.uint64)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        ids = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = ((ids >> bitpos[None, :]) & 1).astype(np.float64)
        s = 2.0 * bits - 1.0
        e = -0.5 * np.einsum("bi,ij,bj->b", s, J, s, optimize=True) - s @ h
        k = int(np.argmin(e))
        if float(e[k]) < best_e:
            best_e = float(e[k])
            best_s = s[k].astype(np.int8)
    return {"energy": best_e, "state": best_s, "n_states": total, "reference_type": "exact"}
