from __future__ import annotations
import numpy as np


def spectral_radius_linearized(J, beta: float, q: float, xi: float, d_diag=None) -> float:
    """Mechanistic local spectral approximation from the proposal.

    A(q,xi)=(1-q)I + q D (beta J + xi I).
    D defaults to I, giving a conservative/easy-to-compare reference map rather
    than a formal stochastic convergence certificate.
    """
    J = np.asarray(J, dtype=np.float64)
    n = J.shape[0]
    D = np.eye(n) if d_diag is None else np.diag(np.asarray(d_diag, dtype=np.float64))
    A = (1.0 - q) * np.eye(n) + q * D @ (beta * J + xi * np.eye(n))
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def spectral_edges(J):
    vals = np.linalg.eigvalsh(np.asarray(J, dtype=np.float64))
    return {"lambda_min": float(vals[0]), "lambda_max": float(vals[-1]), "spectral_radius_J": float(np.max(np.abs(vals)))}
