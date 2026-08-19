from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import torch

@dataclass
class IsingProblem:
    J: torch.Tensor
    h: torch.Tensor
    name: str = "ising"
    metadata: Optional[Dict[str, Any]] = None

    @property
    def n(self) -> int:
        return int(self.h.numel())

    def to(self, device=None, dtype=None) -> "IsingProblem":
        return IsingProblem(
            self.J.to(device=device, dtype=dtype),
            self.h.to(device=device, dtype=dtype),
            self.name,
            dict(self.metadata or {}),
        )


def _symmetrize_zero_diag(J: np.ndarray) -> np.ndarray:
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J


def normalize_ising(J: np.ndarray, h: np.ndarray, mode: str = "field_rms") -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale J,h by one positive scalar. Optimum states are unchanged."""
    if mode in (None, "none"):
        return J, h, 1.0
    if mode == "field_rms":
        # Expected random-spin coupling-field RMS plus bias RMS.
        row_power = np.sum(J * J, axis=1)
        scale = float(np.sqrt(np.mean(row_power + h * h)))
    elif mode == "max_row_l1":
        scale = float(np.max(np.sum(np.abs(J), axis=1) + np.abs(h)))
    elif mode == "spectral":
        scale = float(np.max(np.abs(np.linalg.eigvalsh(J))))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    scale = max(scale, 1e-12)
    return J / scale, h / scale, scale


def make_er_maxcut(
    n: int = 128,
    p: float = 0.25,
    seed: int = 0,
    weighted: bool = False,
    normalize: str = "field_rms",
) -> IsingProblem:
    """Dense Erdős–Rényi Max-Cut encoded as Ising minimization.

    Cut maximization: C = sum_{i<j} w_ij (1-s_i s_j)/2.
    Ignoring a constant, minimizing H uses J_ij = -w_ij/2 under
    H=-1/2 s^T J s - h^T s.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < p
    mask = np.triu(mask, 1)
    if weighted:
        W = rng.uniform(0.5, 1.5, size=(n, n)) * mask
    else:
        W = mask.astype(np.float64)
    W = W + W.T
    J = -0.5 * W
    h = np.zeros(n, dtype=np.float64)
    J, h, scale = normalize_ising(J, h, normalize)
    return IsingProblem(
        torch.tensor(J), torch.tensor(h), "er_maxcut",
        {"n": n, "p": p, "seed": seed, "weighted": weighted, "scale": scale, "W": W},
    )


def make_sk(
    n: int = 128,
    seed: int = 0,
    field_std: float = 0.0,
    normalize: str = "field_rms",
) -> IsingProblem:
    """Sherrington–Kirkpatrick-like signed dense random Ising instance."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.normal(0.0, 1.0 / math.sqrt(n), size=(n, n)), 1)
    J = upper + upper.T
    h = rng.normal(0.0, field_std, size=n) if field_std > 0 else np.zeros(n)
    J, h, scale = normalize_ising(J, h, normalize)
    return IsingProblem(torch.tensor(J), torch.tensor(h), "sk", {"n": n, "seed": seed, "scale": scale})


def make_signed_er(
    n: int = 128,
    p: float = 0.25,
    seed: int = 0,
    normalize: str = "field_rms",
) -> IsingProblem:
    rng = np.random.default_rng(seed)
    mask = np.triu(rng.random((n, n)) < p, 1)
    signs = rng.choice([-1.0, 1.0], size=(n, n))
    upper = mask * signs
    J = upper + upper.T
    h = np.zeros(n)
    J, h, scale = normalize_ising(J, h, normalize)
    return IsingProblem(torch.tensor(J), torch.tensor(h), "signed_er", {"n": n, "p": p, "seed": seed, "scale": scale})


def qubo_to_ising(Q: np.ndarray, linear: Optional[np.ndarray] = None, normalize: str = "field_rms") -> IsingProblem:
    """Convert E(x)=x^T Q x + linear^T x, x in {0,1}, to Ising.

    Uses x=(s+1)/2 and H=-1/2 s^T J s-h^T s+const.
    Q is symmetrized. Diagonal quadratic terms become constants + linear fields
    in the spin representation, so J's diagonal is removed before dynamics.
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q = 0.5 * (Q + Q.T)
    n = Q.shape[0]
    c = np.zeros(n, dtype=np.float64) if linear is None else np.asarray(linear, dtype=np.float64)
    ones = np.ones(n)
    h = -0.5 * (Q @ ones + c)
    J = -0.5 * Q
    np.fill_diagonal(J, 0.0)
    J, h, scale = normalize_ising(J, h, normalize)
    return IsingProblem(torch.tensor(J), torch.tensor(h), "qubo", {"Q": Q, "linear": c, "scale": scale})


def random_euclidean_tsp(n_cities: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_cities, 2))
    delta = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.sum(delta * delta, axis=-1))
    return coords, D


def build_tsp_qubo(D: np.ndarray, A: float = 4.0, B: float = 1.0) -> np.ndarray:
    """Build symmetric Q for the proposal's N^2 full-resident TSP encoding.

    E = A*sum_i(sum_p x_ip-1)^2 + A*sum_p(sum_i x_ip-1)^2
        + B*sum_p,sum_ij d_ij x_i,p x_j,p+1.
    Q is defined so x^T Q x equals this energy up to an additive constant.
    """
    D = np.asarray(D, dtype=np.float64)
    N = D.shape[0]
    M = N * N
    Q = np.zeros((M, M), dtype=np.float64)

    def idx(i: int, p: int) -> int:
        return i * N + p

    # Each variable appears in one city constraint and one position constraint.
    for i in range(N):
        for p in range(N):
            Q[idx(i, p), idx(i, p)] += -2.0 * A

    # Pair coefficient in the squared constraint is 2A. In x^T Q x,
    # symmetric off-diagonal entries contribute 2*Q_ab*x_a*x_b, hence Q_ab += A.
    for i in range(N):
        for p in range(N):
            for q in range(p + 1, N):
                a, b = idx(i, p), idx(i, q)
                Q[a, b] += A; Q[b, a] += A
    for p in range(N):
        for i in range(N):
            for j in range(i + 1, N):
                a, b = idx(i, p), idx(j, p)
                Q[a, b] += A; Q[b, a] += A

    # Tour distance, cyclic position p -> p+1.
    for p in range(N):
        pn = (p + 1) % N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                a, b = idx(i, p), idx(j, pn)
                val = 0.5 * B * D[i, j]
                Q[a, b] += val
                Q[b, a] += val
    return Q


def make_tsp_ising(
    n_cities: int = 10,
    seed: int = 0,
    A: float = 4.0,
    B: float = 1.0,
    normalize: str = "field_rms",
) -> Tuple[IsingProblem, np.ndarray, np.ndarray]:
    coords, D = random_euclidean_tsp(n_cities, seed)
    Q = build_tsp_qubo(D, A=A, B=B)
    prob = qubo_to_ising(Q, normalize=normalize)
    prob.name = "tsp"
    prob.metadata = {**(prob.metadata or {}), "n_cities": n_cities, "seed": seed, "A": A, "B": B, "coords": coords, "D": D}
    return prob, coords, D


def tsp_decode(spins: np.ndarray, n_cities: int, D: np.ndarray) -> Dict[str, Any]:
    x = ((np.asarray(spins).reshape(n_cities, n_cities) + 1) // 2).astype(int)
    row_sums = x.sum(axis=1)
    col_sums = x.sum(axis=0)
    violation = int(np.abs(row_sums - 1).sum() + np.abs(col_sums - 1).sum())
    feasible = violation == 0
    tour = None
    distance = float("nan")
    if feasible:
        tour = [int(np.argmax(x[:, p])) for p in range(n_cities)]
        distance = float(sum(D[tour[p], tour[(p + 1) % n_cities]] for p in range(n_cities)))
    return {"feasible": feasible, "constraint_violation": violation, "tour": tour, "tour_distance": distance, "x": x}


def _ising_energy_np(J: np.ndarray, h: np.ndarray, s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float(-0.5 * s @ np.asarray(J, dtype=np.float64) @ s - np.asarray(h, dtype=np.float64) @ s)


def make_planted_ising(
    n: int = 128,
    p: float = 0.25,
    seed: int = 0,
    field_strength: float = 0.20,
    weight_low: float = 0.5,
    weight_high: float = 1.5,
    normalize: str = "field_rms",
) -> IsingProblem:
    """Generate an Ising instance whose exact global optimum is known by construction.

    A random planted state s* is sampled first. Positive edge weights are then
    gauge-transformed as J_ij = w_ij s*_i s*_j, and the local field is aligned
    with s*. Under H=-1/2 s^T J s-h^T s, every pair term and every field term is
    individually minimized by s*. With field_strength > 0, any state different
    from s* pays a strictly positive field penalty, so s* is a strict global
    minimum (not an optimum estimated by the solver).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    if field_strength <= 0.0:
        raise ValueError("field_strength must be > 0 to make the planted optimum strict")
    if weight_low <= 0.0 or weight_high < weight_low:
        raise ValueError("Require 0 < weight_low <= weight_high")

    rng = np.random.default_rng(seed)
    s_star = rng.choice([-1.0, 1.0], size=n).astype(np.float64)
    mask = np.triu(rng.random((n, n)) < p, 1)
    w_upper = rng.uniform(weight_low, weight_high, size=(n, n)) * mask
    W = w_upper + w_upper.T
    J = W * np.outer(s_star, s_star)
    np.fill_diagonal(J, 0.0)
    h = float(field_strength) * s_star

    J, h, scale = normalize_ising(J, h, normalize)
    exact_e = _ising_energy_np(J, h, s_star)
    return IsingProblem(
        torch.tensor(J),
        torch.tensor(h),
        "planted_ising",
        {
            "n": n,
            "p": p,
            "seed": seed,
            "scale": scale,
            "planted_state": s_star.astype(np.int8),
            "exact_optimum_state": s_star.astype(np.int8),
            "exact_optimum_energy": exact_e,
            "exact_optimum_source": "planted_by_construction",
            "strict_global_optimum": True,
            "field_strength": float(field_strength),
            "weight_low": float(weight_low),
            "weight_high": float(weight_high),
        },
    )


def make_planted_tsp_ising(
    n_cities: int = 10,
    seed: int = 0,
    edge_cost: float = 1.0,
    nonoptimal_gap: float = 0.25,
    nonoptimal_jitter: float = 0.10,
    A: Optional[float] = None,
    B: float = 1.0,
    normalize: str = "field_rms",
) -> Tuple[IsingProblem, np.ndarray]:
    """TSP QUBO with an exact optimal route planted before solver execution.

    The planted Hamiltonian cycle has edge cost ``edge_cost``. Every non-cycle
    edge has cost at least ``edge_cost + nonoptimal_gap``. Therefore every
    genuinely different Hamiltonian cycle is longer than the planted cycle
    (the reverse/rotation representations of the same cycle are equivalent
    optima). The one-hot penalty is selected large enough that infeasible QUBO
    states cannot beat the feasible planted route.
    """
    if n_cities < 3:
        raise ValueError("n_cities must be >= 3")
    if edge_cost <= 0.0 or nonoptimal_gap <= 0.0 or nonoptimal_jitter < 0.0:
        raise ValueError("edge_cost and nonoptimal_gap must be > 0; jitter must be >= 0")
    if B <= 0.0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)
    route = rng.permutation(n_cities).astype(int)
    D = edge_cost + nonoptimal_gap + nonoptimal_jitter * rng.random((n_cities, n_cities))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    for k in range(n_cities):
        i = int(route[k]); j = int(route[(k + 1) % n_cities])
        D[i, j] = D[j, i] = float(edge_cost)

    exact_route_distance = float(n_cities * edge_cost)
    if A is None:
        # Original TSP energy is non-negative. Any constraint violation costs
        # at least A, so A > exact feasible objective is a conservative guard.
        A = 2.0 * B * exact_route_distance + 1.0
    if A <= B * exact_route_distance:
        raise ValueError(
            "A must exceed B * planted route distance so infeasible states cannot undercut the planted route"
        )

    Q = build_tsp_qubo(D, A=A, B=B)
    prob = qubo_to_ising(Q, normalize=normalize)
    prob.name = "planted_tsp"

    x = np.zeros((n_cities, n_cities), dtype=np.int8)
    for pos, city in enumerate(route):
        x[int(city), pos] = 1
    s_star = (2 * x.reshape(-1) - 1).astype(np.int8)
    Jn = prob.J.detach().cpu().numpy().astype(np.float64)
    hn = prob.h.detach().cpu().numpy().astype(np.float64)
    exact_e = _ising_energy_np(Jn, hn, s_star)

    prob.metadata = {
        **(prob.metadata or {}),
        "n_cities": int(n_cities),
        "seed": int(seed),
        "A": float(A),
        "B": float(B),
        "D": D,
        "planted_route": route,
        "exact_optimum_route": route,
        "exact_optimum_route_distance": exact_route_distance,
        "exact_optimum_state": s_star,
        "exact_optimum_energy": exact_e,
        "exact_optimum_source": "planted_tsp_by_construction",
        "strict_route_gap": float(nonoptimal_gap),
        "edge_cost": float(edge_cost),
        "nonoptimal_gap": float(nonoptimal_gap),
        "nonoptimal_jitter": float(nonoptimal_jitter),
    }
    return prob, D
