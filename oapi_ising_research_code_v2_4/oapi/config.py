from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional

XiMode = Literal["none", "fixed", "heuristic", "momentum", "adam", "adamw"]
AnnealMode = Literal["fixed", "monotonic", "cosine", "periodic_restart", "event_restart"]
MaskMode = Literal["exact", "bernoulli"]

@dataclass
class ControllerConfig:
    # Spin-wise oscillation diagnostic
    rho_o: float = 0.90
    controller_rms_norm: bool = True

    # e_i = a * Obar_i - b * F_i - c * R
    a: float = 1.0
    b: float = 0.50
    c: float = 0.0  # proposal recommends starting with c=0

    # inertia controller
    xi_mode: XiMode = "adamw"
    xi0: float = 0.0
    xi_fixed: float = 0.20
    xi_max: float = 1.50
    alpha_xi: float = 0.02
    beta1: float = 0.90
    beta2: float = 0.99
    lambda_xi: float = 0.002
    dxi_max: float = 0.05
    clip_dxi: bool = True

    # direct heuristic baseline
    heuristic_alpha: float = 0.80
    heuristic_gamma: float = 0.50

    # adaptive parallelism q(t)
    adaptive_q: bool = True
    q_init: float = 1.0
    q_fixed: float = 1.0
    q_min: float = 0.125
    q_step: float = 0.125
    slow_interval: int = 25
    o_low: float = 0.05
    o_high: float = 0.15
    dwell_low: int = 4
    dwell_high: int = 2

    eps: float = 1e-8

@dataclass
class AnnealConfig:
    mode: AnnealMode = "event_restart"
    beta_min: float = 0.20
    beta_max: float = 2.00
    eta_max: float = 0.50
    eta_min: float = 0.05
    cycle_steps: int = 1000
    stall_steps: int = 300
    inertia_release: float = 0.50
    # Prevents an event restart from firing every tick while no new best is found.
    # None -> use stall_steps.
    restart_cooldown: Optional[int] = None

@dataclass
class SolverConfig:
    steps: int = 3000
    batch_size: int = 16
    log_every: int = 10
    mask_mode: MaskMode = "exact"
    noise_distribution: Literal["normal", "uniform"] = "normal"
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = 1234
    # Optional target for exact per-tick first-hit tracking.
    # Leave None when only final/best energy is required.
    target_energy: Optional[float] = None
    target_atol: float = 1e-6

    controller: ControllerConfig = None  # type: ignore
    anneal: AnnealConfig = None  # type: ignore

    def __post_init__(self):
        if self.controller is None:
            self.controller = ControllerConfig()
        if self.anneal is None:
            self.anneal = AnnealConfig()

    def to_dict(self):
        return asdict(self)
