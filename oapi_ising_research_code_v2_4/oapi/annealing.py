from __future__ import annotations
import math
import torch
from .config import AnnealConfig


def schedule_values(age: torch.Tensor, cfg: AnnealConfig, mode: str | None = None):
    """Return beta, eta for each batch member from its schedule age."""
    mode = mode or cfg.mode
    if mode == "fixed":
        beta = torch.full_like(age, cfg.beta_max, dtype=torch.float32)
        eta = torch.full_like(age, cfg.eta_min, dtype=torch.float32)
        return beta, eta

    x = torch.clamp(age.to(torch.float32) / max(cfg.cycle_steps, 1), 0.0, 1.0)
    if mode == "monotonic":
        g = x
    else:
        # cosine, periodic_restart, event_restart share the cosine envelope
        g = 0.5 * (1.0 - torch.cos(math.pi * x))
    beta = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * g
    eta = cfg.eta_max - (cfg.eta_max - cfg.eta_min) * g
    return beta, eta
