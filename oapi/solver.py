from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import numpy as np
import torch

from .config import SolverConfig
from .problems import IsingProblem
from .annealing import schedule_values

@dataclass
class SolverResult:
    best_energy: np.ndarray
    final_energy: np.ndarray
    best_state: np.ndarray
    final_state: np.ndarray
    runtime_s: float
    update_opportunities: np.ndarray
    restarts: np.ndarray
    first_hit_tick: np.ndarray
    first_hit_update_ops: np.ndarray
    summary: Dict[str, Any]
    history: Dict[str, np.ndarray]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return dev


def _energy(s: torch.Tensor, J: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    field = s @ J.T
    return -0.5 * torch.sum(s * field, dim=1) - torch.sum(s * h, dim=1)


def _exact_random_mask(q: torch.Tensor, M: int, generator: torch.Generator) -> torch.Tensor:
    B = q.numel()
    p = torch.round(q * M).to(torch.long).clamp(1, M)
    if bool(torch.all(p == M)):
        return torch.ones((B, M), dtype=torch.bool, device=q.device)
    scores = torch.rand((B, M), device=q.device, generator=generator)
    order = torch.argsort(scores, dim=1)
    rank = torch.empty_like(order)
    arange = torch.arange(M, device=q.device).expand(B, M)
    rank.scatter_(1, order, arange)
    return rank < p[:, None]


def _bernoulli_mask(q: torch.Tensor, M: int, generator: torch.Generator) -> torch.Tensor:
    mask = torch.rand((q.numel(), M), device=q.device, generator=generator) < q[:, None]
    # Avoid zero-update batch members at very small q.
    zero = ~mask.any(dim=1)
    if zero.any():
        idx = torch.randint(M, (int(zero.sum()),), device=q.device, generator=generator)
        rows = torch.where(zero)[0]
        mask[rows, idx] = True
    return mask


class IsingSolver:
    """Batched stochastic Ising solver implementing the proposal's OAPI controller."""

    def __init__(self, problem: IsingProblem, cfg: SolverConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
        self.problem = problem.to(self.device, self.dtype)
        self.J, self.h = self.problem.J, self.problem.h
        if self.J.shape != (self.problem.n, self.problem.n):
            raise ValueError("J must be square and match h")
        if not torch.allclose(self.J, self.J.T, atol=1e-5, rtol=1e-5):
            raise ValueError("J must be symmetric")

    def run(self, initial_state: Optional[torch.Tensor] = None) -> SolverResult:
        cfg = self.cfg
        cc = cfg.controller
        ac = cfg.anneal
        B, M = cfg.batch_size, self.problem.n

        gen = torch.Generator(device=self.device)
        gen.manual_seed(cfg.seed)

        if initial_state is None:
            s = torch.where(
                torch.rand((B, M), device=self.device, generator=gen) < 0.5,
                -torch.ones((B, M), device=self.device, dtype=self.dtype),
                torch.ones((B, M), device=self.device, dtype=self.dtype),
            )
        else:
            s = initial_state.to(self.device, self.dtype).clone()
            if s.shape != (B, M):
                raise ValueError(f"initial_state must have shape {(B, M)}")

        prev1 = s.clone(); prev2 = s.clone()
        obar = torch.zeros_like(s)
        m1 = torch.zeros_like(s); m2 = torch.zeros_like(s)
        xi = torch.full_like(s, cc.xi0)
        if cc.xi_mode == "fixed":
            xi.fill_(cc.xi_fixed)
        if cc.xi_mode == "none":
            xi.zero_()

        q0 = cc.q_init if cc.adaptive_q else cc.q_fixed
        q = torch.full((B,), q0, device=self.device, dtype=self.dtype)
        high_count = torch.zeros(B, device=self.device, dtype=torch.long)
        low_count = torch.zeros(B, device=self.device, dtype=torch.long)

        E = _energy(s, self.J, self.h)
        best_E = E.clone(); best_s = s.clone()
        last_best = torch.zeros(B, device=self.device, dtype=torch.long)
        last_restart = torch.full((B,), -10**9, device=self.device, dtype=torch.long)
        restarts = torch.zeros(B, device=self.device, dtype=torch.long)
        update_ops = torch.zeros(B, device=self.device, dtype=torch.long)
        cycle_age = torch.zeros(B, device=self.device, dtype=torch.long)
        first_hit_tick = torch.full((B,), -1, device=self.device, dtype=torch.long)
        first_hit_update_ops = torch.full((B,), -1, device=self.device, dtype=torch.long)
        if cfg.target_energy is not None:
            initial_hit = best_E <= float(cfg.target_energy) + cfg.target_atol
            first_hit_tick = torch.where(initial_hit, torch.zeros_like(first_hit_tick), first_hit_tick)
            first_hit_update_ops = torch.where(initial_hit, torch.zeros_like(first_hit_update_ops), first_hit_update_ops)

        hist = {k: [] for k in ["t", "energy", "best_energy", "O", "q", "xi_mean", "xi_max", "beta", "eta", "dxi_abs_mean", "clip_rate", "stalled"]}

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        beta = torch.full((B,), ac.beta_min, device=self.device, dtype=self.dtype)
        eta = torch.full((B,), ac.eta_max, device=self.device, dtype=self.dtype)

        for t in range(1, cfg.steps + 1):
            # 1) Local field + online diagnostics, from current state s(t)
            I = s @ self.J.T + self.h
            if cc.controller_rms_norm:
                rms = torch.sqrt(torch.mean(I * I, dim=1, keepdim=True) + cc.eps)
                Ictrl = I / rms
            else:
                Ictrl = I

            if t >= 3:
                O2 = ((s == prev2) & (s != prev1)).to(self.dtype)
            else:
                O2 = torch.zeros_like(s)
            obar = cc.rho_o * obar + (1.0 - cc.rho_o) * O2
            O = torch.mean(obar, dim=1)

            C = s * torch.tanh(beta[:, None] * Ictrl)
            F = torch.clamp(-C, min=0.0)
            stalled = (t - last_best) >= ac.stall_steps

            # 2) FAST LOOP: inertia controller
            e = cc.a * obar - cc.b * F - cc.c * stalled.to(self.dtype)[:, None]
            dxi = torch.zeros_like(xi)
            clip_rate = torch.zeros(B, device=self.device, dtype=self.dtype)

            if cc.xi_mode == "heuristic":
                new_xi = cc.xi0 + cc.heuristic_alpha * obar - cc.heuristic_gamma * F
                dxi = new_xi - xi
                xi = torch.clamp(new_xi, 0.0, cc.xi_max)
            elif cc.xi_mode == "momentum":
                m1 = cc.beta1 * m1 + (1.0 - cc.beta1) * e
                dxi = cc.alpha_xi * m1
                if cc.clip_dxi:
                    clip_rate = (torch.abs(dxi) > cc.dxi_max).to(self.dtype).mean(dim=1)
                    dxi = torch.clamp(dxi, -cc.dxi_max, cc.dxi_max)
                xi = torch.clamp(xi + dxi, 0.0, cc.xi_max)
            elif cc.xi_mode in ("adam", "adamw"):
                m1 = cc.beta1 * m1 + (1.0 - cc.beta1) * e
                m2 = cc.beta2 * m2 + (1.0 - cc.beta2) * (e * e)
                mhat = m1 / (1.0 - cc.beta1 ** t)
                vhat = m2 / (1.0 - cc.beta2 ** t)
                dxi = cc.alpha_xi * mhat / (torch.sqrt(vhat) + cc.eps)
                if cc.clip_dxi:
                    clip_rate = (torch.abs(dxi) > cc.dxi_max).to(self.dtype).mean(dim=1)
                    dxi = torch.clamp(dxi, -cc.dxi_max, cc.dxi_max)
                decay = (1.0 - cc.lambda_xi) if cc.xi_mode == "adamw" else 1.0
                xi = torch.clamp(decay * xi + dxi, 0.0, cc.xi_max)
            elif cc.xi_mode == "fixed":
                xi.fill_(cc.xi_fixed)
            elif cc.xi_mode == "none":
                xi.zero_()
            else:
                raise ValueError(f"Unknown xi_mode={cc.xi_mode}")

            # 3) SLOW LOOP: adaptive q with hysteresis+dwell
            if cc.adaptive_q and (t % cc.slow_interval == 0):
                high = O > cc.o_high
                low = O < cc.o_low
                high_count = torch.where(high, high_count + 1, torch.zeros_like(high_count))
                low_count = torch.where(low, low_count + 1, torch.zeros_like(low_count))
                dec = high_count >= cc.dwell_high
                inc = low_count >= cc.dwell_low
                q = torch.where(dec, torch.clamp(q - cc.q_step, min=cc.q_min), q)
                q = torch.where(inc, torch.clamp(q + cc.q_step, max=1.0), q)
                high_count = torch.where(dec, torch.zeros_like(high_count), high_count)
                low_count = torch.where(inc, torch.zeros_like(low_count), low_count)
            elif not cc.adaptive_q:
                q.fill_(cc.q_fixed)

            # 4) Annealing + failure-mode-aware event restart
            cooldown = ac.stall_steps if ac.restart_cooldown is None else ac.restart_cooldown
            stable_stall = (O < cc.o_low) & stalled
            may_restart = (t - last_restart) >= cooldown
            event = stable_stall & may_restart & (ac.mode == "event_restart")
            periodic = torch.zeros_like(event)
            if ac.mode == "periodic_restart":
                periodic = cycle_age >= ac.cycle_steps
            restart = event | periodic
            if restart.any():
                cycle_age = torch.where(restart, torch.zeros_like(cycle_age), cycle_age)
                xi = torch.where(restart[:, None], ac.inertia_release * xi, xi)
                last_restart = torch.where(restart, torch.full_like(last_restart, t), last_restart)
                restarts += restart.to(torch.long)

            sched_mode = ac.mode
            beta_f, eta_f = schedule_values(cycle_age, ac, sched_mode)
            beta = beta_f.to(self.device, self.dtype)
            eta = eta_f.to(self.device, self.dtype)

            # 5) Stochastic proposal + masked commit
            if cfg.noise_distribution == "normal":
                noise = torch.randn((B, M), device=self.device, dtype=self.dtype, generator=gen)
            else:
                noise = 2.0 * torch.rand((B, M), device=self.device, dtype=self.dtype, generator=gen) - 1.0
            z = torch.tanh(beta[:, None] * I) + xi * s + eta[:, None] * noise
            proposal = torch.where(z >= 0, torch.ones_like(s), -torch.ones_like(s))
            mask = _exact_random_mask(q, M, gen) if cfg.mask_mode == "exact" else _bernoulli_mask(q, M, gen)
            new_s = torch.where(mask, proposal, s)
            update_ops += mask.sum(dim=1)

            old_s = s
            s = new_s
            prev2, prev1 = prev1, old_s
            E = _energy(s, self.J, self.h)
            improved = E < best_E
            best_E = torch.where(improved, E, best_E)
            best_s = torch.where(improved[:, None], s, best_s)
            last_best = torch.where(improved, torch.full_like(last_best, t), last_best)

            # Exact first-hit tracking for success/TTS experiments.
            if cfg.target_energy is not None:
                hit_now = (first_hit_tick < 0) & (best_E <= float(cfg.target_energy) + cfg.target_atol)
                first_hit_tick = torch.where(hit_now, torch.full_like(first_hit_tick, t), first_hit_tick)
                first_hit_update_ops = torch.where(hit_now, update_ops, first_hit_update_ops)

            cycle_age += 1

            if t == 1 or t % cfg.log_every == 0 or t == cfg.steps:
                hist["t"].append(t)
                hist["energy"].append(E.detach().cpu().numpy())
                hist["best_energy"].append(best_E.detach().cpu().numpy())
                hist["O"].append(O.detach().cpu().numpy())
                hist["q"].append(q.detach().cpu().numpy())
                hist["xi_mean"].append(xi.mean(dim=1).detach().cpu().numpy())
                hist["xi_max"].append(xi.max(dim=1).values.detach().cpu().numpy())
                hist["beta"].append(beta.detach().cpu().numpy())
                hist["eta"].append(eta.detach().cpu().numpy())
                hist["dxi_abs_mean"].append(dxi.abs().mean(dim=1).detach().cpu().numpy())
                hist["clip_rate"].append(clip_rate.detach().cpu().numpy())
                hist["stalled"].append(stalled.detach().cpu().numpy().astype(np.int8))

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - t0
        history = {k: np.asarray(v) for k, v in hist.items()}
        best_np = best_E.detach().cpu().numpy()
        final_np = E.detach().cpu().numpy()
        q_hist = history["q"]
        o_hist = history["O"]
        xi_hist = history["xi_mean"]
        summary = {
            "problem": self.problem.name,
            "n_spins": M,
            "batch_size": B,
            "steps": cfg.steps,
            "device": str(self.device),
            "runtime_s": runtime,
            "best_energy_mean": float(np.mean(best_np)),
            "best_energy_std": float(np.std(best_np)),
            "final_energy_mean": float(np.mean(final_np)),
            "mean_q": float(np.mean(q_hist)),
            "mean_O": float(np.mean(o_hist)),
            "mean_xi": float(np.mean(xi_hist)),
            "mean_update_opportunities": float(update_ops.float().mean().cpu()),
            "mean_restarts": float(restarts.float().mean().cpu()),
            "target_energy": None if cfg.target_energy is None else float(cfg.target_energy),
            "success_probability": None if cfg.target_energy is None else float((best_E <= float(cfg.target_energy) + cfg.target_atol).float().mean().cpu()),
        }
        return SolverResult(
            best_energy=best_np,
            final_energy=final_np,
            best_state=best_s.detach().cpu().numpy().astype(np.int8),
            final_state=s.detach().cpu().numpy().astype(np.int8),
            runtime_s=runtime,
            update_opportunities=update_ops.detach().cpu().numpy(),
            restarts=restarts.detach().cpu().numpy(),
            first_hit_tick=first_hit_tick.detach().cpu().numpy(),
            first_hit_update_ops=first_hit_update_ops.detach().cpu().numpy(),
            summary=summary,
            history=history,
        )
