#!/usr/bin/env python3
"""Full-resident p-bit experiment for Optimizer-Inspired Adaptive Parallel Ising (OAPI).

Research intent
---------------
This driver implements the experiment described in the attached meeting brief
without decomposing variables into graph partitions / sub-TSP groups.

Every logical variable is resident in one p-bit population:

    ER / SK / signed-ER:  M = N spins
    TSP:                  M = N_city^2 spins

Adaptive q(t) changes only the fraction of resident p-bits updated at a global
clock tick.  It never partitions the optimization variables.

The stochastic proposal uses a paper-style p-bit logistic response.  With
local Ising field I_i = sum_j J_ij s_j + h_i, inertia xi_i and annealing/noise,

    u_i(t) = beta(t) I_i(t) + xi_i(t) s_i(t) + eta(t) z_i(t)
    P[s_i(t+1)=+1] = sigmoid(2 * gain * (u_i(t) - bias))

so E[s_i(t+1)] = tanh(gain * (u_i-bias)) before masked committing.

The controller follows the report:
  * online period-2 oscillation O_i and field-conflict F_i,
  * spin-wise heuristic / momentum / Adam / AdamW inertia,
  * slow hysteretic adaptive q(t),
  * stable-stagnation-triggered annealing restart with inertia release.

This script deliberately does NOT call the old large-TSP four-stage path
(main_gp_gpu -> main_tsp_gpu -> main_merge_gpu -> main_opt_gpu), because that
architecture partitions variables and conflicts with the full-resident premise.
It does reuse the OAPI problem generators, configs, exact solver, annealing,
statistics, and TSP full-resident QUBO/decoder.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Allow execution as scripts/fullresident_oapi_experiment.py from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oapi.annealing import schedule_values
from oapi.config import SolverConfig
from oapi.exact_solvers import ExactBackendUnavailable, solve_exact_ising
from oapi.experiment_utils import method_config
from oapi.problems import (
    IsingProblem,
    make_er_maxcut,
    make_planted_ising,
    make_planted_tsp_ising,
    make_signed_er,
    make_sk,
    make_tsp_ising,
    tsp_decode,
)
from oapi.solver import SolverResult
from oapi.stability import spectral_edges, spectral_radius_linearized
from oapi.statistics import aggregate_paper_metrics, paired_instance_delta_bootstrap


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def _energy(s: torch.Tensor, J: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    # H = -1/2 s^T J s - h^T s
    field = s @ J.T
    return -0.5 * torch.sum(s * field, dim=1) - torch.sum(s * h, dim=1)


def _exact_random_mask(q: torch.Tensor, M: int, gen: torch.Generator) -> torch.Tensor:
    """Update exactly round(q*M) resident p-bits per trajectory.

    q is an update-parallelism control, NOT a variable partition.
    """
    B = q.numel()
    k = torch.round(q * M).to(torch.long).clamp(1, M)
    if bool(torch.all(k == M)):
        return torch.ones((B, M), dtype=torch.bool, device=q.device)
    scores = torch.rand((B, M), device=q.device, generator=gen)
    order = torch.argsort(scores, dim=1)
    ranks = torch.empty_like(order)
    ar = torch.arange(M, device=q.device).expand(B, M)
    ranks.scatter_(1, order, ar)
    return ranks < k[:, None]


def _bernoulli_mask(q: torch.Tensor, M: int, gen: torch.Generator) -> torch.Tensor:
    mask = torch.rand((q.numel(), M), device=q.device, generator=gen) < q[:, None]
    zero = ~mask.any(dim=1)
    if zero.any():
        rows = torch.where(zero)[0]
        idx = torch.randint(M, (int(zero.sum()),), device=q.device, generator=gen)
        mask[rows, idx] = True
    return mask


def _parse_csv_ints(text: str) -> List[int]:
    """Accept '1,2,3' or inclusive range-like '10:14' -> [10,11,12,13]."""
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) == 2:
                a, b = bits
                step = 1 if b >= a else -1
            elif len(bits) == 3:
                a, b, step = bits
                if step == 0:
                    raise ValueError("range step cannot be 0")
            else:
                raise ValueError(f"Bad integer range: {part}")
            out.extend(list(range(a, b, step)))
        else:
            out.append(int(part))
    return out


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _time_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_stability_visualizer(args: argparse.Namespace, out: Path) -> None:
    """Run scripts/visualize_fullresident_stability.py on the completed stability CSV.

    Visualization is enabled by default after the stability phase.  It uses the
    same Python interpreter as this experiment and writes figures inside the same
    timestamped result directory.  A plotting failure is recorded without losing
    the completed simulation unless --strict-visualization is requested.
    """
    if args.no_auto_visualize:
        print("[VIS] Automatic stability visualization disabled.", flush=True)
        return

    csv_path = out / "stability_map_fullresident.csv"
    if not csv_path.exists():
        msg = f"Stability CSV not found: {csv_path}"
        if args.strict_visualization:
            raise FileNotFoundError(msg)
        print(f"[VIS][WARNING] {msg}", flush=True)
        (out / "visualization_error.txt").write_text(msg + "\n", encoding="utf-8")
        return

    visualizer = Path(args.visualizer_script)
    if not visualizer.is_absolute():
        visualizer = ROOT / visualizer
    visualizer = visualizer.resolve()

    if not visualizer.exists():
        msg = f"Visualization script not found: {visualizer}"
        if args.strict_visualization:
            raise FileNotFoundError(msg)
        print(f"[VIS][WARNING] {msg}", flush=True)
        (out / "visualization_error.txt").write_text(msg + "\n", encoding="utf-8")
        return

    fig_dir = out / args.visualization_dirname
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(visualizer),
        "--csv", str(csv_path),
        "--out", str(fig_dir),
        "--dpi", str(args.visualization_dpi),
        "--formats", str(args.visualization_formats),
    ]

    print(f"[VIS] Running: {subprocess.list2cmdline(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        msg = (
            f"Visualization command failed with return code {exc.returncode}.\n"
            f"Command: {subprocess.list2cmdline(cmd)}\n"
        )
        (out / "visualization_error.txt").write_text(msg, encoding="utf-8")
        if args.strict_visualization:
            raise
        print(f"[VIS][WARNING] {msg.strip()}", flush=True)
        return


    print(f"[VIS] Figures saved under: {fig_dir}", flush=True)


# -----------------------------------------------------------------------------
# Full-resident paper-style p-bit solver
# -----------------------------------------------------------------------------

@dataclass
class PBitTransfer:
    gain: float = 1.0
    bias: float = 0.0


class FullResidentPBitSolver:
    """One full-resident stochastic p-bit population with OAPI feedback control."""

    def __init__(self, problem: IsingProblem, cfg: SolverConfig, transfer: PBitTransfer):
        self.cfg = cfg
        self.transfer = transfer
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
        gen.manual_seed(int(cfg.seed))

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

        prev1 = s.clone()
        prev2 = s.clone()

        # Online controller state.
        obar = torch.zeros_like(s)
        m1 = torch.zeros_like(s)
        m2 = torch.zeros_like(s)
        xi = torch.full_like(s, float(cc.xi0))
        if cc.xi_mode == "fixed":
            xi.fill_(float(cc.xi_fixed))
        elif cc.xi_mode == "none":
            xi.zero_()

        q0 = cc.q_init if cc.adaptive_q else cc.q_fixed
        q = torch.full((B,), float(q0), device=self.device, dtype=self.dtype)
        high_count = torch.zeros(B, device=self.device, dtype=torch.long)
        low_count = torch.zeros(B, device=self.device, dtype=torch.long)

        E = _energy(s, self.J, self.h)
        best_E = E.clone()
        best_s = s.clone()
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

        hist = {k: [] for k in [
            "t", "energy", "best_energy", "O", "q", "xi_mean", "xi_max",
            "beta", "eta", "dxi_abs_mean", "clip_rate", "stalled",
            "flip_fraction", "high_q_fraction"
        ]}

        beta = torch.full((B,), float(ac.beta_min), device=self.device, dtype=self.dtype)
        eta = torch.full((B,), float(ac.eta_max), device=self.device, dtype=self.dtype)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        for t in range(1, cfg.steps + 1):
            # -----------------------------------------------------------------
            # 1) Observe full-resident state; controller normalization does not
            #    alter the physical/problem field used by the p-bit proposal.
            # -----------------------------------------------------------------
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

            # Report definition: F_i=max(0,-s_i*tanh(beta*I_tilde_i)).
            conflict = s * torch.tanh(beta[:, None] * Ictrl)
            F = torch.clamp(-conflict, min=0.0)
            stalled = (t - last_best) >= ac.stall_steps

            # -----------------------------------------------------------------
            # 2) FAST local loop: adaptive spin-wise inertia xi_i(t).
            # -----------------------------------------------------------------
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
                m2 = cc.beta2 * m2 + (1.0 - cc.beta2) * e.square()
                mhat = m1 / (1.0 - cc.beta1 ** t)
                vhat = m2 / (1.0 - cc.beta2 ** t)
                dxi = cc.alpha_xi * mhat / (torch.sqrt(vhat) + cc.eps)
                if cc.clip_dxi:
                    clip_rate = (torch.abs(dxi) > cc.dxi_max).to(self.dtype).mean(dim=1)
                    dxi = torch.clamp(dxi, -cc.dxi_max, cc.dxi_max)
                decay = (1.0 - cc.lambda_xi) if cc.xi_mode == "adamw" else 1.0
                xi = torch.clamp(decay * xi + dxi, 0.0, cc.xi_max)
            elif cc.xi_mode == "fixed":
                xi.fill_(float(cc.xi_fixed))
            elif cc.xi_mode == "none":
                xi.zero_()
            else:
                raise ValueError(f"Unknown xi_mode={cc.xi_mode}")

            # -----------------------------------------------------------------
            # 3) SLOW global loop: adaptive update parallelism q(t).
            # -----------------------------------------------------------------
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
                q.fill_(float(cc.q_fixed))

            # -----------------------------------------------------------------
            # 4) Annealing and failure-mode-aware restart.
            #    Only stable stagnation (low O + no best-energy improvement)
            #    triggers the event restart.
            # -----------------------------------------------------------------
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

            beta_f, eta_f = schedule_values(cycle_age, ac, ac.mode)
            beta = beta_f.to(self.device, self.dtype)
            eta = eta_f.to(self.device, self.dtype)

            # -----------------------------------------------------------------
            # 5) Paper-style stochastic p-bit proposal.
            #    All M spins are resident. q(t) only masks which can commit.
            # -----------------------------------------------------------------
            if cfg.noise_distribution == "normal":
                zeta = torch.randn((B, M), device=self.device, dtype=self.dtype, generator=gen)
            else:
                zeta = 2.0 * torch.rand((B, M), device=self.device, dtype=self.dtype, generator=gen) - 1.0

            drive = beta[:, None] * I + xi * s + eta[:, None] * zeta
            logits = 2.0 * float(self.transfer.gain) * (drive - float(self.transfer.bias))
            p_plus = torch.sigmoid(logits)
            u = torch.rand((B, M), device=self.device, dtype=self.dtype, generator=gen)
            proposal = torch.where(u < p_plus, torch.ones_like(s), -torch.ones_like(s))

            if cfg.mask_mode == "exact":
                mask = _exact_random_mask(q, M, gen)
            else:
                mask = _bernoulli_mask(q, M, gen)

            old_s = s
            s = torch.where(mask, proposal, s)
            flip_fraction = (s != old_s).to(self.dtype).mean(dim=1)
            update_ops += mask.sum(dim=1)

            prev2, prev1 = prev1, old_s
            E = _energy(s, self.J, self.h)
            improved = E < best_E
            best_E = torch.where(improved, E, best_E)
            best_s = torch.where(improved[:, None], s, best_s)
            last_best = torch.where(improved, torch.full_like(last_best, t), last_best)

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
                hist["flip_fraction"].append(flip_fraction.detach().cpu().numpy())
                hist["high_q_fraction"].append((q >= 0.75).to(self.dtype).detach().cpu().numpy())

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - t0

        history = {k: np.asarray(v) for k, v in hist.items()}
        best_np = best_E.detach().cpu().numpy()
        final_np = E.detach().cpu().numpy()
        summary = {
            "problem": self.problem.name,
            "n_spins": M,
            "batch_size": B,
            "steps": cfg.steps,
            "device": str(self.device),
            "runtime_s": float(runtime),
            "best_energy_mean": float(np.mean(best_np)),
            "best_energy_std": float(np.std(best_np)),
            "final_energy_mean": float(np.mean(final_np)),
            "mean_q": float(np.mean(history["q"])),
            "mean_O": float(np.mean(history["O"])),
            "mean_xi": float(np.mean(history["xi_mean"])),
            "mean_flip_fraction": float(np.mean(history["flip_fraction"])),
            "time_at_high_q": float(np.mean(history["high_q_fraction"])),
            "mean_update_opportunities": float(update_ops.float().mean().cpu()),
            "mean_restarts": float(restarts.float().mean().cpu()),
            "target_energy": None if cfg.target_energy is None else float(cfg.target_energy),
            "success_probability": None if cfg.target_energy is None else float(
                (best_E <= float(cfg.target_energy) + cfg.target_atol).float().mean().cpu()
            ),
            "proposal_model": "paper_logistic_fullresident",
            "pbit_gain": float(self.transfer.gain),
            "pbit_bias": float(self.transfer.bias),
        }

        return SolverResult(
            best_energy=best_np,
            final_energy=final_np,
            best_state=best_s.detach().cpu().numpy().astype(np.int8),
            final_state=s.detach().cpu().numpy().astype(np.int8),
            runtime_s=float(runtime),
            update_opportunities=update_ops.detach().cpu().numpy(),
            restarts=restarts.detach().cpu().numpy(),
            first_hit_tick=first_hit_tick.detach().cpu().numpy(),
            first_hit_update_ops=first_hit_update_ops.detach().cpu().numpy(),
            summary=summary,
            history=history,
        )


# -----------------------------------------------------------------------------
# Problem / target / method configuration
# -----------------------------------------------------------------------------

def make_problem(
    family: str,
    n: int,
    p: float,
    seed: int,
    tsp_A: float,
    tsp_B: float,
    normalize: str,
) -> IsingProblem:
    if family == "er":
        return make_er_maxcut(n=n, p=p, seed=seed, normalize=normalize)
    if family == "sk":
        return make_sk(n=n, seed=seed, normalize=normalize)
    if family == "signed_er":
        return make_signed_er(n=n, p=p, seed=seed, normalize=normalize)
    if family == "planted":
        return make_planted_ising(n=n, p=p, seed=seed, normalize=normalize)
    if family == "tsp":
        prob, _, _ = make_tsp_ising(n_cities=n, seed=seed, A=tsp_A, B=tsp_B, normalize=normalize)
        return prob
    if family == "planted_tsp":
        # Let the planted-instance helper choose a conservative penalty A so the
        # planted feasible tour is also the exact global QUBO/Ising optimum.
        prob, _ = make_planted_tsp_ising(n_cities=n, seed=seed, A=None, B=tsp_B, normalize=normalize)
        return prob
    raise ValueError(f"Unknown family={family}")


def configure_method(base: SolverConfig, method: str, M: int, fixed_xi: float, fixed_q: float) -> SolverConfig:
    b = copy.deepcopy(base)
    b.controller.xi_fixed = float(fixed_xi)
    b.controller.q_fixed = float(fixed_q)

    if method == "seq":
        b.controller.xi_mode = "none"
        b.controller.adaptive_q = False
        b.controller.q_fixed = 1.0 / max(1, M)  # exactly one resident p-bit per global tick with exact mask
        b.anneal.mode = "monotonic"
        return b

    # Reuse the experiment definitions already present in oapi.experiment_utils.
    c = method_config(b, method)
    if method == "fixed_partial":
        c.controller.q_fixed = float(fixed_q)
    if method == "fixed_pimi":
        c.controller.xi_fixed = float(fixed_xi)
    return c


def load_target_csv(path: Optional[str]) -> Dict[Tuple[str, int, int, float], Dict[str, Any]]:
    """Load a frozen target/reference table.

    Minimum schema: ``instance_seed,target_energy``.
    The exact-optimum CSV produced by the existing
    ``scripts/solve_test_global_optima.py`` is recognized automatically via
    ``optimality_proven`` / ``exact_global_optimum_energy`` / solver-backend
    columns, so proven exact rows retain their provenance.
    """
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"instance_seed", "target_energy"}.issubset(df.columns):
        raise ValueError("target CSV requires instance_seed,target_energy")
    out: Dict[Tuple[str, int, int, float], Dict[str, Any]] = {}
    for r in df.to_dict("records"):
        fam = str(r.get("family", r.get("problem", "*")))
        n = int(r.get("n", -1)) if pd.notna(r.get("n", np.nan)) else -1
        p = float(r.get("p", -1.0)) if pd.notna(r.get("p", np.nan)) else -1.0
        target = float(r["target_energy"])
        proven = bool(r.get("optimality_proven", False)) if pd.notna(r.get("optimality_proven", np.nan)) else False
        exact_energy = r.get("exact_global_optimum_energy", np.nan)
        exact_energy = float(exact_energy) if pd.notna(exact_energy) else np.nan
        if proven and np.isfinite(exact_energy):
            backend = str(r.get("exact_solver_backend", "external"))
            source = f"exact_csv_{backend}"
        else:
            source = str(r.get("reference_type", "target_csv"))
        out[(fam, int(r["instance_seed"]), n, p)] = {
            "target_energy": target,
            "source": source,
            "optimality_proven": proven,
            "exact_global_optimum_energy": exact_energy,
        }
    return out


def lookup_target(target_map: Dict[Tuple[str, int, int, float], Dict[str, Any]], family: str, seed: int, n: int, p: float) -> Optional[Dict[str, Any]]:
    keys = [
        (family, seed, n, p),
        (family, seed, n, -1.0),
        (family, seed, -1, -1.0),
        ("*", seed, n, p),
        ("*", seed, -1, -1.0),
    ]
    for k in keys:
        if k in target_map:
            return dict(target_map[k])
    return None

def resolve_target(
    problem: IsingProblem,
    family: str,
    instance_seed: int,
    n: int,
    p: float,
    target_map: Dict[Tuple[str, int, int, float], Dict[str, Any]],
    exact_backend: str,
    exact_max_spins: int,
    exact_time_limit_s: float,
    exact_threads: int,
    exact_log: bool,
) -> Tuple[Optional[float], str, Dict[str, Any]]:
    md = problem.metadata or {}
    if "exact_optimum_energy" in md:
        return float(md["exact_optimum_energy"]), str(md.get("exact_optimum_source", "metadata_exact")), {}

    csv_ref = lookup_target(target_map, family, instance_seed, n, p)
    if csv_ref is not None:
        return float(csv_ref["target_energy"]), str(csv_ref.get("source", "target_csv")), {
            "optimality_proven": bool(csv_ref.get("optimality_proven", False)),
            "exact_global_optimum_energy": csv_ref.get("exact_global_optimum_energy", np.nan),
        }

    if exact_backend != "none" and problem.n <= exact_max_spins:
        try:
            ex = solve_exact_ising(
                problem,
                backend=exact_backend,
                time_limit_s=exact_time_limit_s,
                threads=exact_threads,
                log=exact_log,
                verify_model=True,
            )
            meta = ex.to_dict()
            if ex.optimality_proven and np.isfinite(ex.energy):
                return float(ex.energy), f"exact_{ex.backend}", meta
            return None, f"exact_unproven_{ex.backend}", meta
        except ExactBackendUnavailable as e:
            return None, "exact_backend_unavailable", {"message": str(e)}

    return None, "none", {}


# -----------------------------------------------------------------------------
# Run collection and publication-style aggregation
# -----------------------------------------------------------------------------

def _row_from_result(
    result: SolverResult,
    j: int,
    *,
    family: str,
    n: int,
    p: float,
    method: str,
    instance_seed: int,
    trajectory_id: int,
    solver_seed: int,
    target_energy: Optional[float],
    target_source: str,
    problem: IsingProblem,
    cfg: SolverConfig,
) -> Dict[str, Any]:
    be = float(result.best_energy[j])
    target = float(target_energy) if target_energy is not None else np.nan
    gap = max(0.0, be - target) if target_energy is not None else np.nan
    rel_gap = 100.0 * gap / max(abs(target), cfg.target_atol) if target_energy is not None else np.nan
    success = int(target_energy is not None and be <= target_energy + cfg.target_atol)
    is_exact_target = bool(
        target_energy is not None
        and (target_source.startswith("exact_") or "planted" in target_source or "metadata_exact" in target_source)
    )
    hist = result.history

    row: Dict[str, Any] = {
        "family": family,
        "n": int(n),
        "p": float(p),
        "n_spins": int(problem.n),
        "method": method,
        "instance_seed": int(instance_seed),
        "trajectory_id": int(trajectory_id),
        "solver_seed_batch": int(solver_seed),
        "best_energy": be,
        "final_energy": float(result.final_energy[j]),
        "target_energy": target,
        "target_source": target_source,
        "target_atol": float(cfg.target_atol),
        "success": success,
        "gap_to_target": gap,
        "relative_gap_to_target_percent": rel_gap,
        "exact_global_optimum_energy": target if is_exact_target else np.nan,
        "energy_gap_to_global_optimum": gap if is_exact_target else np.nan,
        "relative_energy_gap_percent": rel_gap if is_exact_target else np.nan,
        "first_hit_tick": int(result.first_hit_tick[j]),
        "first_hit_update_ops": int(result.first_hit_update_ops[j]),
        "update_opportunities": int(result.update_opportunities[j]),
        "restarts": int(result.restarts[j]),
        "runtime_s_batch": float(result.runtime_s),
        "runtime_s_per_trajectory": float(result.runtime_s / len(result.best_energy)),
        "batch_size_actual": int(len(result.best_energy)),
        "steps": int(cfg.steps),
        "mean_q": float(hist["q"][:, j].mean()),
        "mean_O": float(hist["O"][:, j].mean()),
        "mean_xi": float(hist["xi_mean"][:, j].mean()),
        "clip_rate": float(hist["clip_rate"][:, j].mean()),
        "mean_flip_fraction": float(hist["flip_fraction"][:, j].mean()),
        "time_at_high_q": float(hist["high_q_fraction"][:, j].mean()),
        "proposal_model": "paper_logistic_fullresident",
    }

    md = problem.metadata or {}
    if "n_cities" in md and "D" in md:
        dec = tsp_decode(result.best_state[j], int(md["n_cities"]), np.asarray(md["D"], dtype=float))
        row.update({
            "best_tsp_feasible": int(bool(dec["feasible"])),
            "constraint_violation": int(dec["constraint_violation"]),
            "best_route_distance": float(dec["tour_distance"]) if dec["feasible"] else np.nan,
            "exact_optimum_route_distance": float(md.get("exact_optimum_route_distance", np.nan)),
        })
        if dec["feasible"] and np.isfinite(row["exact_optimum_route_distance"]):
            row["route_gap_to_global_optimum"] = float(row["best_route_distance"] - row["exact_optimum_route_distance"])
        else:
            row["route_gap_to_global_optimum"] = np.nan
    return row


def save_history_npz(path: Path, result: SolverResult, batch_start_trajectory: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"hist_{k}": v for k, v in result.history.items()}
    arrays.update({
        "best_energy": result.best_energy,
        "final_energy": result.final_energy,
        "best_state": result.best_state,
        "final_state": result.final_state,
        "update_opportunities": result.update_opportunities,
        "restarts": result.restarts,
        "first_hit_tick": result.first_hit_tick,
        "first_hit_update_ops": result.first_hit_update_ops,
        "trajectory_id_start": np.array([batch_start_trajectory], dtype=np.int64),
    })
    np.savez_compressed(path, **arrays)


def run_method_trajectories_fullresident(
    problem: IsingProblem,
    family: str,
    n: int,
    p: float,
    method: str,
    base_cfg: SolverConfig,
    transfer: PBitTransfer,
    n_runs: int,
    batch_size: int,
    solver_seed_base: int,
    instance_seed: int,
    target_energy: Optional[float],
    target_source: str,
    fixed_xi: float,
    fixed_q: float,
    save_history: bool,
    history_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    done = 0
    batch_id = 0
    while done < n_runs:
        b = min(batch_size, n_runs - done)
        cfg = configure_method(base_cfg, method, problem.n, fixed_xi, fixed_q)
        cfg.batch_size = b
        cfg.seed = int(solver_seed_base + 1_000_003 * batch_id)
        cfg.target_energy = target_energy
        result = FullResidentPBitSolver(problem, cfg, transfer).run()
        for j in range(b):
            rows.append(_row_from_result(
                result, j,
                family=family, n=n, p=p, method=method,
                instance_seed=instance_seed,
                trajectory_id=done + j,
                solver_seed=cfg.seed,
                target_energy=target_energy,
                target_source=target_source,
                problem=problem,
                cfg=cfg,
            ))
        if save_history:
            save_history_npz(
                history_dir / f"{family}_n{n}_p{p:g}_seed{instance_seed}_{method}_batch{batch_id:04d}.npz",
                result,
                done,
            )
        done += b
        batch_id += 1
        print(
            f"[{family} seed={instance_seed} {method}] {done}/{n_runs} "
            f"best={np.mean(result.best_energy):.6g} O={result.summary['mean_O']:.4f} "
            f"q={result.summary['mean_q']:.3f} xi={result.summary['mean_xi']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def aggregate_safe(runs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Use the existing hierarchical publication aggregation.

    If no target is available, success/TTS columns are mathematically undefined;
    the existing aggregator will return p_success=0 from the all-zero success
    placeholder, so overwrite those fields with NaN for honest reporting.
    """
    summary = aggregate_paper_metrics(runs, n_boot=n_boot, seed=seed)
    target_available = runs.groupby("method")["target_energy"].apply(lambda x: x.notna().any()).to_dict()
    for i, row in summary.iterrows():
        if not target_available.get(row["method"], False):
            for c in [
                "p_success", "p_success_ci_low", "p_success_ci_high",
                "tts_wallclock_s", "tts_wallclock_s_ci_low", "tts_wallclock_s_ci_high",
                "tts_ticks", "tts_ticks_ci_low", "tts_ticks_ci_high",
                "tts_update_opportunities", "tts_update_opportunities_ci_low", "tts_update_opportunities_ci_high",
            ]:
                if c in summary.columns:
                    summary.at[i, c] = np.nan
    return summary


# -----------------------------------------------------------------------------
# Experimental phases from the meeting brief
# -----------------------------------------------------------------------------

def phase_stability(args: argparse.Namespace, out: Path, target_map) -> None:
    """Mechanism reproduction: q-xi map on full-resident dense ER/SK."""
    rows: List[Dict[str, Any]] = []
    q_grid = _parse_csv_floats(args.q_grid)
    xi_grid = _parse_csv_floats(args.xi_grid)
    seeds = _parse_csv_ints(args.instance_seeds)
    families = _parse_csv_strs(args.families)

    for family in families:
        if family not in ("er", "sk", "signed_er"):
            continue
        for seed in seeds:
            problem = make_problem(family, args.n, args.p, seed, args.tsp_A, args.tsp_B, args.normalize)
            Jnp = problem.J.detach().cpu().numpy()
            edge_info = spectral_edges(Jnp)
            for qv in q_grid:
                for xiv in xi_grid:
                    cfg = SolverConfig(
                        steps=args.stability_steps,
                        batch_size=args.stability_batch,
                        log_every=args.log_every,
                        mask_mode=args.mask_mode,
                        noise_distribution=args.noise_distribution,
                        device=args.device,
                        dtype=args.dtype,
                        seed=args.solver_seed + seed,
                    )
                    cfg.controller.xi_mode = "fixed" if xiv > 0 else "none"
                    cfg.controller.xi_fixed = xiv
                    cfg.controller.adaptive_q = False
                    cfg.controller.q_fixed = qv
                    cfg.anneal.mode = "monotonic"
                    res = FullResidentPBitSolver(problem, cfg, PBitTransfer(args.pbit_gain, args.pbit_bias)).run()
                    late = max(1, len(res.history["O"]) // 4)
                    rows.append({
                        "family": family,
                        "n": args.n,
                        "p": args.p,
                        "instance_seed": seed,
                        "q": qv,
                        "xi": xiv,
                        "best_energy_mean": float(np.mean(res.best_energy)),
                        "best_energy_std": float(np.std(res.best_energy)),
                        "O_mean": float(np.mean(res.history["O"][-late:])),
                        "flip_fraction_mean": float(np.mean(res.history["flip_fraction"][-late:])),
                        "runtime_s": float(res.runtime_s),
                        "mean_update_opportunities": float(np.mean(res.update_opportunities)),
                        "spectral_rho_D1_beta_max": spectral_radius_linearized(Jnp, cfg.anneal.beta_max, qv, xiv),
                        **edge_info,
                    })
                    print("[stability]", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "stability_map_fullresident.csv", index=False)


def phase_benchmark(args: argparse.Namespace, out: Path, target_map) -> pd.DataFrame:
    """Main ablation/joint experiment on full-resident ER/SK (and optional TSP)."""
    families = _parse_csv_strs(args.families)
    seeds = _parse_csv_ints(args.instance_seeds)
    methods = _parse_csv_strs(args.methods)
    all_runs: List[pd.DataFrame] = []
    target_rows: List[Dict[str, Any]] = []

    transfer = PBitTransfer(args.pbit_gain, args.pbit_bias)
    base_cfg = SolverConfig(
        steps=args.steps,
        batch_size=args.batch,
        log_every=args.log_every,
        mask_mode=args.mask_mode,
        noise_distribution=args.noise_distribution,
        device=args.device,
        dtype=args.dtype,
        seed=args.solver_seed,
    )

    # Primary controller defaults are inherited from oapi.config, then any
    # explicit CLI overrides below are applied.
    cc, ac = base_cfg.controller, base_cfg.anneal
    cc.rho_o = args.rho_o
    cc.a = args.ctrl_a
    cc.b = args.ctrl_b
    cc.c = args.ctrl_c
    cc.alpha_xi = args.alpha_xi
    cc.beta1 = args.beta1
    cc.beta2 = args.beta2
    cc.lambda_xi = args.lambda_xi
    cc.xi_max = args.xi_max
    cc.dxi_max = args.dxi_max
    cc.o_low = args.o_low
    cc.o_high = args.o_high
    cc.q_min = args.q_min
    cc.q_step = args.q_step
    cc.slow_interval = args.slow_interval
    cc.dwell_low = args.dwell_low
    cc.dwell_high = args.dwell_high
    cc.controller_rms_norm = not args.no_controller_rms_norm

    ac.beta_min = args.beta_min
    ac.beta_max = args.beta_max
    ac.eta_max = args.eta_max
    ac.eta_min = args.eta_min
    ac.cycle_steps = args.cycle_steps
    ac.stall_steps = args.stall_steps
    ac.inertia_release = args.inertia_release
    ac.restart_cooldown = args.restart_cooldown

    for family in families:
        n_this = args.tsp_cities if family in ("tsp", "planted_tsp") else args.n
        p_this = args.p
        for instance_seed in seeds:
            problem = make_problem(family, n_this, p_this, instance_seed, args.tsp_A, args.tsp_B, args.normalize)
            target, target_source, exact_meta = resolve_target(
                problem, family, instance_seed, n_this, p_this, target_map,
                args.exact_backend, args.exact_max_spins,
                args.exact_time_limit_s, args.exact_threads, args.exact_log,
            )
            target_rows.append({
                "family": family,
                "n": n_this,
                "p": p_this,
                "instance_seed": instance_seed,
                "n_spins": problem.n,
                "target_energy": target,
                "target_source": target_source,
                **{f"exact_{k}": v for k, v in exact_meta.items() if k != "state"},
            })
            print(
                f"\n[instance] family={family} n={n_this} spins={problem.n} seed={instance_seed} "
                f"target={target} source={target_source}",
                flush=True,
            )

            for mi, method in enumerate(methods):
                df = run_method_trajectories_fullresident(
                    problem=problem,
                    family=family,
                    n=n_this,
                    p=p_this,
                    method=method,
                    base_cfg=base_cfg,
                    transfer=transfer,
                    n_runs=args.runs_per_method,
                    batch_size=args.batch,
                    solver_seed_base=args.solver_seed + 10_000_019 * instance_seed + 1009 * mi,
                    instance_seed=instance_seed,
                    target_energy=target,
                    target_source=target_source,
                    fixed_xi=args.fixed_xi,
                    fixed_q=args.fixed_q,
                    save_history=args.save_history,
                    history_dir=out / "histories",
                )
                all_runs.append(df)

                # Incremental checkpoint; safe if a long run is interrupted.
                current = pd.concat(all_runs, ignore_index=True)
                current.to_csv(out / "runs_fullresident.csv", index=False)

    runs = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    pd.DataFrame(target_rows).to_csv(out / "targets_fullresident.csv", index=False)
    if runs.empty:
        return runs

    summary_parts = []
    for (family, n_val, p_val), dfg in runs.groupby(["family", "n", "p"], sort=False):
        s = aggregate_safe(dfg, args.bootstrap, args.bootstrap_seed)
        s.insert(0, "family", family)
        s.insert(1, "n", n_val)
        s.insert(2, "p", p_val)
        # Extra report metrics not in the legacy aggregator.
        extras = dfg.groupby("method", sort=False).agg(
            mean_flip_fraction=("mean_flip_fraction", "mean"),
            time_at_high_q=("time_at_high_q", "mean"),
            tsp_feasible_rate=("best_tsp_feasible", "mean") if "best_tsp_feasible" in dfg.columns else ("success", lambda x: np.nan),
        ).reset_index()
        s = s.merge(extras, on="method", how="left")
        summary_parts.append(s)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(out / "summary_fullresident.csv", index=False)

    # Paired instance deltas against a strong reference when available.
    for reference in ["fixed_pimi", "fixed_partial", "par0"]:
        if reference in set(runs.method):
            try:
                paired = paired_instance_delta_bootstrap(
                    runs, reference_method=reference, value_col="best_energy",
                    n_boot=args.bootstrap, seed=args.bootstrap_seed + 77,
                )
                paired.to_csv(out / f"paired_vs_{reference}.csv", index=False)
            except Exception as e:
                (out / f"paired_vs_{reference}_error.txt").write_text(str(e), encoding="utf-8")
            break

    return runs


def phase_transfer(args: argparse.Namespace, out: Path, target_map) -> None:
    """Fixed-controller transfer across unseen size/density/coupling family."""
    sizes = _parse_csv_ints(args.transfer_sizes)
    densities = _parse_csv_floats(args.transfer_densities)
    seeds = _parse_csv_ints(args.transfer_seeds)
    methods = _parse_csv_strs(args.transfer_methods)

    rows: List[pd.DataFrame] = []
    transfer = PBitTransfer(args.pbit_gain, args.pbit_bias)
    base = SolverConfig(
        steps=args.transfer_steps, batch_size=args.batch, log_every=args.log_every,
        mask_mode=args.mask_mode, noise_distribution=args.noise_distribution,
        device=args.device, dtype=args.dtype, seed=args.solver_seed + 500_000,
    )
    # Match benchmark controller overrides.
    base.controller.rho_o = args.rho_o
    base.controller.a = args.ctrl_a; base.controller.b = args.ctrl_b; base.controller.c = args.ctrl_c
    base.controller.alpha_xi = args.alpha_xi; base.controller.beta1 = args.beta1; base.controller.beta2 = args.beta2
    base.controller.lambda_xi = args.lambda_xi; base.controller.xi_max = args.xi_max; base.controller.dxi_max = args.dxi_max
    base.controller.o_low = args.o_low; base.controller.o_high = args.o_high
    base.controller.q_min = args.q_min; base.controller.q_step = args.q_step
    base.controller.slow_interval = args.slow_interval; base.controller.dwell_low = args.dwell_low; base.controller.dwell_high = args.dwell_high
    base.controller.controller_rms_norm = not args.no_controller_rms_norm
    base.anneal.beta_min = args.beta_min; base.anneal.beta_max = args.beta_max
    base.anneal.eta_max = args.eta_max; base.anneal.eta_min = args.eta_min
    base.anneal.cycle_steps = args.cycle_steps; base.anneal.stall_steps = args.stall_steps
    base.anneal.inertia_release = args.inertia_release; base.anneal.restart_cooldown = args.restart_cooldown

    k = 0
    for n in sizes:
        for p in densities:
            for family in ["er", "sk"]:
                for seed in seeds:
                    instance_seed = seed + k
                    k += 1
                    problem = make_problem(family, n, p, instance_seed, args.tsp_A, args.tsp_B, args.normalize)
                    target, source, _ = resolve_target(
                        problem, family, instance_seed, n, p, target_map,
                        args.exact_backend, args.exact_max_spins,
                        args.exact_time_limit_s, args.exact_threads, args.exact_log,
                    )
                    for mi, method in enumerate(methods):
                        df = run_method_trajectories_fullresident(
                            problem, family, n, p, method, base, transfer,
                            args.transfer_runs, args.batch,
                            args.solver_seed + 700_000 + 100_003 * k + 1009 * mi,
                            instance_seed, target, source,
                            args.fixed_xi, args.fixed_q,
                            args.save_history, out / "transfer_histories",
                        )
                        rows.append(df)
    if rows:
        d = pd.concat(rows, ignore_index=True)
        d.to_csv(out / "transfer_runs_fullresident.csv", index=False)
        sum_parts = []
        for (family, n, p), dfg in d.groupby(["family", "n", "p"], sort=False):
            s = aggregate_safe(dfg, args.bootstrap, args.bootstrap_seed + 1000)
            s.insert(0, "family", family); s.insert(1, "n", n); s.insert(2, "p", p)
            sum_parts.append(s)
        pd.concat(sum_parts, ignore_index=True).to_csv(out / "transfer_summary_fullresident.csv", index=False)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Full-resident paper-style p-bit experiment for OAPI; no variable partitioning.",
    )
    ap.add_argument("--phase", choices=["stability", "benchmark", "transfer", "all"], default="benchmark")
    ap.add_argument("--families", default="er,sk", help="er,sk,signed_er,planted,tsp,planted_tsp")
    ap.add_argument("--n", type=int, default=128, help="spins for ER/SK")
    ap.add_argument("--p", type=float, default=0.30, help="ER edge probability")
    ap.add_argument("--instance-seeds", default="10,11,12,13,14")

    # TSP is full-resident N_city^2; no partitions.
    ap.add_argument("--tsp-cities", type=int, default=10)
    ap.add_argument("--tsp-A", type=float, default=4.0)
    ap.add_argument("--tsp-B", type=float, default=1.0)

    # Main stochastic experiment budget.
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--runs-per-method", type=int, default=256)
    ap.add_argument("--batch", type=int, default=25, help="independent trajectories per GPU batch; NOT variable grouping")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--solver-seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--mask-mode", choices=["exact", "bernoulli"], default="exact")
    ap.add_argument("--noise-distribution", choices=["normal", "uniform"], default="normal")
    ap.add_argument("--normalize", choices=["field_rms", "max_row_l1", "spectral", "none"], default="field_rms")

    # Methods map directly to the report's baseline/ablation sequence.
    ap.add_argument(
        "--methods",
        default="seq,par0,fixed_pimi,fixed_partial,heuristic_xi,momentum_xi,adam_xi,adamw_rms,adaptive_q,joint,joint_restart",
    )
    ap.add_argument("--fixed-xi", type=float, default=0.30)
    ap.add_argument("--fixed-q", type=float, default=0.50)

    # p-bit transfer. bias/gain are dimensionless because fields are normalized.
    ap.add_argument("--pbit-gain", type=float, default=1.0)
    ap.add_argument("--pbit-bias", type=float, default=0.0)

    # Controller.
    ap.add_argument("--rho-o", type=float, default=0.90)
    ap.add_argument("--ctrl-a", type=float, default=1.0)
    ap.add_argument("--ctrl-b", type=float, default=0.50)
    ap.add_argument("--ctrl-c", type=float, default=0.0)
    ap.add_argument("--alpha-xi", type=float, default=0.02)
    ap.add_argument("--beta1", type=float, default=0.90)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--lambda-xi", type=float, default=0.002)
    ap.add_argument("--xi-max", type=float, default=1.50)
    ap.add_argument("--dxi-max", type=float, default=0.05)
    ap.add_argument("--o-low", type=float, default=0.05)
    ap.add_argument("--o-high", type=float, default=0.15)
    ap.add_argument("--q-min", type=float, default=0.125)
    ap.add_argument("--q-step", type=float, default=0.125)
    ap.add_argument("--slow-interval", type=int, default=25)
    ap.add_argument("--dwell-low", type=int, default=4)
    ap.add_argument("--dwell-high", type=int, default=2)
    ap.add_argument("--no-controller-rms-norm", action="store_true")

    # Annealing / restart.
    ap.add_argument("--beta-min", type=float, default=0.20)
    ap.add_argument("--beta-max", type=float, default=2.00)
    ap.add_argument("--eta-max", type=float, default=0.50)
    ap.add_argument("--eta-min", type=float, default=0.05)
    ap.add_argument("--cycle-steps", type=int, default=1000)
    ap.add_argument("--stall-steps", type=int, default=300)
    ap.add_argument("--inertia-release", type=float, default=0.50)
    ap.add_argument("--restart-cooldown", type=int, default=None)

    # Stability-map phase.
    ap.add_argument("--stability-steps", type=int, default=1500)
    ap.add_argument("--stability-batch", type=int, default=25)
    ap.add_argument("--q-grid", default="0.0625,0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--xi-grid", default="0,0.1,0.2,0.3,0.4,0.6,0.8")

    # Transfer phase.
    ap.add_argument("--transfer-sizes", default="64,96,128,192")
    ap.add_argument("--transfer-densities", default="0.15,0.30,0.50")
    ap.add_argument("--transfer-seeds", default="500,501")
    ap.add_argument("--transfer-methods", default="fixed_pimi,fixed_partial,joint_restart")
    ap.add_argument("--transfer-steps", type=int, default=3000)
    ap.add_argument("--transfer-runs", type=int, default=64)

    # Exact/frozen target support. For N=128 ER/SK, prefer a precomputed frozen
    # target CSV unless a commercial/open MILP backend can prove optimality.
    ap.add_argument("--target-csv", default=None)
    ap.add_argument("--exact-backend", choices=["none", "auto", "gurobi", "cplex", "scip", "enumeration"], default="none")
    ap.add_argument("--exact-max-spins", type=int, default=64)
    ap.add_argument("--exact-time-limit-s", type=float, default=0.0)
    ap.add_argument("--exact-threads", type=int, default=0)
    ap.add_argument("--exact-log", action="store_true")

    # Automatic visualization after the stability phase.
    ap.add_argument(
        "--visualizer-script",
        default="scripts/visualize_fullresident_stability.py",
        help="Visualization script path, relative to the project root unless absolute",
    )
    ap.add_argument(
        "--visualization-dirname",
        default="figures",
        help="Subdirectory inside the stability result directory for generated figures",
    )
    ap.add_argument("--visualization-dpi", type=int, default=300)
    ap.add_argument("--visualization-formats", default="png,pdf")
    ap.add_argument(
        "--no-auto-visualize",
        action="store_true",
        help="Do not automatically visualize stability_map_fullresident.csv",
    )
    ap.add_argument(
        "--strict-visualization",
        action="store_true",
        help="Treat visualization failure as an experiment error instead of only warning",
    )

    # Statistics/output.
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=777)
    ap.add_argument("--save-history", action="store_true")
    ap.add_argument("--out", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.out is None:
        out = ROOT / "results" / f"fullresident_oapi_{_time_tag()}"
    else:
        out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target_map = load_target_csv(args.target_csv)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "project_root": str(ROOT),
        "full_resident": True,
        "variable_partitioning": False,
        "batch_semantics": "independent stochastic trajectories, not variable groups",
        "proposal_model": "paper_logistic_fullresident",
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")

    print(f"[OUT] {out}")
    print("[MODEL] one full-resident p-bit population; no graph/TSP partitioning")
    print(f"[DEVICE] {_resolve_device(args.device)}")

    if args.phase in ("stability", "all"):
        phase_stability(args, out, target_map)
        run_stability_visualizer(args, out)
    if args.phase in ("benchmark", "all"):
        phase_benchmark(args, out, target_map)
    if args.phase in ("transfer", "all"):
        phase_transfer(args, out, target_map)

    print(f"\n[DONE] Results saved under: {out}")


if __name__ == "__main__":
    main()
