#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_sk
from oapi.solver import IsingSolver
from oapi.stability import spectral_radius_linearized, spectral_edges


def parse_floats(s): return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Phase A: empirical (q, xi) oscillation/stability map")
    ap.add_argument("--problem", choices=["er", "sk"], default="er")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--p", type=float, default=0.35)
    ap.add_argument("--instance-seed", type=int, default=10)
    ap.add_argument("--solver-seed", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--q", default="0.0625,0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--xi", default="0,0.1,0.2,0.3,0.4,0.6,0.8")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "results" / "phase_a"))
    args = ap.parse_args()

    problem = make_er_maxcut(args.n, args.p, args.instance_seed) if args.problem == "er" else make_sk(args.n, args.instance_seed)
    Jnp = problem.J.numpy()
    rows = []
    for q in parse_floats(args.q):
        for xi in parse_floats(args.xi):
            cfg = SolverConfig(steps=args.steps, batch_size=args.batch, seed=args.solver_seed, device=args.device)
            cfg.controller.xi_mode = "fixed" if xi > 0 else "none"
            cfg.controller.xi_fixed = xi
            cfg.controller.adaptive_q = False
            cfg.controller.q_fixed = q
            cfg.anneal.mode = "monotonic"
            res = IsingSolver(problem, cfg).run()
            rows.append({
                "q": q, "xi": xi,
                "best_energy_mean": np.mean(res.best_energy),
                "best_energy_std": np.std(res.best_energy),
                "O_mean": np.mean(res.history["O"][-max(1, len(res.history["O"])//4):]),
                "runtime_s": res.runtime_s,
                "mean_update_opportunities": np.mean(res.update_opportunities),
                "spectral_rho_D1_beta_max": spectral_radius_linearized(Jnp, cfg.anneal.beta_max, q, xi),
            })
            print(rows[-1])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "stability_map.csv", index=False)
    with open(out / "spectral_edges.json", "w") as f: json.dump(spectral_edges(Jnp), f, indent=2)
    print(f"Saved: {out / 'stability_map.csv'}")

if __name__ == "__main__": main()
