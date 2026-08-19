#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_sk, make_signed_er
from oapi.solver import IsingSolver
from oapi.experiment_utils import method_config, result_rows
from oapi.io import save_result

METHODS = ["fixed_pimi", "heuristic_xi", "momentum_xi", "adam_xi", "adamw_xi", "adamw_clip", "adamw_rms"]


def main():
    ap = argparse.ArgumentParser(description="Phase B: Fixed -> Heuristic -> Momentum -> Adam -> AdamW inertia ablation")
    ap.add_argument("--problem", choices=["er", "sk", "signed_er"], default="er")
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--p", type=float, default=0.30)
    ap.add_argument("--instance-seeds", default="10,11,12")
    ap.add_argument("--solver-seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fixed-xi", type=float, default=0.30)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default=str(ROOT / "results" / "phase_b"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for iseed in [int(x) for x in args.instance_seeds.split(",")]:
        if args.problem == "er": problem = make_er_maxcut(args.n, args.p, iseed)
        elif args.problem == "sk": problem = make_sk(args.n, iseed)
        else: problem = make_signed_er(args.n, args.p, iseed)
        for mi, method in enumerate(METHODS):
            base = SolverConfig(steps=args.steps, batch_size=args.batch, seed=args.solver_seed + mi, device=args.device)
            base.controller.xi_fixed = args.fixed_xi
            cfg = method_config(base, method)
            res = IsingSolver(problem, cfg).run()
            rows += result_rows(res, method, iseed, cfg.seed)
            save_result(res, out / f"instance_{iseed}", method)
            print(iseed, method, res.summary)
    pd.DataFrame(rows).to_csv(out / "inertia_ablation_runs.csv", index=False)
    print(f"Saved: {out / 'inertia_ablation_runs.csv'}")

if __name__ == "__main__": main()
