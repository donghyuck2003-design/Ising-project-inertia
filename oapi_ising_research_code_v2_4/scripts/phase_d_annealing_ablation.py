#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_sk
from oapi.solver import IsingSolver
from oapi.experiment_utils import result_rows
from oapi.io import save_result

MODES = ["monotonic", "cosine", "periodic_restart", "event_restart"]


def main():
    ap = argparse.ArgumentParser(description="Phase D: annealing/restart ablation on joint controller")
    ap.add_argument("--problem", choices=["er", "sk"], default="er")
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--p", type=float, default=0.30)
    ap.add_argument("--instance-seeds", default="30,31,32")
    ap.add_argument("--solver-seed", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto"); ap.add_argument("--out", default=str(ROOT / "results" / "phase_d"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); rows=[]
    for iseed in [int(x) for x in args.instance_seeds.split(",")]:
        problem = make_er_maxcut(args.n, args.p, iseed) if args.problem == "er" else make_sk(args.n, iseed)
        for mi, mode in enumerate(MODES):
            cfg = SolverConfig(steps=args.steps, batch_size=args.batch, seed=args.solver_seed+mi, device=args.device)
            cfg.controller.xi_mode="adamw"; cfg.controller.adaptive_q=True
            cfg.anneal.mode=mode
            res=IsingSolver(problem,cfg).run(); rows += result_rows(res, mode, iseed, cfg.seed)
            save_result(res, out/f"instance_{iseed}", mode); print(iseed,mode,res.summary)
    pd.DataFrame(rows).to_csv(out/"annealing_ablation_runs.csv",index=False)

if __name__ == "__main__": main()
