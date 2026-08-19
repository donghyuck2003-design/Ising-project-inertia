#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_signed_er, make_sk
from oapi.solver import IsingSolver
from oapi.experiment_utils import result_rows


def main():
    ap=argparse.ArgumentParser(description="Phase E: fixed-hyperparameter transfer across size/density/coupling family")
    ap.add_argument("--sizes", default="64,96,128,192")
    ap.add_argument("--densities", default="0.15,0.30,0.50")
    ap.add_argument("--steps",type=int,default=3000); ap.add_argument("--batch",type=int,default=12)
    ap.add_argument("--device",default="auto"); ap.add_argument("--out",default=str(ROOT/"results"/"phase_e_transfer"))
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rows=[]
    k=0
    for n in [int(x) for x in args.sizes.split(",")]:
      for p in [float(x) for x in args.densities.split(",")]:
       for family in ["er","signed_er","sk"]:
        seed=500+k; k+=1
        problem=make_er_maxcut(n,p,seed) if family=="er" else (make_signed_er(n,p,seed) if family=="signed_er" else make_sk(n,seed))
        for norm in [False,True]:
          cfg=SolverConfig(steps=args.steps,batch_size=args.batch,seed=9000+k,device=args.device)
          cfg.controller.xi_mode="adamw"; cfg.controller.adaptive_q=True; cfg.controller.controller_rms_norm=norm; cfg.anneal.mode="event_restart"
          res=IsingSolver(problem,cfg).run(); rr=result_rows(res,f"joint_restart_rms{int(norm)}",seed,cfg.seed)
          for r in rr: r.update({"family":family,"n":n,"p":p,"controller_rms_norm":norm})
          rows+=rr; print(family,n,p,norm,res.summary)
    pd.DataFrame(rows).to_csv(out/"transfer_runs.csv",index=False)

if __name__=="__main__": main()
