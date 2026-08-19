#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_tsp_ising, tsp_decode
from oapi.solver import IsingSolver
from oapi.io import save_result


def main():
    ap=argparse.ArgumentParser(description="Full-resident N^2-spin TSP demonstration with fixed A/B")
    ap.add_argument("--cities",type=int,default=8); ap.add_argument("--instance-seed",type=int,default=42)
    ap.add_argument("--A",type=float,default=4.0); ap.add_argument("--B",type=float,default=1.0)
    ap.add_argument("--steps",type=int,default=8000); ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--solver-seed",type=int,default=7000); ap.add_argument("--device",default="auto")
    ap.add_argument("--out",default=str(ROOT/"results"/"tsp")); args=ap.parse_args()
    problem,coords,D=make_tsp_ising(args.cities,args.instance_seed,args.A,args.B)
    cfg=SolverConfig(steps=args.steps,batch_size=args.batch,seed=args.solver_seed,device=args.device)
    cfg.controller.xi_mode="adamw"; cfg.controller.adaptive_q=True; cfg.anneal.mode="event_restart"
    res=IsingSolver(problem,cfg).run(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); save_result(res,out,"tsp_joint_restart")
    rows=[]
    for b,s in enumerate(res.best_state):
        d=tsp_decode(s,args.cities,D)
        rows.append({"batch":b,"best_energy":float(res.best_energy[b]),"feasible":d["feasible"],"constraint_violation":d["constraint_violation"],"tour_distance":d["tour_distance"],"tour":json.dumps(d["tour"])})
    pd.DataFrame(rows).to_csv(out/"tsp_decoded.csv",index=False); np.savetxt(out/"coords.csv",coords,delimiter=",")
    print(pd.DataFrame(rows).sort_values(["feasible","tour_distance"],ascending=[False,True]).head(10).to_string(index=False))
    print("feasible_rate=",float(np.mean([r["feasible"] for r in rows])))

if __name__=="__main__": main()
