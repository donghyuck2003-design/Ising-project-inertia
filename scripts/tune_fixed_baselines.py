#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.benchmark import make_problem
from oapi.solver import IsingSolver


def floats(x): return [float(v) for v in x.split(',')]

def main():
    ap=argparse.ArgumentParser(description='Tune strong fixed-PIMI xi and fixed-partial q baselines on tuning-only instances')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=96); ap.add_argument('--p',type=float,default=.35)
    ap.add_argument('--instance-seeds',default='1,2,3'); ap.add_argument('--solver-seeds',default='100,101')
    ap.add_argument('--xis',default='0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0')
    ap.add_argument('--qs',default='0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0')
    ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--device',default='auto')
    ap.add_argument('--out',default=str(ROOT/'results'/'fixed_baseline_tuning')); args=ap.parse_args()
    rows=[]; iseeds=[int(x) for x in args.instance_seeds.split(',')]; sseeds=[int(x) for x in args.solver_seeds.split(',')]
    for iseed in iseeds:
      problem=make_problem(args.problem,args.n,args.p,iseed)
      for sseed in sseeds:
       for xi in floats(args.xis):
        cfg=SolverConfig(steps=args.steps,batch_size=args.batch,seed=sseed,device=args.device); cfg.anneal.mode='monotonic'; cfg.controller.adaptive_q=False; cfg.controller.q_fixed=1.; cfg.controller.xi_mode='none' if xi==0 else 'fixed'; cfg.controller.xi_fixed=xi
        r=IsingSolver(problem,cfg).run(); rows.append({'family':'fixed_pimi','value':xi,'instance_seed':iseed,'solver_seed':sseed,'best_energy_mean':float(np.mean(r.best_energy)),'O_mean':float(np.mean(r.history['O'])),'mean_q':float(np.mean(r.history['q'])),'runtime_s':r.runtime_s,'update_ops_mean':float(np.mean(r.update_opportunities))})
       for q in floats(args.qs):
        cfg=SolverConfig(steps=args.steps,batch_size=args.batch,seed=sseed,device=args.device); cfg.anneal.mode='monotonic'; cfg.controller.adaptive_q=False; cfg.controller.q_fixed=q; cfg.controller.xi_mode='none'
        r=IsingSolver(problem,cfg).run(); rows.append({'family':'fixed_partial','value':q,'instance_seed':iseed,'solver_seed':sseed,'best_energy_mean':float(np.mean(r.best_energy)),'O_mean':float(np.mean(r.history['O'])),'mean_q':float(np.mean(r.history['q'])),'runtime_s':r.runtime_s,'update_ops_mean':float(np.mean(r.update_opportunities))})
      print('finished instance',iseed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(out/'fixed_baseline_sweep.csv',index=False)
    agg=df.groupby(['family','value'],as_index=False).agg(best_energy_mean=('best_energy_mean','mean'),best_energy_std=('best_energy_mean','std'),O_mean=('O_mean','mean'),runtime_s=('runtime_s','mean'),update_ops_mean=('update_ops_mean','mean'))
    agg.to_csv(out/'fixed_baseline_aggregate.csv',index=False)
    rec={}
    for fam in ['fixed_pimi','fixed_partial']:
        sub=agg[agg.family==fam].sort_values('best_energy_mean')
        rec[fam]={'best_value_by_energy':float(sub.iloc[0].value),'best_energy_mean':float(sub.iloc[0].best_energy_mean),'note':'Select on tuning set only; verify on validation/test sets.'}
    with open(out/'recommended_fixed_baselines.json','w') as f: json.dump(rec,f,indent=2)
    print(json.dumps(rec,indent=2)); print('Saved',out)
if __name__=='__main__': main()
