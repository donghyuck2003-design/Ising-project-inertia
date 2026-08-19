#!/usr/bin/env python
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.benchmark import make_problem, run_method_trajectories, load_target_map
from oapi.statistics import hierarchical_bootstrap_ci


def ints(x): return [int(v) for v in x.split(',') if v.strip()]
def floats(x): return [float(v) for v in x.split(',') if v.strip()]

def main():
    ap=argparse.ArgumentParser(description='Paper-scale tuning of strong fixed-xi and fixed-q baselines on tuning instances only')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=96); ap.add_argument('--p',type=float,default=.35)
    ap.add_argument('--instance-seeds',default='1,2,3,4,5'); ap.add_argument('--targets',default=None)
    ap.add_argument('--xis',default='0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0')
    ap.add_argument('--qs',default='0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0')
    ap.add_argument('--target-atol',type=float,default=1e-6); ap.add_argument('--runs-per-value',type=int,default=64); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--steps',type=int,default=3000); ap.add_argument('--device',default='auto')
    ap.add_argument('--bootstrap',type=int,default=2000); ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'fixed_tuning')); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); targets=load_target_map(args.targets); rows=[]
    iseeds=ints(args.instance_seeds)
    for ii,iseed in enumerate(iseeds):
        problem=make_problem(args.problem,args.n,args.p,iseed)
        for vi,xi in enumerate(floats(args.xis)):
            base=SolverConfig(steps=args.steps,batch_size=args.batch,device=args.device,target_atol=args.target_atol); base.controller.xi_fixed=xi
            method='par0' if xi==0 else 'fixed_pimi'
            df=run_method_trajectories(problem,method,base,args.runs_per_value,args.batch,100000+ii*100000+vi*1000,iseed,targets.get(iseed)); df['family']='fixed_pimi'; df['value']=xi; rows.append(df)
        for vi,q in enumerate(floats(args.qs)):
            base=SolverConfig(steps=args.steps,batch_size=args.batch,device=args.device,target_atol=args.target_atol); base.controller.q_fixed=q
            df=run_method_trajectories(problem,'fixed_partial',base,args.runs_per_value,args.batch,500000+ii*100000+vi*1000,iseed,targets.get(iseed)); df['family']='fixed_partial'; df['value']=q; rows.append(df)
        print('finished instance',iseed)
    runs=pd.concat(rows,ignore_index=True); runs.to_csv(out/'runs.csv',index=False)
    agg=[]
    for (fam,val),sub in runs.groupby(['family','value']):
        e,lo,hi=hierarchical_bootstrap_ci(sub,'best_energy','instance_seed',np.mean,.95,args.bootstrap,seed=2026)
        agg.append({'family':fam,'value':val,'best_energy_mean':e,'best_energy_ci_low':lo,'best_energy_ci_high':hi,'p_success':float(sub.success.mean()) if sub.target_energy.notna().all() else np.nan,'mean_O':float(sub.mean_O.mean()),'mean_q':float(sub.mean_q.mean()),'mean_runtime_s_per_trajectory':float(sub.runtime_s_per_trajectory.mean())})
    agg=pd.DataFrame(agg); agg.to_csv(out/'aggregate.csv',index=False)
    rec={}
    for fam in ['fixed_pimi','fixed_partial']:
        sub=agg[agg.family==fam].copy()
        if sub.p_success.notna().all():
            sub=sub.sort_values(['p_success','best_energy_mean'],ascending=[False,True])
            rule='maximize frozen-target success probability, break ties by mean best energy'
        else:
            sub=sub.sort_values('best_energy_mean')
            rule='minimize mean best energy on tuning instances'
        r=sub.iloc[0]; rec[fam]={'best_value':float(r.value),'best_energy_mean':float(r.best_energy_mean),'p_success':None if pd.isna(r.p_success) else float(r.p_success),'selection_rule':rule}
    with open(out/'recommended_fixed_baselines.json','w') as f: json.dump(rec,f,indent=2)
    print(json.dumps(rec,indent=2)); print('Saved',out)
if __name__=='__main__':main()
