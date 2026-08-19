#!/usr/bin/env python
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.config_io import apply_overrides
from oapi.benchmark import make_problem, run_method_trajectories, load_target_map


def ints(x): return [int(v) for v in x.split(',') if v.strip()]

def main():
    ap=argparse.ArgumentParser(description='Select among top tuning trials on disjoint validation instances')
    ap.add_argument('--tuning-trials',required=True); ap.add_argument('--top-k',type=int,default=5)
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=96); ap.add_argument('--p',type=float,default=.35)
    ap.add_argument('--instance-seeds',default='50,51,52,53,54'); ap.add_argument('--targets',default=None)
    ap.add_argument('--method',default='joint_restart'); ap.add_argument('--target-atol',type=float,default=1e-6); ap.add_argument('--runs-per-instance',type=int,default=100); ap.add_argument('--batch',type=int,default=50); ap.add_argument('--steps',type=int,default=3500); ap.add_argument('--device',default='auto')
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'validation')); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); targets=load_target_map(args.targets)
    trials=pd.read_csv(args.tuning_trials).sort_values('score').head(args.top_k)
    param_cols=[c for c in trials.columns if c.startswith('controller.') or c.startswith('anneal.')]
    results=[]; iseeds=ints(args.instance_seeds)
    for rank,(_,tr) in enumerate(trials.iterrows()):
        ov={c:(int(tr[c]) if c in ['controller.slow_interval','controller.dwell_low','controller.dwell_high','anneal.stall_steps'] else float(tr[c])) for c in param_cols}
        pieces=[]
        for ii,iseed in enumerate(iseeds):
            prob=make_problem(args.problem,args.n,args.p,iseed); base=SolverConfig(steps=args.steps,batch_size=args.batch,device=args.device,target_atol=args.target_atol); apply_overrides(base,ov)
            pieces.append(run_method_trajectories(prob,args.method,base,args.runs_per_instance,args.batch,8000000+rank*100000+ii*1000,iseed,targets.get(iseed)))
        d=pd.concat(pieces,ignore_index=True); has=d.target_energy.notna().all(); p=float(d.success.mean()) if has else np.nan; e=float(d.best_energy.mean()); q=float(d.mean_q.mean())
        score=-1000*p+e-.01*q if has else e-.01*q
        results.append({'rank_from_tuning':rank,'source_trial':int(tr.trial),'validation_score':score,'p_success':p,'best_energy_mean':e,'mean_q':q,**ov})
        print('validation candidate',rank,'score',score,'p',p,'E',e)
    res=pd.DataFrame(results).sort_values('validation_score'); res.to_csv(out/'validation_candidates.csv',index=False)
    best=res.iloc[0]; ov={c:(int(best[c]) if c in ['controller.slow_interval','controller.dwell_low','controller.dwell_high','anneal.stall_steps'] else float(best[c])) for c in param_cols}
    payload={'source_trial':int(best.source_trial),'validation_score':float(best.validation_score),'best_overrides':ov,'validation_instance_seeds':iseeds}
    with open(out/'best_config_validated.json','w') as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2)); print('Saved',out/'best_config_validated.json')
if __name__=='__main__':main()
