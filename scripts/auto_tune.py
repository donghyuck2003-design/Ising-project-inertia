#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.config_io import apply_overrides, load_overrides
from oapi.benchmark import make_problem, run_method_trajectories, load_target_map


def ints(x): return [int(v) for v in x.split(',') if v.strip()]

def loguniform(rng, lo, hi): return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def sample_overrides(rng):
    o_low = float(rng.uniform(.015,.10))
    o_high = float(rng.uniform(max(o_low+.03,.08),.30))
    return {
        'controller.rho_o': float(rng.uniform(.82,.98)),
        'controller.a': 1.0,
        'controller.b': float(rng.uniform(.15,1.25)),
        'controller.alpha_xi': loguniform(rng,.003,.08),
        'controller.beta1': float(rng.choice([.85,.90,.95])),
        'controller.beta2': float(rng.choice([.95,.98,.99,.995])),
        'controller.lambda_xi': loguniform(rng,1e-4,.02),
        'controller.dxi_max': loguniform(rng,.01,.15),
        'controller.xi_max': float(rng.uniform(.6,2.0)),
        'controller.q_min': float(rng.choice([.0625,.125,.25])),
        'controller.q_step': float(rng.choice([.0625,.125,.25])),
        'controller.o_low': o_low,
        'controller.o_high': o_high,
        'controller.slow_interval': int(rng.choice([10,20,25,50,100])),
        'controller.dwell_low': int(rng.choice([2,4,6])),
        'controller.dwell_high': int(rng.choice([1,2,3])),
        'anneal.stall_steps': int(rng.choice([150,250,400,600])),
        'anneal.inertia_release': float(rng.choice([.25,.5,.75])),
    }

def main():
    ap=argparse.ArgumentParser(description='Random hyperparameter search on tuning-only instances')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=96); ap.add_argument('--p',type=float,default=.35)
    ap.add_argument('--instance-seeds',default='1,2,3,4,5'); ap.add_argument('--targets',default=None)
    ap.add_argument('--trials',type=int,default=40); ap.add_argument('--runs-per-instance',type=int,default=64)
    ap.add_argument('--batch',type=int,default=32); ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--method',default='joint_restart'); ap.add_argument('--device',default='auto'); ap.add_argument('--target-atol',type=float,default=1e-6)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--base-config-json',default=None)
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'tuning')); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    targets=load_target_map(args.targets); base_overrides=load_overrides(args.base_config_json)
    rng=np.random.default_rng(args.seed); trial_rows=[]
    iseeds=ints(args.instance_seeds)
    for trial in range(args.trials):
        ov=sample_overrides(rng); per=[]
        for ii,iseed in enumerate(iseeds):
            problem=make_problem(args.problem,args.n,args.p,iseed)
            base=SolverConfig(steps=args.steps,batch_size=args.batch,device=args.device,target_atol=args.target_atol)
            apply_overrides(base,base_overrides); apply_overrides(base,ov)
            df=run_method_trajectories(problem,args.method,base,args.runs_per_instance,args.batch,args.seed+trial*100000+ii*1000,iseed,targets.get(iseed))
            per.append(df)
        allrun=pd.concat(per,ignore_index=True)
        has_targets=allrun.target_energy.notna().all()
        p_success=float(allrun.success.mean()) if has_targets else float('nan')
        energy=float(allrun.best_energy.mean()); mean_q=float(allrun.mean_q.mean()); mean_o=float(allrun.mean_O.mean())
        # Lexicographic paper-oriented score: success first when targets exist,
        # then energy, then a small preference for higher parallelism.
        if has_targets:
            score = -1000.0*p_success + energy - .01*mean_q
        else:
            score = energy - .01*mean_q
        row={'trial':trial,'score':score,'p_success':p_success,'best_energy_mean':energy,'mean_q':mean_q,'mean_O':mean_o,**ov}
        trial_rows.append(row); pd.DataFrame(trial_rows).to_csv(out/'tuning_trials.csv',index=False)
        print(f"trial {trial+1}/{args.trials}: score={score:.6g}, p={p_success:.3f}, E={energy:.6g}, q={mean_q:.3f}")
    df=pd.DataFrame(trial_rows).sort_values('score',ascending=True)
    df.to_csv(out/'tuning_trials.csv',index=False)
    best=df.iloc[0]
    best_overrides={k:(int(best[k]) if k in ['controller.slow_interval','controller.dwell_low','controller.dwell_high','anneal.stall_steps'] else float(best[k])) for k in sample_overrides(np.random.default_rng(1)).keys()}
    payload={'best_trial':int(best.trial),'best_score':float(best.score),'selection_rule':'minimize score; success dominates when frozen targets are supplied','best_overrides':best_overrides,'tuning_instance_seeds':iseeds}
    with open(out/'best_config.json','w',encoding='utf-8') as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2)); print('Saved',out/'best_config.json')
if __name__=='__main__': main()
