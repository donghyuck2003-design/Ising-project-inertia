#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.config_io import apply_overrides, load_overrides
from oapi.benchmark import make_problem, reference_row_map
from oapi.experiment_utils import method_config
from oapi.solver import IsingSolver
from oapi.reporting import set_publication_rc


def _save(fig, base: Path):
    fig.tight_layout(); fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(base.with_suffix('.png'),dpi=300,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Representative ensemble trajectories including exact-global-optimum gap plots')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=128); ap.add_argument('--p',type=float,default=.30)
    ap.add_argument('--instance-seed',type=int,default=200); ap.add_argument('--methods',default='par0,fixed_pimi,joint_restart')
    ap.add_argument('--runs',type=int,default=100); ap.add_argument('--batch',type=int,default=25); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--log-every',type=int,default=10)
    ap.add_argument('--device',default='auto'); ap.add_argument('--config-json',default=None); ap.add_argument('--fixed-xi',type=float,default=.30); ap.add_argument('--fixed-q',type=float,default=.50)
    ap.add_argument('--targets',default=None); ap.add_argument('--target-atol',type=float,default=1e-6)
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'benchmark'/'figures'/'trajectory'))
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    set_publication_rc(); problem=make_problem(args.problem,args.n,args.p,args.instance_seed); overrides=load_overrides(args.config_json)

    exact_optimum=None; target_energy=None; refs=reference_row_map(args.targets)
    if args.instance_seed in refs:
        rr=refs[args.instance_seed]
        try: target_energy=float(rr['target_energy'])
        except Exception: target_energy=None
        try:
            v=float(rr.get('exact_global_optimum_energy',np.nan)); exact_optimum=v if np.isfinite(v) else None
        except Exception: exact_optimum=None

    records=[]
    for mi,method in enumerate([m.strip() for m in args.methods.split(',') if m.strip()]):
        histories=[]; summaries=[]; done=0; bid=0
        while done<args.runs:
            b=min(args.batch,args.runs-done)
            base=SolverConfig(steps=args.steps,batch_size=b,log_every=args.log_every,device=args.device,seed=700000+mi*10000+bid*1000003,target_energy=target_energy,target_atol=args.target_atol)
            base.controller.xi_fixed=args.fixed_xi; base.controller.q_fixed=args.fixed_q; apply_overrides(base,overrides)
            cfg=method_config(base,method); res=IsingSolver(problem,cfg).run(); histories.append(res.history); summaries.append(res.summary)
            done+=b; bid+=1
        # Concatenate trajectory axis across safe GPU batches.
        t=np.asarray(histories[0]['t'])
        for li,tick in enumerate(t):
            for metric in ['best_energy','O','q','xi_mean','beta','eta']:
                v=np.concatenate([np.asarray(h[metric][li],float).reshape(-1) for h in histories])
                records.append({'method':method,'t':int(tick),'metric':metric,'mean':float(v.mean()),'q10':float(np.quantile(v,.10)),'q90':float(np.quantile(v,.90))})
            if exact_optimum is not None:
                v=np.concatenate([np.maximum(np.asarray(h['best_energy'][li],float).reshape(-1)-exact_optimum,0.0) for h in histories])
                records.append({'method':method,'t':int(tick),'metric':'best_energy_gap','mean':float(v.mean()),'q10':float(np.quantile(v,.10)),'q90':float(np.quantile(v,.90))})
        print(method,f'completed runs={args.runs} in batches <= {args.batch}',flush=True)
    df=pd.DataFrame(records); df.to_csv(out/'trajectory_summary.csv',index=False)

    suball=df[df.metric=='best_energy']; fig,ax=plt.subplots(figsize=(5.2,3.3))
    for method,sub in suball.groupby('method',sort=False): ax.plot(sub.t,sub['mean'],label=method); ax.fill_between(sub.t,sub.q10,sub.q90,alpha=.15)
    if exact_optimum is not None: ax.axhline(exact_optimum,linestyle='--',linewidth=1.1,label='Exact global optimum $E^*$')
    ax.set_xlabel('Global tick'); ax.set_ylabel('Best energy'); ax.legend(frameon=False); ax.set_title('Best-energy trajectory'); _save(fig,out/'best_energy_trajectory')

    if exact_optimum is not None:
        gap=df[df.metric=='best_energy_gap'].copy(); fig,ax=plt.subplots(figsize=(5.2,3.3))
        for method,sub in gap.groupby('method',sort=False):
            mean=np.maximum(sub['mean'].to_numpy(float),args.target_atol); q10=np.maximum(sub.q10.to_numpy(float),args.target_atol); q90=np.maximum(sub.q90.to_numpy(float),args.target_atol)
            ax.plot(sub.t,mean,label=method); ax.fill_between(sub.t,q10,q90,alpha=.15)
        ax.set_yscale('log'); ax.set_xlabel('Global tick'); ax.set_ylabel(r'$\max(E_{best}-E^*,\epsilon)$'); ax.legend(frameon=False); ax.set_title('Best-energy gap to exact global optimum (log scale)'); _save(fig,out/'best_energy_trajectory_log_gap')

    for metric,ylabel in [('O','Oscillation $O$'),('q','Parallelism $q$'),('xi_mean','Mean inertia $\\bar{\\xi}$'),('beta','Inverse temperature $\\beta$'),('eta','Noise amplitude $\\eta$')]:
        fig,ax=plt.subplots(figsize=(5.2,3.3))
        for method,sub in df[df.metric==metric].groupby('method',sort=False): ax.plot(sub.t,sub['mean'],label=method); ax.fill_between(sub.t,sub.q10,sub.q90,alpha=.15)
        ax.set_xlabel('Global tick'); ax.set_ylabel(ylabel); ax.legend(frameon=False); ax.set_title(f'{ylabel}: ensemble trajectory'); _save(fig,out/f'trajectory_{metric}')
    print('Saved',out)
if __name__=='__main__':main()
