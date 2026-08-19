#!/usr/bin/env python
from __future__ import annotations
import argparse,sys,json,os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.config_io import apply_overrides,load_overrides
from oapi.benchmark import make_problem,run_method_trajectories,load_target_map,reference_row_map
from oapi.statistics import aggregate_paper_metrics,paired_instance_delta_bootstrap

DEFAULT_METHODS='par0,fixed_pimi,fixed_partial,heuristic_xi,momentum_xi,adam_xi,adamw_clip,adamw_rms,adaptive_q,joint,joint_restart'

def ints(x): return [int(v) for v in x.split(',') if v.strip()]

def _exact_from_ref(row):
    if not row:
        return None
    v=row.get('exact_global_optimum_energy',np.nan)
    try: v=float(v)
    except Exception: return None
    return v if np.isfinite(v) else None

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp=path.with_suffix(path.suffix+'.tmp')
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser(description='Paper-scale trajectory benchmark with resume, exact-optimum gaps, bootstrap CIs and TTS')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er'); ap.add_argument('--n',type=int,default=128); ap.add_argument('--p',type=float,default=.30)
    ap.add_argument('--instance-seeds',default='200,201,202,203,204,205,206,207,208,209')
    ap.add_argument('--methods',default=DEFAULT_METHODS); ap.add_argument('--runs-per-instance',type=int,default=100,
                    help='Desired total trajectories per method per test instance after resume')
    ap.add_argument('--batch',type=int,default=25); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--device',default='auto')
    ap.add_argument('--targets',required=True,help='Frozen target/global-optimum CSV; test seeds must match')
    ap.add_argument('--config-json',default=None,help='Tuned controller JSON, typically validation/best_config_validated.json')
    ap.add_argument('--fixed-xi',type=float,default=.30); ap.add_argument('--fixed-q',type=float,default=.50); ap.add_argument('--target-atol',type=float,default=1e-6)
    ap.add_argument('--solver-seed-base',type=int,default=3000000)
    ap.add_argument('--bootstrap',type=int,default=5000); ap.add_argument('--ci',type=float,default=.95); ap.add_argument('--tts-confidence',type=float,default=.99)
    ap.add_argument('--reference-method',default='joint_restart')
    ap.add_argument('--reuse-existing',action=argparse.BooleanOptionalAction,default=False,
                    help='Reuse runs.csv and append only missing trajectories up to --runs-per-instance')
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'benchmark'))
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.batch <= 0 or args.runs_per_instance <= 0: raise ValueError('--batch and --runs-per-instance must be positive')

    targets=load_target_map(args.targets); refs=reference_row_map(args.targets); overrides=load_overrides(args.config_json)
    methods=[m.strip() for m in args.methods.split(',') if m.strip()]; iseeds=ints(args.instance_seeds)
    missing=[s for s in iseeds if s not in targets]
    if missing: raise ValueError(f'Target CSV is missing test instance seeds: {missing}')

    runs_path=out/'runs.csv'
    existing=pd.DataFrame()
    if args.reuse_existing and runs_path.is_file():
        try:
            # Do not mix wall-clock/TTS measurements from a previous GPU batch
            # size with the new one. Reuse is allowed only when the old manifest
            # reports the same requested batch. Otherwise preserve a backup and
            # rerun the test stage in the same existing folder.
            manifest_path=out/'benchmark_manifest.json'
            previous_batch=None
            if manifest_path.is_file():
                try:
                    previous_batch=json.loads(manifest_path.read_text()).get('arguments',{}).get('batch')
                    previous_batch=int(previous_batch) if previous_batch is not None else None
                except Exception:
                    previous_batch=None
            if previous_batch is not None and previous_batch != args.batch:
                backup=out/f'runs.pre_batch{previous_batch}.csv'
                if not backup.exists():
                    import shutil; shutil.copy2(runs_path,backup)
                print(f'Existing benchmark used batch={previous_batch}, requested batch={args.batch}; preserving {backup.name} and rerunning test trajectories for consistent wall-clock/TTS.',flush=True)
                existing=pd.DataFrame()
            else:
                existing=pd.read_csv(runs_path)
            needed={'method','instance_seed','trajectory_id','batch_id','best_energy'}
            if existing.empty:
                pass
            elif not needed.issubset(existing.columns):
                print('Existing runs.csv lacks resume columns; ignoring it.',flush=True)
                existing=pd.DataFrame()
            else:
                # Keep only requested methods/instances. If there are more than the
                # newly requested count, keep the earliest trajectory IDs deterministically.
                existing=existing[
                    existing['method'].astype(str).isin(methods)
                    & existing['instance_seed'].astype(int).isin(iseeds)
                ].copy()
                existing=existing.sort_values(['instance_seed','method','trajectory_id'])
                existing=existing.groupby(['instance_seed','method'],as_index=False,group_keys=False).head(args.runs_per_instance)
                print(f'Reusing {len(existing)} existing test trajectories from {runs_path}',flush=True)
        except Exception as e:
            print(f'Warning: could not reuse existing runs.csv ({e}); starting benchmark rows from scratch.',flush=True)
            existing=pd.DataFrame()

    all_df=existing.copy()
    for ii,iseed in enumerate(iseeds):
        problem=make_problem(args.problem,args.n,args.p,iseed)
        exact_optimum=_exact_from_ref(refs.get(iseed,{}))
        for mi,method in enumerate(methods):
            if all_df.empty:
                old=pd.DataFrame()
            else:
                old=all_df[(all_df.instance_seed.astype(int)==iseed)&(all_df.method.astype(str)==method)].copy()
            have=len(old)
            if have >= args.runs_per_instance:
                print(f'instance={iseed} method={method} SKIP existing={have}/{args.runs_per_instance}',flush=True)
                continue
            need=args.runs_per_instance-have
            batch_start=int(pd.to_numeric(old.get('batch_id',pd.Series(dtype=float)),errors='coerce').max()+1) if have else 0
            base=SolverConfig(steps=args.steps,batch_size=args.batch,device=args.device)
            base.controller.xi_fixed=args.fixed_xi; base.controller.q_fixed=args.fixed_q; base.target_atol=args.target_atol
            apply_overrides(base,overrides)
            print(f'instance={iseed} method={method} RESUME existing={have} add={need} batch={args.batch}',flush=True)
            df=run_method_trajectories(
                problem,method,base,need,args.batch,
                args.solver_seed_base+ii*1000000+mi*10000,iseed,targets[iseed],
                exact_optimum_energy=exact_optimum,
                trajectory_id_start=have,batch_id_start=batch_start,
            )
            if iseed in refs: df['reference_type']=str(refs[iseed].get('reference_type',''))
            all_df=pd.concat([all_df,df],ignore_index=True)
            all_df=all_df.sort_values(['instance_seed','method','trajectory_id']).reset_index(drop=True)
            atomic_csv(all_df,runs_path)  # checkpoint after every method
            gap_text='NA' if exact_optimum is None else f'{df.energy_gap_to_global_optimum.mean():.6g}'
            print(f'instance={iseed} method={method} added={len(df)} E={df.best_energy.mean():.6g} gap*={gap_text} q={df.mean_q.mean():.3f} checkpointed',flush=True)

    runs=all_df.copy()
    counts=runs.groupby(['instance_seed','method']).size()
    incomplete={(int(i),str(m)):int(c) for (i,m),c in counts.items() if int(c)<args.runs_per_instance}
    expected={(s,m) for s in iseeds for m in methods}
    absent=expected-set((int(i),str(m)) for i,m in counts.index)
    if incomplete or absent:
        raise RuntimeError(f'Benchmark incomplete. partial={incomplete}, absent={sorted(absent)}')

    # Restrict exactly to desired count and requested matrix before statistics.
    runs=runs.sort_values(['instance_seed','method','trajectory_id']).groupby(['instance_seed','method'],as_index=False,group_keys=False).head(args.runs_per_instance)
    atomic_csv(runs,runs_path)
    summary=aggregate_paper_metrics(runs,args.tts_confidence,args.ci,args.bootstrap,seed=2026); summary.to_csv(out/'summary.csv',index=False)
    if args.reference_method in methods:
        pair=paired_instance_delta_bootstrap(runs,args.reference_method,'best_energy',args.ci,args.bootstrap,seed=2027); pair.to_csv(out/'paired_energy_deltas.csv',index=False)
        if runs.energy_gap_to_global_optimum.notna().any():
            pair_gap=paired_instance_delta_bootstrap(runs,args.reference_method,'energy_gap_to_global_optimum',args.ci,args.bootstrap,seed=2028); pair_gap.to_csv(out/'paired_global_optimum_gap_deltas.csv',index=False)
    with open(out/'benchmark_manifest.json','w',encoding='utf-8') as f:
        json.dump({'arguments':vars(args),'methods':methods,'instance_seeds':iseeds,'total_trajectories':len(runs),'trajectories_per_method_per_instance':args.runs_per_instance,'exact_global_optimum_available':bool(runs.exact_global_optimum_energy.notna().all())},f,indent=2)
    print('\n=== summary ==='); print(summary.to_string(index=False)); print('Saved',out)
if __name__=='__main__':main()
