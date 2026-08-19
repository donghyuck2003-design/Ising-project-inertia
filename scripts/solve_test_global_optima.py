#!/usr/bin/env python
from __future__ import annotations
"""Generate test_global_optima.csv using an external exact MILP solver.

The OAPI stochastic solvers do not participate in this stage.  Each test
instance is generated from its seed, converted to one common linearized binary
MILP, and solved by Gurobi, CPLEX, or SCIP.  A row is labelled exact only when
the backend proves OPTIMAL.  Time-limit incumbents are retained as diagnostics
but are never written into exact_global_optimum_energy/target_energy.
"""
import argparse, hashlib, json, math, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oapi.benchmark import make_problem
from oapi.exact_solvers import solve_exact_ising, available_backends, ExactBackendUnavailable


def ints(x): return [int(v) for v in x.split(',') if v.strip()]

def state_hash(state) -> str:
    if state is None:
        return ''
    a=np.ascontiguousarray(np.asarray(state,dtype=np.int8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main():
    ap=argparse.ArgumentParser(description='Solve ER/SK/Ising test instances to proven global optimality')
    ap.add_argument('--problem',choices=['er','signed_er','sk'],default='er')
    ap.add_argument('--n',type=int,default=64); ap.add_argument('--p',type=float,default=.30)
    ap.add_argument('--instance-seeds',default='200,201,202,203,204,205,206,207,208,209')
    ap.add_argument('--backend',choices=['auto','gurobi','cplex','scip','enumeration'],default='auto')
    ap.add_argument('--time-limit-s',type=float,default=0.0,help='Per-instance wall-clock limit; 0 means solver default/no limit')
    ap.add_argument('--threads',type=int,default=0,help='Per-instance solver threads; 0 means backend default')
    ap.add_argument('--solver-log',action='store_true')
    ap.add_argument('--require-optimal',action=argparse.BooleanOptionalAction,default=True,
                    help='Fail after writing CSV if any instance lacks a proven optimal certificate')
    ap.add_argument('--verify-samples',type=int,default=16)
    ap.add_argument('--verification-atol',type=float,default=1e-7)
    ap.add_argument('--enumeration-max-spins',type=int,default=24)
    ap.add_argument('--target-abs-tol',type=float,default=0.0)
    ap.add_argument('--reuse-existing',action=argparse.BooleanOptionalAction,default=False,help='Reuse already proven instance rows in the existing CSV')
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'test_global_optima.csv'))
    args=ap.parse_args()

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; states={}; failures=[]
    done_seeds=set()
    states_path=out.with_name(out.stem+'_states.npz')
    if args.reuse_existing and states_path.is_file():
        try:
            with np.load(states_path) as z:
                states={k:np.asarray(z[k],dtype=np.int8) for k in z.files}
        except Exception as e:
            print(f'Warning: could not reuse existing exact-state archive: {e}',flush=True)
            states={}
    if args.reuse_existing and out.is_file():
        try:
            old=pd.read_csv(out)
            rows=old.to_dict('records')
            if 'optimality_proven' in old.columns:
                ok=old['optimality_proven'].fillna(False).astype(bool)
                done_seeds=set(old.loc[ok,'instance_seed'].astype(int).tolist())
            print(f'Reusing proven exact rows: {sorted(done_seeds)}',flush=True)
        except Exception as e:
            print(f'Warning: could not reuse existing exact CSV: {e}',flush=True)
            rows=[]; done_seeds=set()
    print('Exact backend availability:',available_backends(),flush=True)
    for iseed in ints(args.instance_seeds):
        if iseed in done_seeds:
            print(f'seed={iseed} SKIP (existing proven optimum)',flush=True)
            continue
        problem=make_problem(args.problem,args.n,args.p,iseed)
        try:
            result=solve_exact_ising(
                problem,args.backend,time_limit_s=args.time_limit_s,threads=args.threads,
                log=args.solver_log,verify_model=True,verify_samples=args.verify_samples,
                verification_atol=args.verification_atol,enumeration_max_spins=args.enumeration_max_spins,
            )
            proven=bool(result.optimality_proven and result.state is not None and np.isfinite(result.energy))
            if result.state is not None:
                states[f'seed_{iseed}']=np.asarray(result.state,dtype=np.int8)
            if not proven:
                failures.append(f'{iseed}:{result.backend}:{result.status}')
            exact=float(result.energy) if proven else math.nan
            row={
                'problem':args.problem,'n':args.n,'p':args.p,'instance_seed':iseed,
                'reference_energy':exact,
                'exact_global_optimum_energy':exact,
                'target_energy':exact + args.target_abs_tol if proven else math.nan,
                'target_abs_tol':args.target_abs_tol,
                'exact_global_optimum_known':proven,
                'optimality_proven':proven,
                'reference_type':f'milp_exact_{result.backend}' if proven else f'milp_unproven_{result.backend}',
                'reference_runs':0,'reference_methods':f'external exact MILP backend={result.backend}',
                'exact_solver_backend':result.backend,'exact_solver_status':result.status,
                'exact_solver_version':result.backend_version,'exact_solver_runtime_s':result.runtime_s,
                'exact_solver_best_bound_energy':result.best_bound_energy,
                'exact_solver_relative_gap':result.relative_gap,'exact_solver_nodes':result.nodes,
                'best_incumbent_energy':result.energy,
                'exact_optimum_state_hash':state_hash(result.state) if proven else '',
                'incumbent_state_hash':state_hash(result.state),
                'milp_spin_binary_vars':result.n_spin_vars,
                'milp_product_binary_vars':result.n_product_vars,
                'milp_constraints':result.n_constraints,
            }
            rows=[r for r in rows if int(r.get('instance_seed',-1)) != iseed]
            rows.append(row)
            print(f"seed={iseed} backend={result.backend} status={result.status} "
                  f"proven={proven} E={result.energy:.10g} bound={result.best_bound_energy:.10g} "
                  f"time={result.runtime_s:.3f}s",flush=True)
        except Exception as e:
            failures.append(f'{iseed}:ERROR:{type(e).__name__}:{e}')
            rows=[r for r in rows if int(r.get('instance_seed',-1)) != iseed]
            rows.append({
                'problem':args.problem,'n':args.n,'p':args.p,'instance_seed':iseed,
                'reference_energy':math.nan,'exact_global_optimum_energy':math.nan,'target_energy':math.nan,
                'target_abs_tol':args.target_abs_tol,'exact_global_optimum_known':False,
                'optimality_proven':False,'reference_type':'milp_error','reference_runs':0,
                'reference_methods':'external exact MILP','exact_solver_backend':args.backend,
                'exact_solver_status':'error','exact_solver_version':'','exact_solver_runtime_s':math.nan,
                'exact_solver_best_bound_energy':math.nan,'exact_solver_relative_gap':math.nan,
                'exact_solver_nodes':math.nan,'best_incumbent_energy':math.nan,
                'exact_optimum_state_hash':'','incumbent_state_hash':'',
                'milp_spin_binary_vars':problem.n,'milp_product_binary_vars':math.nan,'milp_constraints':math.nan,
                'error':f'{type(e).__name__}: {e}',
            })
            traceback.print_exc()

        pd.DataFrame(rows).to_csv(out,index=False)
        if states:
            np.savez_compressed(states_path,**states)

    df=pd.DataFrame(rows); df.to_csv(out,index=False)
    all_optimal=bool(len(df)>0 and df['optimality_proven'].fillna(False).astype(bool).all())
    manifest={
        'arguments':vars(args),'all_optimal':all_optimal,'n_instances':len(df),
        'n_optimal':int(df['optimality_proven'].fillna(False).astype(bool).sum()),
        'failures':failures,'backend_availability_at_start':available_backends(),
        'important':'Only rows with optimality_proven=true are exact global optima. Time-limit incumbents are not exact.',
    }
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Saved',out)
    if args.require_optimal and not all_optimal:
        raise RuntimeError('Not every instance reached a proven optimum. See CSV/JSON. ' + ' | '.join(failures))

if __name__=='__main__': main()
