#!/usr/bin/env python
from __future__ import annotations
"""Resume-safe end-to-end OAPI paper workflow.

Completed tuning/validation artifacts can be reused, while unfinished target
creation and paper benchmarks resume in-place.  This is intended for long GPU
runs launched with nohup.
"""
import argparse, json, os, subprocess, sys, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oapi.notify import send_completion_email


def run(cmd):
    cmd=[str(x) for x in cmd]
    print('\n$', ' '.join(cmd), flush=True)
    subprocess.run(cmd,check=True)


def _resolved_target_mode(problem: str, requested: str) -> str:
    if requested != 'auto': return requested
    if problem in {'planted','planted_tsp'}: return 'planted'
    return 'best_known'


def _seed_list(s: str): return [int(x) for x in s.split(',') if x.strip()]


def csv_complete(path: Path, seeds: str, require_target: bool=True, require_optimal: bool=False) -> bool:
    if not path.is_file(): return False
    try: df=pd.read_csv(path)
    except Exception: return False
    if 'instance_seed' not in df.columns: return False
    req=set(_seed_list(seeds)); sub=df[df['instance_seed'].astype(int).isin(req)].copy()
    if set(sub['instance_seed'].astype(int)) != req: return False
    if require_target:
        if 'target_energy' not in sub.columns: return False
        if not np.isfinite(pd.to_numeric(sub['target_energy'],errors='coerce')).all(): return False
    if require_optimal:
        if 'optimality_proven' not in sub.columns: return False
        if not sub['optimality_proven'].fillna(False).astype(bool).all(): return False
    return True


def parse_args():
    ap=argparse.ArgumentParser(description='Full OAPI v2.3 resume-safe paper workflow')
    ap.add_argument('--problem',choices=['er','signed_er','sk','planted','planted_tsp'],default='er')
    ap.add_argument('--n',type=int,default=128); ap.add_argument('--p',type=float,default=.30)
    ap.add_argument('--tuning-seeds',default='1,2,3,4,5'); ap.add_argument('--validation-seeds',default='50,51,52,53,54'); ap.add_argument('--test-seeds',default='200,201,202,203,204,205,206,207,208,209')
    ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--batch',type=int,default=25); ap.add_argument('--device',default='auto')
    ap.add_argument('--fixed-runs-per-value',type=int,default=64); ap.add_argument('--tune-trials',type=int,default=40); ap.add_argument('--tune-runs-per-instance',type=int,default=64)
    ap.add_argument('--validation-runs-per-instance',type=int,default=100); ap.add_argument('--reference-runs-per-method',type=int,default=256); ap.add_argument('--test-runs-per-instance',type=int,default=100)
    ap.add_argument('--top-k-validation',type=int,default=5)
    ap.add_argument('--target-mode',choices=['auto','best_known','exact','planted','milp_exact'],default='auto')
    ap.add_argument('--max-exact-spins',type=int,default=24); ap.add_argument('--target-atol',type=float,default=1e-6)
    ap.add_argument('--exact-backend',choices=['auto','gurobi','cplex','scip','enumeration'],default='auto')
    ap.add_argument('--exact-time-limit-s',type=float,default=0.0); ap.add_argument('--exact-threads',type=int,default=0); ap.add_argument('--exact-solver-log',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=5000); ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'full_workflow'))
    ap.add_argument('--reuse-existing',action=argparse.BooleanOptionalAction,default=False,
                    help='Reuse completed stage outputs and append missing test trajectories in the existing output tree')
    ap.add_argument('--recipient',default=os.environ.get('RECIPIENT','donghyuck200@naver.com'))
    ap.add_argument('--send-mail-script',default=os.environ.get('SEND_MAIL_SCRIPT','/home/onion120/mail/send_mail.sh'))
    ap.add_argument('--no-email',action='store_true')
    return ap.parse_args()


def target_cmd(args,seeds,out_csv,config_json=None,fixed_xi=None,fixed_q=None,mode=None):
    mode=mode or _resolved_target_mode(args.problem,args.target_mode)
    cmd=[sys.executable,ROOT/'scripts'/'estimate_targets.py','--mode',mode,'--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',seeds,'--runs-per-method',args.reference_runs_per_method,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--max-exact-spins',args.max_exact_spins,'--out',out_csv]
    if args.reuse_existing: cmd += ['--reuse-existing']
    if config_json is not None: cmd += ['--config-json',config_json]
    if fixed_xi is not None: cmd += ['--fixed-xi',fixed_xi]
    if fixed_q is not None: cmd += ['--fixed-q',fixed_q]
    return cmd


def workflow(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    target_mode=_resolved_target_mode(args.problem,args.target_mode)
    if target_mode=='planted' and args.problem not in {'planted','planted_tsp'}: raise ValueError('target-mode planted requires planted/planted_tsp')
    if target_mode=='milp_exact' and args.problem not in {'er','signed_er','sk'}: raise ValueError('milp_exact is for er/signed_er/sk')

    tuning_targets=None
    if target_mode in {'planted','exact'}:
        tuning_targets=out/'00_tuning_global_optima.csv'
        if args.reuse_existing and csv_complete(tuning_targets,args.tuning_seeds):
            print('REUSE stage 00:',tuning_targets,flush=True)
        else:
            run(target_cmd(args,args.tuning_seeds,tuning_targets))

    fixed_dir=out/'01_fixed_tuning'; rec_path=fixed_dir/'recommended_fixed_baselines.json'
    if args.reuse_existing and rec_path.is_file():
        print('REUSE stage 01:',rec_path,flush=True)
    else:
        cmd=[sys.executable,ROOT/'scripts'/'tune_fixed_baselines_v2.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.tuning_seeds,'--runs-per-value',args.fixed_runs_per_value,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--target-atol',args.target_atol,'--out',fixed_dir]
        if tuning_targets is not None: cmd += ['--targets',tuning_targets]
        run(cmd)
    rec=json.loads(rec_path.read_text()); fixed_xi=rec['fixed_pimi']['best_value']; fixed_q=rec['fixed_partial']['best_value']

    tune_dir=out/'02_controller_tuning'; tune_best=tune_dir/'best_config.json'; tune_trials=tune_dir/'tuning_trials.csv'
    reuse_tune=False
    if args.reuse_existing and tune_best.is_file() and tune_trials.is_file():
        try: reuse_tune=len(pd.read_csv(tune_trials))>=args.tune_trials
        except Exception: reuse_tune=False
    if reuse_tune:
        print('REUSE stage 02:',tune_best,flush=True)
    else:
        cmd=[sys.executable,ROOT/'scripts'/'auto_tune.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.tuning_seeds,'--trials',args.tune_trials,'--runs-per-instance',args.tune_runs_per_instance,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--target-atol',args.target_atol,'--out',tune_dir]
        if tuning_targets is not None: cmd += ['--targets',tuning_targets]
        run(cmd)

    val_targets=out/'03_validation_targets.csv'; validation_mode='best_known' if target_mode=='milp_exact' else target_mode
    if args.reuse_existing and csv_complete(val_targets,args.validation_seeds):
        print('REUSE stage 03:',val_targets,flush=True)
    else:
        run(target_cmd(args,args.validation_seeds,val_targets,tune_best,fixed_xi,fixed_q,mode=validation_mode))

    val_dir=out/'04_validation'; validated=val_dir/'best_config_validated.json'
    if args.reuse_existing and validated.is_file():
        print('REUSE stage 04:',validated,flush=True)
    else:
        run([sys.executable,ROOT/'scripts'/'validate_tuning.py','--tuning-trials',tune_trials,'--top-k',args.top_k_validation,'--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.validation_seeds,'--targets',val_targets,'--runs-per-instance',args.validation_runs_per_instance,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--target-atol',args.target_atol,'--out',val_dir])

    if target_mode=='milp_exact':
        test_targets=out/'test_global_optima.csv'
        if args.reuse_existing and csv_complete(test_targets,args.test_seeds,require_target=True,require_optimal=True):
            print('REUSE exact test optima:',test_targets,flush=True)
        else:
            cmd=[sys.executable,ROOT/'scripts'/'solve_test_global_optima.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.test_seeds,'--backend',args.exact_backend,'--time-limit-s',args.exact_time_limit_s,'--threads',args.exact_threads,'--target-abs-tol',0.0,'--out',test_targets]
            if args.reuse_existing: cmd += ['--reuse-existing']
            if args.exact_solver_log: cmd += ['--solver-log']
            run(cmd)
    else:
        test_targets=out/'05_test_targets.csv'
        # Always invoke in resume mode if requested; estimate_targets will skip
        # completed seeds and checkpoint each missing seed.
        if args.reuse_existing and csv_complete(test_targets,args.test_seeds):
            print('REUSE stage 05:',test_targets,flush=True)
        else:
            run(target_cmd(args,args.test_seeds,test_targets,validated,fixed_xi,fixed_q))

    bench=out/'06_test_benchmark'
    bench_cmd=[sys.executable,ROOT/'scripts'/'paper_benchmark.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.test_seeds,'--runs-per-instance',args.test_runs_per_instance,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--targets',test_targets,'--config-json',validated,'--fixed-xi',fixed_xi,'--fixed-q',fixed_q,'--target-atol',args.target_atol,'--bootstrap',args.bootstrap,'--out',bench]
    if args.reuse_existing: bench_cmd += ['--reuse-existing']
    run(bench_cmd)
    run([sys.executable,ROOT/'scripts'/'make_paper_figures.py','--runs',bench/'runs.csv','--summary',bench/'summary.csv','--out',bench/'figures'])
    first_test=args.test_seeds.split(',')[0]
    run([sys.executable,ROOT/'scripts'/'paper_trajectory_ensemble.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seed',first_test,'--runs',min(100,args.test_runs_per_instance),'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--config-json',validated,'--fixed-xi',fixed_xi,'--fixed-q',fixed_q,'--targets',test_targets,'--target-atol',args.target_atol,'--out',bench/'figures'/'trajectory'])

    manifest={'version':'2.3.0','target_mode_resolved':target_mode,'fixed_xi':fixed_xi,'fixed_q':fixed_q,'validated_config':str(validated),'test_targets':str(test_targets),'benchmark':str(bench),'arguments':vars(args)}
    (out/'WORKFLOW_COMPLETE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\nFULL WORKFLOW COMPLETE:',out); return out,manifest


def main():
    args=parse_args(); status='SUCCESS'; err=''; out=Path(args.out)
    try: workflow(args)
    except Exception as exc:
        status='FAILED'; err=f'{type(exc).__name__}: {exc}'; traceback.print_exc(); raise
    finally:
        if not args.no_email:
            body=('OAPI paper workflow finished.\n\n'+f'Status={status}\nError={err or "None"}\nProblem={args.problem}, n={args.n}, p={args.p}\nTargetMode={_resolved_target_mode(args.problem,args.target_mode)}\nBatch={args.batch}\nTestRunsPerInstance={args.test_runs_per_instance}\nReuseExisting={args.reuse_existing}\nDevice={args.device}\nOutput={out.resolve()}\nFinishedAt={datetime.now().astimezone().isoformat()}\n')
            send_completion_email(args.recipient,f'[{status}] OAPI v2.3 paper workflow finished',body,script_info='scripts/full_paper_workflow.py',send_mail_script=args.send_mail_script)
if __name__=='__main__': main()
