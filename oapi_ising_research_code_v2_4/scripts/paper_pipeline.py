#!/usr/bin/env python
from __future__ import annotations
"""Convenience orchestrator for the frozen-target -> test benchmark -> figures workflow.

This intentionally does not tune on the test set. Run auto_tune.py separately on
TUNING seeds, then pass its best_config.json here.
"""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    print('\n$', ' '.join(map(str,cmd)), flush=True)
    subprocess.run([str(x) for x in cmd],check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--problem',default='er',choices=['er','signed_er','sk','planted','planted_tsp']); ap.add_argument('--n',type=int,default=128); ap.add_argument('--p',type=float,default=.30)
    ap.add_argument('--test-seeds',default='200,201,202,203,204,205,206,207,208,209')
    ap.add_argument('--runs-per-instance',type=int,default=100); ap.add_argument('--reference-runs-per-method',type=int,default=256)
    ap.add_argument('--batch',type=int,default=50); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--device',default='auto')
    ap.add_argument('--config-json',required=True,help='best_config.json from tuning-only auto_tune.py')
    ap.add_argument('--fixed-xi',type=float,default=.30); ap.add_argument('--fixed-q',type=float,default=.50)
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'pipeline'))
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    targets=out/'targets.csv'; bench=out/'benchmark'
    run([sys.executable,ROOT/'scripts'/'estimate_targets.py','--mode','best_known','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.test_seeds,'--runs-per-method',args.reference_runs_per_method,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--config-json',args.config_json,'--fixed-xi',args.fixed_xi,'--fixed-q',args.fixed_q,'--out',targets])
    run([sys.executable,ROOT/'scripts'/'paper_benchmark.py','--problem',args.problem,'--n',args.n,'--p',args.p,'--instance-seeds',args.test_seeds,'--runs-per-instance',args.runs_per_instance,'--batch',args.batch,'--steps',args.steps,'--device',args.device,'--targets',targets,'--config-json',args.config_json,'--fixed-xi',args.fixed_xi,'--fixed-q',args.fixed_q,'--out',bench])
    run([sys.executable,ROOT/'scripts'/'make_paper_figures.py','--runs',bench/'runs.csv','--summary',bench/'summary.csv','--out',bench/'figures'])
    print('\nPipeline complete:',out)
if __name__=='__main__':main()
