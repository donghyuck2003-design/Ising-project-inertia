#!/usr/bin/env python
"""Prepare exact global optima *before* OAPI benchmark routes are run.

For planted/planted_tsp problems this does not solve the instance. The optimum
state/route is generated first and the couplings/costs are constructed around
it, so the exact global optimum is known by construction.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description='Pre-create exact planted global-optimum manifest')
    ap.add_argument('--problem',choices=['planted','planted_tsp'],default='planted')
    ap.add_argument('--n',type=int,default=128,help='spins for planted; cities for planted_tsp')
    ap.add_argument('--p',type=float,default=.30,help='edge density for planted Ising; ignored by planted_tsp')
    ap.add_argument('--instance-seeds',default='200,201,202,203,204,205,206,207,208,209')
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'global_optima.csv'))
    args=ap.parse_args()
    cmd=[
        sys.executable, str(ROOT/'scripts'/'estimate_targets.py'),
        '--mode','planted','--problem',args.problem,'--n',str(args.n),'--p',str(args.p),
        '--instance-seeds',args.instance_seeds,'--out',args.out,
    ]
    print('$',' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)
    print('Exact global-optimum manifest prepared before solver benchmarking:',args.out)

if __name__=='__main__': main()
