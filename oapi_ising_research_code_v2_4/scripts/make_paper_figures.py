#!/usr/bin/env python
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.reporting import make_all_figures

def main():
    ap=argparse.ArgumentParser(description='Generate publication-quality PDF/PNG figures and LaTeX/CSV table')
    ap.add_argument('--runs',default=str(ROOT/'results'/'paper_v2'/'benchmark'/'runs.csv'))
    ap.add_argument('--summary',default=str(ROOT/'results'/'paper_v2'/'benchmark'/'summary.csv'))
    ap.add_argument('--out',default=str(ROOT/'results'/'paper_v2'/'benchmark'/'figures'))
    args=ap.parse_args(); make_all_figures(args.runs,args.summary,args.out); print('Saved figures/tables to',args.out)
if __name__=='__main__':main()
