#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite-root',required=True)
    args=ap.parse_args(); root=Path(args.suite_root)
    frames=[]
    for p in sorted(root.glob('*/06_test_benchmark/summary.csv')):
        df=pd.read_csv(p)
        df.insert(0,'benchmark',p.parents[1].name)
        frames.append(df)
    if not frames:
        print('No benchmark summaries found under',root); return
    out=pd.concat(frames,ignore_index=True)
    out.to_csv(root/'combined_summary.csv',index=False)
    preferred=[c for c in ['benchmark','method','p_success','exact_optimum_hit_rate','global_optimum_gap_mean','best_energy_mean','mean_q','mean_O','tts_wallclock_s'] if c in out]
    out[preferred].to_csv(root/'combined_main_metrics.csv',index=False)
    print(out[preferred].to_string(index=False))
    print('Saved',root/'combined_summary.csv')

if __name__=='__main__': main()
