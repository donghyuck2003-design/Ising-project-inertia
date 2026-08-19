#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description='Plot mean controller trajectories from *_history.npz')
    ap.add_argument('npz'); ap.add_argument('--out',default=None); args=ap.parse_args()
    z=np.load(args.npz); t=z['t']; out=Path(args.out) if args.out else Path(args.npz).with_suffix('')
    out.mkdir(parents=True,exist_ok=True)
    for key in ['best_energy','energy','O','q','xi_mean','xi_max','beta','eta','dxi_abs_mean','clip_rate']:
        if key not in z: continue
        y=z[key]; mean=y.mean(axis=1); lo=np.quantile(y,.25,axis=1); hi=np.quantile(y,.75,axis=1)
        fig,ax=plt.subplots(figsize=(7,4.2)); ax.plot(t,mean,label='mean'); ax.fill_between(t,lo,hi,alpha=.2,label='IQR'); ax.set_xlabel('global tick'); ax.set_ylabel(key); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{key}.png',dpi=180); plt.close(fig)
    print('Saved',out)
if __name__=='__main__': main()
