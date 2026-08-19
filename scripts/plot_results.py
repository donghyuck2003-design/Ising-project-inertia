#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description="Simple paper-development plots from ablation CSV")
    ap.add_argument("csv"); ap.add_argument("--out",default="plots"); args=ap.parse_args()
    df=pd.read_csv(args.csv); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if "method" not in df: raise ValueError("CSV needs a method column")
    g=df.groupby("method",as_index=False).agg(best_energy=("best_energy","mean"),mean_q=("mean_q","mean"),mean_O=("mean_O","mean"),runtime=("runtime_s_total_batch","mean"))
    for y in ["best_energy","mean_q","mean_O","runtime"]:
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(g["method"],g[y]); ax.set_ylabel(y); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(out/f"{y}.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(g["mean_q"],g["best_energy"])
    for _,r in g.iterrows(): ax.annotate(r["method"],(r["mean_q"],r["best_energy"]),fontsize=8)
    ax.set_xlabel("Mean effective parallelism q"); ax.set_ylabel("Mean best Ising energy"); fig.tight_layout(); fig.savefig(out/"pareto_q_vs_energy.png",dpi=180); plt.close(fig)
    g.to_csv(out/"method_summary.csv",index=False)

if __name__=="__main__": main()
