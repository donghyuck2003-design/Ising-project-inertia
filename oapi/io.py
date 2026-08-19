from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def save_result(result, out_dir, prefix="run"):
    out = ensure_dir(out_dir)
    with open(out / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        out / f"{prefix}_states.npz",
        best_state=result.best_state,
        final_state=result.final_state,
        best_energy=result.best_energy,
        final_energy=result.final_energy,
        update_opportunities=result.update_opportunities,
        restarts=result.restarts,
        first_hit_tick=result.first_hit_tick,
        first_hit_update_ops=result.first_hit_update_ops,
    )
    np.savez_compressed(out / f"{prefix}_history.npz", **result.history)


def history_to_frame(result):
    h = result.history
    rows = []
    for li, t in enumerate(h["t"]):
        B = len(h["energy"][li])
        for b in range(B):
            rows.append({
                "t": int(t), "batch": b,
                "energy": float(h["energy"][li, b]),
                "best_energy": float(h["best_energy"][li, b]),
                "O": float(h["O"][li, b]),
                "q": float(h["q"][li, b]),
                "xi_mean": float(h["xi_mean"][li, b]),
                "xi_max": float(h["xi_max"][li, b]),
                "beta": float(h["beta"][li, b]),
                "eta": float(h["eta"][li, b]),
                "dxi_abs_mean": float(h["dxi_abs_mean"][li, b]),
                "clip_rate": float(h["clip_rate"][li, b]),
                "stalled": int(h["stalled"][li, b]),
            })
    return pd.DataFrame(rows)
