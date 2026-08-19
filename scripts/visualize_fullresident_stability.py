#!/usr/bin/env python3
"""Visualize full-resident p-bit Ising stability sweeps.

Designed for stability_map_fullresident.csv produced by
fullresident_oapi_experiment.py --phase stability.

Main outputs (per problem family):
  1) period-2 oscillation heatmap O(q, xi)
  2) seed-relative best-energy-gap heatmap (lower is better)
  3) q-vs-oscillation curves for each xi
  4) q-vs-energy-gap curves for each xi
  5) oscillation-vs-energy-gap scatter
  6) linearized spectral-radius heatmap, if available

Also writes:
  stability_aggregated.csv
  stability_findings.txt

Example
-------
python scripts/visualize_fullresident_stability.py \
  --csv results/fullresident_stability_20260819_155728/stability_map_fullresident.csv \
  --out results/fullresident_stability_20260819_155728/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED = {
    "family",
    "instance_seed",
    "q",
    "xi",
    "best_energy_mean",
    "O_mean",
    "flip_fraction_mean",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results/fullresident_stability_20260819_155728/stability_map_fullresident.csv"),
        help="Input stability_map_fullresident.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <csv_dir>/figures",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated figure formats, e.g. png,pdf",
    )
    p.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate heatmap cells with numerical values",
    )
    return p.parse_args()


def clean_family_name(x: str) -> str:
    x = str(x).strip()
    return {"er": "Dense ER Max-Cut", "sk": "SK"}.get(x.lower(), x.upper())


def validate(df: pd.DataFrame) -> None:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input CSV is empty.")
    for c in ["q", "xi", "best_energy_mean", "O_mean", "flip_fraction_mean"]:
        if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all():
            raise ValueError(f"Column {c!r} contains non-numeric or non-finite values.")


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    numeric = [
        "q", "xi", "best_energy_mean", "best_energy_std", "O_mean",
        "flip_fraction_mean", "runtime_s", "mean_update_opportunities",
        "spectral_rho_D1_beta_max", "lambda_min", "lambda_max", "spectral_radius_J",
    ]
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Comparing raw energy across different random graph instances can be misleading.
    # Subtract the best parameter-grid result observed for each individual instance.
    # 0 therefore means "best tested (q, xi) point for that seed"; lower is better.
    d["energy_gap_to_seed_best"] = (
        d["best_energy_mean"]
        - d.groupby(["family", "instance_seed"])["best_energy_mean"].transform("min")
    )

    agg_spec = {
        "n_instances": ("instance_seed", "nunique"),
        "best_energy_mean": ("best_energy_mean", "mean"),
        "best_energy_across_seed_std": ("best_energy_mean", "std"),
        "energy_gap_to_seed_best_mean": ("energy_gap_to_seed_best", "mean"),
        "energy_gap_to_seed_best_std": ("energy_gap_to_seed_best", "std"),
        "O_mean": ("O_mean", "mean"),
        "O_across_seed_std": ("O_mean", "std"),
        "flip_fraction_mean": ("flip_fraction_mean", "mean"),
        "flip_fraction_across_seed_std": ("flip_fraction_mean", "std"),
    }
    if "runtime_s" in d:
        agg_spec["runtime_s_mean"] = ("runtime_s", "mean")
    if "mean_update_opportunities" in d:
        agg_spec["mean_update_opportunities"] = ("mean_update_opportunities", "mean")
    if "spectral_rho_D1_beta_max" in d:
        agg_spec["spectral_rho_D1_beta_max"] = ("spectral_rho_D1_beta_max", "mean")
    if "spectral_radius_J" in d:
        agg_spec["spectral_radius_J"] = ("spectral_radius_J", "mean")

    agg = (
        d.groupby(["family", "q", "xi"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["family", "q", "xi"])
        .reset_index(drop=True)
    )
    return d, agg


def save_figure(fig: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> None:
    for ext in formats:
        ext = ext.strip().lower().lstrip(".")
        if not ext:
            continue
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def annotated_heatmap(
    table: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_stem: Path,
    formats: list[str],
    dpi: int,
    annotate: bool,
    fmt: str,
) -> None:
    values = table.to_numpy(dtype=float)
    q_values = list(table.columns)
    xi_values = list(table.index)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    im = ax.imshow(values, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(q_values)), [f"{q:g}" for q in q_values])
    ax.set_yticks(np.arange(len(xi_values)), [f"{xi:g}" for xi in xi_values])
    ax.set_xlabel("Update parallelism q")
    ax.set_ylabel("Inertia ξ")
    ax.set_title(title)

    if annotate:
        finite = values[np.isfinite(values)]
        mid = float(np.nanmedian(finite)) if finite.size else 0.0
        for iy in range(values.shape[0]):
            for ix in range(values.shape[1]):
                v = values[iy, ix]
                if not np.isfinite(v):
                    continue
                ax.text(ix, iy, format(v, fmt), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    save_figure(fig, out_stem, formats, dpi)


def line_by_xi(
    a: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    formats: list[str],
    dpi: int,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for xi, g in a.groupby("xi", sort=True):
        g = g.sort_values("q")
        ax.plot(g["q"], g[y], marker="o", label=f"ξ={xi:g}")
    ax.set_xlabel("Update parallelism q")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    save_figure(fig, out_stem, formats, dpi)


def scatter_oscillation_gap(
    d: pd.DataFrame,
    family_label: str,
    out_stem: Path,
    formats: list[str],
    dpi: int,
) -> None:
    # Every point is one (instance seed, q, xi) measurement. q is encoded by marker size;
    # xi is deliberately not color-coded so the figure remains robust in grayscale.
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    sizes = 25 + 120 * d["q"].to_numpy(dtype=float)
    ax.scatter(d["O_mean"], d["energy_gap_to_seed_best"], s=sizes, alpha=0.55)
    ax.set_xlabel("Period-2 oscillation O")
    ax.set_ylabel("Energy gap to best tested setting per instance")
    ax.set_title(f"{family_label}: oscillation vs optimization degradation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_stem, formats, dpi)


def write_findings(d: pd.DataFrame, agg: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("Full-resident stability sweep summary")
    lines.append("=====================================")
    lines.append("")
    lines.append(
        "Energy gap is computed within each family/instance seed relative to the lowest "
        "best_energy_mean observed anywhere in the tested (q, xi) grid. Therefore 0 means "
        "best tested setting, not necessarily the proven global optimum."
    )

    for family in sorted(d["family"].astype(str).unique()):
        g = d[d["family"].astype(str) == family]
        a = agg[agg["family"].astype(str) == family]
        label = clean_family_name(family)
        lines.extend(["", f"[{label}]", "-"])
        q_list = [float(v) for v in sorted(g["q"].unique())]
        xi_list = [float(v) for v in sorted(g["xi"].unique())]
        lines.append(
            f"instances={g['instance_seed'].nunique()}, q={q_list}, xi={xi_list}"
        )

        best = a.loc[a["best_energy_mean"].idxmin()]
        lines.append(
            f"Lowest mean energy over seeds: q={best.q:g}, xi={best.xi:g}, "
            f"E={best.best_energy_mean:.6g}, O={best.O_mean:.6g}."
        )

        q1 = a[np.isclose(a["q"], 1.0)]
        if not q1.empty:
            no_inertia = q1[np.isclose(q1["xi"], 0.0)]
            if not no_inertia.empty:
                r = no_inertia.iloc[0]
                lines.append(
                    f"Fully parallel, no inertia (q=1, xi=0): E={r.best_energy_mean:.6g}, "
                    f"gap={r.energy_gap_to_seed_best_mean:.6g}, O={r.O_mean:.6g}."
                )
            o_min = float(q1["O_mean"].min())
            o_max = float(q1["O_mean"].max())
            if np.isclose(o_min, o_max):
                lines.append(
                    f"Across all tested q=1 inertia values, O is effectively unchanged "
                    f"({o_min:.6g} to {o_max:.6g})."
                )
            else:
                stable_q1 = q1.loc[q1["O_mean"].idxmin()]
                lines.append(
                    f"Lowest oscillation among q=1 settings: xi={stable_q1.xi:g}, "
                    f"O={stable_q1.O_mean:.6g}, gap={stable_q1.energy_gap_to_seed_best_mean:.6g}."
                )

        # A descriptive threshold only, not a statistical significance claim.
        low_osc = a[a["O_mean"] <= 0.01]
        if not low_osc.empty:
            maxq = low_osc["q"].max()
            candidates = low_osc[np.isclose(low_osc["q"], maxq)]
            cand = candidates.loc[candidates["energy_gap_to_seed_best_mean"].idxmin()]
            lines.append(
                f"Highest q with mean O<=0.01: q={maxq:g}; best point there xi={cand.xi:g}, "
                f"gap={cand.energy_gap_to_seed_best_mean:.6g}."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    out = (args.out or (csv_path.parent / "figures")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]

    df = pd.read_csv(csv_path)
    validate(df)
    detail, agg = prepare(df)

    agg.to_csv(out / "stability_aggregated.csv", index=False)
    write_findings(detail, agg, out / "stability_findings.txt")

    for family in sorted(agg["family"].astype(str).unique()):
        a = agg[agg["family"].astype(str) == family].copy()
        d = detail[detail["family"].astype(str) == family].copy()
        label = clean_family_name(family)
        slug = str(family).lower().replace(" ", "_")

        # Heatmap: period-2 oscillation
        t = a.pivot(index="xi", columns="q", values="O_mean").sort_index().sort_index(axis=1)
        annotated_heatmap(
            t,
            f"{label}: period-2 oscillation",
            "Mean period-2 oscillation O",
            out / f"{slug}_heatmap_oscillation",
            formats,
            args.dpi,
            args.annotate,
            ".3f",
        )

        # Heatmap: seed-relative energy degradation
        t = a.pivot(index="xi", columns="q", values="energy_gap_to_seed_best_mean").sort_index().sort_index(axis=1)
        annotated_heatmap(
            t,
            f"{label}: optimization degradation",
            "Mean energy gap to best tested setting",
            out / f"{slug}_heatmap_energy_gap",
            formats,
            args.dpi,
            args.annotate,
            ".2f",
        )

        # Curves make the stabilization boundary easier to see than a heatmap alone.
        line_by_xi(
            a,
            "O_mean",
            "Mean period-2 oscillation O",
            f"{label}: oscillation vs update parallelism",
            out / f"{slug}_q_vs_oscillation",
            formats,
            args.dpi,
        )
        line_by_xi(
            a,
            "energy_gap_to_seed_best_mean",
            "Mean energy gap to best tested setting",
            f"{label}: solution degradation vs update parallelism",
            out / f"{slug}_q_vs_energy_gap",
            formats,
            args.dpi,
        )

        scatter_oscillation_gap(
            d,
            label,
            out / f"{slug}_oscillation_vs_energy_gap",
            formats,
            args.dpi,
        )

        if "spectral_rho_D1_beta_max" in a.columns and a["spectral_rho_D1_beta_max"].notna().any():
            t = a.pivot(index="xi", columns="q", values="spectral_rho_D1_beta_max").sort_index().sort_index(axis=1)
            annotated_heatmap(
                t,
                f"{label}: linearized spectral-radius diagnostic",
                "ρ[A(q, ξ)] using stored D=1 approximation",
                out / f"{slug}_heatmap_spectral_rho",
                formats,
                args.dpi,
                args.annotate,
                ".2f",
            )

    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Families: {', '.join(map(str, sorted(df['family'].unique())))}")
    print(f"Figures/results written to: {out}")
    print(f"Aggregated table: {out / 'stability_aggregated.csv'}")
    print(f"Text summary: {out / 'stability_findings.txt'}")


if __name__ == "__main__":
    main()
