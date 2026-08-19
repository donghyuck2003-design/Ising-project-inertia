from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_publication_rc():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def _method_boxplot(data, methods, ylabel, title):
    fig, ax = plt.subplots(figsize=(max(5.5, .55 * len(methods)), 3.6))
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    return fig, ax


def plot_energy_distribution(runs: pd.DataFrame, out: Path):
    """Legacy distribution plus user-facing best_energy figure.

    If exact global optima are available for every run, ``best_energy.png``
    plots the gap E_best-E* so multiple instances with different absolute
    optimum energies can be compared on one common axis. Gap=0 is exact.
    Otherwise it falls back to the raw best-energy distribution.
    """
    methods = list(runs.method.drop_duplicates())
    raw = [runs.loc[runs.method == m, 'best_energy'].to_numpy(float) for m in methods]
    fig, ax = _method_boxplot(raw, methods, 'Best Ising energy',
                              'Best-energy distribution across independent trajectories')
    _save(fig, out / 'fig_energy_distribution')

    has_exact = (
        'exact_global_optimum_energy' in runs.columns
        and runs['exact_global_optimum_energy'].notna().all()
        and 'energy_gap_to_global_optimum' in runs.columns
        and runs['energy_gap_to_global_optimum'].notna().all()
    )
    if has_exact:
        gap = [runs.loc[runs.method == m, 'energy_gap_to_global_optimum'].to_numpy(float) for m in methods]
        fig, ax = _method_boxplot(
            gap, methods, r'$E_{best}-E^*$',
            'Best energy compared with the exact global optimum'
        )
        ax.axhline(0.0, linestyle='--', linewidth=1.0, label='Exact global optimum')
        ax.legend(frameon=False)
        _save(fig, out / 'best_energy')

        positive = runs['energy_gap_to_global_optimum'].to_numpy(float)
        tol = runs['target_atol'].to_numpy(float) if 'target_atol' in runs.columns else np.full(len(runs), 1e-6)
        loggap = np.maximum(positive, tol)
        tmp = runs.copy(); tmp['_log_gap'] = loggap
        gap_data = [tmp.loc[tmp.method == m, '_log_gap'].to_numpy(float) for m in methods]
        fig, ax = _method_boxplot(
            gap_data, methods, r'$\max(E_{best}-E^*,\epsilon)$',
            'Exact-global-optimum gap (log scale)'
        )
        ax.set_yscale('log')
        _save(fig, out / 'best_energy_log_gap')
    else:
        fig, ax = _method_boxplot(raw, methods, 'Best Ising energy',
                                  'Best energy (exact global optimum unavailable)')
        _save(fig, out / 'best_energy')


def plot_success_ci(summary: pd.DataFrame, out: Path):
    s = summary.copy(); x = np.arange(len(s))
    y = s.p_success.to_numpy(float)
    lo = y - s.p_success_ci_low.to_numpy(float); hi = s.p_success_ci_high.to_numpy(float) - y
    fig, ax = plt.subplots(figsize=(max(5.5, .55 * len(s)), 3.4))
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt='o', capsize=3)
    ax.set_xticks(x, s.method, rotation=45, ha='right'); ax.set_ylim(-.02, 1.02)
    ax.set_ylabel('Success probability'); ax.set_title('Success probability with 95% bootstrap CI')
    _save(fig, out / 'fig_success_probability')


def plot_tts(summary: pd.DataFrame, out: Path):
    s = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=['tts_wallclock_s']).copy()
    x = np.arange(len(s)); y = s.tts_wallclock_s.to_numpy(float)
    lo = np.maximum(0, y - s.tts_wallclock_s_ci_low.to_numpy(float)); hi = np.maximum(0, s.tts_wallclock_s_ci_high.to_numpy(float) - y)
    fig, ax = plt.subplots(figsize=(max(5.5, .55 * len(s)), 3.4))
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt='o', capsize=3)
    ax.set_xticks(x, s.method, rotation=45, ha='right'); ax.set_yscale('log')
    ax.set_ylabel('TTS$_{0.99}$ (s, throughput-normalized)'); ax.set_title('Wall-clock time-to-solution')
    _save(fig, out / 'fig_tts_wallclock')


def plot_parallelism_success(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    ax.scatter(summary.mean_q, summary.p_success)
    for r in summary.itertuples():
        ax.annotate(str(r.method), (r.mean_q, r.p_success), xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set_xlabel('Mean effective parallelism $\\bar{q}$'); ax.set_ylabel('Success probability')
    ax.set_title('Optimization–parallelism trade-off')
    _save(fig, out / 'fig_parallelism_success')


def plot_oscillation_energy(summary: pd.DataFrame, out: Path):
    ycol = 'global_optimum_gap_mean' if 'global_optimum_gap_mean' in summary.columns else 'best_energy_mean'
    ylabel = r'Mean $E_{best}-E^*$' if ycol == 'global_optimum_gap_mean' else 'Mean best energy'
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    ax.scatter(summary.mean_O, summary[ycol])
    for r in summary.itertuples():
        ax.annotate(str(r.method), (r.mean_O, getattr(r, ycol)), xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set_xlabel('Mean period-2 oscillation statistic $O$'); ax.set_ylabel(ylabel)
    ax.set_title('Oscillation vs optimization quality')
    _save(fig, out / 'fig_oscillation_energy')


def write_tables(summary: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    preferred = [
        'method','n_trajectories','n_instances','best_energy_mean',
        'global_optimum_gap_mean','relative_global_optimum_gap_percent_mean','exact_optimum_hit_rate',
        'p_success','mean_q','mean_O','tts_wallclock_s','tts_ticks',
        'tts_update_opportunities','mean_restarts','route_gap_to_global_optimum_mean'
    ]
    cols = [c for c in preferred if c in summary]
    summary[cols].to_csv(out / 'table_main_metrics.csv', index=False)
    try:
        tex = summary[cols].to_latex(index=False, float_format=lambda x: f'{x:.4g}', escape=True)
        (out / 'table_main_metrics.tex').write_text(tex, encoding='utf-8')
    except Exception as e:
        (out / 'table_main_metrics_tex_error.txt').write_text(str(e), encoding='utf-8')


def make_all_figures(runs_csv: str | Path, summary_csv: str | Path, out_dir: str | Path):
    set_publication_rc()
    runs = pd.read_csv(runs_csv); summary = pd.read_csv(summary_csv); out = Path(out_dir)
    plot_energy_distribution(runs, out)
    plot_success_ci(summary, out)
    plot_tts(summary, out)
    plot_parallelism_success(summary, out)
    plot_oscillation_energy(summary, out)
    write_tables(summary, out)
