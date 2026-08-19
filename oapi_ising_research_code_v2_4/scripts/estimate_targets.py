#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, json, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oapi.config import SolverConfig
from oapi.config_io import apply_overrides, load_overrides
from oapi.benchmark import make_problem, run_method_trajectories
from oapi.references import exact_ground_state


def ints(x): return [int(v) for v in x.split(',') if v.strip()]


def state_hash(state) -> str:
    a = np.ascontiguousarray(np.asarray(state, dtype=np.int8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def atomic_csv(df: pd.DataFrame, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def compatible_existing(out: Path, args) -> pd.DataFrame:
    if not (args.reuse_existing and out.is_file()):
        return pd.DataFrame()
    try:
        df = pd.read_csv(out)
    except Exception:
        return pd.DataFrame()
    needed = {'problem','n','p','instance_seed','target_energy'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    ok = (
        df['problem'].astype(str).eq(str(args.problem)).all()
        and pd.to_numeric(df['n'], errors='coerce').eq(args.n).all()
        and np.allclose(pd.to_numeric(df['p'], errors='coerce'), args.p, rtol=0, atol=1e-12)
    )
    if not ok:
        raise ValueError(f'Existing target CSV is incompatible with current problem/n/p: {out}')
    return df.copy()


def main():
    ap = argparse.ArgumentParser(description="Create frozen target/global-optimum energies for success/TTS experiments")
    ap.add_argument('--mode', choices=['exact','best_known','planted'], default='best_known')
    ap.add_argument('--problem', choices=['er','signed_er','sk','planted','planted_tsp'], default='er')
    ap.add_argument('--n', type=int, default=128); ap.add_argument('--p', type=float, default=.30)
    ap.add_argument('--instance-seeds', default='100,101,102,103,104')
    ap.add_argument('--methods', default='fixed_pimi,fixed_partial,joint_restart')
    ap.add_argument('--runs-per-method', type=int, default=256)
    ap.add_argument('--batch', type=int, default=25); ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--solver-seed-base', type=int, default=900000)
    ap.add_argument('--device', default='auto'); ap.add_argument('--config-json', default=None)
    ap.add_argument('--fixed-xi', type=float, default=.30); ap.add_argument('--fixed-q', type=float, default=.50)
    ap.add_argument('--max-exact-spins', type=int, default=24)
    ap.add_argument('--target-abs-tol', type=float, default=0.0, help='Target = reference energy + abs tol')
    ap.add_argument('--reuse-existing', action=argparse.BooleanOptionalAction, default=False,
                    help='Reuse completed instance rows in --out and checkpoint each newly completed instance atomically')
    ap.add_argument('--out', default=str(ROOT/'results'/'paper_v2'/'targets.csv'))
    args = ap.parse_args()

    if args.mode == 'planted' and args.problem not in {'planted','planted_tsp'}:
        raise ValueError("--mode planted requires --problem planted or planted_tsp")
    if args.batch <= 0 or args.runs_per_method <= 0:
        raise ValueError('--batch and --runs-per-method must be positive')

    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    existing = compatible_existing(out, args)
    requested = ints(args.instance_seeds)
    rows = existing.to_dict('records') if not existing.empty else []
    done_seeds = set()
    if not existing.empty:
        expected_reference_runs = args.runs_per_method * len([m for m in args.methods.split(',') if m.strip()])
        for r in existing.itertuples():
            try:
                finite_target = np.isfinite(float(r.target_energy))
                ref_type = str(getattr(r, 'reference_type', ''))
                ref_runs = int(getattr(r, 'reference_runs', 0) or 0)
                enough_budget = (ref_type != 'best_known_disjoint_budget') or (ref_runs >= expected_reference_runs)
                if finite_target and enough_budget:
                    done_seeds.add(int(r.instance_seed))
            except Exception:
                pass
    if done_seeds:
        print(f'Reusing {len(done_seeds)} completed target instance(s) from {out}: {sorted(done_seeds)}', flush=True)

    overrides = load_overrides(args.config_json)
    for ii, iseed in enumerate(requested):
        if iseed in done_seeds:
            print(f'instance={iseed} SKIP (existing target)', flush=True)
            continue

        problem = make_problem(args.problem, args.n, args.p, iseed)
        exact_state = None
        exact_route_distance = np.nan
        exact_route = ''

        if args.mode == 'exact':
            ref = exact_ground_state(problem, max_spins=args.max_exact_spins)
            energy = float(ref['energy'])
            exact_state = ref['state']
            meta = {
                'reference_type':'exact_exhaustive',
                'reference_runs':0,
                'reference_methods':'exhaustive',
                'exact_global_optimum_known':True,
            }
        elif args.mode == 'planted':
            md = problem.metadata or {}
            if 'exact_optimum_energy' not in md or 'exact_optimum_state' not in md:
                raise ValueError(f"Problem {args.problem} does not expose planted exact optimum metadata")
            energy = float(md['exact_optimum_energy'])
            exact_state = np.asarray(md['exact_optimum_state'], dtype=np.int8)
            if 'exact_optimum_route_distance' in md:
                exact_route_distance = float(md['exact_optimum_route_distance'])
            if 'exact_optimum_route' in md:
                exact_route = '-'.join(map(str, np.asarray(md['exact_optimum_route']).tolist()))
            meta = {
                'reference_type':str(md.get('exact_optimum_source','planted_by_construction')),
                'reference_runs':0,
                'reference_methods':'none; optimum planted before solver execution',
                'exact_global_optimum_known':True,
            }
        else:
            best = float('inf')
            methods = [m.strip() for m in args.methods.split(',') if m.strip()]
            for mi, method in enumerate(methods):
                base = SolverConfig(steps=args.steps, batch_size=args.batch, device=args.device)
                base.controller.xi_fixed = args.fixed_xi
                base.controller.q_fixed = args.fixed_q
                apply_overrides(base, overrides)
                print(f'instance={iseed} reference_method={method} runs={args.runs_per_method} batch={args.batch}', flush=True)
                df = run_method_trajectories(
                    problem, method, base, args.runs_per_method, args.batch,
                    args.solver_seed_base + ii*100000 + mi*10000, iseed, target_energy=None
                )
                best = min(best, float(df.best_energy.min()))
            energy = best
            meta = {
                'reference_type':'best_known_disjoint_budget',
                'reference_runs':args.runs_per_method*len(methods),
                'reference_methods':','.join(methods),
                'exact_global_optimum_known':False,
            }

        row = {
            'problem':args.problem,
            'n':args.n,
            'p':args.p,
            'instance_seed':iseed,
            'reference_energy':energy,
            'exact_global_optimum_energy':energy if meta['exact_global_optimum_known'] else np.nan,
            'target_energy':energy+args.target_abs_tol,
            'target_abs_tol':args.target_abs_tol,
            'exact_optimum_state_hash':state_hash(exact_state) if exact_state is not None else '',
            'exact_optimum_route_distance':exact_route_distance,
            'exact_optimum_route':exact_route,
            **meta,
        }
        # Remove a stale/unusable row for the same seed, append the completed row,
        # and checkpoint immediately. A later SIGSEGV therefore loses at most one seed.
        rows = [r for r in rows if int(r.get('instance_seed', -1)) != iseed]
        rows.append(row)
        ordered = pd.DataFrame(rows)
        if 'instance_seed' in ordered.columns:
            ordered = ordered.sort_values('instance_seed').reset_index(drop=True)
        atomic_csv(ordered, out)
        done_seeds.add(iseed)
        print(f"instance={iseed} target={energy+args.target_abs_tol:.9g} ({meta['reference_type']}) checkpointed", flush=True)

    final = pd.read_csv(out) if out.is_file() else pd.DataFrame(rows)
    final = final[final['instance_seed'].astype(int).isin(requested)].copy()
    missing = sorted(set(requested) - set(final['instance_seed'].astype(int)))
    if missing:
        raise RuntimeError(f'Target generation incomplete; missing instance seeds: {missing}')
    with open(out.with_suffix('.json'),'w',encoding='utf-8') as f:
        json.dump({'arguments':vars(args),'n_targets':len(final),'completed_instance_seeds':sorted(requested)},f,indent=2)
    print('Saved',out)

if __name__=='__main__': main()
