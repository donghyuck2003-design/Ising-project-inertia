#!/usr/bin/env python
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.benchmark import make_problem
from oapi.exact_solvers import verify_formulation, solve_exact_ising, ising_energy

for fam in ('er','signed_er','sk'):
    p=make_problem(fam,10,0.30,123)
    verify_formulation(p,samples=128,atol=1e-9)
    r=solve_exact_ising(p,'enumeration',enumeration_max_spins=12)
    assert r.optimality_proven and r.state is not None
    assert np.isclose(r.energy,ising_energy(p,r.state),atol=1e-9)
    print(f'{fam}: exact E={r.energy:.10g}, product_vars={r.n_product_vars}')
print('EXACT SOLVER FORMULATION SMOKE TEST: PASS')
