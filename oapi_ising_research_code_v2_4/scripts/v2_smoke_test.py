#!/usr/bin/env python
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_planted_ising, make_planted_tsp_ising
from oapi.references import exact_ground_state
from oapi.solver import IsingSolver
from oapi.statistics import aggregate_paper_metrics

# Existing exact-enumeration + first-hit path.
p=make_er_maxcut(10,.35,123)
ref=exact_ground_state(p,max_spins=12)
cfg=SolverConfig(steps=100,batch_size=8,seed=321,target_energy=ref['energy'],target_atol=1e-6,device='cpu')
r=IsingSolver(p,cfg).run()
assert r.first_hit_tick.shape==(8,)
assert r.first_hit_update_ops.shape==(8,)
rows=[]
for i in range(8):
    gap=max(0.0,float(r.best_energy[i]-ref['energy']))
    rows.append({'method':'joint_restart','instance_seed':123,'best_energy':r.best_energy[i], 'mean_q':r.history['q'][:,i].mean(), 'mean_O':r.history['O'][:,i].mean(), 'update_opportunities':r.update_opportunities[i], 'restarts':r.restarts[i], 'success':int(r.best_energy[i] <= ref['energy']+cfg.target_atol), 'runtime_s_per_trajectory':r.runtime_s/8, 'steps':cfg.steps, 'first_hit_tick':r.first_hit_tick[i], 'first_hit_update_ops':r.first_hit_update_ops[i], 'target_atol':cfg.target_atol, 'energy_gap_to_global_optimum':gap, 'relative_energy_gap_percent':100*gap/max(abs(ref['energy']),cfg.target_atol)})
s=aggregate_paper_metrics(pd.DataFrame(rows),n_boot=50)
assert len(s)==1 and 'tts_wallclock_s' in s.columns and 'global_optimum_gap_mean' in s.columns

# New planted Ising: metadata optimum must equal exhaustive optimum.
pp=make_planted_ising(n=12,p=.35,seed=99)
pr=exact_ground_state(pp,max_spins=12)
assert abs(float(pp.metadata['exact_optimum_energy'])-float(pr['energy'])) < 1e-9
assert np.array_equal(np.asarray(pp.metadata['exact_optimum_state'],dtype=np.int8),np.asarray(pr['state'],dtype=np.int8))

# New planted TSP: 4 cities -> 16 Ising spins, small enough to exhaustively verify
# that the pre-planted route/QUBO state is a true global optimum.
tp,_=make_planted_tsp_ising(n_cities=4,seed=77)
tr=exact_ground_state(tp,max_spins=16)
assert abs(float(tp.metadata['exact_optimum_energy'])-float(tr['energy'])) < 1e-9

print('V2.1 SMOKE TEST PASSED', {
    'enumerated_target':ref['energy'],
    'planted_ising_exact':pp.metadata['exact_optimum_energy'],
    'planted_tsp_exact':tp.metadata['exact_optimum_energy'],
    'p_success':float(s.iloc[0].p_success),
})
