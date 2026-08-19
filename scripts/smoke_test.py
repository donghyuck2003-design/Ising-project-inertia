#!/usr/bin/env python
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.config import SolverConfig
from oapi.problems import make_er_maxcut, make_tsp_ising, tsp_decode
from oapi.solver import IsingSolver

p=make_er_maxcut(24,0.4,1)
cfg=SolverConfig(steps=80,batch_size=4,log_every=10,seed=2)
cfg.controller.xi_mode="adamw"; cfg.controller.adaptive_q=True; cfg.anneal.mode="event_restart"
r=IsingSolver(p,cfg).run()
assert r.best_state.shape==(4,24)
assert np.all(np.isfinite(r.best_energy))
assert np.all((r.history["q"]>=cfg.controller.q_min-1e-6)&(r.history["q"]<=1+1e-6))

p2,coords,D=make_tsp_ising(4,2,A=5,B=1)
cfg2=SolverConfig(steps=50,batch_size=2,log_every=10,seed=3)
r2=IsingSolver(p2,cfg2).run(); tsp_decode(r2.best_state[0],4,D)
print("SMOKE TEST PASSED",r.summary)
