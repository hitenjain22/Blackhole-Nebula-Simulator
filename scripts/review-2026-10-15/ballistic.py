"""Gravity-only (ballistic) run of the cloud at high particle number, recording x and p only, to
measure where real stream crossings happen near pericentre and how fast the streams meet.
Usage: uv run python ballistic.py OUT.npz n=50000 radius=100 t0=380000 t1=392000 dt=100"""
import sys, time
import numpy as np
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.cloud import cloud_ic
from nebula.sim.coupled import CoupledConfig

out = sys.argv[1]
o = dict(a.split('=') for a in sys.argv[2:])
n = int(o.get('n', 50000)); radius = float(o.get('radius', 100))
t0, t1, dt = float(o.get('t0', 380000)), float(o.get('t1', 392000)), float(o.get('dt', 100))
ic = cloud_ic(n, radius)
cfg = BlockConfig(coupled=CoupledConfig(), dt_max=4096.0, hydro=False)
st = blocks.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, cfg, t_stop=t1)
T, X, P = [st.t], [st.x.copy()], [st.p.copy()]
nxt = t0; w = time.perf_counter()
while st.n < st.n_stop:
    blocks.advance(st, cfg)
    if st.t >= nxt:
        T.append(st.t); X.append(st.x.copy()); P.append(st.p.copy()); nxt = st.t + dt
        if len(T) % 10 == 0:
            print(f"t={st.t:.0f} wall={time.perf_counter()-w:.0f}s rmin={np.linalg.norm(st.x,axis=1).min():.2f}", flush=True)
np.savez(out, t=np.array(T), x=np.array(X), p=np.array(P), x0=ic.x)
print('saved', out, len(T), 'snapshots, N', len(ic.x), 'wall', time.perf_counter()-w, flush=True)
