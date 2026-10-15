"""Cooler hot-blob (u = 4e-8) past T_END_BLOB, to see the debris-return plunge. visc=off option."""
import sys, time
import numpy as np
from dataclasses import replace
from nebula.sim import blocks, blob
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
out = sys.argv[1]
opts = dict(a.split('=') for a in sys.argv[2:])
cc = CoupledConfig()
if opts.get('visc') == 'off': cc = replace(cc, hydro=replace(cc.hydro, viscosity=False))
cfg = BlockConfig(coupled=cc, dt_max=256.0, hydro=(opts.get('hydro', 'on') != 'off'))
x, v, u, m, h = blob._initial(cc)
T_END = float(opts.get('tend', 26000))
st = blocks.initialise(x, v, u, m, h, cfg, t_stop=T_END)
snaps = {k: [] for k in ('t', 'x', 'p', 'u', 'h')}
next_t = 0.0; err = None; w0 = time.perf_counter()
def record():
    snaps['t'].append(st.t); snaps['x'].append(st.x.copy()); snaps['p'].append(st.p.copy()); snaps['u'].append(st.u.copy()); snaps['h'].append(st.h.copy())
try:
    while st.n < st.n_stop:
        blocks.advance(st, cfg)
        if st.t >= next_t:
            record(); next_t = st.t + (5.0 if st.t > 17000 else 100.0)
except Exception as e:
    err = repr(e); print('STOPPED', err, st.t, flush=True)
record()
np.savez_compressed(out, err=str(err), mass=st.mass, **{k: np.array(v) for k, v in snaps.items()})
print('saved', out, 'wall', time.perf_counter() - w0, 't', st.t)
