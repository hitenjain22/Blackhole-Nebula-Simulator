"""Diagnostic run (review 2026-10-15): the Phase 5 cloud on block steps, snapshots dense near pericentre.
Usage: uv run python run_cloud.py OUT.npz [key=value ...]
  n=1000            particles (cloud_ic target)
  hydro=off         gravity only (run A)
  visc=off          no artificial viscosity (run B)
  cond=pressure|off conductivity signal / conductivity off (runs D, E)
  alpha=0           switch off, constant alpha (run F);  beta=0  no quadratic term (run G)
  kernel=cubic      cubic spline at its own eta (run H);  hmax=20  cap h (crashes: see D11)
  tend=390000 tdense=360000 dtdense=25
Only settings change (dataclasses.replace); nothing in src/ is edited.
"""
import sys, time
import numpy as np
from dataclasses import replace
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.cloud import cloud_ic
from nebula.sim.coupled import CoupledConfig
from nebula.sim.sph import COND_PRESSURE, COND_VELOCITY

out = sys.argv[1]
opts = dict(a.split('=') for a in sys.argv[2:])
n_target = int(opts.get('n', 1000))
cc = CoupledConfig()
hc = cc.hydro
if opts.get('cond') == 'pressure': hc = replace(hc, conductivity_signal=COND_PRESSURE)
if opts.get('cond') == 'off': hc = replace(hc, conductivity=False)
if opts.get('visc') == 'off': hc = replace(hc, viscosity=False)
if 'alpha' in opts: hc = replace(hc, switch=False, alpha_const=float(opts['alpha']))
if 'hmax' in opts: hc = replace(hc, h_max=float(opts['hmax']))
if 'beta' in opts: hc = replace(hc, beta=float(opts['beta']))
if 'kernel' in opts: 
    from nebula.physics.kernels import get_kernel
    hc = replace(hc, kernel=opts['kernel'], eta=get_kernel(opts['kernel']).hfact)
cc = replace(cc, hydro=hc)
cfg = BlockConfig(coupled=cc, dt_max=4096.0, hydro=(opts.get('hydro','on')!='off'))
ic = cloud_ic(n_target)
print('N', len(ic.x), 'kernel', hc.kernel, 'eta', hc.eta, 'cond', hc.conductivity, hc.conductivity_signal, 'visc', hc.viscosity, 'switch', hc.switch, hc.alpha_const, 'beta', hc.beta, 'hmax', hc.h_max, flush=True)
T_END = float(opts.get('tend', 390000))
T_DENSE = float(opts.get('tdense', 360000))
DT_DENSE = float(opts.get('dtdense', 25.0))
DT_SPARSE = 4096.0
st = blocks.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, cfg, t_stop=T_END)
snaps = {k: [] for k in ('t', 'x', 'p', 'u', 'h', 'rho', 'alpha', 'level', 'ahyd', 'dudt')}
next_t = 0.0
w0 = time.perf_counter()
def record(st):
    snaps['t'].append(st.t); snaps['x'].append(st.x.copy()); snaps['p'].append(st.p.copy())
    snaps['u'].append(st.u.copy()); snaps['h'].append(st.h.copy()); snaps['rho'].append(st.rho.copy())
    snaps['alpha'].append(st.alpha.copy()); snaps['level'].append(st.level.copy())
    snaps['ahyd'].append(st.a_hydro.copy()); snaps['dudt'].append(st.dudt.copy())
err = None
try:
    while st.n < st.n_stop:
        blocks.advance(st, cfg)
        if st.t >= next_t:
            record(st)
            next_t = st.t + (DT_DENSE if st.t >= T_DENSE else DT_SPARSE)
            if len(snaps['t']) % 20 == 0:
                r = np.linalg.norm(st.x, axis=1)
                L = np.linalg.norm(np.cross(st.x, st.p), axis=1)
                print(f"t={st.t:.0f} wall={time.perf_counter()-w0:.0f}s rmin={r.min():.2f} Lmin={L.min():.3f} umax={st.u.max():.2e} hmax={st.h.max():.1f} steps={st.particle_steps}", flush=True)
except Exception as e:
    err = repr(e)
    print('STOPPED:', err, 't=', st.t, flush=True)
record(st)
np.savez_compressed(out, err=str(err), mass=st.mass, **{k: np.array(v) for k, v in snaps.items()})
print('saved', out, 'wall', time.perf_counter() - w0, flush=True)
