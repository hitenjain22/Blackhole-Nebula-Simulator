"""Diagnostic run (review 2026-10-15) WITH the plan's swallowing radius: particles inside
r_acc = 4 are removed after each system step (logged with t, L, u), so the run continues past
the first plunge and shows what the bulk of the cloud does at pericentre.
Usage: uv run python run_cloud_acc.py OUT.npz [n=1000] [tend=410000] [tdense=380000] [dtdense=100] [visc=off]
Removal mid-hierarchy: the particle's rows are deleted from every per-particle array and the
neighbour cache is reset (others keep their steps). Snapshots keep N rows (NaN once removed).
ckpt=PATH: pickle the state every 10 min of wall time and resume from it if it exists (survives
container restarts; the neighbour cache is rebuilt on resume, as after a removal)."""
import sys, time, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from dataclasses import replace, fields
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.cloud import cloud_ic
from nebula.sim.coupled import CoupledConfig

out = sys.argv[1]
opts = dict(a.split('=') for a in sys.argv[2:])
R_ACC = 4.0
RECON = opts.get('recon', 'off') == 'on'
if RECON:
    import recon; recon.install()
cc = CoupledConfig()
if opts.get('visc') == 'off': cc = replace(cc, hydro=replace(cc.hydro, viscosity=False))
cfg = BlockConfig(coupled=cc, dt_max=4096.0)
T_ON = float(opts.get('von', -1))  # viscosity off until t = T_ON, then on (an 'ideal switch' test)
cfg_on = cfg
if T_ON > 0: cfg = BlockConfig(coupled=replace(cc, hydro=replace(cc.hydro, viscosity=False)), dt_max=4096.0)
ic = cloud_ic(int(opts.get('n', 1000)), float(opts.get('radius', 100)))
N = len(ic.x)
T_END = float(opts.get('tend', 410000)); T_DENSE = float(opts.get('tdense', 380000)); DT_DENSE = float(opts.get('dtdense', 100))
st = blocks.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, cfg, t_stop=T_END)
ids = np.arange(N)
CKPT = opts.get('ckpt')
print('N', N, 'visc', cc.hydro.viscosity, 'von', T_ON, 'recon', RECON, 'radius', opts.get('radius', 100), flush=True)

def remove(st, rows):
    n = len(st.x); keep = np.ones(n, bool); keep[rows] = False
    for f in fields(st):
        v = getattr(st, f.name)
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            setattr(st, f.name, v[keep])
    st.neighbour_cache = None; st.last_pairs = None
    if RECON: recon.prune(keep)
    return keep

snaps = {k: [] for k in ('t', 'x', 'p', 'u', 'h')}
def record():
    snaps['t'].append(st.t)
    for k in ('x', 'p', 'u', 'h'):
        a = getattr(st, k); full = np.full((N,) + a.shape[1:], np.nan); full[ids] = a; snaps[k].append(full)
accreted = []  # (t, id, L, u, steps since start)
next_t = 0.0; err = None; w0 = time.perf_counter(); last_print = 0.0; last_ckpt = time.perf_counter()
if CKPT and os.path.exists(CKPT):
    with open(CKPT, 'rb') as f: ck = pickle.load(f)
    st, ids, accreted, next_t = ck['st'], ck['ids'], ck['accreted'], ck['next_t']
    if RECON: recon.STORE['dv'] = ck['dv']
    print('resumed from', CKPT, 't =', st.t, 'N =', len(ids), flush=True)
def save_ckpt():
    nc, lp = st.neighbour_cache, st.last_pairs
    st.neighbour_cache = None; st.last_pairs = None; st.active_counts = []
    with open(CKPT + '.tmp', 'wb') as f:
        pickle.dump(dict(st=st, ids=ids, accreted=accreted, next_t=next_t, dv=recon.STORE['dv'] if RECON else None), f)
    os.replace(CKPT + '.tmp', CKPT)
    st.neighbour_cache, st.last_pairs = nc, lp
try:
    while st.n < st.n_stop:
        if T_ON > 0 and st.t >= T_ON and cfg is not cfg_on:
            cfg = cfg_on; print('viscosity on at t =', st.t, flush=True)
        blocks.advance(st, cfg)
        r = np.linalg.norm(st.x, axis=1)
        gone = np.flatnonzero(r < R_ACC)
        if len(gone):
            L = np.linalg.norm(np.cross(st.x[gone], st.p[gone]), axis=1)
            for g, l in zip(gone, L):
                accreted.append((st.t, ids[g], l, st.u[g]))
                print(f"  accreted id {ids[g]} at t={st.t:.1f} L={l:.2f} u={st.u[g]:.1e}", flush=True)
            ids = ids[remove(st, gone)]
        if st.t >= next_t:
            record(); next_t = st.t + (DT_DENSE if st.t >= T_DENSE else 4096.0)
        if CKPT and time.perf_counter() - last_ckpt > float(opts.get('ckpt_every', 600)):
            save_ckpt(); last_ckpt = time.perf_counter()
        if time.perf_counter() - last_print > 60:
            last_print = time.perf_counter(); r = np.linalg.norm(st.x, axis=1); L = np.linalg.norm(np.cross(st.x, st.p), axis=1)
            print(f"t={st.t:.0f} wall={last_print-w0:.0f}s N={len(ids)} rmin={r.min():.2f} Lmin={L.min():.3f} nL4={(L<4).sum()} nL5={(L<5).sum()} umax={st.u.max():.2e} steps={st.particle_steps}", flush=True)
except Exception as e:
    err = repr(e); print('STOPPED:', err, 't=', st.t, flush=True)
record()
np.savez_compressed(out, err=str(err), accreted=np.array(accreted).reshape(-1, 4), mass=ic.mass * np.ones(N), **{k: np.array(v) for k, v in snaps.items()})
print('saved', out, 'wall', time.perf_counter() - w0, 'accreted', len(accreted), flush=True)
