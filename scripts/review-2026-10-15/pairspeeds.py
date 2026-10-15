"""For particles near the hole in a snapshot: approach speeds of neighbour pairs (what the
viscosity sees), raw and after linear reconstruction (least-squares gradients)."""
import sys, numpy as np
from scipy.spatial import cKDTree
from nebula.physics.gravity import gn_v_from_p
d = np.load(sys.argv[1]); t = d['t']
for tt in [float(a) for a in sys.argv[2:]]:
    i = np.searchsorted(t, tt); x = d['x'][i]; p = d['p'][i]; h = d['h'][i]
    ok = np.isfinite(x[:, 0]); idx = np.flatnonzero(ok); x = x[ok]; p = p[ok]; h = h[ok]
    v = gn_v_from_p(x, p); r = np.linalg.norm(x, axis=1)
    tree = cKDTree(x); grads = {}
    def grad(a):
        if a in grads: return grads[a]
        nb = [b for b in tree.query_ball_point(x[a], 2 * h[a]) if b != a]
        if len(nb) < 6: grads[a] = None; return None
        dx = x[nb] - x[a]; dv = v[nb] - v[a]
        G, *_ = np.linalg.lstsq(dx, dv, rcond=None)  # dv = dx @ G  -> G[j,k] = dv_k/dx_j
        grads[a] = G.T; return G.T
    near = np.flatnonzero(r < 40)
    raw, rec, sep = [], [], []
    for a in near:
        ga = grad(a)
        for b in tree.query_ball_point(x[a], 2 * h[a]):
            if b == a: continue
            dxab = x[a] - x[b]; rr = np.linalg.norm(dxab); rh = dxab / rr
            w = (v[a] - v[b]) @ rh
            gb = grad(b)
            if ga is None or gb is None: wr = w
            else:
                da = ga @ dxab; db = gb @ dxab; pa = da @ dxab; pb = db @ dxab
                A = pa / pb if pb != 0 else 0; phi = min(1, 4 * A / (1 + A) ** 2) if A > 0 else 0
                wr = ((v[a] - 0.5 * phi * da) - (v[b] + 0.5 * phi * db)) @ rh
            raw.append(min(w, 0)); rec.append(min(wr, 0)); sep.append(rr)
    raw = -np.array(raw); rec = -np.array(rec)
    print(f"t={t[i]:.0f} particles r<40: {len(near)}  pairs {len(raw)}  approach speed raw: median {np.median(raw):.1e} p99 {np.percentile(raw,99):.1e} max {raw.max():.1e} | reconstructed: median {np.median(rec):.1e} p99 {np.percentile(rec,99):.1e} max {rec.max():.1e}  (c_s ~ 5e-5; beta*w^2 heating scale p99: {2*np.percentile(rec,99)**2:.1e})")
