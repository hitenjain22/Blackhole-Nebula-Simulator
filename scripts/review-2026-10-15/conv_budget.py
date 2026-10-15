"""Squeeze budget in every direction (gravity-only run). For each particle near the hole, fit a
linear velocity field over its k nearest current neighbours; the symmetric part's negative
eigenvalues (lambda < 0) are the directions in which the gas is being squeezed. The gas's spread
along each such direction (sigma) times |lambda| is the speed at which it converges, and
1/2 sum (lambda sigma)^2 is the most heat a shock could make from that squeeze. Rotation and the
bending of the orbit sit in the antisymmetric part and are left out.
Usage: uv run python conv_budget.py ball.npz t1 t2 ... [k=64] [rmax=300]"""
import sys, numpy as np
from scipy.spatial import cKDTree
from nebula.physics.gravity import gn_v_from_p
args = [a for a in sys.argv[2:] if '=' not in a]
o = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
k = int(o.get('k', 64)); rmax = float(o.get('rmax', 300))
d = np.load(sys.argv[1]); t = d['t']
for tt in [float(a) for a in args]:
    i = int(np.argmin(abs(t - tt))); x = d['x'][i]; p = d['p'][i]
    r = np.linalg.norm(x, axis=1); sel = np.flatnonzero(r < rmax)
    v = np.zeros_like(x); v[sel] = np.array([gn_v_from_p(x[j], p[j]) for j in sel])
    xs = x[sel]; tree = cKDTree(xs); dist, nb = tree.query(xs, k)
    e_all = np.zeros(len(sel)); e_vert = np.zeros(len(sel))
    for m in range(len(sel)):
        idx = sel[nb[m]]; dx = x[idx] - x[sel[m]]
        A = np.hstack([np.ones((k, 1)), dx]); coef, *_ = np.linalg.lstsq(A, v[idx], rcond=None)
        G = coef[1:].T; S = 0.5 * (G + G.T); lam, vec = np.linalg.eigh(S)
        dxc = dx - dx.mean(0)
        for j in range(3):
            if lam[j] < 0:
                sig = np.sqrt(np.mean((dxc @ vec[:, j]) ** 2))
                e = 0.5 * (lam[j] * sig) ** 2
                e_all[m] += e
                if abs(vec[2, j]) > 0.7: e_vert[m] += e
    q = np.percentile(e_all, [50, 90, 99]); qv = np.percentile(e_vert, [50, 99])
    print(f"t={t[i]:.0f} near={len(sel)} ball~{np.median(dist[:, -1]):.2f} | all squeezes: p50 {q[0]:.1e} p90 {q[1]:.1e} p99 {q[2]:.1e} max {e_all.max():.1e} | vertical only: p50 {qv[0]:.1e} p99 {qv[1]:.1e}", flush=True)
