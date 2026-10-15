"""Real stream crossings in a gravity-only run (review 2026-10-15). For each particle near the
hole, fit the deformation gradient F = dx(t)/dx(0) over its 16 nearest INITIAL neighbours;
det F < 0 means its patch has turned inside out, i.e. it has passed through a caustic and now
overlaps other gas (a fold). Where a particle's current neighbours include gas from the other
side of the fold (det F of the opposite sign), the two flows really do run into each other. The
smooth part of the velocity (a linear fit to this side's neighbours, extrapolated) is removed, so
dv is the jump between the flows, and u = dv^2 / 8 is the heat a perfectly sticky collision makes.
Usage: uv run python folds.py ball.npz t1 t2 ... [rmax=300] [k=32]"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from nebula.physics.gravity import gn_v_from_p

args = [a for a in sys.argv[2:] if '=' not in a]
o = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
rmax = float(o.get('rmax', 300)); k = int(o.get('k', 32))
d = np.load(sys.argv[1]); t = d['t']; x0 = d['x'][0]
_, lag = cKDTree(x0).query(x0, 17)
for tt in [float(a) for a in args]:
    i = int(np.argmin(abs(t - tt))); x = d['x'][i]; p = d['p'][i]
    r = np.linalg.norm(x, axis=1); sel = np.flatnonzero(r < rmax)
    if len(sel) == 0: print(f"t={t[i]:.0f}: none near"); continue
    v = np.array([gn_v_from_p(x[j], p[j]) for j in sel]); vfull = np.zeros_like(x); vfull[sel] = v
    det = np.zeros(len(x)); det[:] = np.nan
    for a in sel:
        dx0 = x0[lag[a, 1:]] - x0[a]; dx = x[lag[a, 1:]] - x[a]
        F, *_ = np.linalg.lstsq(dx0, dx, rcond=None); det[a] = np.linalg.det(F)
    tree = cKDTree(x[sel]); dist, nb = tree.query(x[sel], k + 1)
    ucross = np.zeros(len(sel)); approach = np.zeros(len(sel))
    for m, a in enumerate(sel):
        bs = sel[nb[m, 1:]]
        other = bs[np.sign(det[bs]) != np.sign(det[a])]
        same = np.concatenate([[a], bs[np.sign(det[bs]) == np.sign(det[a])]])
        if len(other) and len(same) >= 6:
            # remove the smooth part: fit v = v0 + G (x - x_a) to this side of the fold only,
            # extrapolate it to the other side's particles, keep the jump
            A = np.hstack([np.ones((len(same), 1)), x[same] - x[a]])
            coef, *_ = np.linalg.lstsq(A, vfull[same], rcond=None)
            pred = np.hstack([np.ones((len(other), 1)), x[other] - x[a]]) @ coef
            dv = vfull[other] - pred; dxx = x[other] - x[a]
            ucross[m] = np.mean(np.sum(dv**2, axis=1)) / 8
            approach[m] = np.max(-np.sum(dv * dxx, axis=1) / np.linalg.norm(dxx, axis=1))
    f = ucross > 0
    line = f"t={t[i]:.0f} near={len(sel)} folded={np.sum(det[sel] < 0)} touching-a-fold={f.sum()} spacing~{np.median(dist[:, -1]):.2f}"
    if f.sum():
        q = np.percentile(ucross[f], [50, 90, 99]); qa = np.percentile(approach[f], [50, 90, 99])
        line += f" | u=dv^2/8: p50 {q[0]:.1e} p90 {q[1]:.1e} p99 {q[2]:.1e} max {ucross.max():.1e} | approach speed p50 {qa[0]:.4f} p90 {qa[1]:.4f} max {approach.max():.4f}"
    print(line, flush=True)
