"""Vertical squeeze budget in a gravity-only run: 1/2 v_z^2 (v_z = velocity across the cloud
centre's orbital plane, z = 0) for gas near the hole. If the nozzle shock turned ALL of that
motion into heat, this is the most it could make."""
import sys, numpy as np
from nebula.physics.gravity import gn_v_from_p
d = np.load(sys.argv[1]); t = d['t']; best = np.zeros(d['x'].shape[1])
for i in range(len(t)):
    x = d['x'][i]; p = d['p'][i]; r = np.linalg.norm(x, axis=1); s = np.flatnonzero(r < 300)
    if len(s) == 0: continue
    vz = np.array([gn_v_from_p(x[j], p[j])[2] for j in s]); e = 0.5 * vz**2
    best[s] = np.maximum(best[s], e)
    if i % 20 == 0:
        q = np.percentile(e, [50, 99]); print(f"t={t[i]:.0f} near={len(s)} 1/2 vz^2 p50 {q[0]:.1e} p99 {q[1]:.1e} max {e.max():.1e}")
b = best[best > 0]; q = np.percentile(b, [50, 90, 99])
print(f"each particle's largest 1/2 vz^2 near the hole: p50 {q[0]:.1e} p90 {q[1]:.1e} p99 {q[2]:.1e} max {b.max():.1e} (n={len(b)})")
