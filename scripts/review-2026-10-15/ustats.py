"""Percentiles of internal energy u for particles within r < rmax, at given times (hydro runs)."""
import sys, numpy as np
o = dict(a.split('=') for a in sys.argv[2:] if '=' in a); rmax = float(o.get('rmax', 300))
d = np.load(sys.argv[1], allow_pickle=True); t = d['t']
print(sys.argv[1].split('/')[-1], 'times', t[0], '..', t[-1], len(t))
for tt in [float(a) for a in sys.argv[2:] if '=' not in a]:
    i = int(np.argmin(abs(t - tt))); x = d['x'][i]; u = d['u'][i]; r = np.linalg.norm(x, axis=1)
    s = r < rmax
    if s.sum() == 0: print(f"t={t[i]:.0f} none"); continue
    q = np.percentile(u[s], [50, 90, 99])
    print(f"t={t[i]:.0f} near={s.sum()} u50 {q[0]:.1e} u90 {q[1]:.1e} u99 {q[2]:.1e} max {u[s].max():.1e}")
