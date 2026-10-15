"""Script 12 of the plan, generalised: cloud radius R and number of passages K from the command
line. Same test-particle model (uniform sphere at apocentre, all with the apocentre velocity, GN
acceleration, DOP853, pressure ignored). Prints, for each passage, the pericentre-time spread and
whether outgoing gas reaches r_SI before the last incoming gas has passed it (streams cross).
Run reference/scripts/10_units_orbits_crossing.py first (it writes orbit_consts.npy here).
Usage: python crash_by_radius.py R K [N=200]   (about 1-10 min)"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
L, E, eps, rSI, tp = np.load('orbit_consts.npy')
ra = 5000.0; R = float(sys.argv[1]); K = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
def acc(x, v):
    r = np.linalg.norm(x); rd = x @ v / r; h2 = np.sum(np.cross(x, v) ** 2)
    return -(1 / r**3) * (1 - 2 / r) ** 2 * x + 2 * rd / (r * (r - 2)) * v - 3 * h2 / r**5 * x
f = ra / (ra - 2); vt = L / (f * ra)
rng = np.random.default_rng(2); pts = []
while len(pts) < N:
    q = rng.uniform(-1, 1, 3)
    if q @ q <= 1: pts.append(q * R)
def rhs(t, s): return np.concatenate([s[3:], acc(s[:3], s[3:])])
def ev_peri(t, s): return s[:3] @ s[3:]
ev_peri.direction = 1
def ev_si(t, s): return np.linalg.norm(s[:3]) - rSI
TP = []; SI = []
for q in pts:
    x0 = np.array([ra, 0, 0.]) + q; v0 = np.array([0, vt, 0.])
    sol = solve_ivp(rhs, [0, (K + 0.1) * 2 * 396224 ], np.concatenate([x0, v0]), rtol=1e-10, atol=1e-10, events=[ev_peri, ev_si], method='DOP853')
    TP.append(sol.t_events[0][:K]); SI.append(sol.t_events[1][:2 * K])
TP = np.array(TP); SI = np.array(SI)
for k in range(K):
    t = TP[:, k]; tin = SI[:, 2 * k]; tout = SI[:, 2 * k + 1]
    cross = tin.max() > tout.min()
    print(f"R={R:g} passage {k+1}: peri t mean {t.mean():.0f}, full spread {t.max()-t.min():.0f} | streams cross: {'YES, fraction still incoming %.2f' % np.mean(tin > tout.min()) if cross else 'no'}", flush=True)
    if cross: break
