"""Angular-momentum change since the start, at chosen times, for gas that has passed closest
approach (its distance from the hole has reached a minimum and is now rising, or r_min < 60)."""
import sys, numpy as np
d = np.load(sys.argv[1]); t = d['t']; X = d['x']; P = d['p']
L0 = np.linalg.norm(np.cross(X[0], P[0]), axis=1)
R = np.linalg.norm(X, axis=2); rmin_sofar = np.fmin.accumulate(np.where(np.isnan(R), np.inf, R), axis=0)
for tt in [float(a) for a in sys.argv[2:]]:
    i = int(np.argmin(abs(t - tt))); L = np.linalg.norm(np.cross(X[i], P[i]), axis=1)
    ok = ~np.isnan(L); passed = ok & (rmin_sofar[i] < 60)
    dl = L[passed] / L0[passed] - 1
    print(f"{sys.argv[1].split('/')[-1]} t={t[i]:.0f}: passed {passed.sum()} of {ok.sum()}; dL/L rms {np.std(dl)*100:.2f}%, worst 1% {np.percentile(dl,1)*100:+.1f}%, L<4 {np.sum(L[ok]<4)}, L<6 {np.sum(L[ok]<6)}")
