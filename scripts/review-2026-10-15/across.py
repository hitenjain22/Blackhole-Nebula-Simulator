"""Across-the-hole diagnostic: for particles with r < RNEAR, neighbours within 2h whose
direction from the hole differs by > 90 degrees (kernel spans the hole)."""
import sys, numpy as np
from scipy.spatial import cKDTree
f = sys.argv[1]; RNEAR = 60.0
d = np.load(f)
t = d['t']; X = d['x']; H = d['h']; U = d['u']; P = d['p']
times = [float(a) for a in sys.argv[2:]] or [384000, 384600, 385200, 385800, 386400, 387000, 388000, 389500]
print(f, 'N', X.shape[1])
for tt in times:
    i = np.searchsorted(t, tt)
    if i >= len(t): break
    x = X[i]; h = H[i]; r = np.linalg.norm(x, axis=1)
    near = np.where(r < RNEAR)[0]
    if len(near) == 0:
        print(f"t={t[i]:8.0f} none within r<{RNEAR:.0f}"); continue
    tree = cKDTree(x)
    across = 0; pairs = 0; maxang = 0
    for a in near:
        nb = tree.query_ball_point(x[a], 2 * h[a])
        nb = [b for b in nb if b != a]
        cosang = (x[nb] @ x[a]) / (r[nb] * r[a])
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        pairs += len(nb); across += np.sum(ang > 90); maxang = max(maxang, ang.max() if len(nb) else 0)
    L = np.linalg.norm(np.cross(x, P[i]), axis=1)
    print(f"t={t[i]:8.0f} n(r<{RNEAR:.0f})={len(near):4d} med 2h/r={np.median(2*h[near]/r[near]):5.2f} max 2h/r={np.max(2*h[near]/r[near]):5.2f} "
          f"across-hole pairs={across}/{pairs} ({100*across/max(pairs,1):.0f}%) max angle={maxang:5.1f}  u_max(near)={U[i,near].max():.1e}  Lmin={L.min():.2f}")
