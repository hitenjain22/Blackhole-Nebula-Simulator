import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f)
    t = d['t']; X = d['x']; P = d['p']; U = d['u']
    L = np.linalg.norm(np.cross(X[-1], P[-1]), axis=1)
    L0 = np.linalg.norm(np.cross(X[0], P[0]), axis=1)
    i = np.searchsorted(t, 384000)
    Li = np.linalg.norm(np.cross(X[i], P[i]), axis=1)
    print(f"{f.split('/')[-1]:12s} t_end={t[-1]:.0f} err={str(d['err'])[:40]:40s} N={len(L)} L<4 (doomed): {np.sum(L<4)}  L<6: {np.sum(L<6)}  Lmin@384k={Li.min():.3f} Lmin_end={L.min():.3f}  umax_end={U[-1].max():.1e}  totL drift={(np.sum(d['mass']*L)-np.sum(d['mass']*L0))/np.sum(d['mass']*L0):+.1e}")
