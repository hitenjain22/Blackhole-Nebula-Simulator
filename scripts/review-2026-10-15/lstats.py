import sys, numpy as np
for f in sys.argv[1:]:
    d = np.load(f); t = d['t']; X = d['x']; P = d['p']; U = d['u']
    L0 = np.linalg.norm(np.cross(X[0], P[0]), axis=1)
    print(f.split('/')[-1], 'N', X.shape[1])
    for tt in (160000, 241920, 323840, 361000, 370000, 380000, 383000, 384600):
        i = np.searchsorted(t, tt)
        if i >= len(t): break
        L = np.linalg.norm(np.cross(X[i], P[i]), axis=1); dl = L / L0 - 1
        r = np.linalg.norm(X[i], axis=1); lead = r < np.percentile(r, 5)
        print(f"  t={t[i]:7.0f}  dL/L0: p1={np.percentile(dl,1)*100:+6.2f}% rms={np.std(dl)*100:5.2f}%  leading 5%: mean {dl[lead].mean()*100:+6.2f}%  u median {np.median(U[i]):.1e} (x{np.median(U[i])/np.median(U[0]):.1f})")
