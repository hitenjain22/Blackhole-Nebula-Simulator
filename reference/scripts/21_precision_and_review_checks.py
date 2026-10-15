"""Script 21: extra-precision values and review checks (plan time, Oct 2026).

Prints, in order:
  1. EOS test points to 10 significant figures, plus the two extreme points (expected-values §9).
  2. Shadow angles and ray deflections at high precision with mpmath (§3).
  3. Cloud-orbit constants at high precision (§2).
  4. Generalized-leapfrog energy error on the hot-blob ledger orbit r_a = 500, r_p = 25 (§2, Phase 4 test 4).
  5. Colour LUT interpolation error, linear vs log-log, for 1,024 / 2,048 / 4,096 entries (§7).
  6. Blackbody band fractions at 1e4, 1e5, 1e6 K (§7, Phase 9 test 14).
  7. Doppler test-ring ratio (§7, Phase 9 test 12).
Needs numpy, scipy, mpmath and colour-science. Run from any folder: it finds script 02 next to itself.
"""
import os, warnings
import numpy as np
import mpmath as mp
from scipy.optimize import brentq

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 1. EOS
kB = 1.380649e-23; m_p = 1.67262192369e-27; mu = 0.6; sigma = 5.670374419e-8
c = 2.99792458e8; a_rad = 4 * sigma / c; gam = 5 / 3

def u_of(rho, T): return 1.5 * kB * T / (mu * m_p) + a_rad * T**4 / rho
def P_of(rho, T): return rho * kB * T / (mu * m_p) + a_rad * T**4 / 3
def G1(rho, T):
    b = (rho * kB * T / (mu * m_p)) / P_of(rho, T)
    return b + (4 - 3 * b)**2 * (gam - 1) / (b + 12 * (gam - 1) * (1 - b)), b
def T_of(rho, u):
    return np.exp(brentq(lambda lT: np.log(u_of(rho, np.exp(lT))) - np.log(u), 0, 40, xtol=1e-15))

print('== 1. EOS (10 significant figures)')
for rho, T in ((1e-3, 1e4), (1e-6, 1e6), (1e-9, 1e5), (1e-9, 1e7), (1e3, 1e3), (1e-15, 1e8)):
    g, b = G1(rho, T)
    print('rho %.0e T %.0e: u %.10e P %.10e beta_gas %.10e Gamma1 %.10f' % (rho, T, u_of(rho, T), P_of(rho, T), b, g))
rho = 1.6586853e-6; u = 4.415186e14; T = T_of(rho, u); g, b = G1(rho, T)
print('rho %.7e u %.6e: T %.10e P %.10e beta_gas %.10e Gamma1 %.10f' % (rho, u, T, P_of(rho, T), b, g))

# ---------------------------------------------------------------- 2. rays
mp.mp.dps = 30
def deflection(b):
    b = mp.mpf(b)
    f = lambda uu: 2 * uu**3 - uu**2 + 1 / b**2
    roots = mp.polyroots([2, -1, 0, 1 / b**2], maxsteps=200, extraprec=60)
    u0 = [mp.re(r) for r in roots if abs(mp.im(r)) < 1e-20 and 0 < mp.re(r) < mp.mpf(1) / 3][0]
    g = lambda t: 2 * u0 * t / mp.sqrt(f(u0 * (1 - t**2)))   # u = u0 (1 - t^2) removes the turning-point singularity
    return mp.re(2 * mp.quad(g, [0, mp.mpf('0.5'), 1]) - mp.pi)
print('\n== 2. Rays')
for b in ('1000', '50', '10', '6', '5.3'):
    print('deflection b=%s: %s rad' % (b, mp.nstr(deflection(b), 13)))
for D in (20, 50, 100, 1000):
    D = mp.mpf(D)
    print('shadow D=%s: %s deg' % (mp.nstr(D, 5), mp.nstr(mp.degrees(mp.asin(3 * mp.sqrt(3) / D * mp.sqrt(1 - 2 / D))), 12)))

# ---------------------------------------------------------------- 3. cloud orbit
print('\n== 3. Cloud orbit r_p = 25, r_a = 5000')
rp, ra = mp.mpf(25), mp.mpf(5000)
L2 = ((1 - 2 / rp) - (1 - 2 / ra)) / ((1 - 2 / ra) / ra**2 - (1 - 2 / rp) / rp**2)
E = mp.sqrt((1 - 2 / ra) * (1 + L2 / ra**2))
print('L', mp.nstr(mp.sqrt(L2), 15), 'E', mp.nstr(E, 15), 'H', mp.nstr((E**2 - 1) / 2, 15))
print('v_apo = L/(f(r_a) r_a), f = r/(r-2):', mp.nstr(mp.sqrt(L2) / (ra / (ra - 2) * ra), 12))
print('r_circ', mp.nstr((L2 + mp.sqrt(L2**2 - 12 * L2)) / 2, 12))

# ---------------------------------------------------------------- 4. GLF on the blob orbit
src = open(os.path.join(HERE, '02_generalized_leapfrog_check.py')).read()
exec(src.split('x=np.array([7.,3.,1.])')[0])   # brings in p_from_v, dHdp, dHdq, H
def glf(x, p, h, it=12):
    ph = p.copy()
    for _ in range(it):
        new = p - 0.5 * h * dHdq(x, ph)
        done = np.max(np.abs(new - ph)) < 1e-16 * np.max(np.abs(new)); ph = new
        if done: break
    x1 = x.copy()
    for _ in range(it):
        new = x + 0.5 * h * (dHdp(x, ph) + dHdp(x1, ph))
        done = np.max(np.abs(new - x1)) < 1e-16 * np.max(np.abs(new)); x1 = new
        if done: break
    return x1, ph - 0.5 * h * dHdq(x1, ph)
print('\n== 4. Hot-blob orbit r_a = 500, r_p = 25 (gravity only, quantized adaptive steps)')
rpb, rab = 25., 500.
Lb = np.sqrt(((1 - 2 / rpb) - (1 - 2 / rab)) / ((1 - 2 / rab) / rab**2 - (1 - 2 / rpb) / rpb**2))
print('ell', repr(Lb))
for eta in (0.005, 0.0025, 0.00125):
    x = np.array([rab, 0, 0.]); p = np.array([0, Lb / rab, 0.]); e0 = H(x, p)
    t = 0.; n = 0; emax = 0.; passed = False; rprev = rab
    while True:
        r = np.sqrt(x @ x); dt = 2.**10
        while dt > eta * r**1.5: dt /= 2
        x, p = glf(x, p, dt); t += dt; n += 1
        rn = np.sqrt(x @ x); emax = max(emax, abs(H(x, p) - e0) / abs(e0))
        if rn < 100: passed = True
        if passed and rn >= rab * 0.999 and rn < rprev: break
        rprev = rn
    print('eta %.5f: steps %d, t_orbit %.1f, max|dH/H| %.3e, end |dH/H| %.3e' % (eta, n, t, emax, abs(H(x, p) - e0) / abs(e0)))

# ---------------------------------------------------------------- 5. colour LUT
import colour
h_P = 6.62607015e-34
cm = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']; wl = cm.wavelengths; V = cm.values
sel = (wl >= 360) & (wl <= 830); wl = wl[sel]; V = V[sel]
def chroma(T):
    lam = wl * 1e-9; B = 2 * h_P * c**2 / lam**5 / np.expm1(h_P * c / (lam * kB * T))
    return (B[:, None] * V).sum(0) * 1e-9 / (sigma * T**4 / np.pi)
print('\n== 5. Colour LUT interpolation error (log T grid 1e3..1e8)')
for n in (1024, 2048, 4096):
    lT = np.linspace(np.log(1e3), np.log(1e8), n); tab = np.array([chroma(np.exp(v)) for v in lT])
    mids = 0.5 * (lT[1:] + lT[:-1]); idx = np.unique(np.linspace(0, n - 2, 400).astype(int))
    elin = []; elog = []
    for i in idx:
        ex = chroma(np.exp(mids[i]))
        elin.append(np.max(np.abs(0.5 * (tab[i] + tab[i + 1]) / ex - 1)))
        elog.append(np.max(np.abs(np.exp(0.5 * (np.log(tab[i]) + np.log(tab[i + 1]))) / ex - 1)))
    elin = np.array(elin); elog = np.array(elog); Tm = np.exp(mids[idx])
    print('n=%d: linear %.2e | log-log %.2e (T>=2000 K: %.2e)' % (n, elin.max(), elog.max(), elog[Tm >= 2000].max()))

# ---------------------------------------------------------------- 6. band fractions
print('\n== 6. Band fractions of sigma T^4')
hm = mp.mpf('6.62607015e-34'); cm_ = mp.mpf(299792458); km = mp.mpf('1.380649e-23')
keV = lambda Ek: hm * cm_ / (mp.mpf(Ek) * 1000 * mp.mpf('1.602176634e-19'))
def frac(T, l1, l2):
    fx = lambda xx: xx**3 / mp.expm1(xx)
    x1 = hm * cm_ / (l2 * km * T); x2 = hm * cm_ / (l1 * km * T)
    return mp.quad(fx, [x1, (x1 + x2) / 2, x2]) * 15 / mp.pi**4
bands = [('IR 1-5 um', mp.mpf('1e-6'), mp.mpf('5e-6')), ('optical 380-780 nm', mp.mpf('380e-9'), mp.mpf('780e-9')),
         ('UV 100-300 nm', mp.mpf('100e-9'), mp.mpf('300e-9')), ('EUV 12.4-100 nm', keV('0.1'), mp.mpf('100e-9')),
         ('soft X-ray 0.1-2 keV', keV(2), keV('0.1'))]
for T in ('1e4', '1e5', '1e6'):
    print('T=%s: ' % T + ' | '.join('%s %s' % (nm, mp.nstr(frac(mp.mpf(T), l1, l2), 7)) for nm, l1, l2 in bands))

# ---------------------------------------------------------------- 7. Doppler ring
print('\n== 7. Doppler test ring')
for r in (1000., 10.):
    beta = 1 / np.sqrt(r - 2)
    print('r=%g: beta %.8f, ((1+beta)/(1-beta))^4 = %.6f' % (r, beta, ((1 + beta) / (1 - beta))**4))
