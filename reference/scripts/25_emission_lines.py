import warnings; warnings.filterwarnings('ignore')
# Phase 14 N1 expected values (expected-values §12): emission-line colours and the H alpha fit.
#   uv run --no-project --python 3.12 --with colour-science --with pyneb --with numpy \
#       python reference/scripts/25_emission_lines.py
import numpy as np, colour, pyneb as pn
cm = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
wl = cm.wavelengths; sel = (wl >= 360) & (wl <= 830); wl = wl[sel]; v = cm.values[sel]
lines = [('Halpha', 656.28), ('Hbeta', 486.13), ('Hgamma', 434.05), ('OIII_5007', 500.68),
         ('OIII_4959', 495.89), ('NII_6584', 658.35), ('NII_6548', 654.80),
         ('SII_6716', 671.65), ('SII_6731', 673.08)]
print('line | lambda nm | CIE 1931 2deg x y (linear interpolation of the 1 nm table)')
for n, lam in lines:
    xyz = np.array([np.interp(lam, wl, v[:, k]) for k in range(3)])
    print('%-10s %7.2f  %.9f %.9f' % (n, lam, xyz[0] / xyz.sum(), xyz[1] / xyz.sum()))
print('\nH alpha, Dong & Draine 2011 fit: 2.82e-26 T4^(-0.942 - 0.031 ln T4) [erg cm^3 s^-1 sr^-1]')
h1 = pn.RecAtom('H', 1)
for T in (5e3, 1e4, 2e4):
    t4 = T / 1e4; fit = 2.82e-26 * t4 ** (-0.942 - 0.031 * np.log(t4))
    pyn = h1.getEmissivity(T, 100.0, label='3_2') / (4 * np.pi)
    print('T %6.0f: fit %.10e   PyNeb case B %.6e   fit / PyNeb %.4f' % (T, fit, pyn, fit / pyn))
    hb = h1.getEmissivity(T, 100.0, label='4_2'); hg = h1.getEmissivity(T, 100.0, label='5_2')
    print('          PyNeb Ha/Hb %.4f   Hg/Hb %.4f' % (h1.getEmissivity(T, 100.0, label='3_2') / hb, hg / hb))
