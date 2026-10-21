"""Disc viscosity through the artificial-viscosity term (phases/phase-07-viscosity-drain.md 7.2,
Option B). Pure formulas; the SPH pair sum lives in sim/sph.py.

Source: Lodato & Price 2010 (MNRAS 405, 1212); Price et al. 2018 (Phantom, PASA 35, e031,
arXiv:1702.03930) §2.6.1, checked in the paper's text 2026-10-21 (Eqs. 119-124); after
Artymowicz & Lubow 1994 and Murray 1996. Units: code units (G = c = M = 1).

The AV pair term becomes a Navier-Stokes viscosity with fixed bulk/shear ratio (Eq. 119) when it
is applied to approaching AND receding pairs, with the signal speed alpha_AV c_s, a constant
alpha_AV and an extra factor h/|r_ab| (Eq. 120, its linear part):

    q^a_ab = -rho_a h_a / (2 |r_ab|) alpha_AV c_s,a (v_ab . r_hat_ab)

entering the momentum equation like a pressure and heating with
du_a/dt = -1/(Omega_a rho_a) sum_b m_b alpha_AV c_s,a h_a/|r_ab| 1/2 (v_ab . r_hat)^2 F_ab(h_a),
which conserves energy and (pair forces along r_ab) angular momentum exactly. The equivalent
viscosities (Eqs. 121-122; continuum limit: the sum (Eq. 119) reduces to kernel moments with
int r dW/dr dV = -3 for any normalized kernel, so the 1/10 holds for our Wendland C4 too)

    nu_AV = alpha_AV c_s h / 10,     zeta_AV = 5/3 nu_AV,

and with the Shakura-Sunyaev nu = alpha_SS c_s H (Eq. 123):

    alpha_SS = alpha_AV <h> / (10 H)                      (Eq. 124)

with <h> the mean smoothing length in an annulus and H the scale height. The effective alpha_SS
therefore depends on resolution (<h>/H): report the measured one (phase-07 7.2).
"""

import numpy as np


def disc_q(rho, h, r, alpha_av, c_s, vdotr):
    """Pair pressure q^a_ab = -rho_a h_a/(2 r) alpha_AV c_s,a (v_ab . r_hat) (Eq. 120, linear
    part), for every pair, approaching or receding."""
    return (
        -np.asarray(rho)
        * np.asarray(h)
        / (2.0 * np.asarray(r))
        * alpha_av
        * np.asarray(c_s)
        * np.asarray(vdotr)
    )


def nu_av(alpha_av, c_s, h):
    """Equivalent shear viscosity nu_AV = alpha_AV c_s h / 10 (Eq. 121)."""
    return np.asarray(alpha_av) * np.asarray(c_s) * np.asarray(h) / 10.0


def alpha_ss_from_av(alpha_av, h_mean, scale_height):
    """alpha_SS = alpha_AV <h> / (10 H) (Eq. 124)."""
    return np.asarray(alpha_av) * np.asarray(h_mean) / (10.0 * np.asarray(scale_height))


def alpha_av_for(alpha_ss, h_mean, scale_height):
    """The alpha_AV that gives alpha_SS (inverse of Eq. 124): 10 alpha_SS H / <h>."""
    return 10.0 * np.asarray(alpha_ss) * np.asarray(scale_height) / np.asarray(h_mean)
