"""Thin-disk relations for setting up the ring-spreading test (phases/phase-07-viscosity-drain.md
7.4; Lodato & Price 2010 §3). Newtonian point mass, code units (G = M = 1). Pure formulas.

    Omega_K = R^(-3/2),  H = c_s / Omega_K                       (vertical hydrostatic balance)
    rho(R, z) = Sigma / (sqrt(2 pi) H) exp(-z^2 / (2 H^2))       (isothermal vertical profile)
    v_phi^2 = R^2 (R^2 + z^2)^(-3/2) + c_s^2 d ln P_mid / d ln R  (radial balance: gravity's
              radial part at height z plus the midplane pressure gradient, O((H/R)^2);
              Pringle 1981 §2)
with P_mid = c_s^2 rho_mid and, for c_s ~ R^(-q), H ~ R^(3/2 - q):
    d ln P_mid / d ln R = -2 q + d ln Sigma / d ln R - (3/2 - q)
A Gaussian ring Sigma ~ exp(-(R - r0)^2 / (2 w^2)) has d ln Sigma / d ln R = -R (R - r0) / w^2.
"""

import numpy as np


def omega_k(r):
    """Keplerian angular frequency R^(-3/2)."""
    return np.asarray(r, float) ** -1.5


def scale_height(c_s, r):
    """H = c_s / Omega_K."""
    return np.asarray(c_s) / omega_k(r)


def gaussian_ring_dlnsigma(r, r0: float, width: float):
    """d ln Sigma / d ln R = -R (R - r0) / w^2 for Sigma ~ exp(-(R - r0)^2 / (2 w^2))."""
    r = np.asarray(r, float)
    return -r * (r - r0) / width**2


def vphi_pressure_supported(r, c_s, dlnsigma_dlnr, q: float = 0.75, z=0.0):
    """Circular speed with the pressure gradient: v_phi^2 = R^2 (R^2 + z^2)^(-3/2)
    + c_s^2 dlnP/dlnR, dlnP/dlnR = -2q + dlnSigma/dlnR - (3/2 - q) (c_s ~ R^-q)."""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    dlnp = -2.0 * q + np.asarray(dlnsigma_dlnr) - (1.5 - q)
    return np.sqrt(r**2 * (r**2 + z**2) ** -1.5 + np.asarray(c_s) ** 2 * dlnp)
