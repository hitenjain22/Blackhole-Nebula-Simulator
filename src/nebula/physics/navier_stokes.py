"""Physical (Navier-Stokes) viscosity for the disk drain (phases/phase-07-viscosity-drain.md 7.2,
Option A). Pure formulas; the SPH pair sums live in sim/sph.py.

Source: Price et al. 2018 (Phantom, PASA 35, e031, arXiv:1702.03930) §2.6.2 and §2.7, checked in
the paper's text 2026-10-21 (Eqs. 123, 125, 129, 133-136, 145); following Flebbe et al. 1994 and
Lodato & Price 2010 (MNRAS 405, 1212). Units: code units (G = c = M = 1), velocities in c.

Stress tensor (Eq. 129), with shear coefficient eta = rho nu and bulk zeta = rho zeta_v:

    S^ij = [P - (zeta - 2/3 eta) dv^k/dx^k] delta^ij - eta (dv^i/dx^j + dv^j/dx^i)

so S = P delta - Sigma_v, where ``viscous_stress`` returns the viscous part Sigma_v. The
equation of motion is dv^i/dt = -(1/rho) dS^ij/dx^j (Eq. 127); the heating (Eq. 136) is

    Lambda = (zeta_v - 2/3 nu) (div v)^2 + nu/2 sum_ij (dv^i/dx^j + dv^j/dx^i)^2,

positive definite when the same gradient G^ij = dv^i/dx^j enters the stress and the heating
(Phantom's difference operator, Eq. 131: sim/sph.py). Disk viscosity (Shakura & Sunyaev 1973,
Phantom Eqs. 123, 125): nu = alpha_SS c_s H with H = c_s / Omega_K, Omega_K = r^(-3/2) (Keplerian,
Newtonian: the GN circular frequency differs by 1-2% at r ~ 50; phase-07 7.2). Explicit time-step
limit (Eq. 145): dt <= C_visc h^2 / nu, C_visc = 0.25 (Brookshaw 1994).
"""

import numpy as np

C_VISC = 0.25


def ss_viscosity(alpha, c_s, r):
    """Shakura-Sunyaev kinematic viscosity nu = alpha c_s^2 / Omega_K, Omega_K = r^(-3/2)."""
    return np.asarray(alpha) * np.asarray(c_s) ** 2 * np.asarray(r, float) ** 1.5


def viscous_stress(grad_v, rho, nu, zeta_v=0.0):
    """Viscous part of the stress (Eq. 129), Sigma_v = rho [(zeta_v - 2/3 nu) tr(G) delta
    + nu (G + G^T)], for gradients G (..., 3, 3) with G[i, j] = dv^i/dx^j."""
    g = np.asarray(grad_v, float)
    rho, nu, zeta_v = (np.asarray(a, float)[..., None, None] for a in (rho, nu, zeta_v))
    div = np.trace(g, axis1=-2, axis2=-1)[..., None, None]
    return rho * ((zeta_v - 2.0 / 3.0 * nu) * div * np.eye(3) + nu * (g + np.swapaxes(g, -1, -2)))


def viscous_heating(grad_v, nu, zeta_v=0.0):
    """Lambda (Eq. 136): (zeta_v - 2/3 nu) (div v)^2 + nu/2 sum (G + G^T)_ij^2 per unit mass."""
    g = np.asarray(grad_v, float)
    t = g + np.swapaxes(g, -1, -2)
    div = np.trace(g, axis1=-2, axis2=-1)
    return (np.asarray(zeta_v) - 2.0 / 3.0 * np.asarray(nu)) * div**2 + 0.5 * np.asarray(
        nu
    ) * np.sum(t * t, axis=(-2, -1))


def viscous_dt(h, nu, c_visc: float = C_VISC):
    """Explicit time-step limit dt = C_visc h^2 / nu (Eq. 145); inf where nu = 0."""
    nu = np.asarray(nu, float)
    with np.errstate(divide="ignore"):
        return np.where(nu > 0, c_visc * np.asarray(h, float) ** 2 / nu, np.inf)
