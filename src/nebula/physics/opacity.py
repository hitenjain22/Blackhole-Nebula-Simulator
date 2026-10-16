"""Opacities for the gas (phases/phase-06-shocks-cooling-ring.md 6.2 "LTE" diagnostics; reused by
Phase 9). Physical units (*_si): kappa in m^2/kg, rho in kg/m^3, T in K.

    kappa_es  = 0.02 (1 + X) m^2/kg  (electron scattering, fully ionized; units.KAPPA_ES = 0.034)
    kappa_ff  = 3.68e22 g_ff (1 - Z)(1 + X) rho[g/cm^3] T^(-7/2)  cm^2/g    (Kramers free-free)
              = 3.68e18 g_ff (1 - Z)(1 + X) rho[kg/m^3] T^(-7/2)  m^2/kg
    kappa_eff = sqrt(kappa_abs (kappa_abs + kappa_es))  (effective, thermalizing opacity)
    tau = kappa rho L  over a length L (e.g. tau_h with L = h)

Sources: Kramers' free-free Rosseland-mean form as in Carroll & Ostlie, An Introduction to Modern
Astrophysics, §9.2 (Eq. 9.22), and standard lecture notes (checked 2026-10-16); the plan
(phase-06 6.2) asked for the coefficient to be checked, as it was written from memory there.
Gaunt factor g_ff = 1 (O(1) correction, as the plan's estimates). Metal bound-free absorption
(up to ~20x more, phase-06 6.2) is not included: kappa_abs here is a lower bound, so optical
depths from it are lower bounds too.
"""

import numpy as np

from nebula import units

KFF_SI = 3.68e18  # m^2/kg per (kg/m^3) per K^(-7/2): 3.68e22 cm^2/g x 0.1 (cm^2/g -> m^2/kg)
# x 1e-3 (rho in g/cm^3 -> kg/m^3)


def kappa_es_si(x_h: float = units.X_HYDROGEN) -> float:
    """Electron-scattering opacity 0.02 (1 + X) m^2/kg (fully ionized gas)."""
    return 0.02 * (1.0 + x_h)


def kappa_ff_si(rho_si, t_k, x_h: float = units.X_HYDROGEN, z: float = units.Z_METALS,
                gaunt: float = 1.0):  # fmt: skip
    """Kramers free-free (Rosseland mean) opacity 3.68e18 g (1-Z)(1+X) rho T^(-7/2) m^2/kg."""
    return (
        KFF_SI
        * gaunt
        * (1.0 - z)
        * (1.0 + x_h)
        * np.asarray(rho_si, float)
        * np.asarray(t_k, float) ** (-3.5)
    )


def kappa_eff_si(kappa_abs, kappa_es):
    """Effective (thermalization) opacity sqrt(kappa_abs (kappa_abs + kappa_es)): kappa_abs when
    absorption dominates, the geometric mean sqrt(kappa_abs kappa_es) when scattering does."""
    ka = np.asarray(kappa_abs, float)
    return np.sqrt(ka * (ka + np.asarray(kappa_es, float)))


def optical_depth(kappa, rho_si, length_m):
    """tau = kappa rho L (dimensionless)."""
    return np.asarray(kappa) * np.asarray(rho_si) * np.asarray(length_m)
