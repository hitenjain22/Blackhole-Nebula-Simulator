"""Equations of state (phases/phase-03-sph-core.md 3.1; Phase 6 adds gas + radiation).

Code units throughout: rho in code density, u (specific internal energy) in c^2,
P in code pressure (rho c^2), sound speeds in c.

Ideal gas (adiabatic index gamma; Phantom, Price et al. 2018, Eq. 27-28):
    P = (gamma - 1) rho u,        c_s = sqrt(gamma P / rho) = sqrt(gamma (gamma - 1) u)
Isothermal:
    P = c_s^2 rho
"""

import numpy as np


def ideal_gas_pressure(rho, u, gamma: float):
    """P = (gamma - 1) rho u."""
    return (gamma - 1.0) * np.asarray(rho) * np.asarray(u)


def ideal_gas_sound_speed(u, gamma: float):
    """Adiabatic sound speed sqrt(gamma (gamma - 1) u)."""
    return np.sqrt(gamma * (gamma - 1.0) * np.asarray(u))


def isothermal_pressure(rho, c_s: float):
    """P = c_s^2 rho."""
    return c_s * c_s * np.asarray(rho)
