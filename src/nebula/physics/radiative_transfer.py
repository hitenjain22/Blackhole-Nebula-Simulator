"""Radiative transfer along a ray (phases/phase-08-gas-rendering.md 8.4).

Source: Rybicki & Lightman 1979, *Radiative Processes in Astrophysics*, ch. 1 (Eqs. 1.23-1.30):
along a path, dI/dtau = S - I with dtau = alpha ds (alpha: absorption coefficient, 1/length;
S = j / alpha: source function). For S constant across a step of optical depth dtau the exact
solution is I_out = I_in e^-dtau + S (1 - e^-dtau). Marching front to back from the camera
(light that reaches the camera from a step is dimmed by the transmission T of everything in
front of it):

    I += T S (1 - e^-dtau),    T *= e^-dtau

This is exact for piecewise-constant S and alpha, tends to I += T j ds when dtau -> 0 (thin
gas) and to I += T S when dtau >> 1 (opaque gas). 1 - e^-dtau is evaluated as -expm1(-dtau)
so thin steps keep full precision.

Phase 8's placeholder emission model (8.4; Phase 9 replaces it): grey gas, S = sigma T^4 / pi
(frequency-integrated blackbody intensity) and alpha = kappa rho with a chosen grey opacity.

Pure functions; NumPy arrays or scalars.
"""

import numpy as np

from nebula import units


def transfer_step(intensity, transmission, source, dtau):
    """One front-to-back step: returns (I + T S (1 - e^-dtau), T e^-dtau).

    intensity, source: same units (any intensity unit); transmission: dimensionless (1 at the
    camera); dtau: optical depth of the step (>= 0). Rybicki & Lightman 1979 Eq. 1.30.
    """
    dtau = np.asarray(dtau, float)
    return intensity + transmission * source * (-np.expm1(-dtau)), transmission * np.exp(-dtau)


def blackbody_intensity_si(t_k):
    """Frequency-integrated blackbody intensity B = sigma T^4 / pi [W m^-2 sr^-1], T in K.

    Rybicki & Lightman 1979 Eq. 1.43 (sigma T^4 = pi B). Phase 8 placeholder source function.
    """
    t = np.asarray(t_k, float)
    return units.SIGMA_SB * t**4 / np.pi


def grey_absorption(rho, kappa):
    """Absorption coefficient alpha = kappa rho [1/length] (rho in mass/length^3, kappa in
    length^2/mass, same unit system). Rybicki & Lightman 1979 Eq. 1.18."""
    return np.asarray(kappa, float) * np.asarray(rho, float)
