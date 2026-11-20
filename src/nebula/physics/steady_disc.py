"""Steady thin accretion disc between the torus and the innermost stable orbit: the radial
shapes of its temperature and surface density (render-time inner disc, render/inner_disc.py;
DECISIONS 2026-11-20). Code units (G = c = M = 1), temperatures in K. Pure formulas.

Steady disc with zero torque at the inner edge r_in (Shakura & Sunyaev 1973; Pringle 1981
Eqs. 3.10-3.11; Frank, King & Raine 2002, "Accretion Power in Astrophysics", Eqs. 5.41-5.43):

    sigma_SB T^4 = (3 Mdot / (8 pi r^3)) (1 - sqrt(r_in / r))         (each face)
    nu Sigma     = (Mdot / (3 pi)) (1 - sqrt(r_in / r))

With an alpha viscosity nu = alpha c_s H = alpha c_s^2 / Omega, c_s^2 ~ T and Omega = r^(-3/2),
nu ~ T r^(3/2), so Sigma ~ (1 - sqrt(r_in / r)) r^(-3/2) / T. Both are used as shapes,
normalised to the values measured at the torus's inner edge (r_ref), so Mdot and alpha drop out:
the disc continues the simulated gas inward instead of adding a free brightness.

Newtonian disc: the relativistic factors of Novikov & Thorne 1973 (Page & Thorne 1974) are left
out (a look, not a measurement). For a non-spinning hole r_in = 6 (the ISCO,
physics.orbits.isco_radius); a spinning hole moves it inward (Phase 11).
"""

import numpy as np


def flux_shape(r, r_in: float):
    """f(r) = r^-3 (1 - sqrt(r_in / r)) for r > r_in, else 0: the radial shape of a steady
    disc's emitted flux sigma_SB T^4 (FKR Eq. 5.43). Peaks at r = (49/36) r_in."""
    r = np.asarray(r, float)
    inside = r > r_in
    return np.where(inside, r**-3.0 * (1.0 - np.sqrt(r_in / np.where(inside, r, r_in))), 0.0)


def temperature(r, r_in: float, r_ref: float, t_ref: float):
    """T(r) = t_ref [f(r) / f(r_ref)]^(1/4) [K]: the steady-disc temperature (FKR Eq. 5.43)
    through t_ref at r_ref (> r_in); 0 at and inside r_in. Hottest at r = (49/36) r_in, where
    for r_in = 6, r_ref = 50 it is 2.66 t_ref."""
    return t_ref * (flux_shape(r, r_in) / flux_shape(r_ref, r_in)) ** 0.25


def surface_density(r, r_in: float, r_ref: float, sigma_ref: float):
    """Sigma(r) = sigma_ref g(r) / g(r_ref), g = (1 - sqrt(r_in / r)) r^(-3/2) / T(r) [code
    mass per area]: steady disc nu Sigma = Mdot (1 - sqrt(r_in / r)) / (3 pi) (FKR Eq. 5.41) with
    alpha viscosity nu ~ T r^(3/2) and T from ``temperature``. Equivalently
    g ~ (1 - sqrt(r_in / r))^(3/4) r^(-3/4); 0 at and inside r_in."""

    def g(x):
        x = np.asarray(x, float)
        inside = x > r_in
        safe = np.where(inside, x, 2.0 * r_in)
        return np.where(inside, (1.0 - np.sqrt(r_in / safe)) ** 0.75 * safe**-0.75, 0.0)

    return sigma_ref * g(r) / g(r_ref)


def circular_gamma_beta(r):
    """|gamma beta| of a circular Schwarzschild orbit measured by a static observer: the local
    speed is v = 1 / sqrt(r - 2) (Misner, Thorne & Wheeler 1973 Ex. 25.20; Bardeen, Press &
    Teukolsky 1972 with a = 0), so gamma beta = v / sqrt(1 - v^2) = 1 / sqrt(r - 3).
    Needs r > 3 (the photon orbit); along the orbit's direction of motion. Consistent with
    physics.redshift.circular_orbit_redshift (tests/physics/test_steady_disc.py)."""
    return 1.0 / np.sqrt(np.asarray(r, float) - 3.0)


def circular_gamma_beta_kerr(r, a: float):
    """|gamma beta| of a prograde circular equatorial orbit around a spinning hole, measured by
    the local zero-angular-momentum observer (Bardeen, Press & Teukolsky 1972, Eq. 3.10):

        v = (r^2 - 2 a sqrt(r) + a^2) / (sqrt(Delta) (r^(3/2) + a)),  Delta = r^2 - 2 r + a^2

    gamma beta = v / sqrt(1 - v^2). a = 0: v = 1 / sqrt(r - 2), i.e. circular_gamma_beta. v = 1/2
    at the innermost stable orbit for both a = 0 (r = 6) and a -> 1 (r -> 1). Needs r outside
    the prograde photon orbit."""
    r = np.asarray(r, float)
    delta = r * r - 2.0 * r + a * a
    v = (r * r - 2.0 * a * np.sqrt(r) + a * a) / (np.sqrt(delta) * (r**1.5 + a))
    return v / np.sqrt(1.0 - v * v)
