"""Gas 4-velocity from the GN state and the photon redshift factor g
(phases/phase-09-colour-physics.md 9.2-9.3; expected values in reference/expected-values.md §6).

Units: G = c = M = 1; the hole is non-spinning (Schwarzschild), horizon at r = 2.

Sources: Tejeda & Rosswog 2013 (GN Hamiltonian; its energy is eps = (E^2 - 1)/2 with E the
Schwarzschild energy at infinity, Eqs. 2.6, 2.8); the static-frame form of the 4-velocity
derived for phase-09 9.2 (checked to 7e-16 at plan time, expected-values §6, script 16);
g = (k.u)_obs / (k.u)_em for a static observer (Misner, Thorne & Wheeler 1973 §25.2, or any GR
text); circular-orbit cross-check: u^t = 1/sqrt(1 - 3/r), Omega = r^(-3/2) (Bardeen, Press &
Teukolsky 1972 with a = 0).

Pure functions; NumPy arrays or scalars, vectors on the last axis.
"""

import numpy as np

from nebula.physics.gravity import gn_hamiltonian


def gn_static_frame_velocity(x, p):
    """Gas 4-velocity in the static observer's frame from the GN state (position x, canonical
    momentum p per unit mass): returns (gamma (...,), gamma beta (..., 3)).

        E = sqrt(1 + 2 H),  gamma = E / sqrt(1 - 2/r)
        gamma beta = sqrt(1 - 2/r) p_r r_hat + p_perp,  p_r = p . r_hat,  p_perp = p - p_r r_hat

    phase-09 9.2 (exact for GN: the canonical momentum is the spatial part of the covariant
    Schwarzschild 4-velocity). gamma^2 - |gamma beta|^2 = 1 holds identically (test 9).
    gamma beta is in Cartesian components of the static orthonormal frame (radial direction
    along r_hat, angular directions as in flat space). Requires r > 2.
    """
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    r = np.linalg.norm(x, axis=-1)
    r_hat = x / r[..., None]
    lapse = np.sqrt(1.0 - 2.0 / r)
    gamma = np.sqrt(1.0 + 2.0 * gn_hamiltonian(x, p)) / lapse
    p_r = np.sum(p * r_hat, axis=-1)
    gamma_beta = p + ((lapse - 1.0) * p_r)[..., None] * r_hat
    return gamma, gamma_beta


def redshift_factor(r, gamma_beta, n_hat, r_cam=np.inf):
    """g = nu_received / nu_emitted for light leaving gas at radius r toward a static camera at
    r_cam (dimensionless):

        g = [sqrt(1 - 2/r) / sqrt(1 - 2/r_cam)] / [gamma (1 - beta . n_hat)]
          = [sqrt(1 - 2/r) / sqrt(1 - 2/r_cam)] / [gamma - (gamma beta) . n_hat]

    gamma_beta: (..., 3) gas velocity in the static frame at the gas (gamma = sqrt(1 + |gb|^2));
    n_hat: (..., 3) unit direction the light travels, in the same frame (minus the backward-traced
    camera ray's direction). phase-09 9.3; first factor gravitational, second Doppler.
    """
    gb = np.asarray(gamma_beta, float)
    gamma = np.sqrt(1.0 + np.sum(gb * gb, axis=-1))
    grav = np.sqrt(1.0 - 2.0 / np.asarray(r, float)) / np.sqrt(1.0 - 2.0 / r_cam)
    return grav / (gamma - np.sum(gb * np.asarray(n_hat, float), axis=-1))


def circular_orbit_redshift(r, b):
    """Cross-check form for gas on a circular Schwarzschild orbit seen from infinity:
    g = 1 / (u^t (1 - Omega b)), u^t = 1/sqrt(1 - 3/r), Omega = r^(-3/2), b the photon's impact
    parameter (b = +-r/sqrt(1 - 2/r) for light leaving along / against the orbital motion,
    0 for light leaving radially). phase-09 9.3; Bardeen, Press & Teukolsky 1972 (a = 0)."""
    r = np.asarray(r, float)
    return np.sqrt(1.0 - 3.0 / r) / (1.0 - r**-1.5 * np.asarray(b, float))
