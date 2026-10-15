"""Gravity of a non-spinning black hole: the generalized Newtonian (GN) description.

Source: Tejeda & Rosswog 2013, MNRAS, doi:10.1093/mnras/stt853 (arXiv:1303.4068), "An accurate
Newtonian description of particle motion around a Schwarzschild black hole"; checked against the
paper's Eqs. (2.11) and (A.5) on 2026-10-14. Plan: phases/phase-01-orbits.md "Background"
and Step 1.2.

Units: G = c = M = 1 (lengths GM/c^2, times GM/c^3; the paper's r_g = 1). Position x and
velocity v are with respect to the GN coordinate time t. The horizon is at r = 2.

GN orbits have exactly the Schwarzschild shape r(phi) (precession, innermost stable orbit at
r = 6, marginally bound orbit at r = 4); only the timing differs slightly near the hole.

All functions accept a single particle (shape (3,)) or many (shape (N, 3)); vectors are on
the last axis. Pure functions: no state, no I/O.
"""

import numpy as np


def _radius(x: np.ndarray) -> np.ndarray:
    """|x| with a trailing axis kept, so it broadcasts against (..., 3) arrays."""
    return np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1, keepdims=True)


def gn_lagrangian(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """GN Lagrangian per unit mass (Tejeda & Rosswog 2013, Eq. 2.11):

        L = 1/2 f |v|^2 + 1/2 (f^2 - f) rdot^2 + 1/r,   f = r/(r - 2),  rdot = (x.v)/r

    (equal to the paper's 1/2 [r^2 rdot^2/(r-2)^2 + r^3 phidot^2/(r-2)] + 1/r). Units of c^2.
    Returns shape (...,) (no trailing axis).
    """
    r = _radius(x)[..., 0]
    f = r / (r - 2.0)
    rdot = np.sum(x * v, axis=-1) / r
    return 0.5 * f * np.sum(v * v, axis=-1) + 0.5 * (f * f - f) * rdot**2 + 1.0 / r


def gn_accel(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """GN acceleration d^2x/dt^2 from the Euler-Lagrange equations of ``gn_lagrangian``
    (Tejeda & Rosswog 2013, Eq. A.5):

        a = -(1/r^3)(1 - 2/r)^2 x  +  [2 rdot / (r (r - 2))] v  -  [3 |x cross v|^2 / r^5] x

    Units: c^4/(GM). Used for cross-checks, plots and the DOP853 reference integrator; the
    simulation itself integrates the canonical form (``gn_hamiltonian``).
    """
    r = _radius(x)
    rdot = _dot(x, v) / r
    h2 = np.sum(np.cross(x, v) ** 2, axis=-1, keepdims=True)
    return (
        -((1.0 - 2.0 / r) ** 2) * x / r**3 + 2.0 * rdot / (r * (r - 2.0)) * v - 3.0 * h2 / r**5 * x
    )


def _a_c(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of the GN Hamiltonian: A(r) = (r - 2)/r, C(r) = -2 (r - 2)/r^4."""
    return (r - 2.0) / r, -2.0 * (r - 2.0) / r**4


def gn_p_from_v(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Canonical momentum per unit mass p = dL/dv (phase-01 "Hamiltonian form"):

        p = f v + g rdot r_hat,   f = r/(r - 2),  g = 2r/(r - 2)^2,  rdot = (x.v)/r

    Units of c. For orbits, |x cross p| is the conserved angular momentum (= Schwarzschild L).
    """
    r = _radius(x)
    f = r / (r - 2.0)
    g = 2.0 * r / (r - 2.0) ** 2
    rdot = _dot(x, v) / r
    return f * v + g * rdot * x / r


def gn_v_from_p(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Velocity from canonical momentum, v = dH/dp = A p + C (p.x) x (inverse of
    ``gn_p_from_v``), with A = (r - 2)/r, C = -2 (r - 2)/r^4. Units of c."""
    a, c = _a_c(_radius(x))
    return a * p + c * _dot(p, x) * x


def gn_dH_dp(x: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: N802 (physics notation)
    """Hamilton's equation dx/dt = dH/dp; identical to ``gn_v_from_p``."""
    return gn_v_from_p(x, p)


def gn_hamiltonian(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """GN Hamiltonian per unit mass (phase-01 "Hamiltonian form"):

        H = 1/2 A |p|^2 + 1/2 C (p.x)^2 - 1/r,   A = (r - 2)/r,  C = -2 (r - 2)/r^4

    Equals the conserved GN energy eps = (E^2 - 1)/2, E the Schwarzschild energy at infinity
    (Tejeda & Rosswog 2013, Eqs. 2.6, 2.8). Units of c^2. Returns shape (...,).
    """
    r = _radius(x)
    a, c = _a_c(r)
    return (0.5 * a * _dot(p, p) + 0.5 * c * _dot(p, x) ** 2 - 1.0 / r)[..., 0]


def gn_dH_dx(x: np.ndarray, p: np.ndarray) -> np.ndarray:  # noqa: N802 (physics notation)
    """Gradient of ``gn_hamiltonian`` with respect to x (dp/dt = -dH/dx):

        dH/dx = [1/2 A'(r) |p|^2 + 1/2 C'(r) (p.x)^2] x/r + C (p.x) p + x/r^3
        A'(r) = 2/r^2,   C'(r) = (6r - 16)/r^5

    Units of c^4/(GM) (force per unit mass).
    """
    r = _radius(x)
    _, c = _a_c(r)
    da = 2.0 / r**2
    dc = (6.0 * r - 16.0) / r**5
    px = _dot(p, x)
    return (0.5 * da * _dot(p, p) + 0.5 * dc * px**2) * x / r + c * px * p + x / r**3


def gn_angmom(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Angular momentum vector per unit mass, x cross p (conserved exactly by GN motion; its
    magnitude equals the Schwarzschild L). Units of GM/c."""
    return np.cross(x, p)


def gn_mass_matrix_inv(x: np.ndarray) -> np.ndarray:
    """The matrix that turns momentum into velocity, v = Minv p:

        Minv = A(r) I + C(r) x x^T,   A = (r - 2)/r,  C = -2 (r - 2)/r^4

    Shape (..., 3, 3). Used in Phase 4 consistency checks and Phase 11 (its inverse turns
    spin accelerations into forces on p).
    """
    r = _radius(x)[..., 0]
    a, c = _a_c(r)
    eye = np.eye(3)
    return a[..., None, None] * eye + c[..., None, None] * x[..., :, None] * x[..., None, :]
