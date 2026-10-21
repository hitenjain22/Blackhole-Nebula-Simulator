"""One-dimensional viscous accretion disk (phases/phase-07-viscosity-drain.md 7.3).

Keplerian, axisymmetric, thin disk (Pringle 1981, ARA&A 19, 137, Eq. 3.11; Lynden-Bell & Pringle
1974, MNRAS 168, 603):

    dSigma/dt = (1 / (2 pi r)) dMdot/dr,     Mdot(r) = 6 pi r^(1/2) d(nu Sigma r^(1/2))/dr

Mdot > 0 is inflow. Units: any consistent ones (in the project G = M = 1, lengths GM/c^2, times
GM/c^3; nu in length^2/time; Sigma in mass/length^2). Newtonian Keplerian: the zero-torque inner
edge at r_in (6 for a non-spinning hole) stands in for the relativistic inner edge.

Finite volume on cells uniform in sqrt(r): each cell's mass changes only through the mass
fluxes at its two edges, so mass is conserved to round-off. With G = nu Sigma r^(1/2) at cell
centres, the flux through the edge between cells i-1 and i is

    Mdot_e = 6 pi sqrt(r_e) (G_i - G_(i-1)) / (r_i - r_(i-1)),

at the inner edge (zero torque: nu Sigma = 0 at r_in) Mdot_in = 6 pi sqrt(r_in) G_0 / (r_0 - r_in)
(what crosses it is accreted), and at the outer edge Mdot = 0 (closed) by default. For nu
independent of Sigma the system dM/dt = A M is linear and tridiagonal; it is advanced with
Crank-Nicolson, after a few backward-Euler steps (Rannacher 1984) that damp the oscillations
Crank-Nicolson leaves on sharp initial rings. The accreted mass is booked with the same
time weighting as the interior update, so disk + accreted = initial mass to round-off.

Pure functions: arrays in, arrays out.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ive


@dataclass(frozen=True)
class Grid:
    edges: np.ndarray  # (n + 1,) cell edges, uniform in sqrt(r)
    centres: np.ndarray  # (n,) centres (midpoints in sqrt(r))
    areas: np.ndarray  # (n,) annulus areas pi (r_out^2 - r_in^2)


def make_grid(r_in: float, r_out: float, n: int) -> Grid:
    """Cells uniform in sqrt(r) between r_in and r_out."""
    x = np.linspace(np.sqrt(r_in), np.sqrt(r_out), n + 1)
    edges = x**2
    centres = (0.5 * (x[1:] + x[:-1])) ** 2
    return Grid(edges, centres, np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))


def lbp_sigma(x, tau):
    """Lynden-Bell & Pringle (1974) spreading ring with constant nu, Sigma R0^2 / m:

        Sigma pi R0^2 / m = tau^-1 x^(-1/4) exp(-(1 - x)^2 / tau) ive(1/4, 2x/tau)

    x = r / R0, tau = 12 nu t / R0^2 (expected-values §5; ive = exponentially scaled modified
    Bessel function, so large arguments do not overflow). Returns Sigma R0^2 / m."""
    x = np.asarray(x, float)
    return np.exp(-((1.0 - x) ** 2) / tau) * ive(0.25, 2.0 * x / tau) / (tau * x**0.25 * np.pi)


def rate_matrix(grid: Grid, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dM/dt = A M for the cell masses M (Sigma = M / area): A as banded rows (3, n) for
    scipy.linalg.solve_banded (upper, diagonal, lower), and the row b with Mdot_in = b . M."""
    r, re, area = grid.centres, grid.edges, grid.areas
    n = len(r)
    g = nu * np.sqrt(r) / area  # G_i = g_i M_i
    a = np.zeros((3, n))
    # interior edge between cells i-1 and i: Mdot_e = c_e (G_i - G_(i-1)); cell i loses it
    # (inflow leaves through its inner edge), cell i-1 gains it
    c = 6.0 * np.pi * np.sqrt(re[1:-1]) / (r[1:] - r[:-1])  # edges 1..n-1
    a[1, 1:] -= c * g[1:]  # d M_i / d M_i
    a[0, 1:] += c * g[1:]  # d M_(i-1) / d M_i  (upper diagonal)
    a[1, :-1] -= c * g[:-1]  # d M_(i-1) / d M_(i-1)
    a[2, :-1] += c * g[:-1]  # d M_i / d M_(i-1)  (lower diagonal)
    # inner edge (zero torque): Mdot_in = c_in G_0 leaves cell 0
    c_in = 6.0 * np.pi * np.sqrt(re[0]) / (r[0] - re[0])
    a[1, 0] -= c_in * g[0]
    b = np.zeros(n)
    b[0] = c_in * g[0]
    return a, b


def _apply(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """A M for banded A (rows: upper, diagonal, lower)."""
    out = a[1] * m
    out[:-1] += a[0, 1:] * m[1:]
    out[1:] += a[2, :-1] * m[:-1]
    return out


@dataclass
class DiskState:
    mass: np.ndarray  # (n,) cell masses
    t: float
    accreted: float  # mass that crossed r_in so far
    supplied: float = 0.0  # mass added by a source so far


def evolve(
    grid: Grid, nu: np.ndarray, state: DiskState, t_end: float, dt: float, n_rannacher: int = 4,
    source=None, record_every: int = 0,
):  # fmt: skip
    """Advance to t_end with steps dt (the last one shortened). ``source``: (cell index, mass
    per time) fed into that cell. The first ``n_rannacher`` steps of this call are backward
    Euler (half the step each, Rannacher 1984), the rest Crank-Nicolson. Returns the new state
    and, if record_every > 0, a list of (t, accreted, Mdot_in) every that many steps."""
    a, b = rate_matrix(grid, nu)
    n = len(grid.centres)
    eye = np.zeros((3, n))
    eye[1] = 1.0
    m = state.mass.copy()
    t, acc, sup = state.t, state.accreted, state.supplied
    log = []
    k = 0
    rann = n_rannacher
    while t < t_end * (1.0 - 1e-14):
        h = min(dt, t_end - t)
        src = np.zeros(n)
        if source is not None:
            src[source[0]] = source[1]
        if rann > 0:  # two backward-Euler half steps
            for _ in range(2):
                m_new = solve_banded((1, 1), eye - 0.5 * h * a, m + 0.5 * h * src)
                acc += 0.5 * h * float(b @ m_new)
                sup += 0.5 * h * float(src.sum())
                m = m_new
            rann -= 1
        else:  # Crank-Nicolson
            m_new = solve_banded((1, 1), eye - 0.5 * h * a, m + 0.5 * h * _apply(a, m) + h * src)
            acc += 0.5 * h * float(b @ m + b @ m_new)
            sup += h * float(src.sum())
            m = m_new
        t += h
        k += 1
        if record_every and k % record_every == 0:
            log.append((t, acc, float(b @ m)))
    return DiskState(m, t, acc, sup), log
