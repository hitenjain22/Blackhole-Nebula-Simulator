"""Initial conditions for the SPH ring-spreading test (phases/phase-07-viscosity-drain.md 7.4;
Lodato & Price 2010 §3): a Gaussian ring of gas around the hole, locally isothermal, in
vertical and radial balance.

Plan values: r0 = 50, Gaussian width 5, H/R = 0.1 at r0, c_s ~ R^(-3/4) (so nu = alpha
c_s^2 / Omega_K is constant). Particles are placed with a scrambled Halton sequence (low
discrepancy) rather than independent random numbers, so the starting density noise, which the
shock viscosity would otherwise turn into spurious spreading, is small. Formulas from
physics/thin_disk.py and physics/eos.py.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from nebula.physics.eos import locally_isothermal_cs
from nebula.physics.kernels import DEFAULT_ETA
from nebula.physics.thin_disk import gaussian_ring_dlnsigma, scale_height, vphi_pressure_supported

R0 = 50.0
WIDTH = 5.0
H_OVER_R = 0.1


@dataclass(frozen=True)
class Ring:
    x: np.ndarray
    v: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    h: np.ndarray  # first guess (density-based)
    c_s0: float  # sound speed at r0 (HydroConfig isothermal_cs, with iso_r0 = r0)
    r0: float
    width: float


def ring_setup(n: int, r0: float = R0, width: float = WIDTH, h_over_r: float = H_OVER_R,
               q: float = 0.75, seed: int = 0, n_sigma: float = 4.0) -> Ring:  # fmt: skip
    """n particles, total mass 1, in a Gaussian ring (cut at +-n_sigma widths)."""
    c_s0 = h_over_r * r0**-0.5  # H/R = c_s / (R Omega_K) = c_s sqrt(R) at r0
    s = qmc.Halton(d=3, scramble=True, seed=seed).random(n)
    # radius: mass in dR ~ Sigma R dR, inverted on a fine table
    grid = np.linspace(max(r0 - n_sigma * width, 1e-3), r0 + n_sigma * width, 20001)
    pdf = grid * np.exp(-0.5 * ((grid - r0) / width) ** 2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    rr = np.interp(s[:, 0], cdf / cdf[-1], grid)
    phi = 2.0 * np.pi * s[:, 1]
    cs = locally_isothermal_cs(rr, c_s0, r0, q)
    hz = scale_height(cs, rr)
    z = hz * ndtri(np.clip(s[:, 2], 1e-9, 1.0 - 1e-9))
    x = np.c_[rr * np.cos(phi), rr * np.sin(phi), z]
    vphi = vphi_pressure_supported(rr, cs, gaussian_ring_dlnsigma(rr, r0, width), q, z)
    v = np.c_[-vphi * np.sin(phi), vphi * np.cos(phi), np.zeros(n)]
    mass = np.full(n, 1.0 / n)
    # density estimate for the first h: Sigma(R) / (sqrt(2 pi) H) exp(-z^2 / 2H^2)
    sigma = np.exp(-0.5 * ((rr - r0) / width) ** 2) / (cdf[-1] * 2.0 * np.pi)
    rho = sigma / (np.sqrt(2.0 * np.pi) * hz) * np.exp(-0.5 * (z / hz) ** 2)
    h = DEFAULT_ETA * (mass / rho) ** (1.0 / 3.0)
    u = 1.5 * cs**2  # not evolved (isothermal); kept positive for the bookkeeping
    return Ring(x, v, u, mass, h, c_s0, r0, width)
