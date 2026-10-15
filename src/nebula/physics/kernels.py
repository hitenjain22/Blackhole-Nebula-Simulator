"""SPH smoothing kernels (phases/phase-03-sph-core.md 3.1).

W(r, h) = (C_norm / h^3) f(q),   q = r / h,   compact support r < 2h (both kernels here).
Derived quantities (Price et al. 2018, Phantom, PASA 35, e031, arXiv:1702.03930, Eqs. 15-16):
    dW/dr = (C_norm / h^4) f'(q)                          (gradient: r_hat_ab dW/dr)
    dW/dh = -(C_norm / h^4) [3 f(q) + q f'(q)]

Kernels (checked against Phantom Eqs. 17 and 21 on 2026-10-14):
    M4 cubic spline (default; Monaghan & Lattanzio 1985), C_norm = 1/pi in 3D:
        f(q) = 1 - 3/2 q^2 + 3/4 q^3  (0 <= q < 1);  1/4 (2 - q)^3  (1 <= q < 2);  0  (q >= 2)
    Wendland C4 (Wendland 1995; Dehnen & Aly 2012), C_norm = 495 / (256 pi) in 3D:
        f(q) = (1 - q/2)^6 (35/12 q^2 + 3 q + 1)  (q < 2);  0  (q >= 2)
Units: r and h in code lengths; W in 1/length^3. Pure functions, NumPy arrays or scalars.

The kernel choice lives here only: ``get_kernel(name)``, default ``DEFAULT_KERNEL``. Numba
copies used inside the SPH kernels name the function they mirror and are tested against it.
"""

from dataclasses import dataclass

import numpy as np

R_KERN = 2.0  # support radius in units of h, both kernels


def cubic_f(q):
    """Cubic spline shape f(q) (dimensionless)."""
    q = np.asarray(q, float)
    return np.where(
        q < 1.0, 1.0 - 1.5 * q**2 + 0.75 * q**3, np.where(q < 2.0, 0.25 * (2.0 - q) ** 3, 0.0)
    )


def cubic_df(q):
    """df/dq of the cubic spline."""
    q = np.asarray(q, float)
    return np.where(q < 1.0, -3.0 * q + 2.25 * q**2, np.where(q < 2.0, -0.75 * (2.0 - q) ** 2, 0.0))


def wendland_c4_f(q):
    """Wendland C4 shape f(q), support 2h."""
    q = np.asarray(q, float)
    x = np.clip(1.0 - 0.5 * q, 0.0, None)
    return np.where(q < 2.0, x**6 * (35.0 / 12.0 * q**2 + 3.0 * q + 1.0), 0.0)


def wendland_c4_df(q):
    """df/dq of the Wendland C4 kernel: -(7/3) q (1 - q/2)^5 (5 q + 2) (derived from f)."""
    q = np.asarray(q, float)
    x = np.clip(1.0 - 0.5 * q, 0.0, None)
    return np.where(q < 2.0, -7.0 / 3.0 * q * x**5 * (5.0 * q + 2.0), 0.0)


@dataclass(frozen=True)
class Kernel:
    name: str
    c_norm: float  # 3D normalization
    hfact: float  # default eta = h / (m/rho)^(1/3) (Phantom Table 1)
    f: object
    df: object

    def w(self, r, h):
        """W(r, h) in 1/length^3."""
        h = np.asarray(h, float)
        return self.c_norm / h**3 * self.f(np.asarray(r, float) / h)

    def dw_dr(self, r, h):
        """dW/dr (<= 0) in 1/length^4."""
        h = np.asarray(h, float)
        return self.c_norm / h**4 * self.df(np.asarray(r, float) / h)

    def dw_dh(self, r, h):
        """dW/dh in 1/length^4."""
        h = np.asarray(h, float)
        q = np.asarray(r, float) / h
        return -self.c_norm / h**4 * (3.0 * self.f(q) + q * self.df(q))

    def expected_neighbours(self, eta: float | None = None) -> float:
        """Mean neighbour number for h = eta (m/rho)^(1/3): (4 pi / 3) (R_kern eta)^3."""
        eta = self.hfact if eta is None else eta
        return 4.0 * np.pi / 3.0 * (R_KERN * eta) ** 3


CUBIC = Kernel("cubic", 1.0 / np.pi, 1.2, cubic_f, cubic_df)
WENDLAND_C4 = Kernel("wendland_c4", 495.0 / (256.0 * np.pi), 1.6, wendland_c4_f, wendland_c4_df)
_KERNELS = {k.name: k for k in (CUBIC, WENDLAND_C4)}

# Project default (docs/DECISIONS.md, Step 3.6): Wendland C4 with eta = 1.6 (~137 neighbours).
# The plan's original default, the cubic spline at eta = 1.2, failed the sound-wave
# convergence test (its glass noise floor, 0.34%, scatters small waves; Wendland: 0.13%).
DEFAULT_KERNEL = "wendland_c4"
DEFAULT_ETA = WENDLAND_C4.hfact


def get_kernel(name: str = DEFAULT_KERNEL) -> Kernel:
    """The kernel selected by name in a run config ("cubic" or "wendland_c4")."""
    try:
        return _KERNELS[name]
    except KeyError:
        raise ValueError(f"unknown kernel {name!r}; choose from {sorted(_KERNELS)}") from None
