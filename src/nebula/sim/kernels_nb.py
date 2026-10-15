"""Numba copies of the SPH kernel shapes for the hot loops (CLAUDE.md rule 1).

Each function mirrors one in nebula.physics.kernels and is tested against it
(tests/sim/test_density.py). Kernel ids: 0 = cubic spline, 1 = Wendland C4.
"""

import numba
import numpy as np

from nebula.physics.kernels import CUBIC, WENDLAND_C4

KERNEL_IDS = {"cubic": 0, "wendland_c4": 1}
C_NORM = np.array([CUBIC.c_norm, WENDLAND_C4.c_norm])


@numba.njit(cache=True)
def kernel_f(q, kid):
    """Mirrors physics.kernels.cubic_f (kid 0) / wendland_c4_f (kid 1)."""
    if q >= 2.0:
        return 0.0
    if kid == 0:
        if q < 1.0:
            return 1.0 - 1.5 * q * q + 0.75 * q * q * q
        t = 2.0 - q
        return 0.25 * t * t * t
    x = 1.0 - 0.5 * q
    x2 = x * x
    return x2 * x2 * x2 * (35.0 / 12.0 * q * q + 3.0 * q + 1.0)


@numba.njit(cache=True)
def kernel_df(q, kid):
    """Mirrors physics.kernels.cubic_df (kid 0) / wendland_c4_df (kid 1)."""
    if q >= 2.0:
        return 0.0
    if kid == 0:
        if q < 1.0:
            return -3.0 * q + 2.25 * q * q
        t = 2.0 - q
        return -0.75 * t * t
    x = 1.0 - 0.5 * q
    x2 = x * x
    return -7.0 / 3.0 * q * x2 * x2 * x * (5.0 * q + 2.0)
