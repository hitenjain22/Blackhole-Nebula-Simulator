"""Step 3.1: SPH kernels (phases/phase-03-sph-core.md 3.2 test 1; expected-values §4;
Phantom Table 1 for the kernels' variance)."""

import numpy as np
import pytest
from scipy.integrate import quad

from nebula.physics.kernels import CUBIC, WENDLAND_C4, get_kernel
from tests.helpers import check

KERNELS = [CUBIC, WENDLAND_C4]


@pytest.mark.parametrize("k", KERNELS, ids=lambda k: k.name)
def test_normalization(k):
    """Integral of W over all space = 1, for two smoothing lengths."""
    for h in (1.0, 0.37):
        val, _ = quad(
            lambda r, h=h: 4 * np.pi * r * r * k.w(r, h),
            0,
            2 * h,
            points=[h],
            epsabs=0,
            epsrel=1e-13,
            limit=200,
        )
        check(f"{k.name}: integral W dV = 1 (h={h})", 1.0, val, rtol=1e-8)


@pytest.mark.parametrize(
    "k,variance", [(CUBIC, 9 / 10), (WENDLAND_C4, 8 / 13)], ids=["cubic", "wendland_c4"]
)
def test_variance_matches_phantom_table(k, variance):
    """Phantom Table 1: sigma^2 / h^2 = integral r^2 W dV (the full 3D second moment)."""
    val, _ = quad(
        lambda q: 4 * np.pi * q**4 * k.c_norm * k.f(q),
        0,
        2,
        points=[1.0],
        epsabs=0,
        epsrel=1e-13,
    )
    check(f"{k.name}: kernel variance sigma^2/h^2", variance, val, rtol=1e-10)


def test_cubic_values():
    """§4: W(0) = 1/(pi h^3) = 0.3183098862/h^3, W(h) = 0.0795774715/h^3, W(2h) = 0."""
    for h in (1.0, 0.5):
        check(f"cubic W(0, h={h}) h^3", 0.3183098862, CUBIC.w(0.0, h) * h**3, rtol=1e-9)
        check(f"cubic W(h, h={h}) h^3", 0.0795774715, CUBIC.w(h, h) * h**3, rtol=1e-9)
        check(f"cubic W(2h, h={h})", 0.0, CUBIC.w(2 * h, h))
    q = np.array([1 - 1e-12, 1 + 1e-12])
    check("cubic f continuous at q=1", CUBIC.f(q[0]), CUBIC.f(q[1]), atol=1e-11)
    check("cubic f' continuous at q=1", CUBIC.df(q[0]), CUBIC.df(q[1]), atol=1e-11)


def test_wendland_values():
    check("Wendland C4 W(0, 1)", 495 / (256 * np.pi), WENDLAND_C4.w(0.0, 1.0), rtol=1e-14)
    check("Wendland C4 W(2h)", 0.0, WENDLAND_C4.w(2.0, 1.0))
    q = np.linspace(0, 1.999, 2000)
    check("Wendland C4 positive inside support", True, bool(np.all(WENDLAND_C4.f(q) > 0)))


@pytest.mark.parametrize("k", KERNELS, ids=lambda k: k.name)
def test_derivatives_vs_finite_differences(k):
    """dW/dr and dW/dh against central differences of W (step 1e-6 h), 400 points."""
    rng = np.random.default_rng(5)
    h = rng.uniform(0.3, 3.0, 400)
    r = rng.uniform(0.0, 2.0, 400) * h
    r = r[np.abs(r / h - 1.0) > 1e-3]  # cubic spline: second derivative jumps at q = 1
    h = h[: len(r)]
    eps = 1e-6 * h
    fd_r = (k.w(r + eps, h) - k.w(r - eps, h)) / (2 * eps)
    fd_h = (k.w(r, h + eps) - k.w(r, h - eps)) / (2 * eps)
    scale = np.abs(k.c_norm / h**4)  # natural size of a derivative at this h
    check(f"{k.name}: dW/dr vs finite differences", fd_r / scale, k.dw_dr(r, h) / scale, atol=1e-7)
    check(f"{k.name}: dW/dh vs finite differences", fd_h / scale, k.dw_dh(r, h) / scale, atol=1e-7)


def test_neighbour_numbers():
    """§4: (4 pi/3)(2 eta)^3 = 57.9058 for eta = 1.2; Phantom Table 1: 137 for C4, hfact 1.6."""
    check("cubic, eta=1.2: neighbours", 57.9058, CUBIC.expected_neighbours(1.2), rtol=1e-6)
    check("Wendland C4, eta=1.6: neighbours", 137.0, WENDLAND_C4.expected_neighbours(), atol=0.5)
    check("get_kernel('cubic') is the default", CUBIC, get_kernel())
    with pytest.raises(ValueError):
        get_kernel("gaussian")
