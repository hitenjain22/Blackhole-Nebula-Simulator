"""Steady thin disc shapes (physics/steady_disc.py) and the kernel's axis spread
(physics/kernels.Kernel.axis_rms), each against an independent fact."""

import numpy as np
from scipy.integrate import quad

from nebula.physics import steady_disc as sd
from nebula.physics.kernels import get_kernel
from nebula.physics.redshift import circular_orbit_redshift, redshift_factor
from tests.helpers import check


def test_luminosity_is_mdot_over_2_r_in():
    """Both faces of a steady disc radiate L = G M Mdot / (2 r_in) (FKR Eq. 5.20; Newtonian
    binding energy at r_in, half radiated): L / Mdot = 2 x (3 / 8 pi) x int f(r) 2 pi r dr."""
    for r_in in (6.0, 2.32088):
        val = quad(lambda r, ri=r_in: float(sd.flux_shape(r, ri)) * r, r_in, np.inf, limit=200)[0]
        lum = 2.0 * 3.0 / (8.0 * np.pi) * 2.0 * np.pi * val
        check(f"L / Mdot = 1 / (2 r_in), r_in = {r_in}", 1.0 / (2.0 * r_in), lum, rtol=1e-8)


def test_temperature_peak_and_far_slope():
    r = np.linspace(6.0001, 30.0, 400001)
    t = sd.temperature(r, 6.0, 50.0, 1.0)
    check("T peak at r = 49/36 r_in", 49.0 / 36.0 * 6.0, float(r[np.argmax(t)]), atol=1e-3)
    check("T(r_ref) = t_ref", 3.0e5, float(sd.temperature(50.0, 6.0, 50.0, 3.0e5)), rtol=1e-14)
    far = sd.temperature(np.array([1e8, 2e8]), 6.0, 50.0, 1.0)
    check("T ~ r^(-3/4) far out", 2.0**0.75, float(far[0] / far[1]), rtol=1e-3)
    check(
        "T = 0 at and inside r_in", 0.0, float(np.abs(sd.temperature([3.0, 6.0], 6, 50, 1)).max())
    )


def test_surface_density_is_the_steady_disc_law():
    """Sigma ~ (1 - sqrt(r_in/r)) r^(-3/2) / T(r) (nu Sigma = Mdot (1 - sqrt(r_in/r)) / 3 pi
    with nu ~ T r^(3/2)), written directly from the definition."""
    r = np.geomspace(6.01, 500.0, 300)
    direct = (1.0 - np.sqrt(6.0 / r)) * r**-1.5 / sd.temperature(r, 6.0, 50.0, 1.0)
    ratio = sd.surface_density(r, 6.0, 50.0, 1.0) / direct
    check("Sigma / steady-disc law is constant", 0.0, float(np.ptp(ratio) / ratio.mean()),
          atol=1e-12)  # fmt: skip
    check("Sigma(r_ref) = sigma_ref", 2e-5, float(sd.surface_density(50.0, 6.0, 50.0, 2e-5)),
          rtol=1e-14)  # fmt: skip


def test_circular_velocity_gives_the_circular_orbit_redshift():
    """gamma beta = 1/sqrt(r - 3) along the orbit, through the general redshift formula, equals
    the circular-orbit form g = sqrt(1 - 3/r) / (1 - Omega b) (Bardeen et al. 1972, a = 0) for
    light leaving forward, backward (b = +-r / sqrt(1 - 2/r)) and radially (b = 0)."""
    r = np.array([4.0, 6.0, 10.0, 45.0])
    gb = np.stack([np.zeros_like(r), sd.circular_gamma_beta(r), np.zeros_like(r)], -1)  # +y
    b_max = r / np.sqrt(1.0 - 2.0 / r)
    for name, n, b in (("forward", (0, 1, 0), b_max), ("backward", (0, -1, 0), -b_max),
                       ("radial", (1, 0, 0), 0.0 * r)):  # fmt: skip
        g = redshift_factor(r, gb, np.broadcast_to(np.array(n, float), gb.shape))
        check(f"circular orbit redshift, {name}", circular_orbit_redshift(r, b), g, rtol=1e-12)


def test_kernel_axis_rms_against_a_3d_integral():
    """sqrt(<z^2>) of W over a 3D grid (direct sum) vs the 1D-quadrature formula."""
    n = 161
    ax = np.linspace(-2.0, 2.0, n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    q = np.sqrt(x * x + y * y + z * z)
    for name in ("wendland_c4", "cubic"):
        k = get_kernel(name)
        w = k.f(q)
        direct = float(np.sqrt(np.sum(w * z * z) / np.sum(w)))
        check(f"{name}: axis rms, 3D sum vs formula", direct, k.axis_rms(), rtol=2e-3)
    check("cubic spline: <z^2> = 0.3 h^2 exactly", 0.3, get_kernel("cubic").axis_rms() ** 2,
          rtol=1e-10)  # fmt: skip


def test_kerr_circular_velocity():
    """Bardeen et al. 1972: a = 0 gives the Schwarzschild speed; v = 1/2 at the ISCO for a = 0
    and for a -> 1; the speed grows inward."""
    from nebula.physics.orbits import isco_radius

    r = np.geomspace(3.5, 500.0, 50)
    check("Kerr circular speed, a = 0 = Schwarzschild", sd.circular_gamma_beta(r),
          sd.circular_gamma_beta_kerr(r, 0.0), rtol=1e-12)  # fmt: skip

    def speed(gb):
        return gb / np.sqrt(1.0 + gb * gb)

    check("v at the ISCO, a = 0", 0.5, float(speed(sd.circular_gamma_beta_kerr(6.0, 0.0))),
          rtol=1e-12)  # fmt: skip
    a = 1.0 - 1e-9
    v = float(speed(sd.circular_gamma_beta_kerr(isco_radius(a), a)))
    check("v at the ISCO, a -> 1", 0.5, v, atol=2e-3)
    gb = sd.circular_gamma_beta_kerr(np.geomspace(isco_radius(0.9), 100.0, 40), 0.9)
    check("a = 0.9: speed falls outward", True, bool(np.all(np.diff(gb) < 0)))
