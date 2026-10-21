"""Navier-Stokes viscosity formulas (physics/navier_stokes.py) on hand-worked cases."""

import numpy as np

from nebula.physics.navier_stokes import (
    ss_viscosity,
    viscous_dt,
    viscous_heating,
    viscous_stress,
)
from tests.helpers import check


def test_ss_viscosity_hand_value():
    """nu = alpha c_s^2 / Omega_K = 0.1 x 1e-4 x 50^1.5 = 3.5355339e-3."""
    check("nu = alpha c_s^2 r^(3/2)", 3.5355339059327378e-3, float(ss_viscosity(0.1, 0.01, 50.0)),
          rtol=1e-14)  # fmt: skip


def test_simple_shear():
    """v_x = s y: G[0, 1] = s; Sigma_v = rho nu s (e_x e_y + e_y e_x); Lambda = nu s^2."""
    s, rho, nu = 0.3, 2.0, 0.05
    g = np.zeros((3, 3))
    g[0, 1] = s
    want = np.zeros((3, 3))
    want[0, 1] = want[1, 0] = rho * nu * s
    check("simple shear: viscous stress", want, viscous_stress(g, rho, nu), atol=1e-15)
    check("simple shear: heating nu s^2", nu * s**2, float(viscous_heating(g, nu)), rtol=1e-14)


def test_isotropic_compression_has_no_shear_viscosity():
    """G = c I (uniform compression): shear viscosity gives no stress and no heating; bulk
    viscosity gives Sigma_v = 3 rho zeta_v c I and Lambda = zeta_v (3c)^2."""
    c, rho, nu, zv = -0.2, 1.5, 0.07, 0.01
    g = c * np.eye(3)
    check("compression, shear only: stress", np.zeros((3, 3)), viscous_stress(g, rho, nu),
          atol=1e-15)  # fmt: skip
    check("compression, shear only: heating", 0.0, float(viscous_heating(g, nu)), atol=1e-15)
    check("compression, bulk: stress", 3 * rho * zv * c * np.eye(3), viscous_stress(g, rho, nu, zv),
          atol=1e-15)  # fmt: skip
    check("compression, bulk: heating", zv * (3 * c) ** 2, float(viscous_heating(g, nu, zv)),
          rtol=1e-13)  # fmt: skip


def test_heating_positive_and_equals_stress_contraction(rng):
    """Lambda >= 0 for any gradient (shear part), and Lambda = Sigma_v : G / rho (Eq. 133-136)."""
    g = rng.normal(0.0, 1.0, (2000, 3, 3))
    rho, nu = rng.uniform(0.5, 2.0, 2000), rng.uniform(0.01, 0.1, 2000)
    lam = viscous_heating(g, nu)
    check("heating >= 0 for 2,000 random gradients", True, bool(np.all(lam >= 0.0)))
    contraction = np.sum(viscous_stress(g, rho, nu) * g, axis=(-2, -1)) / rho
    check("Lambda = Sigma_v : G / rho", contraction, lam, rtol=1e-12)


def test_viscous_dt():
    check("dt = 0.25 h^2 / nu", 0.25 * 4.0 / 0.5, float(viscous_dt(2.0, 0.5)), rtol=1e-15)
    check("dt = inf where nu = 0", True, bool(np.isinf(viscous_dt(1.0, 0.0))))
