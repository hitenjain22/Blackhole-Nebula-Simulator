"""Exact Sedov solution (expected-values §4; verified value from reference/scripts/22)."""

import numpy as np

from nebula.physics.sedov import sedov_solution
from tests.helpers import check


def test_sedov_constant():
    check(
        "§4 Sedov xi0, gamma = 5/3 (verified, script 22)",
        1.151666,
        sedov_solution(5 / 3).xi0,
        rtol=5e-6,
    )
    check(
        "Sedov xi0, gamma = 1.4 (classic 1.033; script 22: 1.032777)",
        1.032777,
        sedov_solution(1.4).xi0,
        rtol=5e-6,
    )


def test_sedov_profile_limits():
    s = sedov_solution(5 / 3)
    rho, v, p = s.profile([1.0, 2.0])
    check(
        "just behind the shock: density 4 rho0 (strong shock, gamma 5/3)", 4.0, rho[0], rtol=1e-12
    )
    check("ahead of the shock: undisturbed", (1.0, 0.0, 0.0), (rho[1], v[1], p[1]))
    rho_c, *_ = s.profile([0.05])
    check("near the centre the gas is evacuated (rho < 1e-3 rho0)", True, bool(rho_c[0] < 1e-3))
    # mass inside the blast equals the swept-up mass (4 pi/3 rho0 R^3)
    xi = np.linspace(1e-6, 1, 200_001)
    psi, _, _ = s.profile(xi)
    mass = np.trapezoid(4 * np.pi * xi**2 * psi, xi)
    check("swept-up mass conserved: int 4 pi xi^2 psi = 4 pi / 3", 4 * np.pi / 3, mass, rtol=1e-6)
