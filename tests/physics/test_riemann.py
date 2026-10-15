"""Exact Riemann solver vs expected-values §4 (Sod, gamma = 1.4, 4/3, 5/3)."""

import numpy as np
import pytest

from nebula.physics.riemann import sample, solve_riemann
from tests.helpers import check

# §4: (p*, u*, rho*L, rho*R, shock speed, rarefaction head, tail), printed to 6 decimals.
TABLE = {
    1.4: (0.303130, 0.927453, 0.426319, 0.265574, 1.752156, -1.183216, -0.070273),
    4 / 3: (0.306011, 0.953205, 0.411437, 0.278585, 1.728999, -1.154701, -0.042628),
    5 / 3: (0.293945, 0.841195, 0.479689, 0.229806, 1.844473, -1.290994, -0.169401),
}


@pytest.mark.parametrize("gamma", list(TABLE), ids=["1.4", "4/3", "5/3"])
def test_sod_star_state(gamma):
    sol = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma)
    got = (
        sol.p_star,
        sol.u_star,
        sol.rho_star_l,
        sol.rho_star_r,
        sol.s_r_head,
        sol.s_l_head,
        sol.s_l_tail,
    )
    # printed to 6 decimals: rounding 5e-7, tolerance 10x that
    check(
        f"§4 Sod gamma={gamma:.4g}: p*, u*, rho*L, rho*R, shock, head, tail",
        np.array(TABLE[gamma]),
        np.array(got),
        atol=5e-6,
    )


def test_sod_profile_at_t02():
    """§4: at t = 0.2 (diaphragm at 0): head -0.23664, tail -0.01405, contact 0.18549, shock
    0.35043; specific internal energy 1.777600 left of the contact, 2.853541 right of it."""
    sol = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    t = 0.2
    expected = np.array([-0.23664, -0.01405, 0.18549, 0.35043])
    got = np.array([sol.s_l_head, sol.s_l_tail, sol.u_star, sol.s_r_head]) * t
    check("§4 t=0.2 positions: head, tail, contact, shock", expected, got, atol=5e-5)
    x = np.array([0.1, 0.25])
    rho, _, p = sample(sol, x, t)
    check("§4 u left/right of contact", np.array([1.777600, 2.853541]), p / (0.4 * rho), atol=5e-6)
    rho, v, p = sample(sol, np.array([-0.5, 0.5, -0.12]), t)
    check("undisturbed states far left / far right", (1.0, 0.125), (rho[0], rho[1]))
    # inside the fan: isentropic, P / rho^gamma = left value
    check("rarefaction fan is isentropic", 1.0, p[2] / rho[2] ** 1.4, rtol=1e-12)
