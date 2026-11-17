"""Gas 4-velocity and redshift factor (phase-09 9.2-9.3; tests 9 and 10 of 9.7).
Expected values: reference/expected-values.md §6."""

import numpy as np

from nebula.physics.gravity import gn_hamiltonian
from nebula.physics.redshift import (
    circular_orbit_redshift,
    gn_static_frame_velocity,
    redshift_factor,
)
from tests.helpers import check

# §6: r -> (beta, gamma, g approaching, receding, transverse, static), camera at infinity
TABLE = {
    6.0: (0.50000000, 1.15470054, 1.41421356, 0.47140452, 0.70710678, 0.81649658),
    10.0: (0.35355339, 1.06904497, 1.29424459, 0.61812119, 0.83666003, 0.89442719),
    50.8896237: (0.14301831, 1.01038673, 1.13196905, 0.84869746, 0.97007674, 0.98015267),
}


def circular_state(r):
    """GN state on the circular orbit of radius r in the x-y plane, moving along +y at (r, 0, 0):
    p_r = 0, |p_perp| = l / r with l = r / sqrt(r - 3) (phase-09 9.3)."""
    return np.array([r, 0.0, 0.0]), np.array([0.0, (r / np.sqrt(r - 3.0)) / r, 0.0])


def test_9_four_velocity_identity():
    rng = np.random.default_rng(16)
    n = 1000
    r = np.exp(rng.uniform(np.log(2.5), np.log(1e4), n))
    d = rng.normal(size=(n, 3))
    x = d / np.linalg.norm(d, axis=1)[:, None] * r[:, None]
    p = rng.normal(size=(n, 3)) * np.sqrt(2.0 / r)[:, None] * rng.uniform(0.0, 3.0, (n, 1))
    gamma, gb = gn_static_frame_velocity(x, p)
    viol = np.abs(gamma**2 - np.sum(gb * gb, axis=1) - 1.0) / gamma**2
    check("gamma^2 - |gamma beta|^2 = 1, 1,000 GN states (max rel)", 0.0, float(viol.max()),
          atol=1e-12)  # fmt: skip
    # the identity behind it (phase-09 9.2): 1 + 2H = (1-2/r)(1 + l^2/r^2) + (1-2/r)^2 p_r^2
    rh = x / r[:, None]
    pr = np.sum(p * rh, axis=1)
    l2 = np.sum(np.cross(x, p) ** 2, axis=1)
    lhs = 1.0 + 2.0 * gn_hamiltonian(x, p)
    rhs = (1 - 2 / r) * (1 + l2 / r**2) + (1 - 2 / r) ** 2 * pr**2
    check("1 + 2H identity, 1,000 states (max rel)", 0.0, float(np.max(np.abs(lhs / rhs - 1))),
          atol=1e-12)  # fmt: skip


def test_10_redshift_table():
    for r, (beta, gam, g_app, g_rec, g_tr, g_st) in TABLE.items():
        x, p = circular_state(r)
        gamma, gb = gn_static_frame_velocity(x, p)
        check(f"r = {r:g}: gamma (§6)", gam, float(gamma), atol=1e-8)
        check(f"r = {r:g}: beta (§6)", beta, float(np.linalg.norm(gb) / gamma), atol=1e-8)
        v = gb / np.linalg.norm(gb)
        for name, n_hat, want in (("approaching", v, g_app), ("receding", -v, g_rec),
                                  ("transverse", np.array([1.0, 0, 0]), g_tr)):  # fmt: skip
            check(f"r = {r:g}: g {name} (§6)", want, float(redshift_factor(r, gb, n_hat)),
                  atol=1e-8)  # fmt: skip
        check(f"r = {r:g}: g static emitter (§6)", g_st,
              float(redshift_factor(r, np.zeros(3), v)), atol=1e-8)  # fmt: skip
        # cross-check form: b = +-r/sqrt(1-2/r) along/against the motion, b = 0 transverse
        b = r / np.sqrt(1 - 2 / r)
        for name, n_hat, bb in (("approaching", v, b), ("receding", -v, -b),
                                ("transverse", np.array([1.0, 0, 0]), 0.0)):  # fmt: skip
            check(f"r = {r:g}: g {name} vs 1/(u^t(1 - Omega b))",
                  float(circular_orbit_redshift(r, bb)), float(redshift_factor(r, gb, n_hat)),
                  rtol=1e-12)  # fmt: skip
    check("camera at r = 50, static emitter at r = 10 (§6)", 0.91287093,
          float(redshift_factor(10.0, np.zeros(3), np.array([1.0, 0, 0]), r_cam=50.0)),
          atol=1e-8)  # fmt: skip
