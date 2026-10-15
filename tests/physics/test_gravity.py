"""Step 1.2: GN gravity (phases/phase-01-orbits.md, tests 1-6; expected-values §2 circular orbits).

Vector results are compared per vector (error length / vector length): helpers.check.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from nebula.physics.gravity import (
    gn_accel,
    gn_angmom,
    gn_dH_dp,
    gn_dH_dx,
    gn_hamiltonian,
    gn_lagrangian,
    gn_mass_matrix_inv,
    gn_p_from_v,
    gn_v_from_p,
)
from tests.helpers import check


def random_states(rng, n, r_min=4.0, r_max=1e4):
    """n positions with r log-uniform in [r_min, r_max], random directions; velocities in
    random directions with speed 0.1-1.2 times the local escape speed sqrt(2/r)."""
    r = np.exp(rng.uniform(np.log(r_min), np.log(r_max), n))
    nx = rng.normal(size=(n, 3))
    x = r[:, None] * nx / np.linalg.norm(nx, axis=1, keepdims=True)
    nv = rng.normal(size=(n, 3))
    speed = rng.uniform(0.1, 1.2, n) * np.sqrt(2.0 / r)
    v = speed[:, None] * nv / np.linalg.norm(nv, axis=1, keepdims=True)
    return x, v


def lagrangian_paper(x, v):
    """Tejeda & Rosswog 2013 Eq. (A.4), written independently of the project code:
    L = 1/2 [ (x.v)^2/(r-2)^2 + |x cross v|^2/(r(r-2)) ] + 1/r."""
    r = np.linalg.norm(x)
    return 0.5 * ((x @ v) ** 2 / (r - 2) ** 2 + np.sum(np.cross(x, v) ** 2) / (r * (r - 2))) + 1 / r


def euler_lagrange_accel_fd(x, v):
    """Acceleration from the Euler-Lagrange equations by finite differences of the Lagrangian
    alone: M a = dL/dx - J v, with M = d2L/dv dv and J = d2L/dv dx (central differences)."""
    hx = 1e-4 * np.linalg.norm(x)
    hv = 0.5 * np.linalg.norm(v)  # L is exactly quadratic in v: central differences are exact
    e = np.eye(3)
    lag = lagrangian_paper
    dldx = np.array([(lag(x + hx * e[i], v) - lag(x - hx * e[i], v)) / (2 * hx) for i in range(3)])
    m = np.empty((3, 3))
    j = np.empty((3, 3))
    for i in range(3):
        for k in range(3):
            m[i, k] = (
                lag(x, v + hv * e[i] + hv * e[k])
                - lag(x, v + hv * e[i] - hv * e[k])
                - lag(x, v - hv * e[i] + hv * e[k])
                + lag(x, v - hv * e[i] - hv * e[k])
            ) / (4 * hv * hv)
            j[i, k] = (
                lag(x + hx * e[k], v + hv * e[i])
                - lag(x - hx * e[k], v + hv * e[i])
                - lag(x + hx * e[k], v - hv * e[i])
                + lag(x - hx * e[k], v - hv * e[i])
            ) / (4 * hx * hv)
    return np.linalg.solve(m, dldx - j @ v)


def circular_state(r, rotation=None):
    """GN circular orbit at radius r: tangential speed v = (r - 2)/(r sqrt(r - 3)) (§2)."""
    vt = (r - 2.0) / (r * np.sqrt(r - 3.0))
    rotation = np.eye(3) if rotation is None else rotation
    return rotation @ np.array([r, 0.0, 0.0]), rotation @ np.array([0.0, vt, 0.0])


def random_rotation(rng):
    q, rr = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(rr))


def test_round_trip_v_p_v(rng):
    """Test 1: v -> p -> v to 1e-14 at random points, r from 4 to 1e4."""
    x, v = random_states(rng, 1000)
    check(
        "v -> p -> v round trip (1,000 points)",
        v,
        gn_v_from_p(x, gn_p_from_v(x, v)),
        rtol=1e-14,
        per_vector=True,
    )
    check(
        "Minv p = gn_v_from_p",
        gn_v_from_p(x, v),
        (gn_mass_matrix_inv(x) @ v[..., None])[..., 0],
        rtol=1e-14,
        per_vector=True,
    )


def test_lagrangian_matches_paper(rng):
    x, v = random_states(rng, 50)
    paper = np.array([lagrangian_paper(a, b) for a, b in zip(x, v, strict=True)])
    check("gn_lagrangian = paper Eq. A.4 (50 points)", paper, gn_lagrangian(x, v), rtol=1e-13)


def test_accel_vs_euler_lagrange_finite_differences(rng):
    """Test 2 (independent derivation): finite-differenced Euler-Lagrange acceleration from
    the paper's Lagrangian matches the closed form gn_accel at 20 random points to 1e-6."""
    x, v = random_states(rng, 20)
    fd = np.array([euler_lagrange_accel_fd(a, b) for a, b in zip(x, v, strict=True)])
    check(
        "gn_accel vs Euler-Lagrange FD (20 points)", fd, gn_accel(x, v), rtol=1e-6, per_vector=True
    )


def test_hamilton_equations_reproduce_accel(rng):
    """Test 3: integrate Hamilton's equations (dH/dp, -dH/dx) a short time forward and back;
    the central difference of v = dH/dp matches gn_accel to 1e-6."""
    x, v = random_states(rng, 20)
    fd = []
    for x0, v0 in zip(x, v, strict=True):
        r = np.linalg.norm(x0)
        delta = 1e-4 * min(r**1.5, r / np.linalg.norm(v0))

        def rhs(_t, s):
            return np.concatenate([gn_dH_dp(s[:3], s[3:]), -gn_dH_dx(s[:3], s[3:])])

        s0 = np.concatenate([x0, gn_p_from_v(x0, v0)])
        ends = []
        for sign in (1.0, -1.0):
            sol = solve_ivp(rhs, (0, sign * delta), s0, method="DOP853", rtol=1e-13, atol=1e-30)
            ends.append(gn_v_from_p(sol.y[:3, -1], sol.y[3:, -1]))
        fd.append((ends[0] - ends[1]) / (2 * delta))
    check(
        "Hamilton's equations vs gn_accel (20 points)",
        np.array(fd),
        gn_accel(x, v),
        rtol=1e-6,
        per_vector=True,
    )


# §2 circular orbits: r -> (v, H, l^2), printed to 11-12 significant figures.
CIRCULAR_TABLE = {
    6.0: (0.38490017946, -0.0555555555556, 12.0),
    10.0: (0.30237157841, -0.0428571428571, 14.2857142857),
    50.0: (0.14003039184, -0.00978723404255, 53.1914893617),
    1000.0: (0.031606977125, -4.99498495487e-4, 1003.00902708),
}


def test_circular_orbits(rng):
    """Test 4: circular orbits at r = 6, 10, 50, 1000, in a randomly oriented plane."""
    rot = random_rotation(rng)
    for r, (v_tab, h_tab, l2_tab) in CIRCULAR_TABLE.items():
        x, v = circular_state(r, rot)
        a = gn_accel(x, v)
        rhat = x / r
        a_radial = a @ rhat
        speed = np.linalg.norm(v)
        p = gn_p_from_v(x, v)
        check(f"r={r:g}: centripetal accel -v^2/r", -(speed**2) / r, a_radial, rtol=1e-12)
        check(
            f"r={r:g}: tangential accel / |a|",
            0.0,
            np.linalg.norm(a - a_radial * rhat) / np.linalg.norm(a),
            atol=1e-12,
        )
        check(
            f"r={r:g}: H = (4-r)/(2r(r-3))",
            (4 - r) / (2 * r * (r - 3)),
            gn_hamiltonian(x, p),
            rtol=1e-13,
        )
        check(
            f"r={r:g}: |x cross p|^2 = r^2/(r-3)",
            r * r / (r - 3),
            np.sum(gn_angmom(x, p) ** 2),
            rtol=1e-13,
        )
        check(f"r={r:g}: §2 table v", v_tab, speed, rtol=1e-9)
        check(f"r={r:g}: §2 table H", h_tab, gn_hamiltonian(x, p), rtol=1e-9)
        check(f"r={r:g}: §2 table l^2", l2_tab, np.sum(gn_angmom(x, p) ** 2), rtol=1e-9)


def test_isco_at_6():
    """Test 5: the circular-orbit angular momentum L^2(r), computed with the GN functions, is
    smallest at r = 6. Root of its central-difference derivative (step 1e-4: the root moves
    by ~h^2/6 = 2e-9, well inside 1e-8 relative)."""

    def l2(r):
        x, v = circular_state(r)
        return np.sum(gn_angmom(x, gn_p_from_v(x, v)) ** 2)

    h = 1e-4
    r_min = brentq(lambda r: (l2(r + h) - l2(r - h)) / (2 * h), 5.0, 7.0, xtol=1e-14)
    check("ISCO: minimum of L^2(r)", 6.0, r_min, rtol=1e-8)


def test_far_field_newton(rng):
    """Test 6: at r = 1e6 the acceleration is Newton's -x/r^3 to 1e-5."""
    r = 1e6
    rot = random_rotation(rng)
    x, v = circular_state(r, rot)
    v = rot @ (np.array([0.3, 1.0, 0.2]) * np.linalg.norm(v))  # not exactly circular
    check("far field r=1e6: GN vs Newton", -x / r**3, gn_accel(x, v), rtol=1e-5, per_vector=True)


def test_vectorized_equals_scalar(rng):
    """Many-particle calls give exactly the single-particle results."""
    x, v = random_states(rng, 50)
    p = gn_p_from_v(x, v)
    for name, fn, args in (
        ("gn_accel", gn_accel, (x, v)),
        ("gn_dH_dx", gn_dH_dx, (x, p)),
        ("gn_hamiltonian", gn_hamiltonian, (x, p)),
    ):
        one_by_one = np.array([fn(a, b) for a, b in zip(*args, strict=True)])
        check(
            f"{name}: (N,3) call = per-particle calls",
            True,
            bool(np.array_equal(one_by_one, fn(*args))),
        )
