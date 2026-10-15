"""Step 3.3: density and smoothing length (phases/phase-03-sph-core.md 3.2 test 2; §4)."""

import numba
import numpy as np
import pytest

from nebula.physics.kernels import CUBIC, WENDLAND_C4
from nebula.sim.ic import fcc_lattice, smoothing_length_uniform
from nebula.sim.kernels_nb import kernel_df, kernel_f
from nebula.sim.neighbours import find_neighbours, minimum_image
from nebula.sim.sph import (
    FLAG_AT_H_MAX,
    FLAG_OK,
    compute_density,
    density_numba,
    density_reference,
)
from tests.helpers import check


def blob(rng, n=2000):
    """Uneven test set: a Gaussian clump inside a sparse uniform background (open box)."""
    core = rng.normal(0.0, 0.15, size=(n // 2, 3))
    bg = rng.uniform(-1.0, 1.0, size=(n - n // 2, 3))
    pos = np.concatenate([core, bg])
    mass = np.full(n, 1.0 / n)
    return pos, mass


def test_numba_kernel_copies():
    q = np.linspace(0, 2.2, 2001)
    for kid, k in ((0, CUBIC), (1, WENDLAND_C4)):
        f = np.array([kernel_f(x, kid) for x in q])
        df = np.array([kernel_df(x, kid) for x in q])
        check(f"Numba kernel f = physics ({k.name})", k.f(q), f, atol=1e-15)
        check(f"Numba kernel f' = physics ({k.name})", k.df(q), df, atol=1e-15)


@pytest.mark.parametrize("kernel", ["cubic", "wendland_c4"])
def test_lattice_density(kernel):
    """Plan test 2: particles on a close-packed lattice, periodic box, density 1: computed
    rho within 0.5%, h consistent with rho, neighbours ~ the kernel's expected number."""
    pos = fcc_lattice(10)
    n = len(pos)
    k = CUBIC if kernel == "cubic" else WENDLAND_C4
    h0 = np.full(n, smoothing_length_uniform(n, 1.0, k.hfact) * 1.1)
    res = compute_density(pos, np.full(n, 1.0 / n), h0, box=1.0, kernel_name=kernel, eta=k.hfact)
    check(f"{kernel}: lattice density = 1 (0.5%)", np.ones(n), res.rho, rtol=5e-3)
    check(f"{kernel}: all converged", n, int(np.sum(res.flags == FLAG_OK)))
    check(
        f"{kernel}: h = eta (m/rho)^(1/3)",
        k.hfact * (1.0 / n / res.rho) ** (1 / 3),
        res.h,
        rtol=1e-4,
    )
    nb = res.neighbours
    inside = [
        int(
            np.sum(
                np.linalg.norm(minimum_image(pos[nb.of(i)] - pos[i], nb.box), axis=1) < 2 * res.h[i]
            )
        )
        for i in range(0, n, 97)
    ]
    # A lattice puts neighbours in discrete shells, so the count can only be near the
    # continuum value (4 pi/3)(2 eta)^3; 10% allows one shell of difference.
    check(
        f"{kernel}: neighbours within 2h vs (4pi/3)(2 eta)^3",
        k.expected_neighbours(),
        np.mean(inside),
        rtol=0.1,
    )


def test_numba_matches_reference(rng):
    """Same algorithm, two implementations: agree to 1e-12 on an uneven set (tight tol)."""
    pos, mass = blob(rng)
    h0 = np.full(len(pos), 0.05)
    first = compute_density(pos, mass, h0, h_max=0.5)
    h_search = first.h * 1.25  # lists must be built for the radius the passes are given
    nb = find_neighbours(pos, h_search)
    ref = density_reference(pos, mass, first.h, nb, h_search, tol=1e-12, h_max=0.5)
    fast = density_numba(pos, mass, first.h, nb, h_search, tol=1e-12, h_max=0.5)
    check("Numba vs reference: h", ref.h, fast.h, rtol=1e-12)
    check("Numba vs reference: rho", ref.rho, fast.rho, rtol=1e-12)
    check("Numba vs reference: Omega", ref.omega, fast.omega, rtol=1e-12)
    check("Numba vs reference: flags", ref.flags, fast.flags)


def test_threads_identical(rng):
    pos, mass = blob(rng)
    h0 = np.full(len(pos), 0.05)
    numba.set_num_threads(1)
    one = compute_density(pos, mass, h0, h_max=0.5)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    many = compute_density(pos, mass, h0, h_max=0.5)
    check("1 thread vs all threads: identical rho", True, bool(np.array_equal(one.rho, many.rho)))


def test_omega_matches_finite_difference(rng):
    """Omega = 1 - (dh/drho) d rho_sum/dh with dh/drho = -h/(3 rho): check d rho_sum/dh by
    central differences at the converged h, 50 particles of the uneven set."""
    pos, mass = blob(rng)
    res = compute_density(pos, mass, np.full(len(pos), 0.05), h_max=0.5, tol=1e-12)
    nb = find_neighbours(pos, np.full(len(pos), 0.6))
    for name, idx in (("clump", range(0, 50)), ("background", range(1000, 1050))):
        fd = []
        for i in idx:
            r = np.linalg.norm(pos[nb.of(i)] - pos[i], axis=1)
            h, e = res.h[i], 1e-6 * res.h[i]
            drho = (
                np.sum(mass[nb.of(i)] * CUBIC.w(r, h + e))
                - np.sum(mass[nb.of(i)] * CUBIC.w(r, h - e))
            ) / (2 * e)
            fd.append(1 + h / (3 * res.rho[i]) * drho)
        check(
            f"Omega vs finite differences ({name})", np.array(fd), res.omega[list(idx)], rtol=1e-6
        )


def test_edges_and_isolated_particle():
    """A half-filled box (sharp edge to vacuum) converges everywhere; a lone particle far away
    has no root below h_max: capped and flagged, never NaN."""
    pos = fcc_lattice(8)
    pos = pos[pos[:, 0] < 0.5]  # vacuum for x > 0.5
    pos = np.concatenate([pos, [[5.0, 5.0, 5.0]]])
    n = len(pos)
    h0 = np.full(n, smoothing_length_uniform(4 * 8**3, 1.0))
    res = compute_density(pos, np.full(n, 1 / (4 * 8**3)), h0, h_max=0.4)
    check("edge particles: all converged", n - 1, int(np.sum(res.flags[:-1] == FLAG_OK)))
    # Open box: every face is a vacuum edge. Compare a particle at the x = 0.5 face (mid y, z)
    # with one at the slab's centre: fewer neighbours at the edge -> lower density, larger h.
    lattice = pos[:-1]
    edge = int(np.argmin(np.linalg.norm(lattice - [0.5, 0.5, 0.5], axis=1)))
    centre = int(np.argmin(np.linalg.norm(lattice - [0.25, 0.5, 0.5], axis=1)))
    check(
        "edge particle: lower density than centre",
        True,
        bool(res.rho[edge] < 0.8 * res.rho[centre]),
    )
    check("edge particle: larger h than centre", True, bool(res.h[edge] > res.h[centre]))
    check("isolated particle: flagged at h_max", FLAG_AT_H_MAX, int(res.flags[-1]))
    check("isolated particle: h = h_max", 0.4, res.h[-1], rtol=1e-12)
    check("no NaN anywhere", False, bool(np.any(np.isnan(res.rho) | np.isnan(res.omega))))
