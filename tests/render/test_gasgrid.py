"""Phase 8 step 1: depositing particles onto render grids (phases/phase-08-gas-rendering.md 8.3,
tests 1 and 2 of 8.6, plus the rule-2 comparison of the Numba deposition with the readable
reference)."""

import numba
import numpy as np

from nebula.render.gasgrid import (
    CartesianBox,
    LogSphericalGrid,
    deposit_box,
    deposit_box_reference,
    deposit_log,
    deposit_log_reference,
)
from tests.helpers import check

G = LogSphericalGrid(r_min=2.0, r_max=200.0, nr=24, nth=12, nph=24)
B = CartesianBox(centre=(10.0, -5.0, 3.0), size=40.0, n=16)


def _log_particles(seed=0, n=300):
    """Random particles across the grid, with h from 0.1 to 10 local cells, plus particles
    straddling r_max, sitting on the phi = 0 seam and near both poles."""
    rng = np.random.default_rng(seed)
    r = G.r_min * np.exp(rng.uniform(0.3, G.nr - 0.3, n) * G.dlnr)
    th = np.arccos(rng.uniform(-1, 1, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    ph[:30] = rng.uniform(-0.05, 0.05, 30) % (2 * np.pi)  # on the seam
    th[30:45] = rng.uniform(0, 0.08, 15)  # near the north pole
    th[45:60] = np.pi - rng.uniform(0, 0.08, 15)  # near the south pole
    r[60:80] = G.r_max * rng.uniform(0.8, 1.2, 20)  # straddling / beyond the outer edge
    x = np.c_[r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]
    cell = r * G.dlnr  # local radial cell size
    h = cell * 10 ** rng.uniform(-1, 1, n)
    m = rng.uniform(0.5, 2.0, n)
    t = 10 ** rng.uniform(3.5, 7, n)
    v = rng.normal(0, 0.1, (n, 3))
    return x, h, m, t, v


def _box_particles(seed=1, n=300):
    rng = np.random.default_rng(seed)
    x = np.asarray(B.centre) + rng.uniform(-0.6, 0.6, (n, 3)) * B.size  # some outside the box
    h = B.dx * 10 ** rng.uniform(-1, 1, n)
    m = rng.uniform(0.5, 2.0, n)
    t = 10 ** rng.uniform(3.5, 7, n)
    v = rng.normal(0, 0.1, (n, 3))
    return x, h, m, t, v


def test_log_deposition_mass_ledger():
    """8.6 test 1: grid mass + mass beyond the grid = sum m, to 1e-12 (h 0.1-10 cells, edges,
    phi seam, poles)."""
    x, h, m, t, v = _log_particles()
    gg = deposit_log(x, h, m, t, v, G)
    check(
        "log grid: grid + outside mass / sum m",
        1.0,
        (gg.mass.sum() + gg.outside) / m.sum(),
        rtol=1e-12,
    )
    check("log grid: some particles used trilinear", True, gg.n_trilinear > 0)
    check("log grid: some mass beyond r_max", True, gg.outside > 0)
    r = np.linalg.norm(x, axis=1)
    inner = (r + 2 * h < 0.9 * G.r_max) & (r - 2 * h > 1.1 * G.r_min)  # support inside the grid
    gi = deposit_log(x[inner], h[inner], m[inner], t[inner], v[inner], G)
    check("log grid: inner particles, outside mass", 0.0, gi.outside / m[inner].sum(), atol=1e-12)
    check(
        "log grid: inner particles, grid mass / sum m",
        1.0,
        gi.mass.sum() / m[inner].sum(),
        rtol=1e-12,
    )


def test_box_deposition_mass_ledger():
    """8.6 test 1 for a detail box: box mass + mass outside = sum m, to 1e-12."""
    x, h, m, t, v = _box_particles()
    gb = deposit_box(x, h, m, t, v, B)
    check(
        "box: box + outside mass / sum m", 1.0, (gb.mass.sum() + gb.outside) / m.sum(), rtol=1e-12
    )
    check("box: some particles used trilinear", True, gb.n_trilinear > 0)


def test_log_deposition_matches_reference():
    """Rule 2: the Numba deposition equals the readable reference (all fields)."""
    x, h, m, t, v = _log_particles(seed=2, n=150)
    fast = deposit_log(x, h, m, t, v, G)
    ref = deposit_log_reference(x, h, m, t, v, G)
    check("log: numba vs reference, cell mass", ref.mass, fast.mass, atol=1e-12 * ref.mass.max())
    check("log: numba vs reference, outside", ref.outside, fast.outside, rtol=1e-12)
    check("log: numba vs reference, trilinear count", ref.n_trilinear, fast.n_trilinear)
    for name in ("rho", "t_mass", "t_em", "vel"):
        a, b = getattr(ref, name), getattr(fast, name)
        check(f"log: numba vs reference, {name}", a, b, atol=1e-6 * float(np.abs(a).max()))


def test_box_deposition_matches_reference():
    x, h, m, t, v = _box_particles(seed=3, n=150)
    fast = deposit_box(x, h, m, t, v, B)
    ref = deposit_box_reference(x, h, m, t, v, B)
    check("box: numba vs reference, cell mass", ref.mass, fast.mass, atol=1e-12 * ref.mass.max())
    check("box: numba vs reference, outside", ref.outside, fast.outside, rtol=1e-12)
    for name in ("rho", "t_mass", "t_em", "vel"):
        a, b = getattr(ref, name), getattr(fast, name)
        check(f"box: numba vs reference, {name}", a, b, atol=1e-6 * float(np.abs(a).max()))


def test_deposition_deterministic_across_threads():
    """8.6 test 2: 1 vs 2 threads give identical grids (bit for bit)."""
    x, h, m, t, v = _log_particles(seed=4)
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = deposit_log(x, h, m, t, v, G)
        ab = deposit_box(x, h, m, t, v, B)
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        b = deposit_log(x, h, m, t, v, G)
        bb = deposit_box(x, h, m, t, v, B)
    finally:
        numba.set_num_threads(old)
    check(
        "1 vs 2 threads: log grid identical",
        True,
        bool(np.array_equal(a.mass, b.mass) and np.array_equal(a.t_em, b.t_em)),
    )
    check("1 vs 2 threads: box identical", True, bool(np.array_equal(ab.mass, bb.mass)))


def test_uniform_fields_preserved():
    """Constant T and v in every particle give exactly that T, T_em and v in every filled cell
    (the weighting cannot invent structure)."""
    x, h, m, _, _ = _log_particles(seed=5)
    t = np.full(len(h), 2.5e4)
    v = np.tile([0.01, -0.02, 0.03], (len(h), 1))
    gg = deposit_log(x, h, m, t, v, G)
    filled = gg.mass > 0
    check("uniform T: mass-weighted T", 2.5e4, gg.t_mass[filled], rtol=1e-6)
    check("uniform T: emission-weighted T", 2.5e4, gg.t_em[filled], rtol=1e-6)
    check("uniform v: vx", 0.01, gg.vel[filled][:, 0], rtol=1e-6)


def test_log_grid_volumes_fill_the_shell():
    """Cell volumes add up to the shell volume 4/3 pi (r_max^3 - r_min^3)."""
    check(
        "log grid: sum of cell volumes",
        4 / 3 * np.pi * (G.r_max**3 - G.r_min**3),
        G.volumes().sum() * G.nph,
        rtol=1e-12,
    )
