"""Sparse brick grid (DECISIONS 2026-11-16): deposition, sampling and the step rule."""

import numpy as np
import pytest

from nebula.render.bricks import (
    BRICK,
    BrickLattice,
    deposit_bricks,
    deposit_bricks_reference,
    exit_distance,
    fine_particles,
    home_allocated,
    home_brick,
    sample_bricks,
    texture_from_bricks,
    unpack_keys,
)
from nebula.render.gasgrid import CartesianBox, LogSphericalGrid, deposit_box_reference
from nebula.render.march import T_REF, sample_box, texture_from_gasgrid
from tests.helpers import check

DX = 0.5
BOX = CartesianBox(centre=(0.0, 0.0, 0.0), size=32.0, n=64)  # lower corner -16 = -32 dx


def particles(n, seed=3, spread=10.0):
    """Random particles well inside BOX, h from 0.1 (trilinear fallback) to 1.5."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-spread, spread, (n, 3))
    h = rng.uniform(0.1, 1.5, n)
    m = rng.uniform(0.5, 2.0, n)
    t = rng.uniform(5e3, 5e5, n)
    return x, h, m, t


def as_box(bg):
    """Brick cell masses placed into BOX's (n, n, n) array (cells outside the box: error)."""
    out = np.zeros(BOX.shape)
    off = int(round(-BOX.lo[0] / DX))
    bi, bj, bk = unpack_keys(bg.keys)
    for s in range(len(bg.keys)):
        a, b, c = (int(v) * BRICK + off for v in (bi[s], bj[s], bk[s]))
        block = bg.mass[s]
        if block.any():
            out[a:a + BRICK, b:b + BRICK, c:c + BRICK] += block  # fmt: skip
    return out


def test_reference_equals_aligned_box():
    """The brick lattice is gasgrid's deposition on a fixed lattice: on a box aligned with it
    and holding every support, deposit_box_reference gives the same cell masses."""
    x, h, m, t = particles(40)
    lat = BrickLattice(dx=DX)
    bg = deposit_bricks_reference(x, h, m, t, lat)
    gg = deposit_box_reference(x, h, m, t, np.zeros_like(x), BOX)
    check("bricks = aligned box: cell masses", gg.mass, as_box(bg), rtol=1e-12, atol=1e-15)
    check("bricks = aligned box: trilinear-fallback particles", gg.n_trilinear, bg.n_trilinear)
    check("box held everything (outside mass)", 0.0, gg.outside, atol=1e-15)


def test_fast_matches_reference():
    """Rule 2: the Numba deposition matches the readable one (same keys, same cell values)."""
    x, h, m, t = particles(300, seed=5)
    lat = BrickLattice(dx=DX)
    ref = deposit_bricks_reference(x, h, m, t, lat)
    fast = deposit_bricks(x, h, m, t, lat)
    check("fast bricks: same allocated bricks", ref.keys, fast.keys, rtol=0.0, atol=0.0)
    check("fast bricks: cell masses", ref.mass, fast.mass, rtol=1e-12, atol=1e-15)
    check("fast bricks: emission temperature", ref.t_em, fast.t_em, rtol=1e-6, atol=1e-3)
    check("fast bricks: trilinear-fallback count", ref.n_trilinear, fast.n_trilinear)


def test_mass_conserved_and_deterministic():
    """Every particle's mass lands on the bricks exactly (no outside: the lattice has no
    edge); two runs give identical grids."""
    x, h, m, t = particles(20000, seed=8, spread=60.0)
    lat = BrickLattice(dx=1.0)
    a = deposit_bricks(x, h, m, t, lat)
    b = deposit_bricks(x, h, m, t, lat)
    check("bricks: total mass = particle mass", m.sum(), a.mass.sum(), rtol=1e-12)
    check("bricks: deterministic", a.mass, b.mass, rtol=0.0, atol=0.0)


def test_sampling_matches_box_sampling():
    """sample_bricks is trilinear between the same cell values as march.sample_box (points
    kept 1 cell inside the box, where the box's face clamping does not act)."""
    x, h, m, t = particles(40)
    lat = BrickLattice(dx=DX)
    tex = texture_from_bricks(deposit_bricks_reference(x, h, m, t, lat), T_REF)
    btex = texture_from_gasgrid(deposit_box_reference(x, h, m, t, np.zeros_like(x), BOX))
    rng = np.random.default_rng(11)
    pts = np.concatenate([x[:20] + rng.normal(0, 0.7, (20, 3)), rng.uniform(-15, 15, (20, 3))])
    keys = tex.keys.tolist()
    got = np.array([sample_bricks(tex, p, keys) for p in pts])
    want = np.array([sample_box(btex, p) for p in pts])
    check("brick samples = box samples (rho, rho T^4)", want, got, rtol=1e-6, atol=1e-12)


def test_nonzero_sample_has_allocated_home_brick():
    """The step rule's promise: wherever the brick sample is non-zero, the home cell's brick is
    allocated (so the marcher takes fine steps there)."""
    x, h, m, t = particles(60, seed=13)
    lat = BrickLattice(dx=DX)
    tex = texture_from_bricks(deposit_bricks(x, h, m, t, lat), T_REF)
    keys = tex.keys.tolist()
    rng = np.random.default_rng(17)
    pts = x[rng.integers(0, len(x), 4000)] + rng.normal(0, 2.0, (4000, 3))
    bad = sum(1 for p in pts if sample_bricks(tex, p, keys)[0] > 0 and
              not home_allocated(tex, p, keys))  # fmt: skip
    n_gas = sum(1 for p in pts if sample_bricks(tex, p, keys)[0] > 0)
    check(f"non-zero samples without an allocated home brick (of {n_gas})", 0, bad)


def test_exit_distance_leaves_home_brick():
    """exit_distance ends a step just past the home brick's far face: not before it."""
    rng = np.random.default_rng(19)
    ok = 0
    for _ in range(500):
        p = rng.uniform(-40, 40, 3)
        e = rng.normal(size=3)
        e /= np.linalg.norm(e)
        d = exit_distance(p, e, DX)
        same_before = np.array_equal(home_brick(p + (d - 2e-4 * DX) * e, DX), home_brick(p, DX))
        moved_after = not np.array_equal(home_brick(p + d * e, DX), home_brick(p, DX))
        ok += same_before and moved_after and d <= np.sqrt(3) * BRICK * DX + 1e-3
    check("exit distance: inside before, next brick after, <= brick diagonal (500 rays)", 500,
          ok)  # fmt: skip


@pytest.mark.parametrize(
    "x, h, want",
    [
        ([100.0, 0, 0], 1.0, True),
        ([100.0, 0, 0], 20.0, False),  # support 40 > 4 bricks of 8 cells
        ([5.0, 0, 0], 0.5, False),  # inside r_in: the log grid is finer there
        ([3000.0, 0, 0], 1.0, False),  # beyond r_fine
    ],
)
def test_fine_particle_selection(x, h, want):
    lat = BrickLattice(dx=1.0, r_fine=1000.0)
    g = LogSphericalGrid()
    got = bool(fine_particles(np.array([x]), np.array([h]), lat, g)[0])
    check(f"fine particle at r {np.linalg.norm(x):g}, h {h:g} (r_in {lat.r_in(g):.1f})", want,
          got)  # fmt: skip
