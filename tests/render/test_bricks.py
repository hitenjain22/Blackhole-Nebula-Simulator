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


def _tau_fine(tex, a, b, kappa, n=20000):
    """Optical depth along a -> b by brute force: sample_bricks every 1/n of the segment."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ts = (np.arange(n) + 0.5) / n
    rho = [sample_bricks(tex, a + t * (b - a))[0] for t in ts]
    return kappa * np.sum(rho) * np.linalg.norm(b - a) / n


def test_march_through_bricks_misses_nothing():
    """The readable marcher with bricks: optical depth along straight rays (some crossing
    long stretches of empty bricks before thin gas) equals a brute-force integral of the
    brick field (midpoint steps <= dx / 2 vs 20,000 samples: 1e-3)."""
    from nebula.render.march import GreyBlackbody, march_segment, texture_from_fields

    rng = np.random.default_rng(23)
    x = np.concatenate([rng.normal([60.0, 0, 0], 1.5, (150, 3)),
                        rng.normal([-40.0, 25, 0], 0.6, (60, 3))])  # fmt: skip
    h = np.concatenate([np.full(150, 0.8), np.full(60, 0.35)])
    lat = BrickLattice(dx=0.5)
    tex = texture_from_bricks(deposit_bricks(x, h, 1e-3, np.full(len(h), 2e4), lat), T_REF)
    g = LogSphericalGrid(r_max=500.0)
    empty = texture_from_fields(g, np.zeros(g.shape), np.zeros(g.shape))
    kappa = 30.0
    model = GreyBlackbody(kappa=kappa)
    rays = [([200.0, 3.0, 1.0], [-200.0, -1.0, -0.5]), ([60.0, -150.0, 0.4], [60.5, 150.0, 0.0]),
            ([-40.0, 25.2, -100.0], [-39.8, 24.9, 100.0]), ([100.0, 80.0, 2.0], [-90.0, -10, 0])]
    for a, b in rays:
        _, trans, n = march_segment([empty, tex], model, a, b)
        want = _tau_fine(tex, a, b, kappa)
        check(f"brick march: optical depth, ray {a} ({n} steps)", want, -np.log(trans), rtol=1e-3,
              atol=1e-6)  # fmt: skip


@pytest.fixture(scope="module")
def stream_scene():
    """A thin curved stream (fine particles -> bricks) plus a few big particles (-> log grid)
    near the hole, for the GPU-vs-CPU comparison."""
    from nebula.render.march import GreyBlackbody
    from nebula.render.snapshot_scene import textures_from_particles

    rng = np.random.default_rng(29)
    ang = rng.uniform(-1.2, 1.2, 3000)
    rad = 45.0 + 8.0 * ang**2
    x = np.stack([rad * np.cos(ang), rad * np.sin(ang), rng.normal(0, 0.6, 3000)], axis=1)
    x[:, :2] += rng.normal(0, 0.6, (3000, 2))
    h = np.full(3000, 0.7)
    big = rng.normal([-60.0, 10.0, 0.0], 8.0, (100, 3))
    x = np.concatenate([x, big])
    h = np.concatenate([h, np.full(100, 40.0)])  # support 80 > 4 bricks: log grid
    t = np.concatenate([rng.uniform(1e4, 3e5, 3000), np.full(100, 2e4)])
    g = LogSphericalGrid(r_max=400.0)
    texs = textures_from_particles(x, h, 3e-5, t, np.zeros_like(x), g, bricks=BrickLattice(0.5))
    return texs, GreyBlackbody(kappa=40.0)


def test_scene_splits_particles_and_keeps_mass(stream_scene):
    texs, _ = stream_scene
    check("brick scene: log grid + brick texture", ("Texture", "BrickTexture"),
          tuple(type(t).__name__ for t in texs))  # fmt: skip


SHIFT_PX = 0.02  # GPU ray aim precision allowance, in pixels (see the docstring below)


def test_gpu_matches_cpu_with_bricks(stream_scene):
    """Rule 2 / 8.6 test 7 for bricks: the WGSL brick sampler and step rule against the
    readable reference (bent rays, camera 150 from the hole), on pixels crossing the stream,
    the big log-grid particles and empty sky.

    Where the gas is seen (I > 1e-6 of the frame's peak) each pixel must agree to 1e-3, or lie
    within the CPU values for the camera turned by SHIFT_PX pixels either way (x and y). Why:
    the GPU traces rays in float32 (test_sky_directions_match_exact_rays allows 2e-4 rad, here
    0.008 px), and the bricks resolve the stream's edges finely enough that so small a shift
    changes I there by up to 0.7%; the difference did not shrink with 5x finer steps, and all
    33 lit pixels fit a 0.02 px shift (the one exception, a brightness maximum, agrees to
    1.4e-4) (measured 2026-11-16; AI_LOG). Fainter pixels are the far tails of kernels (I down
    to 1e-16 of the peak): there the check is absolute, |dI| < 1e-9 of the peak."""
    from nebula.render.bent_rays import render_reference
    from nebula.render.camera import Camera
    from nebula.render.gpu_march import render_gas, upload_scene

    texs, model = stream_scene
    base = dict(position=(20.0, -140.0, 45.0), look_at=(0.0, 0.0, 0.0), vfov_deg=50.0,
                width=64, height=36)  # fmt: skip
    cam = Camera(**base)
    gpu, timings = render_gas(cam, upload_scene(texs, model), n_sub=1)
    peak = gpu[..., 0].max()
    rng = np.random.default_rng(31)
    seen = np.argwhere(gpu[..., 0] > 1e-6 * peak)
    faint = np.argwhere((gpu[..., 0] > 0) & (gpu[..., 0] <= 1e-6 * peak))
    dark = np.argwhere(gpu[..., 0] == 0)
    pick = np.concatenate([seen,
                           faint[rng.choice(len(faint), min(15, len(faint)), replace=False)],
                           dark[rng.choice(len(dark), min(10, len(dark)), replace=False)]])
    pixels = [(int(c), int(r)) for r, c in pick]
    cpu = render_reference(cam, texs, model, pixels=pixels)
    rows, cols = pick[:, 0], pick[:, 1]
    g, c = gpu[rows, cols], cpu[rows, cols]
    ns = len(seen)
    seen_px = pixels[:ns]
    dist = np.linalg.norm(np.asarray(base["position"]) - np.asarray(base["look_at"]))
    turn = dist * 2.0 * cam.tan_half_vfov / cam.height * SHIFT_PX
    shifted = []
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        la = np.asarray(base["look_at"]) + turn * (sx * cam.right + sy * cam.up_cam)
        ref = render_reference(Camera(**{**base, "look_at": tuple(la)}), texs, model,
                               pixels=seen_px)  # fmt: skip
        shifted.append(ref[rows[:ns], cols[:ns], 0])
    shifted = np.array(shifted)
    gi, ci = g[:ns, 0], c[:ns, 0]
    close = np.abs(gi - ci) <= 1e-3 * ci
    bracket = (gi >= shifted.min(0) * (1 - 1e-3)) & (gi <= shifted.max(0) * (1 + 1e-3))
    rel = np.abs(gi - ci) / ci
    check(f"bricks GPU vs CPU: {ns} seen pixels within 1e-3 or a {SHIFT_PX} px ray shift "
          f"(within 1e-3: {close.sum()}; largest difference {rel.max():.1e})", ns,
          int((close | bracket).sum()))  # fmt: skip
    check("bricks GPU vs CPU: |dI| / peak, faint and dark pixels", np.zeros(len(pick) - ns),
          np.abs(g[ns:, 0] - c[ns:, 0]) / peak, atol=1e-9)  # fmt: skip
    check("bricks GPU vs CPU: transmission", c[:, 1], g[:, 1], atol=1e-3)
    check("bricks GPU vs CPU: mean steps per ray within 2%", c[:, 3].mean(), g[:, 3].mean(),
          rtol=0.02)  # fmt: skip
    check("bricks: slowest GPU tile < 1 s", True, timings.max < 1.0)
