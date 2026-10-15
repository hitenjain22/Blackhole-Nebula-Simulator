"""Step 2.4: procedural star field and cube map (phases/phase-02-blackhole-render.md Step 2.4)."""

import numpy as np
import pytest

from nebula.render.starfield import (
    LUMA_WEIGHTS,
    StarFieldConfig,
    build_cube_map,
    cube_gpu_bytes,
    generate_stars,
    placeholder_blackbody_rgb,
    sample_cube,
    texel_directions,
    texel_solid_angles,
    upload_cube,
)
from tests.helpers import check

SMALL = StarFieldConfig(n_stars=5000, face_size=256, band=False)


@pytest.fixture(scope="module")
def small_cube():
    return build_cube_map(SMALL)


def test_texel_solid_angles():
    omega = texel_solid_angles(256)
    check("texel solid angles sum to 4 pi (6 faces)", 4 * np.pi, 6 * omega.sum(), rtol=1e-12)
    check("corner texel smaller than centre texel", True, bool(omega[0, 0] < omega[128, 128]))


def test_cube_convention_matches_gpu():
    """Each texel stores its own direction; the GPU sampling it at texel centres (and at random
    directions, including edges and corners) must return that direction. A wrong face order
    or orientation gives errors of order 1."""
    size = 64
    faces = np.stack([texel_directions(f, size) for f in range(6)]).astype(np.float32)
    sky = upload_cube([faces])
    centres = np.concatenate([texel_directions(f, size)[::7, ::5].reshape(-1, 3) for f in range(6)])
    got = sample_cube(sky, centres)[:, :3]
    check("GPU cube sample at texel centres = texel direction", centres, got, atol=1e-3)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(5000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    got = sample_cube(sky, d)[:, :3]
    texel = np.pi / 2 / size
    check("GPU cube sample anywhere ~ direction (seamless)", d, got, atol=texel, per_vector=True)


def test_star_light_conserved(small_cube):
    """Every star's light is deposited exactly: sum(L x solid angle) = total flux (luminance)."""
    omega = texel_solid_angles(SMALL.face_size)
    lum = small_cube.faces.astype(np.float64) @ LUMA_WEIGHTS
    total = np.sum(lum * omega[None]) * small_cube.scale
    check(
        "total starlight in the map = sum of star fluxes",
        small_cube.total_star_flux,
        total,
        rtol=1e-6,
    )
    for k, level in enumerate(small_cube.mips[:4], start=1):
        om = texel_solid_angles(level.shape[1])
        tot = np.sum((level.astype(np.float64) @ LUMA_WEIGHTS) * om[None]) * small_cube.scale
        check(f"mip level {k}: starlight kept", small_cube.total_star_flux, tot, rtol=0.01)


def test_star_counts_slope():
    """N(< m) rises as 10^(0.5 m): fit of log10 N(<m) for m in [0, 7]."""
    stars = generate_stars(StarFieldConfig(n_stars=100_000, band=False))
    m = np.linspace(0, 7, 15)
    counts = np.array([np.sum(stars.magnitude < x) for x in m])
    slope = np.polyfit(m, np.log10(counts), 1)[0]
    check("star counts: d log10 N(<m) / dm", 0.5, slope, atol=0.02)
    check(
        "all magnitudes within [-1, 8]",
        True,
        bool(stars.magnitude.min() >= -1.0 and stars.magnitude.max() <= 8.0),
    )


def test_hotter_stars_are_bluer():
    t = np.array([3000.0, 5000.0, 10000.0, 30000.0])
    rgb = placeholder_blackbody_rgb(t)
    ratio = rgb[:, 2] / rgb[:, 0]
    check("blue/red ratio rises with temperature", True, bool(np.all(np.diff(ratio) > 0)))
    check("unit luminance colours", np.ones(4), rgb @ LUMA_WEIGHTS, rtol=1e-12)


def test_band_lies_on_its_plane():
    cfg = StarFieldConfig(n_stars=2000, face_size=128, band=True)
    cube = build_cube_map(cfg)
    pole = np.asarray(cfg.band_pole) / np.linalg.norm(cfg.band_pole)
    lat = np.concatenate([texel_directions(f, 128) @ pole for f in range(6)]).ravel()
    lum = (cube.faces.reshape(-1, 3).astype(np.float64)) @ LUMA_WEIGHTS
    on, off = np.median(lum[np.abs(lat) < 0.05]), np.median(lum[np.abs(lat) > 0.6])
    check("band: median brightness on plane / far from it > 10", True, bool(on > 10 * off))


def test_deterministic_and_memory():
    a = build_cube_map(StarFieldConfig(n_stars=1000, face_size=64))
    b = build_cube_map(StarFieldConfig(n_stars=1000, face_size=64))
    check("same seed -> identical sky", True, bool(np.array_equal(a.faces, b.faces)))
    check("GPU memory, 2048 per face with mips (MB)", 268.4, cube_gpu_bytes(2048) / 1e6, atol=1.0)
    check("within the plan's ~270 MB", True, bool(cube_gpu_bytes(2048) < 280e6))
