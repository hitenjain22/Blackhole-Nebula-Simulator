"""Phase 8 step 2: CPU reference ray march (phases/phase-08-gas-rendering.md 8.4; tests 3-5 of
8.6, plus the transfer step, sampling and box blending they rest on)."""

import numpy as np
import pytest

from nebula.physics.radiative_transfer import (
    blackbody_intensity_si,
    grey_absorption,
    transfer_step,
)
from nebula.render.gas_scenes import box_centres, sphere_volume_fraction
from nebula.render.gasgrid import CartesianBox, LogSphericalGrid
from nebula.render.march import (
    BLEND_CELLS,
    GreyBlackbody,
    march_polyline,
    march_segment,
    sample,
    sample_box,
    sample_log,
    temperature_from_sample,
    texture_from_fields,
)
from tests.helpers import check

BOX = CartesianBox(centre=(0.0, 0.0, 0.0), size=100.0, n=128)
R_SPHERE = 40.0  # 51 cells
RHO0, T0 = 1e-8, 1e4
S0 = float(blackbody_intensity_si(T0))


@pytest.fixture(scope="module")
def sphere_fraction():
    return sphere_volume_fraction(BOX, R_SPHERE)


def test_transfer_step_matches_exact_slab():
    """physics.transfer_step: one step and 1000 sub-steps through a uniform slab both give
    S (1 - e^-tau) (Rybicki & Lightman Eq. 1.30), and the thin limit gives j ds."""
    for tau in (1e-8, 0.1, 1.0, 10.0):
        i1, t1 = transfer_step(0.0, 1.0, 3.0, tau)
        i, t = 0.0, 1.0
        for _ in range(1000):
            i, t = transfer_step(i, t, 3.0, tau / 1000)
        check(f"transfer step, tau {tau:g}: one step", 3.0 * -np.expm1(-tau), i1, rtol=1e-14)
        check(f"transfer step, tau {tau:g}: 1000 steps", float(i1), i, rtol=1e-12)
        check(f"transfer step, tau {tau:g}: transmission", np.exp(-tau), t, rtol=1e-12)
    check("thin limit: I = S tau", 3.0 * 1e-8, transfer_step(0.0, 1.0, 3.0, 1e-8)[0], rtol=1e-8)
    check("alpha = kappa rho", 6.0, float(grey_absorption(2.0, 3.0)), rtol=1e-15)


def test_box_sampling_reproduces_linear_field():
    """Trilinear sampling reproduces a field linear in x, y, z exactly (float32 rounding)."""
    x, y, z = box_centres(BOX)
    rho = 5.0 + 0.01 * x - 0.02 * y + 0.03 * z
    tex = texture_from_fields(BOX, rho, np.full(rho.shape, T0))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-48, 48, (200, 3))
    got = np.array([sample_box(tex, p)[0] for p in pts])
    want = 5.0 + pts @ np.array([0.01, -0.02, 0.03])
    check("box sampling: linear field", want, got, rtol=1e-6)


def test_log_sampling_linear_in_index_and_seam():
    """Log-grid sampling: a field linear in (ln r, theta) index space is reproduced exactly;
    a smooth periodic field in phi is continuous across the 0 / 2 pi seam."""
    g = LogSphericalGrid(r_min=2.0, r_max=2000.0, nr=48, nth=24, nph=48)
    i, j, k = np.meshgrid(np.arange(g.nr), np.arange(g.nth), np.arange(g.nph), indexing="ij")
    tex = texture_from_fields(g, 1.0 + 0.5 * i + 0.25 * j + 0.0 * k, np.full(i.shape, T0))
    rng = np.random.default_rng(1)
    r = g.r_min * np.exp(rng.uniform(1.0, g.nr - 1.0, 100) * g.dlnr)
    th = rng.uniform(1.0, g.nth - 1.0, 100) * g.dth
    ph = rng.uniform(0, 2 * np.pi, 100)
    pts = np.c_[r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]
    got = np.array([sample_log(tex, p)[0] for p in pts])
    want = 1.0 + 0.5 * (np.log(r / g.r_min) / g.dlnr - 0.5) + 0.25 * (th / g.dth - 0.5)
    check("log sampling: linear in index space", want, got, rtol=1e-6)
    tex2 = texture_from_fields(g, 2.0 + np.cos((k + 0.5) * g.dph), np.full(i.shape, T0))
    a = sample_log(tex2, np.array([100 * np.cos(1e-7), -100 * np.sin(1e-7), 0.0]))[0]
    b = sample_log(tex2, np.array([100 * np.cos(1e-7), 100 * np.sin(1e-7), 0.0]))[0]
    check("log sampling: continuous across phi seam", a, b, rtol=1e-6)
    check("log sampling: zero inside r_min", 0.0, sample_log(tex, np.array([1.0, 0, 0]))[0])


def test_temperature_recovered_from_texture():
    """T = T_REF (rho T^4 / rho)^(1/4) gives back the cell temperature."""
    for t in (3e3, 1e4, 2.5e6, 1e8):
        tex = texture_from_fields(BOX, np.full((1, 1, 1), 1e-6), np.full((1, 1, 1), t))
        check(
            f"temperature from texture, {t:g} K",
            t,
            temperature_from_sample(float(tex.rho[0, 0, 0]), float(tex.rho_t4[0, 0, 0])),
            rtol=1e-6,
        )


def test_box_blends_into_log_grid_without_jumps():
    """A detail box (value 3) inside the log grid (value 1): the sampled value goes from 3
    inside to 1 outside over BLEND_CELLS cells, monotonically and with no jump."""
    g = LogSphericalGrid(r_min=2.0, r_max=2000.0, nr=48, nth=24, nph=48)
    b = CartesianBox(centre=(500.0, 0.0, 0.0), size=100.0, n=32)
    base = texture_from_fields(g, np.ones(g.shape), np.full(g.shape, T0))
    box = texture_from_fields(b, np.full(b.shape, 3.0), np.full(b.shape, T0))
    xs = np.linspace(530.0, 560.0, 3001)
    vals = np.array([sample([base, box], np.array([xx, 0.0, 0.0]))[0] for xx in xs])
    check("blend: value deep inside the box", 3.0, vals[0], rtol=1e-6)
    check("blend: value outside the box", 1.0, vals[-1], rtol=1e-6)
    check("blend: monotonic", True, bool(np.all(np.diff(vals) <= 1e-9)))
    check(
        "blend: largest change per 0.01 length",
        2.0 * 0.01 / (BLEND_CELLS * b.dx),
        float(np.max(-np.diff(vals))),
        rtol=0.01,
    )


def test_thin_emissive_sphere(sphere_fraction):
    """8.6 test 3: transparent uniform sphere, straight rays: I(b) = j 2 sqrt(R^2 - b^2)
    within 1% for b < 0.95 R."""
    tex = texture_from_fields(BOX, RHO0 * sphere_fraction, np.full(BOX.shape, T0))
    model = GreyBlackbody(kappa=1.0)  # tau through the sphere 8e-7: transparent
    j = RHO0 * 1.0 * S0
    bs = np.linspace(0.0, 0.95 * R_SPHERE, 12)
    ratio = []
    for b in bs:
        i, _, _ = march_segment([None, tex], model, (b, 0.3, -49.0), (b, 0.3, 49.0))
        ratio.append(i / (j * 2.0 * np.sqrt(R_SPHERE**2 - b**2 - 0.09)))
    check("test 3: thin sphere I / (j 2 sqrt(R^2-b^2))", 1.0, np.array(ratio), rtol=0.01)


def test_opaque_sphere(sphere_fraction):
    """8.6 test 4: opaque uniform sphere: a uniform disk of intensity S; edge (10% to 90%)
    sharper than 1 cell."""
    tex = texture_from_fields(BOX, RHO0 * sphere_fraction, np.full(BOX.shape, T0))
    model = GreyBlackbody(kappa=10.0 / RHO0)  # alpha = 10 per length: 7.8 per cell
    inner = [
        march_segment([None, tex], model, (b, 0.3, -49.0), (b, 0.3, 49.0))[0]
        for b in np.linspace(0.0, R_SPHERE - 2 * BOX.dx, 8)
    ]
    check("test 4: opaque sphere, I / S inside", 1.0, np.array(inner) / S0, rtol=0.01)
    bs = np.linspace(R_SPHERE - 1.5 * BOX.dx, R_SPHERE + 1.5 * BOX.dx, 121)
    prof = (
        np.array(
            [march_segment([None, tex], model, (b, 0.0, -49.0), (b, 0.0, 49.0))[0] for b in bs]
        )
        / S0
    )
    b10 = bs[np.argmax(prof < 0.9)]
    b90 = bs[np.argmax(prof < 0.1)]
    check("test 4: edge width 90% -> 10% (cells)", True, (b90 - b10) / BOX.dx < 1.0)
    print(
        f"    opaque edge: 90% at b = {b10:.3f}, 10% at b = {b90:.3f}, width "
        f"{(b90 - b10) / BOX.dx:.2f} cells (R = {R_SPHERE})"
    )
    outside = march_segment(
        [None, tex], model, (R_SPHERE + 2 * BOX.dx, 0, -49.0), (R_SPHERE + 2 * BOX.dx, 0, 49.0)
    )[0]
    check("test 4: nothing outside the sphere", 0.0, outside / S0, atol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 1.0, 10.0])
def test_uniform_slab_box_and_log_grid(tau):
    """8.6 test 5: uniform slab of optical depth tau: I = S (1 - e^-tau) within 1%, marching a
    detail box and the log grid (where the step size changes with r)."""
    model = GreyBlackbody(kappa=1.0)
    length = 60.0
    rho = tau / length
    tb = texture_from_fields(BOX, np.full(BOX.shape, rho), np.full(BOX.shape, T0))
    i, t, n = march_segment([None, tb], model, (0.0, 0.0, -30.0), (0.0, 0.0, 30.0))
    check(f"test 5 box: tau {tau:g}, I / S", -np.expm1(-tau), i / S0, rtol=0.01)
    if tau <= 1.0:  # thicker slabs stop early once the transmission is below T_STOP
        check(f"test 5 box: tau {tau:g}, steps >= 2 per cell", True, n >= 2 * length / BOX.dx - 1)
    g = LogSphericalGrid(r_min=2.0, r_max=2000.0, nr=96, nth=48, nph=96)
    tg = texture_from_fields(g, np.full(g.shape, rho), np.full(g.shape, T0))
    i2, _ = march_polyline([tg], model, [np.array([20.0, 5.0, 3.0]), np.array([80.0, 5.0, 3.0])])
    check(f"test 5 log grid: tau {tau:g}, I / S", -np.expm1(-tau), i2 / S0, rtol=0.01)


def test_polyline_equals_single_segment():
    """Splitting a straight ray into pieces does not change the result (step 3 feeds bent rays
    as polylines). The field varies over several cells, as deposited gas does; convergence on
    real frames is 8.6 test 10."""
    g = LogSphericalGrid(r_min=2.0, r_max=2000.0, nr=48, nth=24, nph=48)
    i, j, k = np.meshgrid(np.arange(g.nr), np.arange(g.nth), np.arange(g.nph), indexing="ij")
    rho = 1e-3 * (1 + 0.8 * np.sin(i / 3.0) * np.cos((j + k) / 4.0))
    tex = texture_from_fields(g, rho, 1e4 * (1 + 0.5 * np.cos(i / 5.0)))
    model = GreyBlackbody(kappa=5.0)
    a, b = np.array([30.0, -40.0, 10.0]), np.array([-60.0, 70.0, -5.0])
    whole = march_polyline([tex], model, [a, b])
    pieces = march_polyline([tex], model, [a + f * (b - a) for f in np.linspace(0, 1, 7)])
    check("polyline vs one segment: I", whole[0], pieces[0], rtol=1e-3)


def test_texture_overflow_guard():
    """Values that would overflow float32 are refused rather than silently becoming inf."""
    with pytest.raises(OverflowError):
        texture_from_fields(BOX, np.full((1, 1, 1), 1.0), np.full((1, 1, 1), 1e15))
