"""Nebula emission lines in the renderer (Phase 14 N1 tests 4 and 6): the GPU's LINES shader
against the CPU reference (march.RelativisticBlackbody with a LineLook), and lines fading into
the blackbody in hot gas."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from nebula.physics.colour import chroma_lut
from nebula.render.bent_rays import render_reference_colour
from nebula.render.camera import Camera
from nebula.render.colour_data import load_cmf, load_emission_tables
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.gpu_march import OUT_COLOUR, render_gas, upload_scene
from nebula.render.march import LineLook, RelativisticBlackbody, march_segment, texture_from_fields
from tests.helpers import check
from tests.render.test_gpu_colour import (
    REAL_CAM,
    REAL_RUN,
    REAL_SNAP,
    _away_from_shadow_edge,
    _compare,
)

ROOT = Path(__file__).resolve().parents[2]


def _xy(v):
    return v[:2] / v.sum()


@pytest.fixture(scope="module")
def colour_data():
    lam, cmf = load_cmf()
    return lam, cmf, chroma_lut(1e3, 1e8, 4096, lam, cmf), load_emission_tables()


def _look(cd, **kw):
    lam, cmf, _, tab = cd
    return LineLook(tables=tab, lam=lam, cmf=cmf, **kw)


def _model(cd, cam, lines):
    lut = cd[2]
    return RelativisticBlackbody(kappa=1.0, r_cam=float(np.linalg.norm(cam.position)), log_t=lut[0],
                                 log_chroma=lut[1], opacity="thermal", lines=lines)  # fmt: skip


def _shell(rho, t_k, r0=12.0, width=4.0, height=2.0):
    """A ring of gas around r0 with soft (Gaussian) edges, rho exp(-((r - r0)/width)^2 -
    (z/height)^2), at temperature T on circular orbits (Doppler shifts move the lines'
    colours), on a log grid. Soft edges as real gas: a box-edged ring makes rays graze a
    density step, where float32 paths shift the optical depth by ~1-2% even without lines
    (Phase 9's note; measured here 2026-11-23: 2.1% at a grazing pixel with no lines)."""
    g = LogSphericalGrid(r_min=4.0, r_max=40.0, nr=96, nth=64, nph=128)
    r = g.r_centres()[:, None, None]
    th = ((np.arange(g.nth) + 0.5) * g.dth)[None, :, None]
    ph = ((np.arange(g.nph) + 0.5) * g.dph)[None, None, :]
    prof = np.exp(-(((r - r0) / width) ** 2) - ((r * np.cos(th)) / height) ** 2)
    rho_f = rho * prof * np.ones(g.shape)
    gb = 1.0 / np.sqrt(r0 - 3.0)  # circular orbit at r0 (physics.steady_disc)
    vel = np.zeros((*g.shape, 3))
    vel[..., 0] = -gb * np.sin(ph) * np.ones(g.shape)
    vel[..., 1] = gb * np.cos(ph) * np.ones(g.shape)
    return texture_from_fields(g, rho_f, np.full(g.shape, t_k), vel)


@pytest.mark.parametrize("case", ["act2_dense_mix", "thin_nebula"])
def test_4_gpu_matches_cpu_with_lines(colour_data, case):
    """N1 test 4 on synthetic rings: the whole picture with lines, X, Y, Z, bolometric GPU vs
    CPU within 1e-3 relative (Phase 9 test 13's limit) at pixels away from the shadow edge and
    at least 1e-3 of the brightest. Dimmer pixels are the float32 march's floor: with no lines
    at all the old march differs by up to 1.1e-3 on these rings' faint tails (10^-6 of the
    brightest; measured 2026-11-23). The lines are checked to be most of the light there (the
    test then tests them). Rings: dense Act-2-like with a half continuum blend and teal; thin
    nebula with the thin-gas continuum. The real-snapshot test below uses Phase 9's criterion."""
    if case == "act2_dense_mix":
        tex = _shell(2.4e-7, 2.5e4)
        look = _look(colour_data, gain=2.0, continuum_mix=0.5, x_o3=0.05)
    else:
        tex = _shell(1e-12, 1.2e4)
        look = _look(colour_data, continuum_mix=1.0, x_o3=0.03)
    cam = Camera(position=(45.0, 0.0, 15.0), vfov_deg=50.0, width=40, height=28)
    pixels = [(i, j) for j in range(1, 28, 3) for i in range(1, 40, 3)]
    rows = np.array([p[1] for p in pixels])
    cols = np.array([p[0] for p in pixels])
    pics = {}
    for tag, lk in (("with", look), ("without", dataclasses.replace(look, gain=0.0))):
        model = _model(colour_data, cam, lk)
        gpu, timings = render_gas(cam, upload_scene([tex], model), n_sub=1, out_mode=OUT_COLOUR)
        check(f"N1 {case} {tag} lines: every GPU value finite", True,
              bool(np.all(np.isfinite(gpu))))  # fmt: skip
        check(f"N1 {case} {tag} lines: slowest tile < 1 s", True, timings.max < 1.0)
        cpu = render_reference_colour(cam, [tex], model, pixels=pixels)
        pics[tag] = (gpu[rows, cols].astype(float), cpu[rows, cols])
    g, c = pics["with"]
    total = c[:, 3]
    sel = _away_from_shadow_edge(cam, cols, rows) & (total >= 1e-3 * total.max())
    share = (total - pics["without"][1][:, 3])[sel] / total[sel]
    check(f"N1 {case}: lines are most of the light at the compared pixels (median share)",
          True, bool(np.median(share) > 0.5))  # fmt: skip
    for k, ch in enumerate(("X", "Y", "Z", "bolometric")):
        rel = np.abs(g[sel, k] - c[sel, k]) / c[sel, k]
        check(f"N1 test 4 {case}: {ch} GPU vs CPU with lines ({sel.sum()} px)",
              np.zeros(sel.sum()), rel, atol=1e-3)  # fmt: skip


def test_4_gpu_matches_cpu_with_lines_real_snapshot(colour_data):
    """N1 test 4 on the real snapshot of Phase 9 test 13 (gas from cool to hot, own velocities):
    256 pixels within 1e-3."""
    if not (REAL_RUN / "snapshots" / f"snap_{REAL_SNAP:05d}.npz").exists():
        pytest.skip(f"run {REAL_RUN.name} not on this machine")
    from nebula.render.snapshot_scene import snapshot_textures

    textures, _ = snapshot_textures(REAL_RUN, REAL_SNAP)
    model = _model(colour_data, REAL_CAM, _look(colour_data, continuum_mix=0.3, x_o3=0.04))
    gpu, _ = render_gas(REAL_CAM, upload_scene(textures, model), n_sub=1, out_mode=OUT_COLOUR)
    pixels = [(i, j) for j in range(1, 64, 4) for i in range(2, 64, 4)]
    _compare("N1 test 4 real snapshot", REAL_CAM, textures, model, pixels, gpu)


def test_6_hot_gas_lines_fade_into_the_blackbody(colour_data):
    """N1 test 6: through a slab of dense gas (the Act 2 cloud's mean density, 50 units) at
    10^5 and 10^6 K, the picture with lines is the blackbody-only picture plus the lines' share
    w = S_lines / (S_lines + S_cont) (S = X + Y + Z): its chromaticity moves by exactly
    w |xy_lines - xy_cont| <= w sqrt 2 (mixing two colours). The tolerance is that bound with w
    measured; w itself must be small there (the hot fade's design, LineLook defaults 4e4 K / 8):
    < 10 % at 10^5 K, < 1e-4 at 10^6 K."""
    lam, cmf, lut, tab = colour_data
    g = LogSphericalGrid(r_min=900.0, r_max=1100.0, nr=8, nth=16, nph=32)
    cam = Camera(position=(5000.0, 0.0, 0.0), vfov_deg=10.0, width=4, height=4)
    for t_k, w_max in ((1e5, 0.1), (1e6, 1e-4)):
        tex = texture_from_fields(g, np.full(g.shape, 1.0 / (4 / 3 * np.pi * 1e6)),
                                  np.full(g.shape, t_k))  # fmt: skip
        a, b = np.array([975.0, 0.0, 0.0]), np.array([1025.0, 0.0, 0.0])
        base = march_segment([tex], _model(colour_data, cam, None), a, b)[0][:3]
        look = _look(colour_data, x_o3=0.05)
        full = march_segment([tex], _model(colour_data, cam, look), a, b)[0][:3]
        lines = full - base
        w = float(lines.sum() / full.sum())
        shift = float(np.linalg.norm(_xy(full) - _xy(base)))
        bound = w * float(np.linalg.norm(_xy(lines) - _xy(base))) if w > 0 else 0.0
        check(f"N1 test 6: {t_k:.0e} K lines' share w (< {w_max})", True, w < w_max)
        check(f"N1 test 6: {t_k:.0e} K chromaticity shift = w |xy_l - xy_c|", bound, shift,
              rtol=1e-6, atol=1e-12)  # fmt: skip
        print(f"    {t_k:.0e} K: w = {w:.2e}, chromaticity shift {shift:.2e}")


def test_saturation_knob():
    """compose_colour(saturation=1) is the Phase 9 picture exactly; saturation 0 makes the gas
    grey (equal sRGB channels before the tone curve: equal 8-bit channels after)."""
    from nebula.render.gas_frames import Exposure, compose_colour

    rng = np.random.default_rng(4)
    pic = np.zeros((6, 7, 8), np.float32)
    pic[..., 0:3] = rng.uniform(0.1, 2.0, (6, 7, 3))
    exp = Exposure(0.5, 0.0)
    check("saturation 1 = the unchanged composition", compose_colour(pic, exp),
          compose_colour(pic, exp, saturation=1.0))  # fmt: skip
    grey = compose_colour(pic, exp, saturation=0.0).astype(int)
    check("saturation 0: grey gas (channels within 1 of 255 levels, dither)", True,
          bool(np.ptp(grey, axis=-1).max() <= 1))  # fmt: skip


def test_nebula_settings_are_checked():
    import dataclasses

    from nebula.render.director import Shot, ShotError

    base = Shot.load(ROOT / "configs/shots/full_story.toml")
    check("[nebula] accepted", 2.0, dataclasses.replace(base, nebula={"gain": 2.0}).nebula["gain"])
    for bad in ({"glow": 1.0}, {"continuum_mix": 1.5}, {"x_o3": -0.1}, {"t_fade": 0.0}):
        with pytest.raises(ShotError):
            dataclasses.replace(base, nebula=bad).validate()
    with pytest.raises(ShotError):
        dataclasses.replace(base, nebula={}, opacity="constant").validate()
