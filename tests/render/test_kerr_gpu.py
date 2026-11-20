"""Spinning hole on the GPU (gas_march.wgsl KERR variant) against its float64 twin
(render/kerr_rays.py, bent_rays with spin): phase-11 11.2 test 6, plus the gas march along
Kerr rays (grey and colour) and the a -> 0 limit against the non-spinning shader."""

import numpy as np
import pytest

from nebula.physics import kerr as K
from nebula.physics.colour import chroma_lut
from nebula.render import kerr_rays as KR
from nebula.render.bent_rays import render_reference, render_reference_colour
from nebula.render.camera import Camera
from nebula.render.colour_data import load_cmf
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.gpu_march import (
    OUT_COLOUR,
    OUT_DIAGNOSTIC,
    OUT_SKY_DIR,
    render_gas,
    upload_scene,
)
from nebula.render.march import GreyBlackbody, RelativisticBlackbody, texture_from_fields
from tests.helpers import check
from tests.render.test_colour_march import ring_textures

SPIN = 0.9
CAM = Camera(position=(30.0, 4.0, 9.0), vfov_deg=40.0, width=48, height=32)


@pytest.fixture(scope="module")
def fog():
    g = LogSphericalGrid()
    tex = texture_from_fields(g, np.full(g.shape, 1e-3), np.full(g.shape, 1e4))
    return [tex], GreyBlackbody(kappa=1.0)


def _cpu_rays(cam, a, pixels):
    """float64 status and sky direction per pixel centre, and whether the status is stable
    under a 1e-4 rad nudge of the direction (rays that close to the critical curve may go
    either way in float32: Phase 2's rule)."""
    c0, amat = KR.camera_map(cam.position, a)
    out = {}
    for i, j in pixels:
        d = cam.directions(i + 0.5, j + 0.5)
        _, st, sky = KR.ray_polyline(cam.position, d, a, 2000.0, c0, amat)
        stable = True
        for k in range(3):
            for sgn in (-1.0, 1.0):
                dd = d.copy()
                dd[k] += sgn * 1e-4
                dd /= np.linalg.norm(dd)
                stable &= KR.ray_polyline(cam.position, dd, a, 2000.0, c0, amat)[1] == st
        out[(i, j)] = (st, sky, stable)
    return out


def test_6_capture_and_sky_directions(fog):
    textures, model = fog
    scene = upload_scene(textures, model)
    gpu, timings = render_gas(CAM, scene, n_sub=1, out_mode=OUT_SKY_DIR, spin=SPIN)
    pixels = [(i, j) for j in range(1, 32, 3) for i in range(1, 48, 3)]
    cpu = _cpu_rays(CAM, SPIN, pixels)
    mismatch, worst, n_cap, n_esc = 0, 0.0, 0, 0
    for (i, j), (st, sky, stable) in cpu.items():
        escaped_gpu = gpu[j, i, 3] > 0.5
        if not stable:
            continue
        if escaped_gpu != (st == KR.RAY_ESCAPED):
            mismatch += 1
            continue
        if st == KR.RAY_ESCAPED:
            n_esc += 1
            worst = max(worst, float(_angle(gpu[j, i, :3], sky)))
        else:
            n_cap += 1
    print(f"    {n_cap} captured, {n_esc} escaped pixels compared")
    check("test 6: captured / escaped GPU = CPU (away from the critical curve)", 0, mismatch)
    check("test 6: both kinds present", True, n_cap > 5 and n_esc > 50)
    check("test 6: sky directions GPU vs CPU (rad)", 0.0, worst, atol=2e-4)
    check("Kerr GPU: slowest tile < 1 s", True, timings.max < 1.0)


def _angle(u, v):
    """Angle between direction arrays (..., 3) after normalising each (float32 GPU vectors are
    unit only to ~1e-5, which arccos of a dot product turns into fake angles of ~1e-2)."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True).clip(1e-30)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-30)
    return np.arcsin(np.clip(np.linalg.norm(np.cross(u, v), axis=-1), 0.0, 1.0))


def test_spin_zero_matches_the_non_spinning_shader(fog):
    """a = 1e-9 (the Kerr path with no spin) against Phase 2's shader on a zoom of the shadow:
    the same rays captured, and sky directions within 2e-4 rad except within 1% of the critical
    impact parameter. There float32 errors are amplified (the Kerr rays integrate 6 numbers
    instead of Phase 2's one; measured 2026-11-20: worst 3e-4 at 0.3-1%, 2e-2 within 0.1%);
    the band is < 0.3 px wide for the video's cameras (shadow radius ~27 px at 1080p).
    A spin of even 1e-4 is not this test: it moves the critical curve by ~2e-4, which near the
    edge is a real change of direction (AI_LOG 2026-11-20)."""
    from nebula.physics.geodesics import B_CRIT
    from nebula.render.ring_samples import impact_parameter_image

    textures, model = fog
    scene = upload_scene(textures, model)
    cam = Camera(position=(30.0, 4.0, 9.0), vfov_deg=12.0, width=160, height=100)
    schw, _ = render_gas(cam, scene, n_sub=1, out_mode=OUT_SKY_DIR)
    kerr, _ = render_gas(cam, scene, n_sub=1, out_mode=OUT_SKY_DIR, spin=1e-9)
    py, px = np.mgrid[0 : cam.height, 0 : cam.width]
    rel = impact_parameter_image(cam, px + 0.5, py + 0.5) / B_CRIT - 1.0
    esc_s, esc_k = schw[..., 3] > 0.5, kerr[..., 3] > 0.5
    check("a = 0: pixels captured / escaped differently", 0, int(np.sum(esc_s != esc_k)))
    far = esc_s & esc_k & (rel > 0.01)
    worst = float(_angle(schw[far, :3], kerr[far, :3]).max())
    check(f"a = 0: sky directions Kerr vs Phase 2 shader, {far.sum()} px (rad)", 0.0, worst,
          atol=2e-4)  # fmt: skip


def test_gas_march_along_kerr_rays_grey(fog):
    """Diagnostic intensity GPU vs CPU along Kerr rays through a uniform fog: within 1e-3
    relative away from the critical curve (Phase 8 test 7's tolerance)."""
    textures, model = fog
    gpu, _ = render_gas(CAM, upload_scene(textures, model), n_sub=1, out_mode=OUT_DIAGNOSTIC,
                        spin=SPIN)  # fmt: skip
    pixels = [(i, j) for j in range(2, 32, 6) for i in range(2, 48, 6)]
    stable = _cpu_rays(CAM, SPIN, pixels)
    pixels = [px for px in pixels if stable[px][2]]
    cpu = render_reference(CAM, textures, model, n_sub=1, pixels=pixels, spin=SPIN)
    cols = np.array([p[0] for p in pixels])
    rows = np.array([p[1] for p in pixels])
    g, c = gpu[rows, cols].astype(float), cpu[rows, cols]
    check(f"Kerr grey: intensity GPU vs CPU ({len(pixels)} pixels)", c[:, 0], g[:, 0], rtol=1e-3)
    check("Kerr grey: captured fraction GPU vs CPU", c[:, 2], g[:, 2])


def test_gas_march_along_kerr_rays_colour():
    """A spinning ring of gas at r = 10 (strong Doppler colours), true colour: X, Y, Z,
    bolometric within 1e-3 relative (Phase 9 test 13's tolerance) at lit, stable pixels."""
    lam, cmf = load_cmf()
    lt, lc = chroma_lut(1e3, 1e8, 4096, lam, cmf)
    tex, _ = ring_textures(10.0, 2.0, 1.5, nr=120, nth=64, nph=256, r_lo=6.0, r_hi=16.0)
    cam = Camera(position=(50.0, 0.0, 20.0), vfov_deg=40.0, width=48, height=32)
    _, f_cam, _ = K.metric_parts(np.asarray(cam.position, float), SPIN)
    r_cam = float(np.linalg.norm(cam.position))
    model = RelativisticBlackbody(kappa=1.0, r_cam=r_cam, log_t=lt, log_chroma=lc, spin=SPIN,
                                  f_cam=float(f_cam))  # fmt: skip
    gpu, _ = render_gas(cam, upload_scene([tex], model), n_sub=1, out_mode=OUT_COLOUR, spin=SPIN)
    pixels = [(i, j) for j in range(1, 32, 3) for i in range(1, 48, 3)]
    stable = _cpu_rays(cam, SPIN, pixels)
    pixels = [px for px in pixels if stable[px][2]]
    cpu = render_reference_colour(cam, [tex], model, pixels=pixels, spin=SPIN)
    cols = np.array([p[0] for p in pixels])
    rows = np.array([p[1] for p in pixels])
    g, c = gpu[rows, cols].astype(float), cpu[rows, cols]
    lit = c[:, 3] > 1e-3 * c[:, 3].max()
    for k, ch in enumerate(("X", "Y", "Z", "bolometric")):
        rel = np.abs(g[lit, k] - c[lit, k]) / c[lit, k]
        check(f"Kerr colour: {ch} relative ({lit.sum()} lit pixels)", np.zeros(lit.sum()), rel,
              atol=1e-3)  # fmt: skip
    check("Kerr colour: lit pixels present (>= 10 of the sample)", True, lit.sum() >= 10)
