"""Phase 9 step 9e: the GPU colour march (shaders/gas_march.wgsl colour mode) against the CPU
reference (bent_rays.render_reference_colour with march.RelativisticBlackbody); test 13 of
phase-09 9.7 (CLAUDE.md rule 2)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from nebula.physics.colour import chroma_lut
from nebula.physics.geodesics import schw_shadow_angle
from nebula.render.bent_rays import render_reference, render_reference_colour
from nebula.render.camera import Camera
from nebula.render.colour_data import load_cmf
from nebula.render.gpu_march import OUT_COLOUR, OUT_DIAGNOSTIC, render_gas, upload_scene
from nebula.render.march import GreyBlackbody, RelativisticBlackbody
from tests.helpers import check
from tests.render.test_colour_march import ring_textures

ROOT = Path(__file__).resolve().parents[2]
REAL_RUN, REAL_SNAP = ROOT / "runs" / "R6.1_100k", 150  # as Phase 8 test 7 (torus, day 273)
REAL_KAPPA = 1e7
REAL_CAM = Camera(position=(60.0, 0.0, 10.0), vfov_deg=90.0, width=64, height=64)


@pytest.fixture(scope="module")
def lut():
    lam, cmf = load_cmf()
    return chroma_lut(1e3, 1e8, 4096, lam, cmf)


def _model(lut, kappa, cam):
    return RelativisticBlackbody(kappa=kappa, r_cam=float(np.linalg.norm(cam.position)),
                                 log_t=lut[0], log_chroma=lut[1])  # fmt: skip


def _away_from_shadow_edge(cam, cols, rows):
    """Pixels more than 2 px from the shadow edge and not on the photon orbit's impact
    parameter (as Phase 8 test 7: there a float32 ray can be captured or escape)."""
    d = cam.directions(cols + 0.5, rows + 0.5)
    pos = np.asarray(cam.position, float)
    r_cam = np.linalg.norm(pos)
    psi = np.arccos(np.clip(d @ (-pos / r_cam), -1.0, 1.0))
    px_angle = 2.0 * cam.tan_half_vfov / cam.height * np.cos(psi) ** 2
    return np.abs(psi - schw_shadow_angle(r_cam)) / px_angle > 2.0


def _compare(name, cam, textures, model, pixels, gpu):
    cpu = render_reference_colour(cam, textures, model, pixels=pixels)
    cols = np.array([p[0] for p in pixels])
    rows = np.array([p[1] for p in pixels])
    g, c = gpu[rows, cols], cpu[rows, cols]
    far = _away_from_shadow_edge(cam, cols, rows)
    lit = far & (c[:, 3] > 1e-6 * c[:, 3].max())
    worst = 0.0
    for k, ch in enumerate(("X", "Y", "Z", "bolometric")):
        rel = np.abs(g[lit, k] - c[lit, k]) / c[lit, k]
        worst = max(worst, float(rel.max()))
        check(f"test 13 {name}: {ch} relative ({lit.sum()} lit pixels)", np.zeros(lit.sum()),
              rel, atol=1e-3)  # fmt: skip
    check(f"test 13 {name}: transmission, absolute", c[far, 4], g[far, 4], atol=1e-3)
    check(f"test 13 {name}: captured fraction", c[far, 5], g[far, 5])
    print(f"    {name}: worst colour difference {worst:.2e}; lit {lit.sum()} of {len(pixels)}")


@pytest.fixture(scope="module")
def real_colour(lut):
    if not (REAL_RUN / "snapshots" / f"snap_{REAL_SNAP:05d}.npz").exists():
        pytest.skip(f"run {REAL_RUN.name} not on this machine")
    from nebula.render.snapshot_scene import snapshot_textures

    textures, _ = snapshot_textures(REAL_RUN, REAL_SNAP)
    model = _model(lut, REAL_KAPPA, REAL_CAM)
    gpu, timings = render_gas(REAL_CAM, upload_scene(textures, model), n_sub=1,
                              out_mode=OUT_COLOUR)  # fmt: skip
    return textures, model, gpu, timings


def test_13_gpu_matches_cpu_colour_real_snapshot(real_colour):
    """Real snapshot (gas with its own velocities, gamma beta from the GN momenta), 256 pixels
    spread over the 64 x 64 image: X, Y, Z and bolometric within 1e-3 relative."""
    textures, model, gpu, timings = real_colour
    pixels = [(i, j) for j in range(1, 64, 4) for i in range(2, 64, 4)]
    _compare("real snapshot", REAL_CAM, textures, model, pixels, gpu)
    check("colour GPU: every pixel finite", True, bool(np.all(np.isfinite(gpu))))
    check("colour GPU: slowest tile < 1 s", True, timings.max < 1.0)


def test_13_gpu_matches_cpu_spinning_ring(lut):
    """A ring of gas on circular orbits at r = 10 (fast, so Doppler colours are strong), seen
    from r = 54 above the plane: the GPU's velocity textures and redshift match the CPU.

    This box-edged ring makes a few rays graze a density step, where float32 ray paths shift
    the optical depth by up to ~0.5% in the already verified grey march as well (Phase 8; the
    same pixels, the same size: AI_LOG 2026-11-17). So the colour code is judged by what it
    adds: (a) chromaticity and bolometric / Y (set by g, the velocities and the LUT) within
    1e-4, and (b) the colour run's GPU - CPU brightness gap equals the grey run's gap at the
    same pixels within 1e-3. The plan's test 13 itself is the real-snapshot test above."""
    tex, _ = ring_textures(10.0, 2.0, 1.5, nr=120, nth=64, nph=256, r_lo=6.0, r_hi=16.0)
    cam = Camera(position=(50.0, 0.0, 20.0), vfov_deg=40.0, width=48, height=32)
    model = _model(lut, 1.0, cam)
    gpu, _ = render_gas(cam, upload_scene([tex], model), n_sub=1, out_mode=OUT_COLOUR)
    pixels = [(i, j) for j in range(0, 32, 2) for i in range(0, 48, 2)]
    cols = np.array([p[0] for p in pixels])
    rows = np.array([p[1] for p in pixels])
    cpu = render_reference_colour(cam, [tex], model, pixels=pixels)
    g, c = gpu[rows, cols], cpu[rows, cols]
    lit = _away_from_shadow_edge(cam, cols, rows) & (c[:, 3] > 1e-6 * c[:, 3].max())
    print(f"    spinning ring: {lit.sum()} lit pixels of {len(pixels)}")
    for k, ch in enumerate(("x", "y")):
        check(f"test 13 ring: chromaticity {ch}, GPU vs CPU", c[lit, k] / c[lit, :3].sum(axis=1),
              g[lit, k] / g[lit, :3].sum(axis=1), atol=1e-4)  # fmt: skip
    check("test 13 ring: bolometric / Y, GPU / CPU - 1", np.zeros(lit.sum()),
          (g[lit, 3] / g[lit, 1]) / (c[lit, 3] / c[lit, 1]) - 1.0, atol=1e-4)  # fmt: skip
    grey = GreyBlackbody(kappa=1.0)
    gg, _ = render_gas(cam, upload_scene([tex], grey), n_sub=1, out_mode=OUT_DIAGNOSTIC)
    gc = render_reference(cam, [tex], grey, pixels=pixels)[rows, cols, 0]
    gg = gg[rows, cols, 0]
    gap_colour = (g[lit, 3] - c[lit, 3]) / c[lit, 3]
    gap_grey = (gg[lit] - gc[lit]) / gc[lit]
    print(
        f"    largest gap: colour {np.abs(gap_colour).max():.2e}, grey {np.abs(gap_grey).max():.2e}"
    )
    check("test 13 ring: colour GPU-CPU gap minus grey gap", np.zeros(lit.sum()),
          gap_colour - gap_grey, atol=1e-3)  # fmt: skip


def test_colour_subnormal_density_residues(lut):
    """Deposited float32 grids can hold subnormal densities (~1e-45, seen in the continuous
    shot's bricks, frame 312) that the GPU flushes to zero in some operations but not others:
    rho_gb / rho and rho_t4 / rho became 0/0 and whole pixels NaN (AI_LOG 2026-11-17). Below
    march.RHO_MIN_COLOUR a sample is empty in both CPU and GPU: every pixel stays finite and
    the image matches the same scene without the residues."""
    tex, _ = ring_textures(10.0, 2.0, 1.5, nr=120, nth=64, nph=256, r_lo=6.0, r_hi=16.0)
    rho = np.asarray(tex.rho, np.float32).copy()
    sub = np.float32(1e-45)  # smallest float32 subnormal
    rho_t4 = np.asarray(tex.rho_t4, np.float32).copy()
    rho_gb = np.asarray(tex.rho_gb, np.float32).copy()
    empty = rho == 0
    rho[empty], rho_t4[empty] = sub, sub
    rho_gb[empty] = sub
    dirty = dataclasses.replace(tex, rho=rho, rho_t4=rho_t4, rho_gb=rho_gb)
    cam = Camera(position=(50.0, 0.0, 20.0), vfov_deg=40.0, width=48, height=32)
    model = _model(lut, 1.0, cam)
    clean, _ = render_gas(cam, upload_scene([tex], model), n_sub=1, out_mode=OUT_COLOUR)
    gpu, _ = render_gas(cam, upload_scene([dirty], model), n_sub=1, out_mode=OUT_COLOUR)
    check("subnormal residues: every pixel finite", True, bool(np.all(np.isfinite(gpu))))
    check("subnormal residues: same image as without them", clean[..., :5], gpu[..., :5],
          rtol=1e-6, atol=0.0)  # fmt: skip
