"""Phase 8 steps 3-4: gas marched along bent rays, CPU reference and GPU
(phases/phase-08-gas-rendering.md 8.4; 8.6 tests 6, 7, 9 and 10, plus the proper path length,
the ray geometry and the stars behind the gas)."""

from pathlib import Path

import numpy as np
import pytest

from nebula.physics.geodesics import (
    _initial_state,
    _trace,
    schw_radial_proper_distance,
    schw_ray_escape_angle,
    schw_shadow_angle,
    schw_static_proper_length,
)
from nebula.physics.radiative_transfer import blackbody_intensity_si
from nebula.render.bent_rays import (
    RAY_CAPTURED,
    RAY_ESCAPED,
    march_bent_ray,
    ray_basis,
    ray_polyline,
    render_reference,
)
from nebula.render.camera import Camera
from nebula.render.gas_scenes import point_source_textures, ring_angle_deg
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.gpu_march import (
    OUT_PICTURE,
    OUT_SKY_DIR,
    render_gas,
    upload_scene,
)
from nebula.render.march import GreyBlackbody, texture_from_fields
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map
from tests.helpers import check

ROOT = Path(__file__).resolve().parents[2]
S0 = float(blackbody_intensity_si(1e4))

# 8.6 test 6 (reference/expected-values.md: script 08): (camera D, source r, primary, secondary)
RINGS = [
    (50.0, 30.0, 11.701931, 5.849751),
    (50.0, 100.0, 14.958880, 5.850477),
    (30.0, 15.0, 15.114892, 9.639931),
]
RING_FOV, RING_N = 36.0, 512  # full image: pixel = 36 / 512 = 0.0703 deg
ZOOM_FOV, ZOOM_H = 0.008, 1024  # zoom on the secondary ring: pixel 7.8e-6 deg (ring width
# 3.5e-5 to 3.4e-4 deg)

# 8.6 test 7: real snapshot (the torus, day 273), camera r ~ 61 slightly above the plane.
REAL_RUN, REAL_SNAP = ROOT / "runs" / "R6.1_100k", 150
REAL_KAPPA = 1e7  # code units: transmissions from ~0 to ~0.9 across the image (step 3 probe)
REAL_CAM = Camera(position=(60.0, 0.0, 10.0), vfov_deg=90.0, width=64, height=64)


def test_proper_length_formula():
    """schw_static_proper_length: tangential steps keep their flat length; radial steps
    summed reproduce the exact radial proper distance (midpoint rule, so 2nd order)."""
    check("proper length: tangential step", 0.3, float(schw_static_proper_length(0.3, 0.0, 7)))
    for r1, r2 in ((3.0, 10.0), (10.0, 2000.0)):
        edges = np.geomspace(r1, r2, 20001)
        mids = 0.5 * (edges[1:] + edges[:-1])
        total = schw_static_proper_length(np.diff(edges), np.diff(edges), mids).sum()
        check(
            f"proper length: radial {r1:g} -> {r2:g}",
            schw_radial_proper_distance(r1, r2),
            float(total),
            rtol=1e-6,
        )


@pytest.fixture(scope="module")
def uniform_scene():
    g = LogSphericalGrid()
    tex = texture_from_fields(g, np.full(g.shape, 1e-3), np.full(g.shape, 1e4))
    return [tex], GreyBlackbody(kappa=1.0)


def test_radial_ray_uses_proper_length(uniform_scene):
    """A radial ray outward from r = 50 through uniform gas (alpha = 1e-3): I = S (1 - e^-tau)
    with tau = alpha x the proper distance 50 -> 2000 = 1953.9 (flat: 1950; the 0.2% longer
    path changes I by 6e-4, far above the 1e-6 tolerance). CPU reference to 1e-6; GPU
    (float32) to 1e-4 against the CPU on four rays a few degrees off radial."""
    textures, model = uniform_scene
    tau = 1e-3 * schw_radial_proper_distance(50.0, 2000.0)
    i, t, status, _ = march_bent_ray(textures, model, (50.0, 0, 0), np.array([1.0, 0, 0]))
    check("radial ray, CPU: I / S", -np.expm1(-tau), i / S0, rtol=1e-6)
    check("radial ray, CPU: transmission", np.exp(-tau), t, rtol=1e-6)
    check("radial ray, CPU: escaped", RAY_ESCAPED, status)
    cam = Camera(position=(50.0, 0, 0), look_at=(100.0, 0, 0), vfov_deg=10, width=2, height=2)
    img, _ = render_gas(cam, upload_scene(textures, model), n_sub=1)
    # 2x2 image: no pixel centre is exactly radial; compare with the CPU on the same rays.
    ref = render_reference(cam, textures, model)
    check("radial-ish rays, GPU vs CPU: I", ref[..., 0], img[..., 0], rtol=1e-4)


def test_cpu_ray_points_follow_exact_geodesic():
    """bent_rays.ray_polyline (float64 RK4, the GPU's scheme) lies on the DOP853 reference ray
    (physics.geodesics) to 1e-6 in r, for rays passing at 1.05x and 2x the critical impact
    parameter and one captured; and ends at r_max / r = 2 with the right status."""
    d_cam = 50.0
    cam = np.array([d_cam, 0.0, 0.0])
    psi_sh = schw_shadow_angle(d_cam)
    for name, psi, want in (
        ("near photon sphere", np.arcsin(1.05 * np.sin(psi_sh)), RAY_ESCAPED),
        ("wide", np.arcsin(2.0 * np.sin(psi_sh)), RAY_ESCAPED),
        ("captured", 0.8 * psi_sh, RAY_CAPTURED),
    ):
        d = np.array([-np.cos(psi), np.sin(psi), 0.0])
        pts, status = ray_polyline(cam, d, 2000.0)
        e1, e2, _, _ = ray_basis(cam, d)
        phi = np.unwrap(np.arctan2(pts @ e2, pts @ e1))
        r = np.linalg.norm(pts, axis=1)
        _, sol = _trace(*_initial_state(d_cam, psi), dense=True)
        inner = slice(0, -1)  # the last point is interpolated to the boundary
        check(f"ray points, {name}: r vs DOP853", 1.0 / sol.sol(phi[inner])[0], r[inner], rtol=1e-6)
        check(f"ray points, {name}: status", want, status)
        end = 2000.0 if want == RAY_ESCAPED else 2.0
        check(f"ray points, {name}: end radius", end, r[-1], rtol=1e-12)


def _ring_images(d_cam, r_src, n_sub=1):
    textures = point_source_textures(r_src)
    scene = upload_scene(textures, GreyBlackbody(kappa=1e-3))
    cam = Camera(position=(d_cam, 0.0, 0.0), vfov_deg=RING_FOV, width=RING_N, height=RING_N)
    full, timings = render_gas(cam, scene, n_sub=n_sub)
    return cam, full, timings, scene


def _zoom_camera(d_cam, psi_deg):
    p0 = np.radians(psi_deg)
    fwd = np.array([-np.cos(p0), 0.0, np.sin(p0)])
    return Camera(
        position=(d_cam, 0.0, 0.0),
        look_at=tuple(np.array([d_cam, 0.0, 0.0]) + fwd),
        vfov_deg=ZOOM_FOV,
        width=4,
        height=ZOOM_H,
    )


@pytest.mark.parametrize("d_cam,r_src,primary,secondary", RINGS)
def test_lensed_point_source_rings(d_cam, r_src, primary, secondary):
    """8.6 test 6 (GPU): a small glowing ball directly behind the hole appears as rings at the
    exact angles, within 1 pixel (pixel = 36 deg / 512). The secondary ring lies 0.005-0.007
    deg outside the shadow and is 3.5e-5 to 3.4e-4 deg wide, far thinner than a pixel, so it is
    measured on a camera zoomed onto +-0.004 deg around the expected angle (pixel 7.8e-6 deg;
    a ring outside that window gives no light and fails). Every tile < 1 s."""
    px = RING_FOV / RING_N
    sh = np.degrees(schw_shadow_angle(d_cam))
    cam, img, timings, scene = _ring_images(d_cam, r_src)
    got1 = ring_angle_deg(cam, img[..., 0], sh + 0.3, 90.0)
    zcam = _zoom_camera(d_cam, secondary)
    zimg, _ = render_gas(zcam, scene, n_sub=2)
    got2 = ring_angle_deg(zcam, zimg[..., 0], sh, sh + 1.0)
    check(f"test 6, D {d_cam:g}, r {r_src:g}: primary ring (deg)", primary, got1, atol=px)
    check(f"test 6, D {d_cam:g}, r {r_src:g}: secondary ring (deg)", secondary, got2, atol=px)
    print(
        f"    errors: primary {(got1 - primary) / px:+.3f} px, secondary "
        f"{(got2 - secondary) / px:+.5f} px = {(got2 - secondary) / (ZOOM_FOV / ZOOM_H):+.2f} "
        f"zoom px; slowest tile {timings.max:.2f} s"
    )
    check(f"test 6, D {d_cam:g}: slowest GPU tile < 1 s", True, timings.max < 1.0)


@pytest.fixture(scope="module")
def real_scene():
    if not (REAL_RUN / "snapshots" / f"snap_{REAL_SNAP:05d}.npz").exists():
        pytest.skip(f"run {REAL_RUN.name} not on this machine")
    from nebula.render.snapshot_scene import snapshot_textures

    textures, _ = snapshot_textures(REAL_RUN, REAL_SNAP)
    model = GreyBlackbody(kappa=REAL_KAPPA)
    gpu, _ = render_gas(REAL_CAM, upload_scene(textures, model), n_sub=1)
    return textures, model, gpu


def _compare_real(real_scene, pixels):
    """8.6 test 7 on the given (column, row) pixels."""
    textures, model, gpu = real_scene
    cpu = render_reference(REAL_CAM, textures, model, pixels=pixels)
    cols = np.array([p[0] for p in pixels])
    rows = np.array([p[1] for p in pixels])
    g, c = gpu[rows, cols], cpu[rows, cols]
    # Distance of each pixel centre from the shadow edge, in pixels (gnomonic camera looking at
    # the hole: angle per pixel at angle psi from the centre is (2 tan(fov/2) / H) cos^2 psi).
    d = REAL_CAM.directions(cols + 0.5, rows + 0.5)
    pos = np.asarray(REAL_CAM.position, float)
    r_cam = np.linalg.norm(pos)
    psi = np.arccos(np.clip(d @ (-pos / r_cam), -1.0, 1.0))
    px_angle = 2.0 * REAL_CAM.tan_half_vfov / REAL_CAM.height * np.cos(psi) ** 2
    far = np.abs(psi - schw_shadow_angle(r_cam)) / px_angle > 2.0
    b = r_cam * np.sin(psi) / np.sqrt(1.0 - 2.0 / r_cam)
    clear = np.abs(b / (3.0 * np.sqrt(3.0)) - 1.0) > 1e-4
    rel = np.abs(g[:, 0] - c[:, 0]) / np.maximum(c[:, 0], 1e-300)
    check(
        f"test 7: GPU vs CPU, I relative ({far.sum()} of {len(pixels)} pixels)",
        np.zeros(far.sum()),
        rel[far],
        atol=1e-3,
    )
    check("test 7: transmission, absolute", c[far, 1], g[far, 1], atol=1e-3)
    check("test 7: capture classification", c[clear, 2], g[clear, 2])
    print(
        f"    worst I difference {rel[far].max():.2e} (median {np.median(rel[far]):.1e}); "
        f"transmission {np.percentile(c[:, 1], [5, 50, 95]).round(3)} (5/50/95%); "
        f"captured {int(c[:, 2].sum())} pixels"
    )


def test_gpu_matches_cpu_real_snapshot_sample(real_scene):
    """8.6 test 7, quick version: 256 pixels spread over the 64 x 64 GPU image (every 4th
    column and row); the full image is the slow test below."""
    _compare_real(real_scene, [(i, j) for j in range(1, 64, 4) for i in range(2, 64, 4)])


@pytest.mark.slow
def test_gpu_matches_cpu_real_snapshot_full(real_scene):
    """8.6 test 7: every pixel of the 64 x 64 image (CPU reference ~10 min)."""
    _compare_real(real_scene, [(i, j) for j in range(64) for i in range(64)])


def test_no_invalid_pixels_at_the_horizon(uniform_scene):
    """Captured rays end exactly at r = 2, where the proper length diverges. In float32 a last
    step can sit at r = 2.0000000 and gave 0 x infinity = NaN in 8 of 1M rays (found on the
    D = 30 ring picture). Every pixel must be finite: empty space down to the hole, and gas
    filling the grid down to the horizon (where the transmission must reach ~0)."""
    cam = Camera(position=(30.0, 0.0, 0.0), vfov_deg=36.0, width=512, height=512)
    ring_scene = upload_scene(point_source_textures(15.0), GreyBlackbody(kappa=1e-3))
    img, _ = render_gas(cam, ring_scene, n_sub=2)
    check("horizon: empty space, all pixels finite", True, bool(np.isfinite(img).all()))
    textures, model = uniform_scene
    img, _ = render_gas(cam, upload_scene(textures, model), n_sub=2)
    check("horizon: gas down to r = 2, all pixels finite", True, bool(np.isfinite(img).all()))


# ---------------------------------------------------------------- step 4: stars, tests 9-10


def test_sky_directions_match_exact_rays():
    """Escaped rays continue to infinity for the star field (gas_march.wgsl escape_angle). Their
    sky directions match the float64 reference (physics.geodesics.schw_ray_escape_angle) with
    Phase 2's limits (tests/render/test_schw_ray_gpu.py): 2e-4 rad for b > 1.05 b_crit, 1e-2 rad
    for b > 1.001 b_crit, and the same capture/escape decision outside +-1e-4 of b_crit."""
    g = LogSphericalGrid()
    empty = upload_scene([texture_from_fields(g, np.zeros(g.shape), np.zeros(g.shape))],
                         GreyBlackbody(kappa=1.0))  # fmt: skip
    cam = Camera(position=(30.0, 4.0, 6.0), vfov_deg=70.0, width=24, height=24)
    out, _ = render_gas(cam, empty, n_sub=1, out_mode=OUT_SKY_DIR)
    px, py = np.meshgrid(np.arange(24) + 0.5, np.arange(24) + 0.5)
    d = cam.directions(px, py).reshape(-1, 3)
    pos = np.asarray(cam.position, float)
    r_cam = float(np.linalg.norm(pos))
    errs, x, esc_ref = [], [], []
    for k, dk in enumerate(d):
        e1, e2, cos_psi, sin_psi = ray_basis(pos, dk)
        psi = float(np.arctan2(sin_psi, cos_psi))
        phi = schw_ray_escape_angle(r_cam, psi)
        x.append(r_cam * sin_psi / np.sqrt(1 - 2 / r_cam) / (3 * np.sqrt(3)))
        esc_ref.append(np.isfinite(phi))
        if np.isfinite(phi) and out.reshape(-1, 4)[k, 3] > 0.5:
            ref = np.cos(phi) * e1 + np.sin(phi) * e2
            gpu = out.reshape(-1, 4)[k, :3].astype(float)
            errs.append(np.arccos(np.clip(ref @ gpu / np.linalg.norm(gpu), -1, 1)))
        else:
            errs.append(np.nan)
    errs, x, esc_ref = np.array(errs), np.array(x), np.array(esc_ref)
    ok = np.isfinite(errs)
    check("sky direction error, b > 1.05 b_crit (rad)", 0.0, errs[ok & (x > 1.05)].max(),
          atol=2e-4)  # fmt: skip
    check("sky direction error, b > 1.001 b_crit (rad)", 0.0, errs[ok & (x > 1.001)].max(),
          atol=1e-2)  # fmt: skip
    away = np.abs(x - 1) > 1e-4
    esc_gpu = out.reshape(-1, 4)[:, 3] > 0.5
    check("sky directions: capture/escape mismatches", 0, int(np.sum((esc_gpu != esc_ref) & away)))
    check("sky directions: rays tested (escaped)", True, bool(ok.sum() > 400))


def test_stars_dimmed_by_gas_transmission():
    """Stars seen through gas that only absorbs (T = 0 K: no emission) are the unobstructed
    stars times the ray's transmission, which itself matches the CPU reference (test 7)."""
    g = LogSphericalGrid()
    rho = np.zeros(g.shape)
    rho[g.r_centres() < 60.0] = 2e-3  # absorbing ball r < 60: tau ~ 0.1-0.3 across the view
    gas = [texture_from_fields(g, rho, np.zeros(g.shape))]
    empty = [texture_from_fields(g, np.zeros(g.shape), np.zeros(g.shape))]
    model = GreyBlackbody(kappa=1.0)
    sky = upload_cube_map(build_cube_map(StarFieldConfig(face_size=128, n_stars=5000)))
    cam = Camera(position=(40.0, 0.0, 3.0), vfov_deg=60.0, width=32, height=32)
    seen, _ = render_gas(cam, upload_scene(gas, model), sky=sky, out_mode=OUT_PICTURE)
    bare, _ = render_gas(cam, upload_scene(empty, model), sky=sky, out_mode=OUT_PICTURE)
    diag, _ = render_gas(cam, upload_scene(gas, model))
    check("stars through absorbing gas: no gas glow", 0.0, float(np.abs(seen[..., 0]).max()))
    lit = bare[..., 1:4].sum(-1) > 0
    check("stars through gas = stars x transmission", (bare[..., 1:4] * diag[..., 1:2])[lit],
          seen[..., 1:4][lit], rtol=1e-5)  # fmt: skip
    pix = [(3, 5), (16, 8), (28, 27), (10, 22)]
    ref = render_reference(cam, gas, model, pixels=pix)
    for i, j in pix:
        check(f"transmission pixel ({i}, {j}) vs CPU", ref[j, i, 1], diag[j, i, 1], rtol=1e-4)
    check("absorbing ball dims the stars", True, bool(diag[..., 1][diag[..., 2] < 0.5].min() < 0.9))


@pytest.fixture(scope="module")
def frame_camera():
    e = np.radians(5.0)
    return np.array([40.0 * np.cos(e), 0.0, 40.0 * np.sin(e)])


def _frame(real_scene, pos, step_fraction=0.5):
    textures, model, _ = real_scene
    cam = Camera(position=tuple(pos), vfov_deg=60.0, width=320, height=180)
    img, _ = render_gas(cam, upload_scene(textures, model, step_fraction), n_sub=2)
    return img[..., 0].astype(float)


def test_temporal_stability(real_scene, frame_camera):
    """8.6 test 9: the same snapshot with the camera moved by 1e-6 (along x, y and z): no pixel
    changes by more than 1e-3 (relative, linear HDR). Nothing random, no flicker."""
    a = _frame(real_scene, frame_camera)
    for axis in range(3):
        moved = frame_camera.copy()
        moved[axis] += 1e-6
        rel = np.abs(_frame(real_scene, moved) - a) / np.maximum(a, 1e-300)
        check(f"test 9: camera moved 1e-6 along {'xyz'[axis]}, largest pixel change", 0.0,
              rel.max(), atol=1e-3)  # fmt: skip


def test_step_size_converged(real_scene, frame_camera):
    """8.6 test 10: halving the march step (half -> quarter cell) changes a real frame by less
    than 0.5% (mean absolute difference / mean brightness, linear HDR)."""
    a = _frame(real_scene, frame_camera)
    b = _frame(real_scene, frame_camera, step_fraction=0.25)
    check("test 10: halved step, mean |change| / mean", 0.0, np.abs(b - a).mean() / a.mean(),
          atol=5e-3)  # fmt: skip
    print(f"    largest single-pixel change {np.max(np.abs(b - a) / a):.2e}")
