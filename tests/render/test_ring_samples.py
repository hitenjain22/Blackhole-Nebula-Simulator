"""Extra rays on the photon ring (render/ring_samples.py, gas_march.wgsl binding 16;
DECISIONS 2026-11-19): the ring map covers every pixel the shadow edge crosses, the GPU with no
map is unchanged, and ring pixels match the float64 CPU reference using the same map."""

import numpy as np
import pytest

from nebula.physics.geodesics import B_CRIT
from nebula.render.bent_rays import render_reference
from nebula.render.camera import Camera, sub_pixel_offsets
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.gpu_march import OUT_DIAGNOSTIC, render_gas, upload_scene
from nebula.render.march import GreyBlackbody, texture_from_fields
from nebula.render.ring_samples import impact_parameter_image, ring_subsample_map
from tests.helpers import check


def test_ring_map_covers_the_shadow_edge():
    """Every pixel whose own area holds both captured and escaping rays (b < b_crit and
    b > b_crit among 8 x 8 sample points) is a ring pixel; ring pixels are a thin band."""
    cam = Camera(position=(60.0, 5.0, 8.0), vfov_deg=40.0, width=320, height=180)
    ring = ring_subsample_map(cam) > 0
    offs = sub_pixel_offsets(8)
    px, py = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    below = np.zeros(ring.shape, bool)
    above = np.zeros(ring.shape, bool)
    for ox, oy in offs:
        b = impact_parameter_image(cam, px + ox, py + oy)
        below |= b < B_CRIT
        above |= b > B_CRIT
    edge = below & above
    check("ring map: shadow-edge pixels found", True, bool(edge.sum() > 50))
    check("ring map: edge pixels not flagged", 0, int((edge & ~ring).sum()))
    check("ring map: thin band (flagged / edge pixels < 6)", True,
          bool(ring.sum() < 6 * edge.sum()))  # fmt: skip


@pytest.fixture(scope="module")
def scene():
    g = LogSphericalGrid()
    tex = texture_from_fields(g, np.full(g.shape, 1e-3), np.full(g.shape, 1e4))
    return [tex], GreyBlackbody(kappa=1.0)


def test_no_map_and_empty_map_change_nothing(scene):
    textures, model = scene
    cam = Camera(position=(30.0, 4.0, 6.0), vfov_deg=70.0, width=32, height=24)
    gpu_scene = upload_scene(textures, model)
    plain, _ = render_gas(cam, gpu_scene, n_sub=2, out_mode=OUT_DIAGNOSTIC)
    empty, _ = render_gas(cam, gpu_scene, n_sub=2, out_mode=OUT_DIAGNOSTIC,
                          sub_map=np.zeros((24, 32), np.uint32))  # fmt: skip
    check("ring map all zero = no map (GPU)", plain, empty, atol=0.0)


def test_ring_pixels_match_cpu_reference(scene):
    """Ring pixels with 64 rays: GPU = CPU reference (same map). float32 rays within ~1e-4 of
    b_crit can be captured on one and escape on the other (Phase 2 limit), so a pixel may
    differ by one or two of its 64 rays: |captured fraction| within 2/64, intensity within
    2/64 of the brightest ray's share (Phase 8 test 7 tolerances elsewhere)."""
    textures, model = scene
    cam = Camera(position=(30.0, 4.0, 6.0), vfov_deg=70.0, width=32, height=24)
    sub_map = ring_subsample_map(cam)
    rows, cols = np.nonzero(sub_map)
    b = impact_parameter_image(cam, cols + 0.5, rows + 0.5)
    pick = np.argsort(np.abs(b - B_CRIT))[:8]  # the pixels right on the shadow edge
    pixels = [(int(cols[k]), int(rows[k])) for k in pick]
    gpu, _ = render_gas(cam, upload_scene(textures, model), n_sub=2, out_mode=OUT_DIAGNOSTIC,
                        sub_map=sub_map)  # fmt: skip
    cpu = render_reference(cam, textures, model, n_sub=2, pixels=pixels, sub_map=sub_map)
    rr = np.array([p[1] for p in pixels])
    cc = np.array([p[0] for p in pixels])
    g, c = gpu[rr, cc].astype(float), cpu[rr, cc]
    check(f"ring pixels ({len(pixels)}): captured fraction GPU vs CPU", c[:, 2], g[:, 2],
          atol=2.0 / 64)  # fmt: skip
    check("ring pixels: intensity GPU vs CPU (2 rays of 64)", c[:, 0], g[:, 0],
          atol=2.0 / 64 * float(np.nanmax(c[:, 0])))  # fmt: skip
    check("ring pixels: partly captured (the shadow edge)", True,
          bool(((c[:, 2] > 0) & (c[:, 2] < 1)).any()))  # fmt: skip
