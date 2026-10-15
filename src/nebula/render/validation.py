"""Validation renders with exact answers (phases/phase-02-blackhole-render.md Step 2.5): shared
by tests/render/test_validation_images.py and scripts/validation_images.py."""

from dataclasses import dataclass

import numpy as np

from nebula.physics.geodesics import schw_einstein_ring_angle, schw_shadow_angle
from nebula.render.camera import MODE_HOLE_STARS, Camera
from nebula.render.frames import render
from nebula.render.measure import (
    angle_to_radius_px,
    fit_circle,
    peak_radius,
    radial_profile,
    radius_px_to_angle,
    shadow_edge,
)
from nebula.render.starfield import LUMA_WEIGHTS, GPUSky, single_star_cube_map, upload_cube_map

D = 50.0


@dataclass
class Measurement:
    name: str
    expected_deg: float
    measured_deg: float
    expected_px: float
    measured_px: float

    @property
    def error_px(self) -> float:
        return self.measured_px - self.expected_px


def shadow_camera() -> Camera:
    """D = 50, field of view so the shadow is ~200 px across (radius ~102 px)."""
    return Camera(position=(D, 0.0, 0.0), look_at=(0, 0, 0), up=(0, 0, 1), vfov_deg=40.0)


def ring_camera() -> Camera:
    """D = 50, wide enough to contain the primary ring (17.87 deg)."""
    return Camera(position=(D, 0.0, 0.0), look_at=(0, 0, 0), up=(0, 0, 1), vfov_deg=48.0)


def measure_shadow(sky: GPUSky, n_sub: int = 4) -> tuple[np.ndarray, list[Measurement], tuple]:
    """Render the shadow test image and fit a circle to the shadow's edge (captured fraction
    0.5). Returns (HDR image, [shadow radius, centre offset], fitted circle)."""
    cam = shadow_camera()
    img, _ = render(
        cam, MODE_HOLE_STARS, n_sub, (sky.scale, 0.0), sky.entries(), "render_stars.wgsl"
    )
    centre = (cam.width / 2, cam.height / 2)
    xs, ys = shadow_edge(img[..., 3], centre, r_max=200.0)
    xc, yc, radius = fit_circle(xs, ys)
    exp_angle = schw_shadow_angle(D)
    m = Measurement(
        "shadow radius",
        np.degrees(exp_angle),
        np.degrees(radius_px_to_angle(cam, radius)),
        angle_to_radius_px(cam, exp_angle),
        radius,
    )
    offset = float(np.hypot(xc - centre[0], yc - centre[1]))
    return img, [m, Measurement("shadow centre offset", 0.0, 0.0, 0.0, offset)], (xc, yc, radius)


def measure_rings(
    n_sub: int = 4, star_sigma_deg: float = 0.5
) -> tuple[np.ndarray, list[Measurement]]:
    """Render one star exactly behind the hole (sky direction -x, camera on +x) and locate the
    primary and secondary Einstein rings as peaks of the azimuthally averaged brightness."""
    cam = ring_camera()
    cube = single_star_cube_map((-1.0, 0.0, 0.0), star_sigma_deg, face_size=512)
    sky = upload_cube_map(cube)
    img, _ = render(
        cam, MODE_HOLE_STARS, n_sub, (sky.scale, 0.0), sky.entries(), "render_stars.wgsl"
    )
    lum = img[..., :3] @ LUMA_WEIGHTS
    centre = (cam.width / 2, cam.height / 2)
    r, prof = radial_profile(lum, centre, r_max=cam.height / 2 - 5)
    out = []
    for order, name, window in ((0, "primary ring", 20.0), (1, "secondary ring", 3.0)):
        ang = schw_einstein_ring_angle(D, order)
        exp_px = angle_to_radius_px(cam, ang)
        got_px = peak_radius(r, prof, exp_px - window, exp_px + window)
        out.append(
            Measurement(
                name, np.degrees(ang), np.degrees(radius_px_to_angle(cam, got_px)), exp_px, got_px
            )
        )
    return img, out
