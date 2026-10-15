"""Step 2.3: camera and image pipeline (phases/phase-02-blackhole-render.md Step 2.3)."""

import numpy as np

from nebula.render.camera import (
    MODE_DEBUG_DIRECTION,
    MODE_DEBUG_GNOMONIC,
    MODE_FLAT_GNOMONIC_GRID,
    Camera,
    sub_pixel_offsets,
)
from nebula.render.frames import (
    render,
    srgb_decode,
    srgb_encode,
    to_uint8,
    tone_map,
)
from tests.helpers import check

# Looking along +z from z = -100 with up = +y: the camera's right is -x (right-handed axes).
STRAIGHT_CAM = Camera(
    position=(0.0, 0.0, -100.0), look_at=(0, 0, 0), up=(0, 1, 0), vfov_deg=60, width=160, height=120
)


def test_camera_basis():
    cam = Camera(position=(17.0, -31.0, 9.0), look_at=(3, 2, -1), up=(0.2, 0.1, 1), vfov_deg=50)
    basis = np.array([cam.right, cam.up_cam, cam.forward])
    check("camera axes orthonormal", np.eye(3), basis @ basis.T, atol=1e-15)
    centre = cam.directions(cam.width / 2, cam.height / 2)
    check("image centre looks at the target", cam.forward, centre, rtol=1e-15, per_vector=True)
    corner = cam.directions(0.0, 0.0)
    px, py = cam.pixel_of_direction(corner)
    check("pixel -> direction -> pixel (top-left corner)", (0.0, 0.0), (px, py), atol=1e-9)
    top = cam.directions(cam.width / 2, 0.0)
    check(
        "vertical field of view",
        cam.vfov_deg / 2,
        np.degrees(np.arccos(top @ cam.forward)),
        rtol=1e-12,
    )
    check(
        "4x4 sub-ray pattern",
        (16, 0.125, 0.875),
        (len(sub_pixel_offsets(4)), sub_pixel_offsets(4).min(), sub_pixel_offsets(4).max()),
    )


def test_gpu_directions_match_python():
    """GPU pixel_dir (render_sky.wgsl) vs Camera.directions, randomly oriented camera."""
    cam = Camera(
        position=(17.0, -31.0, 9.0),
        look_at=(3, 2, -1),
        up=(0.2, 0.1, 1),
        vfov_deg=50,
        width=64,
        height=48,
    )
    img, _ = render(cam, MODE_DEBUG_DIRECTION, n_sub=1)
    j, i = np.mgrid[0:48, 0:64]
    ref = cam.directions(i + 0.5, j + 0.5)
    check("GPU vs Python pixel directions (64x48)", ref, img[..., :3], rtol=1e-6, per_vector=True)


def test_flat_space_rays_are_straight():
    """No hole, straight rays: every pixel centre's direction, in the camera's own axes, sits
    exactly where a pinhole camera puts it (so straight lines on the sky stay straight)."""
    cam = STRAIGHT_CAM
    img, _ = render(cam, MODE_DEBUG_GNOMONIC, n_sub=1)
    j, i = np.mgrid[0 : cam.height, 0 : cam.width]
    x_expected = (2 * (i + 0.5) / cam.width - 1) * cam.tan_half_vfov * cam.aspect
    y_expected = (1 - 2 * (j + 0.5) / cam.height) * cam.tan_half_vfov
    # right = -x and up = +y here, so (d.right, d.up)/(d.fwd) = (-d_x/d_z, d_y/d_z).
    check("flat space: horizontal image coordinate", x_expected, -img[..., 0], atol=1e-6)
    check("flat space: vertical image coordinate", y_expected, img[..., 1], atol=1e-6)


def test_grid_lines_fall_on_predicted_pixels():
    """Sky grid of straight lines (spacing 0.1 in tangent units): rendered lines must be
    straight and in the predicted columns and rows."""
    cam = STRAIGHT_CAM
    spacing, half_width = 0.1, 0.002
    img, _ = render(cam, MODE_FLAT_GNOMONIC_GRID, n_sub=4, sky_params=(spacing, half_width))
    lum = img[..., :3].mean(axis=-1)
    k = np.arange(-8, 9)
    cols, _ = cam.pixel_of_direction(np.column_stack([-k * spacing, 0 * k, np.ones_like(k)]))
    _, rows = cam.pixel_of_direction(np.column_stack([0 * k, k * spacing, np.ones_like(k)]))
    cols = cols[(cols > 1) & (cols < cam.width - 1)]
    rows = rows[(rows > 1) & (rows < cam.height - 1)]

    # Each line's brightness-weighted centre (pixel centres at i + 0.5, background removed,
    # window +-2 px) must lie within half a pixel of the predicted position. A line close to a
    # pixel border lights both pixels equally, so "brightest pixel" would be ambiguous.
    def centres(profile, predicted):
        background = np.median(profile)
        out = []
        for c in predicted:
            idx = np.arange(int(c) - 2, int(c) + 3)
            w = np.clip(profile[idx] - background, 0, None)
            out.append(np.sum(w * (idx + 0.5)) / np.sum(w))
        return np.array(out)

    # Profiles taken away from the other set of lines: rows/columns midway between lines.
    mid_rows = np.round(0.5 * (rows[:-1] + rows[1:])).astype(int)
    mid_cols = np.round(0.5 * (cols[:-1] + cols[1:])).astype(int)
    col_profile = lum[mid_rows].mean(axis=0)
    row_profile = lum[:, mid_cols].mean(axis=1)
    check("vertical lines at predicted columns (px)", cols, centres(col_profile, cols), atol=0.5)
    check("horizontal lines at predicted rows (px)", rows, centres(row_profile, rows), atol=0.5)
    # Straightness: in every row between horizontal lines, each vertical line's centre is the same.
    per_row = np.array([centres(lum[r], cols) for r in mid_rows])
    check(
        "vertical lines straight: centre spread over rows (px)",
        0.0,
        np.ptp(per_row, axis=0).max(),
        atol=0.05,
    )


def test_srgb_transfer():
    x = np.linspace(0, 1, 100_001)
    check("sRGB encode/decode round trip", x, srgb_decode(srgb_encode(x)), atol=1e-7)
    knee = 0.0031308
    lo, hi = srgb_encode(knee - 1e-12), srgb_encode(knee + 1e-12)
    check("sRGB continuous at the 0.0031308 knee", lo, hi, atol=1e-7)
    check("sRGB: mid-grey 0.5 encodes to 0.7354", 0.735357, srgb_encode(0.5), rtol=1e-5)


def test_tone_map_keeps_hue():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 0.3, size=(1000, 3))
    out = tone_map(rgb)
    ratio_in = rgb / rgb.sum(axis=1, keepdims=True)
    ratio_out = out / out.sum(axis=1, keepdims=True)
    check("tone map keeps colour ratios (in gamut)", ratio_in, ratio_out, atol=1e-12)
    bright = tone_map(np.array([[50.0, 10.0, 1.0]]))
    check("very bright pixel: no channel above 1", True, bool(bright.max() <= 1.0))
    check(
        "very bright pixel: order of channels kept",
        True,
        bool(bright[0, 0] >= bright[0, 1] >= bright[0, 2]),
    )


def test_dither_preserves_mean():
    flat = np.full((64, 64, 3), 0.5)
    img8 = to_uint8(flat)
    check("dithered mid-grey: mean level", 127.5, img8.mean(), atol=0.5)
    check("dithered mid-grey uses neighbouring levels only", {127, 128}, set(np.unique(img8)))
