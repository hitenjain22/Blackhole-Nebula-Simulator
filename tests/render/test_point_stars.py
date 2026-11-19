"""Pinpoint stars through the lens map (render/point_stars.py, physics/lensing.py;
DECISIONS 2026-11-19): photometry, position, smooth motion, lensed magnification against the
point-lens formula, and the Numba kernel against the plain-Python reference."""

import numpy as np

from nebula.physics.lensing import (
    point_lens_magnification,
    quad_solid_angle,
    spherical_triangle_solid_angle,
)
from nebula.render.camera import Camera
from nebula.render.point_stars import (
    build_star_index,
    splat_reference,
    splat_stars,
    star_grid_camera,
)
from tests.helpers import check

CAM = Camera(position=(100.0, 0.0, 0.0), vfov_deg=10.0, width=40, height=30)


def _flat_nodes(cam):
    """Node grid of an unlensed camera: directions through cam's pixel centres plus border."""
    g = star_grid_camera(cam)
    px, py = np.meshgrid(np.arange(g.width) + 0.5, np.arange(g.height) + 0.5)
    return g.directions(px, py), np.ones((g.height, g.width))


def _pixel_solid_angles(cam):
    """Sky solid angle of each pixel square (corners at integer pixel coordinates)."""
    px, py = np.meshgrid(np.arange(cam.width + 1.0), np.arange(cam.height + 1.0))
    d = cam.directions(px, py)
    return quad_solid_angle(d[:-1, :-1], d[:-1, 1:], d[1:, 1:], d[1:, :-1])


def _flux_and_centroid(img, omega_pix):
    lum = img[..., 1]
    tot = float((lum * omega_pix).sum())
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]] + 0.5
    return tot, float((lum * xx).sum() / lum.sum()), float((lum * yy).sum() / lum.sum())


def test_solid_angles():
    """An octant is 4 pi / 8; a small square patch is its flat area."""
    x, y, z = np.eye(3)
    check("octant solid angle", np.pi / 2, float(spherical_triangle_solid_angle(x, y, z)),
          rtol=1e-14)  # fmt: skip
    e = 1e-3
    sq = [np.array(v) / np.linalg.norm(v) for v in ([1, 0, 0], [1, e, 0], [1, e, e], [1, 0, e])]
    check("small square: area e^2", e * e, float(quad_solid_angle(*sq)), rtol=1e-5)


def test_grid_camera_shares_the_pixel_scale():
    """Node (j, i) of the grid camera looks along cam's pixel coordinates (i - 0.5, j - 0.5)."""
    g = star_grid_camera(CAM)
    i, j = np.array([0, 7, 41]), np.array([0, 12, 31])
    err = np.abs(g.directions(i + 0.5, j + 0.5) - CAM.directions(i - 0.5, j - 0.5)).max()
    check("grid camera node = pixel centre (direction error)", 0.0, float(err), atol=1e-14)


def test_one_star_flux_position_and_smooth_motion():
    """An unlensed star keeps its flux (sum of I x pixel solid angle) and lands where the
    camera projects it; moving it across pixels changes neither the flux nor the path's
    straightness (no twinkling)."""
    nodes, t = _flat_nodes(CAM)
    omega = _pixel_solid_angles(CAM)
    fluxes, errs = [], []
    for s in np.linspace(0.0, 1.0, 23):
        px, py = 11.3 + 6.7 * s, 9.1 + 4.2 * s
        d = CAM.directions(np.array([px]), np.array([py]))
        img = splat_reference(nodes, t, d, np.array([[1.0, 1.0, 1.0]]))
        f, cx, cy = _flux_and_centroid(img, omega)
        fluxes.append(f)
        errs.append(np.hypot(cx - px, cy - py))
    check("star flux kept while it moves (min)", 1.0, min(fluxes), rtol=2e-3)
    check("star flux kept while it moves (max)", 1.0, max(fluxes), rtol=2e-3)
    check("star position error (px)", 0.0, max(errs), atol=0.02)
    img = splat_reference(nodes, t, CAM.directions(np.array([20.5]), np.array([15.5])),
                          np.array([[1.0, 1.0, 1.0]]))  # fmt: skip
    lit = int((img[..., 1] > 1e-9 * img[..., 1].max()).sum())  # beyond rounding (w ~ 1e-16)
    check("star on a pixel centre: pixels lit", 1, lit)


def test_dimmed_by_gas_and_hidden_by_capture():
    nodes, t = _flat_nodes(CAM)
    d = CAM.directions(np.array([20.2]), np.array([14.7]))
    rgb = np.array([[1.0, 1.0, 1.0]])
    full = splat_reference(nodes, t, d, rgb)
    half = splat_reference(nodes, 0.5 * t, d, rgb)
    check("star through T = 0.5 gas", 0.5 * full, half, rtol=1e-12)
    tc = t.copy()
    tc[15, 21] = -1.0  # a captured node: its four quads draw nothing
    check("captured corner: star not drawn", 0.0, float(splat_reference(nodes, tc, d, rgb).max()))


def _point_lens_nodes(n, theta_e, pix):
    """Thin point lens, small angles: node at image angle theta sees the source plane at
    beta = theta - theta_E^2 theta / |theta|^2 (unit vectors (beta_x, beta_y, 1))."""
    k = (np.arange(n + 2) - (n + 2) / 2 + 0.5) * pix
    tx, ty = np.meshgrid(k, k)
    r2 = tx * tx + ty * ty
    bx, by = tx - theta_e**2 * tx / r2, ty - theta_e**2 * ty / r2
    d = np.stack([bx, by, np.ones_like(bx)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    e = (np.arange(n + 1) - n / 2) * pix  # pixel corners (image plane)
    cx, cy = np.meshgrid(e, e)
    c = np.stack([cx, cy, np.ones_like(cx)], axis=-1)
    c /= np.linalg.norm(c, axis=-1, keepdims=True)
    omega_pix = quad_solid_angle(c[:-1, :-1], c[:-1, 1:], c[1:, 1:], c[1:, :-1])
    return d, np.ones(d.shape[:2]), omega_pix


def test_lensed_star_brightens_as_the_point_lens_formula():
    """A star behind a thin point lens: both images together carry mu(u) F with
    mu = (u^2 + 2) / (u sqrt(u^2 + 4)) (Paczynski 1986), including the flipped inner image."""
    theta_e, n = 0.01, 600
    # Einstein radius 100 px, field +-3 theta_E (both images in view). The bilinear lens map
    # inside each pixel limits the photometry: 0.1-0.6% at >= 100 px per theta_E, up to 3%
    # at 67 px (AI_LOG 2026-11-19); the video's hole spans hundreds of pixels
    pix = 6.0 * theta_e / n
    d, t, omega_pix = _point_lens_nodes(n, theta_e, pix)
    for u in (0.5, 1.0, 2.0):
        star = np.array([[u * theta_e, 0.3 * u * theta_e, 1.0]])
        u_true = float(np.hypot(star[0, 0], star[0, 1]) / theta_e)
        star /= np.linalg.norm(star)
        img = splat_stars(d, t, build_star_index(star, np.array([[1.0, 1.0, 1.0]])))
        mu = float((img[..., 1] * omega_pix).sum())
        check(f"point lens u = {u_true:.3f}: total magnification", point_lens_magnification(u_true),
              mu, rtol=1e-2)  # fmt: skip


def test_fast_kernel_matches_reference():
    """Numba kernel (sorted cell index) = plain-Python reference on a lensed field of 600
    stars, both images, captured and twisted regions included."""
    theta_e, n = 0.01, 60
    pix = 4.0 * theta_e / n
    d, t, _ = _point_lens_nodes(n, theta_e, pix)
    t = t * np.linspace(0.3, 1.0, t.size).reshape(t.shape)
    t[30, 30:33] = -1.0
    rng = np.random.default_rng(7)
    sd = np.column_stack([rng.uniform(-0.03, 0.03, (600, 2)), np.ones(600)])
    sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    rgb = rng.uniform(0.1, 1.0, (600, 3))
    ref = splat_reference(d, t, sd, rgb)
    fast = splat_stars(d, t, build_star_index(sd, rgb))
    check("point stars: pixels lit (reference)", True, bool((ref > 0).any(axis=-1).sum() > 200))
    check("point stars: Numba = reference", ref, fast, rtol=1e-10, atol=1e-12 * ref.max())
