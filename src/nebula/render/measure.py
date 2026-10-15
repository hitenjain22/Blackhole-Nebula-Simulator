"""Measuring rendered images against exact numbers (phases/phase-02-blackhole-render.md
Step 2.5): shadow radius from the captured-fraction channel, ring radii from brightness.

Pixel convention as in camera.py: continuous coordinates, pixel (i, j) centred at
(i + 0.5, j + 0.5). For a pinhole camera looking at the hole, a sky circle of angular radius
psi around the view axis appears as an image circle of radius f tan(psi), with focal length
f = (height / 2) / tan(vfov / 2) in pixels.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from nebula.render.camera import Camera


def focal_length_px(camera: Camera) -> float:
    return (camera.height / 2) / camera.tan_half_vfov


def angle_to_radius_px(camera: Camera, angle_rad: float) -> float:
    """Image radius (pixels from the centre) of a circle of angular radius ``angle_rad``."""
    return focal_length_px(camera) * np.tan(angle_rad)


def radius_px_to_angle(camera: Camera, radius_px: float) -> float:
    return float(np.arctan(radius_px / focal_length_px(camera)))


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample at continuous pixel coordinates (centres at i + 0.5)."""
    return map_coordinates(img, [y - 0.5, x - 0.5], order=1, mode="nearest")


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle through points (algebraic fit, Kasa 1976): (xc, yc, radius)."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (c0, c1, c2), *_ = np.linalg.lstsq(a, b, rcond=None)
    xc, yc = c0 / 2, c1 / 2
    return float(xc), float(yc), float(np.sqrt(c2 + xc * xc + yc * yc))


def shadow_edge(
    captured: np.ndarray, centre: tuple[float, float], r_max: float, n_angles: int = 720
) -> tuple[np.ndarray, np.ndarray]:
    """Points where the captured fraction falls through 0.5, going outward along n_angles
    rays from ``centre``: the shadow's edge, to sub-pixel precision (linear interpolation)."""
    cx, cy = centre
    r = np.arange(0.0, r_max, 0.02)
    xs, ys = [], []
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        x, y = cx + r * np.cos(th), cy + r * np.sin(th)
        a = _sample(captured, x, y)
        k = int(np.argmax(a < 0.5))  # first sample outside
        if k == 0:
            continue
        t = (a[k - 1] - 0.5) / (a[k - 1] - a[k])
        rr = r[k - 1] + t * (r[k] - r[k - 1])
        xs.append(cx + rr * np.cos(th))
        ys.append(cy + rr * np.sin(th))
    return np.array(xs), np.array(ys)


def radial_profile(
    img: np.ndarray,
    centre: tuple[float, float],
    r_max: float,
    dr: float = 0.05,
    n_angles: int = 1440,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged brightness against radius (pixels) around ``centre``."""
    cx, cy = centre
    r = np.arange(0.0, r_max, dr)
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    x = cx + r[None, :] * np.cos(th)[:, None]
    y = cy + r[None, :] * np.sin(th)[:, None]
    return r, _sample(img, x.ravel(), y.ravel()).reshape(x.shape).mean(axis=0)


def peak_radius(r: np.ndarray, profile: np.ndarray, r_lo: float, r_hi: float) -> float:
    """Radius of the brightest point of the profile between r_lo and r_hi, refined by a
    parabola through the three samples around the maximum."""
    sel = np.where((r >= r_lo) & (r <= r_hi))[0]
    k = sel[np.argmax(profile[sel])]
    a, b, _ = np.polyfit(r[k - 1 : k + 2] - r[k], profile[k - 1 : k + 2], 2)
    return float(r[k] - b / (2 * a)) if a < 0 else float(r[k])
