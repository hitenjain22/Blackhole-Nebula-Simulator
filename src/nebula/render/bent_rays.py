"""CPU reference for marching gas along bent light rays (phases/phase-08-gas-rendering.md 8.4).

The readable float64 twin of shaders/gas_march.wgsl (CLAUDE.md rule 2; 8.6 tests 6-7).

Ray geometry (Phase 2, physics/geodesics.py): a ray stays in the plane through the hole
spanned by e1 = camera position / |camera position| and e2 = the part of the viewing direction
perpendicular to e1. With u = 1/r and phi the angle from e1 in that plane, d^2u/dphi^2 = 3u^2 - u.
It is integrated with exactly the RK4 scheme of shaders/schw_ray.wgsl (same step rule, steps on
a 2^-12 grid), here in float64. Each RK4 step joins two points r (cos phi e1 + sin phi e2) by a
straight chord (chords are <= 0.02 rad of arc, shorter near the photon sphere), and the chords
are marched by render.march.march_segment with the static observer's proper length
(physics.geodesics.schw_static_proper_length).

A ray ends:
- captured: u reaches 1/2 (r = 2); the last chord ends exactly at r = 2 (linear in phi);
- escaped: going outward past the grid's r_max; the last chord ends exactly at r_max;
- orbiting: still circling after PHI_CAP (treated as captured, as in Phase 2);
- or early, once the transmission falls below march.T_STOP (opaque).
Exactly radial rays (|sin psi| < 1e-7) are a single straight segment to r = 2 or r_max.
"""

import numpy as np

from nebula.render.camera import Camera, sub_pixel_offsets
from nebula.render.march import T_STOP, march_segment

RAY_ESCAPED, RAY_CAPTURED, RAY_ORBITING = 0, 1, 2  # as render.rays / schw_ray.wgsl
PHI_CAP = 8.0 * np.pi
H_GRID = 2.0**-12
U_HORIZON = 0.5
RADIAL_EPS = 1e-7


def step_size(u: float) -> float:
    """RK4 step in phi: min(0.02, 0.1 |u - 1/3| + 0.002) floored to the 2^-12 grid.
    Mirrors ray_step_size in shaders/schw_ray.wgsl."""
    h = min(0.02, 0.1 * abs(u - 1.0 / 3.0) + 0.002)
    return max(np.floor(h / H_GRID), 1.0) * H_GRID


def _accel(u):
    return 3.0 * u * u - u


def _rk4(u, w, h):
    k1u, k1w = w, _accel(u)
    k2u, k2w = w + 0.5 * h * k1w, _accel(u + 0.5 * h * k1u)
    k3u, k3w = w + 0.5 * h * k2w, _accel(u + 0.5 * h * k2u)
    k4u, k4w = w + h * k3w, _accel(u + h * k3u)
    return (
        u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u),
        w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w),
    )


def ray_basis(cam_pos, d):
    """(e1, e2, cos_psi, sin_psi) of the ray leaving cam_pos in unit direction d; psi is the
    angle from the direction toward the hole. e2 is None for an exactly radial ray."""
    cam_pos = np.asarray(cam_pos, float)
    e1 = cam_pos / np.sqrt(cam_pos @ cam_pos)
    cos_psi = -float(d @ e1)
    perp = d + cos_psi * e1
    sin_psi = float(np.sqrt(perp @ perp))
    if sin_psi < RADIAL_EPS:
        return e1, None, cos_psi, sin_psi
    return e1, perp / sin_psi, cos_psi, sin_psi


def ray_polyline(cam_pos, d, r_max: float):
    """Points along the bent ray from the camera (front to back) and its status (module doc).

    Returns (points (n, 3), status). Consecutive points are RK4 steps (or the end points).
    """
    cam_pos = np.asarray(cam_pos, float)
    d = np.asarray(d, float)
    r_cam = float(np.sqrt(cam_pos @ cam_pos))
    e1, e2, cos_psi, sin_psi = ray_basis(cam_pos, d)
    if e2 is None:
        if cos_psi > 0.0:  # straight at the hole
            return np.array([cam_pos, 2.0 * e1]), RAY_CAPTURED
        return np.array([cam_pos, max(r_max, r_cam) * e1]), RAY_ESCAPED
    u_out = 1.0 / r_max
    u = 1.0 / r_cam
    w = u * np.sqrt(1.0 - 2.0 * u) * cos_psi / sin_psi  # ray_initial_slope
    phi = 0.0
    pts = [cam_pos]

    def point(r, ph):
        return r * (np.cos(ph) * e1 + np.sin(ph) * e2)

    while phi < PHI_CAP:
        h = step_size(u)
        u1, w1 = _rk4(u, w, h)
        if u1 >= U_HORIZON:
            t = (U_HORIZON - u) / (u1 - u)
            pts.append(point(2.0, phi + t * h))
            return np.array(pts), RAY_CAPTURED
        if u1 < u_out and u1 < u:  # leaving the grid outward
            t = min(max((u - u_out) / (u - u1), 0.0), 1.0)
            if t > 0.0:
                pts.append(point(r_max, phi + t * h))
            return np.array(pts), RAY_ESCAPED
        phi += h
        u, w = u1, w1
        pts.append(point(1.0 / u, phi))
    return np.array(pts), RAY_ORBITING


def march_bent_ray(textures, model, cam_pos, d):
    """(I, T, status, n_substeps) for one bent ray through the textures (march.sample
    convention: log grid first, then boxes). Proper path lengths throughout."""
    r_max = textures[0].grid.r_max
    pts, status = ray_polyline(cam_pos, d, r_max)
    intensity, transmission, n = 0.0, 1.0, 0
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        intensity, transmission, k = march_segment(
            textures, model, a, b, intensity, transmission, curved=True
        )
        n += k
        if transmission < T_STOP:
            break
    return intensity, transmission, status, n


def render_reference(camera: Camera, textures, model, n_sub: int = 1, pixels=None, sub_map=None):
    """CPU reference image: per pixel (mean I, mean T of escaped sub-rays (0 if captured),
    captured fraction, mean sub-steps), as the GPU shader writes them. ``pixels``: optional
    list of (column, row) to render (others are NaN). ``sub_map``: per-pixel sub-ray counts
    for photon-ring pixels (render/ring_samples.py), as the shader. Slow: for tests."""
    offs = sub_pixel_offsets(n_sub)
    out = np.full((camera.height, camera.width, 4), np.nan)
    todo = pixels if pixels is not None else np.ndindex(camera.width, camera.height)
    for i, j in todo:
        acc = np.zeros(4)
        for ox, oy in _offsets(offs, n_sub, sub_map, i, j):
            d = camera.directions(i + ox, j + oy)
            intensity, transmission, status, n = march_bent_ray(textures, model, camera.position, d)
            escaped = status == RAY_ESCAPED
            acc += (intensity, transmission if escaped else 0.0, 0.0 if escaped else 1.0, n)
        out[j, i] = acc / len(_offsets(offs, n_sub, sub_map, i, j))
    return out


def _offsets(offs, n_sub, sub_map, i, j):
    """The sub-ray pattern of pixel (i, j): the ring map's m x m if m > n_sub (gas_march.wgsl,
    capped at its MAX_RING_SUB = 8), else the n_sub x n_sub one."""
    if sub_map is not None and min(int(sub_map[j, i]), 8) > n_sub:
        return sub_pixel_offsets(min(int(sub_map[j, i]), 8))
    return offs


def render_reference_colour(
    camera: Camera, textures, model, n_sub: int = 1, pixels=None, sub_map=None
):
    """CPU reference image for a Phase 9 model (march.RelativisticBlackbody): per pixel the mean
    over sub-rays of (X, Y, Z, bolometric) followed by (mean T of escaped sub-rays, captured
    fraction, mean sub-steps); shape (height, width, model.channels + 3). Slow: for tests."""
    offs = sub_pixel_offsets(n_sub)
    nc = model.channels
    out = np.full((camera.height, camera.width, nc + 3), np.nan)
    todo = pixels if pixels is not None else np.ndindex(camera.width, camera.height)
    for i, j in todo:
        acc = np.zeros(nc + 3)
        offs_ij = _offsets(offs, n_sub, sub_map, i, j)
        for ox, oy in offs_ij:
            d = camera.directions(i + ox, j + oy)
            intensity, transmission, status, n = march_bent_ray(textures, model, camera.position, d)
            escaped = status == RAY_ESCAPED
            acc[:nc] += intensity
            acc[nc:] += (transmission if escaped else 0.0, 0.0 if escaped else 1.0, n)
        out[j, i] = acc / len(offs_ij)
    return out
