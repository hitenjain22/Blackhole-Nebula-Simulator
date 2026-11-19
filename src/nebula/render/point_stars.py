"""Pinpoint stars through the lens map (final render; DECISIONS 2026-11-19).

The Phase 2 sky drew stars as 1.5-texel Gaussians in a cube map: at 1080p they were 5-6 px
soft discs, and near the hole the lensed texture smeared into arcs. Here every star is a point:

1. A node grid of rays through the pixel centres plus a one-pixel border (star_grid_camera:
   the same pixel scale, (W + 2) x (H + 2) rays) gives, per node, the sky direction the ray
   escapes to and the gas transmission T along it (GPU mode OUT_SKY_DIR_T; T < 0: captured).
2. Neighbouring nodes span a quad: the patch of sky between four pixel centres. A star inside
   it (inverse bilinear coordinates u, v on the gnomonic plane at the quad's centre) gives the
   four pixels at its corners the radiance F T w / Omega_sky (physics.lensing.
   point_source_radiance; w the bilinear weights, T bilinear in the corners' T): each image
   keeps its lensed flux mu F exactly, and moves smoothly between pixels (no twinkling).
   Images flipped by lensing (secondary images) work the same way; twisted quads (on a
   critical curve), quads with a captured corner, and quads wider than ``max_quad_angle``
   (the photon ring, where a pixel sees a large part of the sky) get no stars.

``splat_reference`` is the plain-Python version (all stars tested against every quad); the
Numba kernel ``_splat_kernel`` is the fast copy (a sorted 3D cell index of the stars), tested
equal to it (CLAUDE.md rule 2).
"""

from dataclasses import dataclass

import numba
import numpy as np

from nebula.physics.lensing import point_source_radiance, quad_solid_angle
from nebula.render.camera import Camera

MAX_QUAD_ANGLE = np.radians(2.0)  # quads wider than this (photon ring) get no stars
CELL = 0.004  # star index cell (chord units on the unit sphere; 0.23 deg)


def star_grid_camera(cam: Camera) -> Camera:
    """A (W + 2) x (H + 2) camera whose pixel centres are cam's pixel centres plus a one-pixel
    border: node (j, i) looks along cam's continuous pixel coordinates (i - 0.5, j - 0.5)."""
    t = cam.tan_half_vfov * (cam.height + 2) / cam.height
    return Camera(position=cam.position, look_at=cam.look_at, up=cam.up,
                  vfov_deg=float(np.degrees(2.0 * np.arctan(t))), width=cam.width + 2,
                  height=cam.height + 2)  # fmt: skip


def _inverse_bilinear(p, a, b, c, d):
    """(u, v) with a + u (b - a) + v (d - a) + u v (a - b + c - d) = p (2D), the solution in
    [0, 1)^2 or None (Quilez, 'Inverse bilinear interpolation', 2010)."""

    def cross(x, y):
        return x[0] * y[1] - x[1] * y[0]

    e, f, g, h = b - a, d - a, a - b + c - d, p - a
    k2, k1, k0 = cross(g, f), cross(e, f) + cross(h, g), cross(h, e)
    if abs(k2) < 1e-14 * max(abs(k1), 1e-300):
        vs = [-k0 / k1] if k1 != 0 else []
    else:
        disc = k1 * k1 - 4.0 * k0 * k2
        if disc < 0:
            return None
        w = np.sqrt(disc)
        vs = [(-k1 - w) / (2.0 * k2), (-k1 + w) / (2.0 * k2)]
    for v in vs:
        den = e + g * v
        if den[0] == 0.0 and den[1] == 0.0:
            continue  # degenerate root: u undefined
        ax = 0 if abs(den[0]) >= abs(den[1]) else 1
        u = (h[ax] - f[ax] * v) / den[ax]
        if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
            return u, v
    return None


def _quad_frame(corners):
    """Centre direction, angular radius, and the gnomonic projection of the corners."""
    c = corners.sum(axis=0)
    c /= np.linalg.norm(c)
    radius = float(np.max(np.arccos(np.clip(corners @ c, -1.0, 1.0))))
    e1 = np.cross(c, [0.0, 0.0, 1.0] if abs(c[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    q = np.column_stack([corners @ e1, corners @ e2]) / (corners @ c)[:, None]
    return c, radius, e1, e2, q


def _twisted(q):
    """True if the projected quad's two triangles (split along 0-2) have opposite orientation."""

    def area(x, y, z):
        return (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])

    return area(q[0], q[1], q[2]) * area(q[0], q[2], q[3]) <= 0.0


def splat_reference(node_dir, node_t, star_dir, star_rgb, max_quad_angle=MAX_QUAD_ANGLE):
    """Plain-Python point stars: node_dir (H+2, W+2, 3) escape directions, node_t (H+2, W+2)
    transmissions (< 0: captured), stars (n, 3) unit directions and (n, 3) RGB flux.
    Returns (H, W, 3) radiance [flux / sr], already times the gas transmission."""
    hn, wn = node_t.shape
    h, w = hn - 2, wn - 2
    out = np.zeros((h, w, 3))
    for row in range(hn - 1):
        for k in range(wn - 1):
            idx = [(row, k), (row, k + 1), (row + 1, k + 1), (row + 1, k)]  # u: k, v: rows
            t4 = np.array([node_t[i] for i in idx])
            if np.any(t4 < 0.0):
                continue
            corners = np.array([node_dir[i] for i in idx], float)
            corners /= np.linalg.norm(corners, axis=1, keepdims=True)
            c, radius, e1, e2, q = _quad_frame(corners)
            if radius > max_quad_angle or _twisted(q):
                continue
            omega = float(quad_solid_angle(*corners))
            if omega <= 0.0:
                continue
            for s in range(len(star_dir)):
                ds = star_dir[s]
                z = ds @ c
                if z <= np.cos(radius) - 1e-12:
                    continue
                p = np.array([ds @ e1, ds @ e2]) / z
                uv = _inverse_bilinear(p, q[0], q[1], q[2], q[3])
                if uv is None:
                    continue
                u, v = uv
                wts = [(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v]
                tt = sum(wi * ti for wi, ti in zip(wts, t4, strict=True))
                rad = point_source_radiance(star_rgb[s] * tt, omega)
                for (nl, nk), wi in zip(idx, wts, strict=True):
                    r, col = nl - 1, nk - 1  # node (nl, nk) is pixel (nl - 1, nk - 1)
                    if 0 <= r < h and 0 <= col < w:
                        out[r, col] += wi * rad
    return out


@dataclass
class StarIndex:
    """Stars sorted by a 3D cell key (cells of side CELL over [-1, 1]^3)."""

    direction: np.ndarray  # (n, 3) float64, sorted
    rgb: np.ndarray  # (n, 3) float64, sorted
    keys: np.ndarray  # (n,) int64, ascending
    n_cell: int
    cell: float


def build_star_index(star_dir, star_rgb, cell: float = CELL) -> StarIndex:
    d = np.asarray(star_dir, float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    n_cell = int(np.ceil(2.0 / cell)) + 1
    ijk = np.floor((d + 1.0) / cell).astype(np.int64)
    keys = (ijk[:, 0] * n_cell + ijk[:, 1]) * n_cell + ijk[:, 2]
    order = np.argsort(keys, kind="stable")
    return StarIndex(d[order], np.asarray(star_rgb, float)[order], keys[order], n_cell, cell)


@numba.njit(cache=True)
def _tri_omega(a, b, c):
    # mirrors physics.lensing.spherical_triangle_solid_angle (Van Oosterom & Strackee 1983)
    cx = b[1] * c[2] - b[2] * c[1]
    cy = b[2] * c[0] - b[0] * c[2]
    cz = b[0] * c[1] - b[1] * c[0]
    num = abs(a[0] * cx + a[1] * cy + a[2] * cz)
    den = 1.0 + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) + (b[0] * c[0] + b[1] * c[1]
          + b[2] * c[2]) + (c[0] * a[0] + c[1] * a[1] + c[2] * a[2])  # fmt: skip
    return 2.0 * np.arctan2(num, den)


@numba.njit(cache=True)
def _cross2(x0, x1, y0, y1):
    return x0 * y1 - x1 * y0


@numba.njit(cache=True, error_model="numpy")
def _splat_quad(row, k, node_dir, node_t, sdir, srgb, keys, n_cell, cell, max_angle, out):
    """One quad of splat_reference (same steps, same order of tests)."""
    hn, wn = node_t.shape
    h, w = hn - 2, wn - 2
    il = (row, row, row + 1, row + 1)
    ik = (k, k + 1, k + 1, k)
    cs = np.empty((4, 3))
    t4 = np.empty(4)
    for m in range(4):
        t4[m] = node_t[il[m], ik[m]]
        if t4[m] < 0.0:
            return
        nrm = np.sqrt(node_dir[il[m], ik[m], 0] ** 2 + node_dir[il[m], ik[m], 1] ** 2
                      + node_dir[il[m], ik[m], 2] ** 2)  # fmt: skip
        for j in range(3):
            cs[m, j] = node_dir[il[m], ik[m], j] / nrm
    c = cs[0] + cs[1] + cs[2] + cs[3]
    c /= np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
    radius = 0.0
    for m in range(4):
        dot = min(max(cs[m, 0] * c[0] + cs[m, 1] * c[1] + cs[m, 2] * c[2], -1.0), 1.0)
        radius = max(radius, np.arccos(dot))
    if radius > max_angle:
        return
    if abs(c[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.array([c[1] * ref[2] - c[2] * ref[1], c[2] * ref[0] - c[0] * ref[2],
                   c[0] * ref[1] - c[1] * ref[0]])  # fmt: skip
    e1 /= np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
    e2 = np.array([c[1] * e1[2] - c[2] * e1[1], c[2] * e1[0] - c[0] * e1[2],
                   c[0] * e1[1] - c[1] * e1[0]])  # fmt: skip
    q = np.empty((4, 2))
    for m in range(4):
        z = cs[m, 0] * c[0] + cs[m, 1] * c[1] + cs[m, 2] * c[2]
        q[m, 0] = (cs[m, 0] * e1[0] + cs[m, 1] * e1[1] + cs[m, 2] * e1[2]) / z
        q[m, 1] = (cs[m, 0] * e2[0] + cs[m, 1] * e2[1] + cs[m, 2] * e2[2]) / z
    a1 = (q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[1, 1] - q[0, 1]) * (q[2, 0] - q[0, 0])
    a2 = (q[2, 0] - q[0, 0]) * (q[3, 1] - q[0, 1]) - (q[2, 1] - q[0, 1]) * (q[3, 0] - q[0, 0])
    if a1 * a2 <= 0.0:
        return
    # mirrors physics.lensing.quad_solid_angle
    omega = _tri_omega(cs[0], cs[1], cs[2]) + _tri_omega(cs[0], cs[2], cs[3])
    if omega <= 0.0:
        return
    cos_r = np.cos(radius) - 1e-12
    chord = 2.0 * np.sin(0.5 * radius) + 1e-9
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    for j in range(3):
        lo[j] = max(int(np.floor((c[j] - chord + 1.0) / cell)), 0)
        hi[j] = min(int(np.floor((c[j] + chord + 1.0) / cell)), n_cell - 1)
    ex, ey = q[1, 0] - q[0, 0], q[1, 1] - q[0, 1]
    fx, fy = q[3, 0] - q[0, 0], q[3, 1] - q[0, 1]
    gx = q[0, 0] - q[1, 0] + q[2, 0] - q[3, 0]
    gy = q[0, 1] - q[1, 1] + q[2, 1] - q[3, 1]
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            base = (ix * n_cell + iy) * n_cell
            s0 = np.searchsorted(keys, base + lo[2])
            s1 = np.searchsorted(keys, base + hi[2] + 1)
            for s in range(s0, s1):
                z = sdir[s, 0] * c[0] + sdir[s, 1] * c[1] + sdir[s, 2] * c[2]
                if z <= cos_r:
                    continue
                px = (sdir[s, 0] * e1[0] + sdir[s, 1] * e1[1] + sdir[s, 2] * e1[2]) / z
                py = (sdir[s, 0] * e2[0] + sdir[s, 1] * e2[1] + sdir[s, 2] * e2[2]) / z
                hx, hy = px - q[0, 0], py - q[0, 1]
                k2 = _cross2(gx, gy, fx, fy)
                k1 = _cross2(ex, ey, fx, fy) + _cross2(hx, hy, gx, gy)
                k0 = _cross2(hx, hy, ex, ey)
                nv = 0
                v_a = 0.0
                v_b = 0.0
                if abs(k2) < 1e-14 * max(abs(k1), 1e-300):
                    if k1 != 0.0:
                        v_a = -k0 / k1
                        nv = 1
                else:
                    disc = k1 * k1 - 4.0 * k0 * k2
                    if disc >= 0.0:
                        sq = np.sqrt(disc)
                        v_a = (-k1 - sq) / (2.0 * k2)
                        v_b = (-k1 + sq) / (2.0 * k2)
                        nv = 2
                found = False
                u = 0.0
                v = 0.0
                for which in range(nv):
                    vv = v_a if which == 0 else v_b
                    dx, dy = ex + gx * vv, ey + gy * vv
                    if dx == 0.0 and dy == 0.0:
                        continue  # degenerate root (u undefined); the reference gets inf/nan
                    if abs(dx) >= abs(dy):
                        uu = (hx - fx * vv) / dx
                    else:
                        uu = (hy - fy * vv) / dy
                    if 0.0 <= uu < 1.0 and 0.0 <= vv < 1.0:
                        u, v, found = uu, vv, True
                        break
                if not found:
                    continue
                w0, w1, w2, w3 = (1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v
                tt = w0 * t4[0] + w1 * t4[1] + w2 * t4[2] + w3 * t4[3]
                for m in range(4):
                    wm = (w0, w1, w2, w3)[m]
                    r, col = il[m] - 1, ik[m] - 1
                    if 0 <= r < h and 0 <= col < w:
                        for j in range(3):
                            # mirrors physics.lensing.point_source_radiance: F T / Omega_sky
                            out[r, col, j] += wm * srgb[s, j] * tt / omega


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _splat_kernel(node_dir, node_t, sdir, srgb, keys, n_cell, cell, max_angle, out):
    """All quads; quad rows of one parity at a time so parallel rows never write the same
    pixel row (quad row j writes pixel rows j - 1 and j)."""
    hn, wn = node_t.shape
    for parity in range(2):
        n_rows = (hn - 1 - parity + 1) // 2
        for jj in numba.prange(n_rows):
            row = 2 * jj + parity
            for k in range(wn - 1):
                _splat_quad(row, k, node_dir, node_t, sdir, srgb, keys, n_cell, cell, max_angle,
                            out)  # fmt: skip


def splat_stars(node_dir, node_t, index: StarIndex, max_quad_angle=MAX_QUAD_ANGLE):
    """Fast point stars (same result as splat_reference): (H, W, 3) radiance [flux / sr]."""
    node_dir = np.ascontiguousarray(node_dir, dtype=np.float64)
    node_t = np.ascontiguousarray(node_t, dtype=np.float64)
    out = np.zeros((node_t.shape[0] - 2, node_t.shape[1] - 2, 3))
    _splat_kernel(node_dir, node_t, index.direction, index.rgb, index.keys, index.n_cell,
                  index.cell, float(max_quad_angle), out)  # fmt: skip
    return out
