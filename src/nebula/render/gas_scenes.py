"""Synthetic gas scenes with exact answers for the Phase 8 marcher tests
(phases/phase-08-gas-rendering.md 8.6): shared by the tests in tests/render and the
scripts/phase8_step*_test_images.py check pictures."""

import numpy as np

from nebula.render.gasgrid import CartesianBox, LogSphericalGrid
from nebula.render.march import texture_from_fields


def box_centres(b: CartesianBox):
    """Cell-centre coordinates (X, Y, Z) of a box, each shaped b.shape."""
    lo = b.lo
    axes = [lo[d] + (np.arange(b.n) + 0.5) * b.dx for d in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def sphere_volume_fraction(b: CartesianBox, radius: float, ns: int = 8) -> np.ndarray:
    """Fraction of each cell's volume inside a sphere of the given radius centred on the box
    centre (ns^3 sub-samples in cells near the surface). This is the smooth edge that
    deposition gives, so a test measures the marcher rather than a staircase sphere."""
    x, y, z = (c - b.centre[d] for d, c in enumerate(box_centres(b)))
    r = np.sqrt(x**2 + y**2 + z**2)
    f = (r < radius).astype(float)
    edge = np.abs(r - radius) < 0.9 * b.dx
    o = (np.arange(ns) + 0.5) / ns - 0.5
    ox, oy, oz = (a.ravel() * b.dx for a in np.meshgrid(o, o, o, indexing="ij"))
    xe, ye, ze = x[edge][:, None] + ox, y[edge][:, None] + oy, z[edge][:, None] + oz
    f[edge] = ((xe**2 + ye**2 + ze**2) < radius**2).mean(1)
    return f


def point_source_textures(r_source: float, radius: float = 0.3, n: int = 64):
    """8.6 test 6 scene: an empty log grid plus a detail box holding a small glowing sphere
    (radius ``radius``, rho 1e-3, T 1e4 K) on the -x axis at r = r_source, i.e. directly
    behind the hole for a camera on the +x axis. Returns the march texture list."""
    g = LogSphericalGrid()
    empty = texture_from_fields(g, np.zeros(g.shape), np.zeros(g.shape))
    b = CartesianBox(centre=(-r_source, 0.0, 0.0), size=4.0 * radius, n=n)
    ball = texture_from_fields(b, 1e-3 * sphere_volume_fraction(b, radius), np.full(b.shape, 1e4))
    return [empty, ball]


def ring_angle_deg(camera, image, psi_min_deg: float, psi_max_deg: float) -> float:
    """Brightness-weighted mean angle (deg) from the direction to the hole (the camera sits on
    the +x axis) of the pixels whose centre lies between psi_min and psi_max."""
    h, w = image.shape
    px, py = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    to_hole = -np.asarray(camera.position, float) / np.linalg.norm(camera.position)
    psi = np.degrees(np.arccos(np.clip(camera.directions(px, py) @ to_hole, -1.0, 1.0)))
    sel = (psi > psi_min_deg) & (psi < psi_max_deg)
    return float((image[sel] * psi[sel]).sum() / image[sel].sum())
