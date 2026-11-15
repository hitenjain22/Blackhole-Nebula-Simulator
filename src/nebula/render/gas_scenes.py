"""Synthetic gas scenes with exact answers for the Phase 8 marcher tests
(phases/phase-08-gas-rendering.md 8.6): shared by tests/render/test_march.py and
scripts/phase8_step2_test_images.py."""

import numpy as np

from nebula.render.gasgrid import CartesianBox


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
