"""Initial particle layouts for SPH tests (phases/phase-03-sph-core.md 3.2).

Lattices here; the relaxed "glass" (needs SPH pressure forces) is built in sim/glass.py.
All layouts fill a box [0, L)^3 with equal-mass particles, offset by half a spacing so no
particle sits on a box face.
"""

import numpy as np


def cubic_lattice(n_side: int, box: float = 1.0) -> np.ndarray:
    """n_side^3 particles on a simple cubic lattice of spacing box / n_side."""
    s = box / n_side
    g = (np.arange(n_side) + 0.5) * s
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def fcc_lattice(n_cells: int, box: float = 1.0) -> np.ndarray:
    """4 n_cells^3 particles on a face-centred cubic (close-packed) lattice, cubic cell of
    side box / n_cells; periodic in the box."""
    a = box / n_cells
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    c = np.arange(n_cells)
    cells = np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)
    pos = (cells[:, None, :] + basis[None, :, :] + 0.25).reshape(-1, 3) * a
    return pos


def smoothing_length_uniform(n_particles: int, volume: float, eta: float = 1.6) -> float:
    """h = eta (m / rho)^(1/3) = eta (V / N)^(1/3) for a uniform distribution (default eta is
    physics.kernels.DEFAULT_ETA, the Wendland C4 value)."""
    return eta * (volume / n_particles) ** (1.0 / 3.0)
