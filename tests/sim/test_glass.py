"""Step 3.4: the stored glass and tiling (phases/phase-03-sph-core.md 3.2 "Initial conditions":
density noise < 1%)."""

import numpy as np

from nebula.sim.glass import density_noise, load_glass, tile_glass
from nebula.sim.ic import smoothing_length_uniform
from nebula.sim.sph import compute_density
from tests.helpers import check


def test_stored_glass_is_smooth():
    glass = load_glass(4096)
    check("glass: particle count", 4096, len(glass))
    check("glass: inside the unit box", True, bool(glass.min() >= 0 and glass.max() < 1))
    check("glass: density noise < 1%", 0.0, density_noise(glass), atol=0.01)


def test_tiled_glass_is_seamless():
    """Two copies side by side in a periodic (2, 1, 1) box: still < 1% noise (no seam)."""
    glass = load_glass(4096)
    pos = tile_glass(glass, (2, 1, 1), (2.0, 1.0, 1.0))
    n = len(pos)
    check("tiled glass: count", 8192, n)
    h0 = np.full(n, smoothing_length_uniform(n, 2.0))
    d = compute_density(pos, np.full(n, 2.0 / n), h0, box=np.array([2.0, 1.0, 1.0]))
    check("tiled glass: mean density 1", 1.0, d.rho.mean(), rtol=5e-3)
    check("tiled glass: density noise < 1%", 0.0, d.rho.std() / d.rho.mean(), atol=0.01)
    seam = np.abs(pos[:, 0] - 1.0) < 0.05
    check("tiled glass: density at the seam (mean)", 1.0, d.rho[seam].mean(), rtol=1e-2)
