"""Shows the comparison-table format (Step 0.6)."""

import numpy as np

from tests.helpers import check


def test_smoke(rng):
    check("smoke: 1 + 1", 2, 1 + 1)
    check("smoke: pi to 1e-12", np.pi, 4 * np.arctan(1.0), rtol=1e-12)
    check("smoke: seeded mean of 10^5 uniforms", 0.5, rng.uniform(size=100_000).mean(), atol=1e-2)
