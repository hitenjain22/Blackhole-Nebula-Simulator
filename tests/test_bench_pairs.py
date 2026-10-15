"""Step 0.3: the Numba neighbour sum must equal the plain NumPy reference."""

import numpy as np

from nebula._bench.pairs import pair_sum_numba, pair_sum_reference, uniform_cube
from tests.helpers import check


def test_numba_matches_reference():
    x = uniform_cube(2_000, neighbours=60, rng=np.random.default_rng(12345))
    check("pair sum, Numba vs NumPy (N=2000)", pair_sum_reference(x), pair_sum_numba(x), rtol=1e-12)
