"""Step 0.3: the Numba neighbour sum must equal the plain NumPy reference."""

import numpy as np

from nebula._bench.pairs import pair_sum_numba, pair_sum_reference, uniform_cube


def test_numba_matches_reference():
    x = uniform_cube(2_000, neighbours=60, rng=np.random.default_rng(12345))
    expected = pair_sum_reference(x)
    actual = pair_sum_numba(x)
    rel = np.max(np.abs(actual - expected) / np.abs(expected))
    verdict = "PASS" if rel < 1e-12 else "FAIL"
    print(f"\npair sum, N=2000 | max rel diff {rel:.2e} | tol 1e-12 | {verdict}")
    assert rel < 1e-12
