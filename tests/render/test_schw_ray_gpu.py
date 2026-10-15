"""Step 2.2: GPU (float32) light rays vs the float64 reference (phases/phase-02 Step 2.2).

The plan's test uses 2,000 viewing angles per camera distance (marked slow: the float64
reference takes ~25-50 s); the everyday run uses 400 with the same requirements.
2,000 (or 400) viewing angles per camera distance, half packed toward the shadow edge. Requirements:
|d phi_inf| < 2e-4 rad for b > 1.05 b_crit; < 1e-2 rad for b > 1.001 b_crit; capture/escape
classification identical for all b outside +-1e-4 (relative) of b_crit.
"""

import numpy as np
import pytest

from nebula.render.ray_validation import compare
from tests.helpers import check


def _check_against_reference(d: float, n: int) -> None:
    c = compare(d, n)
    x = c.b_over_bcrit
    both = np.isfinite(c.ref) & c.gpu_escaped
    err = np.abs(c.gpu - c.ref)
    check(
        f"D={d:g}: max |d phi_inf|, b > 1.05 b_crit (rad)",
        0.0,
        err[both & (x > 1.05)].max(),
        atol=2e-4,
    )
    check(
        f"D={d:g}: max |d phi_inf|, b > 1.001 b_crit (rad)",
        0.0,
        err[both & (x > 1.001)].max(),
        atol=1e-2,
    )
    away = np.abs(x - 1.0) > 1e-4
    mismatches = int(np.sum((np.isfinite(c.ref) != c.gpu_escaped) & away))
    check(f"D={d:g}: capture/escape mismatches outside 1e-4", 0, mismatches)
    check(f"D={d:g}, {n} angles: enough rays escaped by both", True, bool(both.sum() > n // 2))


@pytest.mark.parametrize("d", [50.0, 20.0])
def test_gpu_escape_angles_vs_reference_quick(d):
    _check_against_reference(d, 400)


@pytest.mark.slow
@pytest.mark.parametrize("d", [50.0, 20.0])
def test_gpu_escape_angles_vs_reference_full(d):
    _check_against_reference(d, 2000)
