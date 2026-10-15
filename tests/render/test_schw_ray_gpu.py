"""Step 2.2: GPU (float32) light rays vs the float64 reference (phases/phase-02 Step 2.2).

2,000 viewing angles per camera distance, half packed toward the shadow edge. Requirements:
|d phi_inf| < 2e-4 rad for b > 1.05 b_crit; < 1e-2 rad for b > 1.001 b_crit; capture/escape
classification identical for all b outside +-1e-4 (relative) of b_crit.
"""

import numpy as np
import pytest

from nebula.render.ray_validation import compare
from tests.helpers import check


@pytest.mark.parametrize("d", [50.0, 20.0])
def test_gpu_escape_angles_vs_reference(d):
    c = compare(d)
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
    check(f"D={d:g}: rays tested (escaped by both)", True, bool(both.sum() > 1000))
