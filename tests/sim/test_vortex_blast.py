"""Step 3.7: Gresho-Chan vortex and Sedov blast (phases/phase-03-sph-core.md 3.2 tests 6, 7;
expected-values §4).

Gresho: "the switch must cut the error substantially" is made quantitative here as: with
the switch, the velocity L1 error at t = 1 is at most 70% of the error with constant
alpha = 1 (docs/DECISIONS.md, Step 3.7).
"""

import pytest

from nebula.sim.testproblems import gresho_l1, gresho_setup, run_problem
from tests.helpers import check

SWITCH_RATIO_MAX = 0.7


def _gresho_pair(resolution):
    out = {}
    for switch in (False, True):
        out[switch] = gresho_l1(run_problem(gresho_setup(resolution, switch=switch)))[0]
    return out[False], out[True]


def test_gresho_initial_state_exact():
    l1, *_ = gresho_l1(gresho_setup("low").state)
    check("Gresho t=0: v_phi L1 (set-up exact)", 0.0, l1, atol=1e-12)


@pytest.mark.slow
def test_gresho_switch_low():
    """16 particles per unit length (2,048 particles), t = 1 (~45 s: slow until Step 3.8)."""
    const, sw = _gresho_pair("low")
    check("Gresho low: L1 with switch / L1 with alpha = 1", 0.0, sw / const, atol=SWITCH_RATIO_MAX)
    print(f"   L1 alpha=1 {const:.4f}, switch {sw:.4f}")


@pytest.mark.slow
def test_gresho_switch_high():
    """32 particles per unit length (16,384 particles), t = 1; the error with the switch also
    falls with resolution."""
    const, sw = _gresho_pair("high")
    check("Gresho high: L1 with switch / L1 with alpha = 1", 0.0, sw / const, atol=SWITCH_RATIO_MAX)
    low_sw = gresho_l1(run_problem(gresho_setup("low", switch=True)))[0]
    check("Gresho (switch): L1 falls with resolution", True, bool(sw < low_sw))
    print(f"   high: L1 alpha=1 {const:.4f}, switch {sw:.4f}; low switch {low_sw:.4f}")
