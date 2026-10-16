"""Phase 6 tests 4 and 5 (phases/phase-06-shocks-cooling-ring.md 6.3): shocks with the Phase 6
equation of state, and the self-intersection in miniature. Both run with velocity
reconstruction (D2) on: test 5 is the key check that it still captures strong shocks fully."""

import pytest

from nebula.sim.testproblems import run_problem, slabs_run, slabs_setup, sod_measure, sod_setup
from tests.helpers import check

# plan's gamma = 4/3 exact Sod values (phase-06 6.3 test 4; reproduced by physics.riemann)
SOD43 = {"p_star": 0.306011, "u_star": 0.953205, "rho_star_l": 0.411437, "rho_star_r": 0.278585}


@pytest.mark.slow
def test_radiation_dominated_shock_tube():
    """Test 4: Sod initial state with the EOS forced into the radiation limit P = rho u / 3 (an
    exact gamma = 4/3 gas): the gamma = 4/3 exact solution, with Phase 3's Sod requirements
    (plateaus within 3%, shock within 2 h, post-shock oscillation below 5%)."""
    prob = sod_setup("full", radiation=True)
    m = sod_measure(prob, run_problem(prob))
    check("gamma 4/3 Sod rho*L", SOD43["rho_star_l"], m.rho_star_l, rtol=0.03)
    check("gamma 4/3 Sod rho*R", SOD43["rho_star_r"], m.rho_star_r, rtol=0.03)
    check("gamma 4/3 Sod p*", SOD43["p_star"], m.p_star, rtol=0.03)
    check("gamma 4/3 Sod u*", SOD43["u_star"], m.u_star, rtol=0.03)
    check("gamma 4/3 Sod shock position (in h)", 0.0, abs(m.shock_x - m.shock_x_exact) / m.h_shock,
          atol=2.0)  # fmt: skip
    check("gamma 4/3 Sod post-shock oscillation", 0.0, m.oscillation, atol=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("radiation", [False, True], ids=["gamma 5/3", "gamma 4/3"])
def test_colliding_slabs(radiation):
    """Test 5: two cold slabs collide head-on at +-v. Post-shock gas at rest with u = v^2/2
    within 5%, compression 4 or 7 within 10%, shock speed v/3 or v/6 within 5% (measured
    between t_end/2 and t_end, so the shock's formation is not counted)."""
    prob = slabs_setup(radiation=radiation)
    g = prob.extra["gamma"]
    m = slabs_run(prob)
    label = "4/3" if radiation else "5/3"
    check(f"slabs gamma {label}: compression", (g + 1) / (g - 1), m.rho_post, rtol=0.10)
    check(f"slabs gamma {label}: post-shock u = v^2/2", 0.5, m.u_post, rtol=0.05)
    check(f"slabs gamma {label}: post-shock gas at rest (|v| / v)", 0.0, m.v_post, atol=0.05)
    check(f"slabs gamma {label}: shock speed", (g - 1) / 2, m.shock_speed, rtol=0.05)
    print(f"   max energy error over the run (usual step): {m.energy_error:.2e}")
