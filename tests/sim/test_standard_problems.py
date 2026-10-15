"""Step 3.6: sound wave and Sod shock tube (phases/phase-03-sph-core.md 3.2 tests 4 and 5;
expected-values §4). Full-resolution plan tests are marked slow; everyday versions are coarser
(see each docstring)."""

import numpy as np
import pytest

from nebula.physics.riemann import solve_riemann
from nebula.sim.testproblems import (
    run_problem,
    run_sound,
    sod_measure,
    sod_setup,
    sound_measure,
    sound_setup,
)
from tests.helpers import check

SOD = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)


@pytest.mark.slow
def test_sod_quick():
    """Coarse tube (9,216 particles) to t = 0.1: too coarse to resolve the plateaus between
    their margins, so this checks the shock position and that nothing misbehaves. Marked slow
    (~35 s) to keep the everyday suite near a minute; revisit after the Step 3.8 speed-up."""
    prob = sod_setup("quick")
    st = run_problem(prob)
    m = sod_measure(prob, st)
    check(
        "Sod quick: shock position (in smoothing lengths)",
        0.0,
        abs(m.shock_x - m.shock_x_exact) / m.h_shock,
        atol=2.0,
    )
    check(
        "Sod quick: no NaN, no density flags",
        (False, 0),
        (bool(np.any(np.isnan(st.rho))), st.n_flagged),
    )


@pytest.mark.slow
def test_sod_full():
    """Plan test 5: 36,864 particles to t = 0.2. Plateaus within 3%, shock within 2 local h,
    post-shock oscillations below 5%."""
    prob = sod_setup("full")
    m = sod_measure(prob, run_problem(prob))
    check("§4 Sod rho*L (left of contact)", SOD.rho_star_l, m.rho_star_l, rtol=0.03)
    check("§4 Sod rho*R (right of contact)", SOD.rho_star_r, m.rho_star_r, rtol=0.03)
    check("§4 Sod p* (plateau)", SOD.p_star, m.p_star, rtol=0.03)
    check("§4 Sod u* (plateau)", SOD.u_star, m.u_star, rtol=0.03)
    check(
        "§4 Sod shock position (in smoothing lengths)",
        0.0,
        abs(m.shock_x - m.shock_x_exact) / m.h_shock,
        atol=2.0,
    )
    check("Sod post-shock oscillation (max |v - u*| / u*)", 0.0, m.oscillation, atol=0.05)


def test_sound_wave_quick():
    """32 particles per wavelength: after one period the wave is back within 1% of a
    wavelength."""
    prob = sound_setup("low")
    m = sound_measure(prob, *run_sound(prob))
    check(
        "sound wave (32/wavelength): phase error, wavelengths", 0.0, abs(m.phase_cycles), atol=0.01
    )


@pytest.mark.slow
def test_sound_wave_convergence():
    """Plan test 4: phase error < 1% and the L1 density error decreasing with resolution
    (32 and 64 particles per wavelength)."""
    results = {}
    for res in ("low", "high"):
        prob = sound_setup(res)
        results[res] = sound_measure(prob, *run_sound(prob))
        check(
            f"sound wave ({res}): phase error, wavelengths",
            0.0,
            abs(results[res].phase_cycles),
            atol=0.01,
        )
    check(
        "sound wave: L1 error decreases with resolution",
        True,
        bool(results["high"].l1 < results["low"].l1),
    )
    lo, hi = results["low"].l1, results["high"].l1
    print(f"   L1 low {lo:.4f} -> high {hi:.4f} (ratio {lo / hi:.2f})")
