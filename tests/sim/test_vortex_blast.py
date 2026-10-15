"""Step 3.7: Gresho-Chan vortex and Sedov blast (phases/phase-03-sph-core.md 3.2 tests 6, 7;
expected-values §4).

Gresho: "the switch must cut the error substantially" is made quantitative here as: with
the switch, the velocity L1 error at t = 1 is at most 70% of the error with standard
constant viscosity, alpha = 1 (docs/DECISIONS.md, Step 3.7). Since D2 the scheme also uses
velocity reconstruction; the comparison run stays the *standard* scheme (constant alpha,
no reconstruction: the yardstick, not the thing tested), and a second check requires that
reconstruction does not make the vortex worse (DECISIONS, 2026-10-15, Gresho baseline).
"""

from dataclasses import replace

import pytest

from nebula.sim.hydro import initialise
from nebula.sim.testproblems import Problem, gresho_l1, gresho_setup, run_problem
from tests.helpers import check

SWITCH_RATIO_MAX = 0.7


def _gresho_l1(resolution, switch, reconstruction):
    prob = gresho_setup(resolution, switch=switch)
    cfg = replace(prob.cfg, reconstruction=reconstruction)
    s0 = prob.state
    st = initialise(s0.x, s0.v, s0.u, s0.mass, s0.h, cfg)
    return gresho_l1(run_problem(Problem(st, cfg, prob.t_end, prob.x0, prob.extra)))[0]


def _gresho_runs(resolution):
    """(standard constant viscosity, the scheme as used: switch + reconstruction, the switch
    without reconstruction)."""
    return (
        _gresho_l1(resolution, switch=False, reconstruction=False),
        _gresho_l1(resolution, switch=True, reconstruction=True),
        _gresho_l1(resolution, switch=True, reconstruction=False),
    )


def _check_gresho(label, const, sw, sw_plain):
    check(f"Gresho {label}: L1 with switch (+ recon) / L1 standard alpha = 1", 0.0, sw / const,
          atol=SWITCH_RATIO_MAX)  # fmt: skip
    check(f"Gresho {label}: reconstruction does not make it worse", True, bool(sw <= sw_plain))
    print(f"   {label}: standard alpha=1 {const:.4f}, switch {sw_plain:.4f}, "
          f"switch + recon {sw:.4f}")  # fmt: skip


def test_gresho_initial_state_exact():
    l1, *_ = gresho_l1(gresho_setup("low").state)
    check("Gresho t=0: v_phi L1 (set-up exact)", 0.0, l1, atol=1e-12)


@pytest.mark.slow
def test_gresho_switch_low():
    """16 particles per unit length (2,048 particles), t = 1 (~45 s: slow until Step 3.8)."""
    _check_gresho("low", *_gresho_runs("low"))


@pytest.mark.slow
def test_gresho_switch_high():
    """32 particles per unit length (16,384 particles), t = 1; the error with the switch also
    falls with resolution."""
    const, sw, sw_plain = _gresho_runs("high")
    _check_gresho("high", const, sw, sw_plain)
    low_sw = gresho_l1(run_problem(gresho_setup("low", switch=True)))[0]
    check("Gresho (switch): L1 falls with resolution", True, bool(sw < low_sw))
    print(f"   low switch + recon {low_sw:.4f}")


@pytest.mark.slow
def test_sedov_blast_radius():
    """Plan test 7: 110,592 particles, one global step. Shock radius (density peak) within 5% of
    xi0 (E t^2/rho)^(1/5) at three times with R >= 10 h. Energy (blast + background) conserved.
    The run is shared with scripts/plot_vortex_blast.py through the fingerprinted cache."""
    from nebula.sim.testproblems import cached, sedov_run, sedov_setup

    results = cached("sedov_3tiles", lambda: sedov_run(sedov_setup(3)))
    h_bg = 1.6 / 48
    for res in results:
        check(
            f"Sedov R at t={res['t']:.4f} (R/h = {res['r_exact'] / h_bg:.1f})",
            res["r_exact"],
            res["r_sph"],
            rtol=0.05,
        )
    check(
        "Sedov total energy at the end (E = 1 + 1e-6 background)",
        1.0 + 1e-6,
        results[-1]["energy"],
        rtol=1e-3,
    )
