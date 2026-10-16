"""Step 4.2: gas coupled to the hole through canonical momenta (phases/phase-04-timestepping.md
4.1 and 4.2 test 4; expected-values §2 hot-blob orbit)."""

import numpy as np
import pytest

from nebula.sim import coupled
from nebula.sim.blob import U_BLOB, cached_hot_blob
from nebula.sim.coupled import CoupledConfig
from nebula.sim.glass import load_glass
from nebula.sim.hydro import HydroConfig
from nebula.sim.hydro import initialise as hydro_init
from nebula.sim.hydro import step as hydro_step
from tests.helpers import check


def glass_sphere(radius: float, n_target: int = 500):
    g = load_glass(4096) - 0.5
    r_unit = (n_target / 4096 / (4 / 3 * np.pi)) ** (1 / 3)
    pos = g[np.linalg.norm(g, axis=1) < r_unit] * (radius / r_unit)
    return pos


def test_without_hole_equals_phase3_stepper(rng):
    """hole=False: the coupled step is Phase 3's kick-drift-kick, result for result."""
    pos = glass_sphere(0.35, 700)
    n = len(pos)
    v = rng.normal(0, 0.3, (n, 3))
    u = rng.uniform(0.5, 1.5, n)
    m = np.full(n, 1 / 4096)
    h0 = np.full(n, 1.6 * 4096 ** (-1 / 3))
    hc = HydroConfig()
    a = hydro_init(pos, v, u, m, h0, hc)
    b = coupled.initialise(pos, v, u, m, h0, CoupledConfig(hydro=hc, hole=False))
    dt = 0.05 * float(np.min(a.h / a.vsig_max))
    for _ in range(5):
        a = hydro_step(a, hc, dt)
        b = coupled.step(b, CoupledConfig(hydro=hc, hole=False), dt)
    check("no hole: positions = Phase 3 stepper", a.x, b.x, rtol=1e-14, per_vector=True)
    check("no hole: momenta = Phase 3 velocities", a.v, b.p, rtol=1e-13, per_vector=True)
    check("no hole: internal energies equal", a.u, b.u, rtol=1e-13)


@pytest.mark.slow
def test_hot_blob_ledger():
    """Plan test 4 (shared step), from apocentre through pericentre to t = 18,000 with a blob
    100x cooler than the plan's (both: DECISIONS, Step 4.2): (a) max ledger error < 1e-5 at
    (eta_orb, C_cour) = (0.00125, 0.15); (b) it falls 3-5x from (0.0025, 0.3); (c) angular
    momentum to 1e-10; (d) the wrong coupling (kicks on v) does NOT shrink when the steps are
    halved."""
    fine = cached_hot_blob(0.00125, 0.15)
    coarse = cached_hot_blob(0.0025, 0.3)
    fine_e, coarse_e = fine["max_energy_error"], coarse["max_energy_error"]
    check("hot blob: max |dE| / sum m|H| (eta 0.00125, C 0.15)", 0.0, fine_e, atol=1e-5)
    check("hot blob: error ratio coarse / fine", 4.0, coarse_e / fine_e, atol=1.0)
    check("hot blob: max |dL| / |L|", 0.0, fine["max_angmom_error"], atol=1e-10)
    check("hot blob: heated at pericentre (u rose > 10x)", True, fine["u_max"] > 10 * U_BLOB)
    check(
        "hot blob stays on safe orbits (lowest particle L > 5; capture < 4)",
        True,
        fine["min_particle_angmom"] > 5.0,
    )
    wrong_fine = cached_hot_blob(0.00125, 0.15, wrong=True)["max_energy_error"]
    wrong_coarse = cached_hot_blob(0.0025, 0.3, wrong=True)["max_energy_error"]
    print(f"   right coupling: {coarse_e:.2e} -> {fine_e:.2e}")
    print(f"   wrong coupling: {wrong_coarse:.2e} -> {wrong_fine:.2e}")
    check(
        "wrong coupling: error does not fall with smaller steps (ratio < 2)",
        True,
        bool(wrong_coarse / wrong_fine < 2.0),
    )
    check(
        "wrong coupling: error much larger than right coupling",
        True,
        bool(wrong_fine > 10 * fine_e),
    )


@pytest.mark.slow
def test_hot_blob_ledger_with_cooling_and_full_eos():
    """Phase 6 test 8: the Phase 4 ledger test (as it now runs, D8: blob u = 4e-8, apocentre
    through pericentre to t = 18,000) with beta = 10 cooling and the gas + radiation EOS:
    max |delta(sum m (H + u) + E_rad)| / sum m |H| < 1e-5, falling 3-5x when eta_orb and
    C_cour are halved; and energy is actually radiated."""
    fine = cached_hot_blob(0.00125, 0.15, cooling_beta=10.0, eos="gasrad")
    coarse = cached_hot_blob(0.0025, 0.3, cooling_beta=10.0, eos="gasrad")
    fe, ce = fine["max_energy_error"], coarse["max_energy_error"]
    check("test 8: max ledger error (with E_rad)", 0.0, fe, atol=1e-5)
    check("test 8: error ratio coarse / fine", 4.0, ce / fe, atol=1.0)
    check("test 8: angular momentum", 0.0, fine["max_angmom_error"], atol=1e-10)
    print(f"   ledger {ce:.2e} -> {fe:.2e}")
