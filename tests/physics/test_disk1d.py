"""1D viscous disk (physics/disk1d.py): Phase 7 1D tests 1-6 (phases/phase-07-viscosity-drain.md
7.3; expected values in reference/expected-values.md §5)."""

import mpmath
import numpy as np
import pytest

from nebula.physics.disk1d import DiskState, evolve, lbp_sigma, make_grid
from tests.helpers import check

# expected-values §5: Lynden-Bell & Pringle ring (Sigma R0^2 / m) and mass remaining
LBP_TABLE = {  # tau: (mass remaining, Sigma(x=0.5), Sigma(x=1), Sigma(x=1.5))
    0.016: (1.000000000, 1.957777e-7, 0.7104152, 8.579919e-8),
    0.032: (1.000000000, 3.426427e-4, 0.5027209, 1.500074e-4),
    0.064: (0.999999995, 0.01208259, 0.3560243, 0.005278396),
    0.128: (0.999978021, 0.06064328, 0.2525431, 0.02637038),
    0.256: (0.998280186, 0.1156337, 0.1797739, 0.04972546),
    0.512: (0.981639256, 0.1370121, 0.1290960, 0.05783287),
    1.0: (0.932078868, 0.1218845, 0.09491733, 0.05361829),
    2.0: (0.846486404, 0.08303970, 0.06580134, 0.04389108),
}


def _lbp_run(n: int, taus=(0.064, 0.256, 1.0), nu=1.0, dt_tau=1e-4):
    """LBP ring, R0 = 1, from tau = 0.016; inner edge 1e-6 R0 (zero torque), outer 100 R0.
    The exact solution lets gas reach r = 0; with the plan's inner edge at 0.01 R0 the gas is
    removed early and by tau = 1 the disk holds 1.3% less mass than the exact solution at any
    resolution (L1 1.39% for 400-3,200 cells; 0.12% floor at 1e-4 R0). At 1e-6 R0 the boundary
    error is below the discretization error, which then falls ~4x per doubling of the cells
    (L1 at tau = 1: 0.87%, 0.23%, 0.049% at 400, 800, 1,600; DECISIONS 2026-10-21)."""
    grid = make_grid(1e-6, 100.0, n)
    m0 = lbp_sigma(grid.centres, 0.016) * grid.areas
    state = DiskState(m0, t=0.0, accreted=0.0)
    out = {}
    for tau in taus:
        t_end = (tau - 0.016) / (12.0 * nu)
        state, _ = evolve(grid, np.full(n, nu), state, t_end, dt=dt_tau / (12.0 * nu))
        exact = lbp_sigma(grid.centres, tau)
        sigma = state.mass / grid.areas
        l1 = np.sum(np.abs(sigma - exact) * grid.areas) / np.sum(exact * grid.areas)
        j_disk = np.sum(state.mass * np.sqrt(grid.centres))
        out[tau] = (l1, state.mass.sum(), state.accreted, j_disk, m0.sum())
    return grid, out


def test_lbp_profile_matches_table_and_mpmath():
    """Test 3: the analytic ring (ive form) reproduces the §5 table to its printed 7 significant
    figures, and an independent mpmath evaluation with the unscaled Bessel function to 1e-12."""
    mismatches = []
    for tau, (_, s05, s1, s15) in LBP_TABLE.items():
        for x, want in ((0.5, s05), (1.0, s1), (1.5, s15)):
            got = float(lbp_sigma(x, tau))
            if float(f"{got:.6e}") != want:  # rounded to the table's 7 significant figures
                mismatches.append((tau, x, got, want))
    check("LBP formula vs the §5 table: values differing at 7 significant figures", 0,
          len(mismatches))  # fmt: skip
    mpmath.mp.dps = 40
    worst = 0.0
    for tau in LBP_TABLE:
        for x in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
            ref = (mpmath.exp(-((1 - mpmath.mpf(x)) ** 2) / tau - 2 * mpmath.mpf(x) / tau)
                   * mpmath.besseli(0.25, 2 * mpmath.mpf(x) / tau)
                   / (tau * mpmath.mpf(x) ** 0.25 * mpmath.pi))  # fmt: skip
            got = float(lbp_sigma(x, tau))
            if ref != 0:
                worst = max(worst, abs(got / float(ref) - 1.0))
    check("LBP formula vs mpmath (unscaled Bessel, 40 digits)", 0.0, worst, atol=1e-12)


@pytest.mark.slow
def test_lbp_spreading_ring():
    """Tests 1-2: evolve the exact ring from tau = 0.016: L1 error < 1% and falling with
    resolution; mass remaining as §5 within 2e-3; angular momentum: disk + what the accreted
    gas carried across r_in (sqrt(r_in) per unit mass; zero torque there) stays 1 to 1e-4."""
    # angular momentum is conserved only to the discretization error (second order: dJ/J at
    # tau = 1 is 1.3e-3, 3.3e-4, 8.3e-5 at 800, 1,600, 3,200 cells), hence 3,200 cells
    _, coarse = _lbp_run(1600)
    grid, fine = _lbp_run(3200)
    for tau in (0.064, 0.256, 1.0):
        l1, m_disk, acc, j_disk, m0 = fine[tau]
        check(f"LBP tau = {tau}: L1 error of Sigma (3,200 cells)", 0.0, l1, atol=0.01)
        check(f"LBP tau = {tau}: L1 falls with resolution (1,600 -> 3,200)", True,
              bool(l1 < coarse[tau][0]))  # fmt: skip
        check(f"LBP tau = {tau}: mass remaining", LBP_TABLE[tau][0], m_disk, atol=2e-3)
        j_total = j_disk + acc * np.sqrt(grid.edges[0])
        j0 = np.sum(lbp_sigma(grid.centres, 0.016) * grid.areas * np.sqrt(grid.centres))
        check(f"LBP tau = {tau}: angular momentum (disk + accreted)", j0, j_total, rtol=1e-4)


def test_mass_ledger_exact():
    """Test 6: initial = disk + accreted (+ outer loss: none, the outer edge is closed), to
    1e-12, through Rannacher and Crank-Nicolson steps."""
    grid = make_grid(6.0, 1e4, 300)
    nu = 0.1 * 0.3**2 * np.sqrt(grid.centres)
    sig0 = np.exp(-0.5 * ((grid.centres - 50.0) / 1.5) ** 2)
    m0 = sig0 * grid.areas
    m0 /= m0.sum()
    state, _ = evolve(grid, nu, DiskState(m0, 0.0, 0.0), t_end=40000.0, dt=50.0)
    check("1D mass ledger: disk + accreted = initial", 1.0, state.mass.sum() + state.accreted,
          atol=1e-12)  # fmt: skip
    check("1D: some mass was accreted (the ledger was exercised)", True, state.accreted > 0.1)


@pytest.mark.slow
def test_steady_state_with_feeding():
    """Test 4: constant supply Mdot at r_feed = 50, zero torque at r_in = 6: inside r_feed the
    disk converges to nu Sigma = (Mdot / 3 pi)(1 - sqrt(r_in / r)) within 0.5%."""
    # outer edge 200: the whole disk (including the part outside r_feed, which must fill up
    # before the inner disk is steady) reaches steady state in ~30 viscous times at r_out
    grid = make_grid(6.0, 200.0, 400)
    nu = 0.1 * 0.3**2 * np.sqrt(grid.centres)
    i_feed = int(np.argmin(np.abs(grid.centres - 50.0)))
    mdot = 1.0
    t_visc = 200.0**2 / (0.1 * 0.3**2 * np.sqrt(200.0))
    state, _ = evolve(grid, nu, DiskState(np.zeros(len(grid.centres)), 0.0, 0.0),
                      t_end=30 * t_visc, dt=t_visc / 2000, source=(i_feed, mdot))  # fmt: skip
    r = grid.centres
    inside = (r > 7.0) & (r < 0.8 * r[i_feed])
    want = mdot / (3 * np.pi) * (1 - np.sqrt(6.0 / r[inside]))
    got = nu[inside] * state.mass[inside] / grid.areas[inside]
    check("steady state inside r_feed: nu Sigma vs (Mdot/3pi)(1 - sqrt(r_in/r))", want, got,
          rtol=0.005)  # fmt: skip
    check("steady state: accretion rate at r_in = supply", mdot, float(_mdot_in(grid, nu, state)),
          rtol=0.005)  # fmt: skip


def _mdot_in(grid, nu, state):
    from nebula.physics.disk1d import rate_matrix

    _, b = rate_matrix(grid, nu)
    return b @ state.mass


@pytest.mark.slow
def test_self_similar_drain():
    """Test 5: nu = alpha (H/r)^2 r^(1/2), Gaussian ring of width 3% at r0 = 50.8896, zero torque
    at 6, outer edge 5,000: half-mass at 0.659 t_nu and peak Mdot t_nu = 1.945 within 3%
    (independent plan-time solver, expected-values §5)."""
    r0, alpha, hr = 50.8896, 0.1, 0.3
    grid = make_grid(6.0, 5000.0, 800)
    nu = alpha * hr**2 * np.sqrt(grid.centres)
    t_nu = r0**2 / (alpha * hr**2 * np.sqrt(r0))
    sig = np.exp(-0.5 * ((grid.centres - r0) / (0.03 * r0)) ** 2)
    m0 = sig * grid.areas
    m0 /= m0.sum()
    state, log = evolve(grid, nu, DiskState(m0, 0.0, 0.0), t_end=1.5 * t_nu, dt=t_nu / 4000,
                        record_every=4)  # fmt: skip
    t = np.array([e[0] for e in log])
    acc = np.array([e[1] for e in log])
    mdot = np.array([e[2] for e in log])
    t_half = float(np.interp(0.5, acc, t))
    print(f"   t_half {t_half / t_nu:.4f} t_nu; peak Mdot t_nu {mdot.max() * t_nu:.4f} at "
          f"{t[np.argmax(mdot)] / t_nu:.4f} t_nu")  # fmt: skip
    check("drain: half the mass accreted at (t_nu)", 0.659, t_half / t_nu, rtol=0.03)
    check("drain: peak Mdot x t_nu", 1.945, mdot.max() * t_nu, rtol=0.03)
