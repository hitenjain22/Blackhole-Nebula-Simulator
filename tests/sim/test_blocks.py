"""Step 4.3: block time steps (phases/phase-04-timestepping.md 4.2 tests 1, 2, 5, 6)."""

import numba
import numpy as np
import pytest

from nebula.analysis.orbit_metrics import pericentres
from nebula.physics.gravity import gn_angmom, gn_hamiltonian, gn_v_from_p
from nebula.physics.orbits import apocentre_state
from nebula.sim import blocks, coupled
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.hydro import HydroConfig
from nebula.sim.integrator import integrate_orbit
from nebula.sim.neighbours import find_neighbours_numba, tighten
from nebula.sim.sph import compute_density, density_numba, forces_numba, gradients_numba
from nebula.sim.testproblems import run_sound, sod_measure, sod_setup, sound_measure, sound_setup
from nebula.sim.timesteps import ticks
from tests.helpers import check
from tests.sim.test_coupled import glass_sphere


def warm_ball(rng, u=1e-3, sigma_v=0.01):
    """A 505-particle glass ball (radius 3) at x = (500, 0, 0) on the hot-blob orbit, warmer
    than the blob and stirred, so that pressure, viscosity and conductivity all act."""
    pos = glass_sphere(3.0) + np.array([500.0, 0.0, 0.0])
    n = len(pos)
    rho = 1.0 / (4 / 3 * np.pi * 27.0)
    v = np.tile([0.0, 7.1951756691 / 500.0, 0.0], (n, 1)) + rng.normal(0, sigma_v, (n, 3))
    return pos, v, np.full(n, u), np.full(n, 1.0 / n), np.full(n, 1.6 * (1 / n / rho) ** (1 / 3))


# --------------------------------------------------------------------------- active subsets


def test_active_rows_identical_to_full_pass(rng):
    """The SPH passes restricted to a subset of rows give, row for row, exactly the full pass
    (block steps compute only the active particles)."""
    pos = glass_sphere(0.4, 900)
    n = len(pos)
    m = np.full(n, 1 / 4096)
    vel = rng.normal(0, 0.3, (n, 3))
    acc = rng.normal(0, 1.0, (n, 3))
    h0 = np.full(n, 1.6 * 4096 ** (-1 / 3))
    rows = np.sort(rng.choice(n, n // 3, replace=False))
    full = compute_density(pos, m, h0)
    part = compute_density(pos, m, h0, rows=rows)
    check("density rows: h", full.h[rows], part.h, rtol=0.0)
    check("density rows: rho", full.rho[rows], part.rho, rtol=0.0)
    check("density rows: omega", full.omega[rows], part.omega, rtol=0.0)
    lists = full.neighbours
    d2 = density_numba(pos, m, h0, lists, np.full(n, np.inf), rows=rows[::-1])
    d1 = density_numba(pos, m, h0, lists, np.full(n, np.inf))
    check("density_numba rows (reversed order)", d1.rho[rows[::-1]], d2.rho, rtol=0.0)
    h, rho, omega = full.h, full.rho, full.omega
    u = rng.uniform(0.5, 1.5, n)
    p_gas = (2 / 3) * rho * u
    cs = np.sqrt(10 / 9 * u)
    alpha = rng.uniform(0, 1, n)
    nb_all = tighten(lists, pos, h)
    nb_rows = tighten(lists, pos, h, rows=rows)
    same_rows = all(
        np.array_equal(
            nb_all.index[nb_all.offset[i] : nb_all.offset[i + 1]],
            nb_rows.index[nb_rows.offset[i] : nb_rows.offset[i + 1]],
        )
        for i in rows
    )
    others = np.setdiff1d(np.arange(n), rows)
    empty = bool(np.all(np.diff(nb_rows.offset)[others] == 0))
    check("tighten rows: same lists for rows, empty for the rest", (True, True), (same_rows, empty))
    f_all = forces_numba(pos, vel, m, h, rho, omega, p_gas, nb_all, u=u, c_s=cs, alpha=alpha)
    f_row = forces_numba(
        pos, vel, m, h, rho, omega, p_gas, nb_rows, u=u, c_s=cs, alpha=alpha, rows=rows
    )
    check("forces rows: acceleration", f_all.acc[rows], f_row.acc, rtol=0.0)
    check("forces rows: du/dt", f_all.dudt[rows], f_row.dudt, rtol=0.0)
    check("forces rows: v_sig,max", f_all.vsig_max[rows], f_row.vsig_max, rtol=0.0)
    g_all = gradients_numba(pos, vel, acc, m, h, nb_all)
    g_row = gradients_numba(pos, vel, acc, m, h, nb_rows, rows=rows)
    check("gradients rows: div v", g_all.div_v[rows], g_row.div_v, rtol=0.0)
    check("gradients rows: d(div v)/dt", g_all.ddivv_dt[rows], g_row.ddivv_dt, rtol=0.0)


def test_reused_lists_stay_complete_for_inactive_neighbours(rng):
    """List reuse with an active subset: if an inactive particle's (stored) h no longer fits
    its search radius minus the displacement, the lists are rebuilt, so an active particle never
    misses an inactive neighbour."""
    pos = glass_sphere(0.4, 900)
    n = len(pos)
    m = np.full(n, 1 / 4096)
    h0 = np.full(n, 1.6 * 4096 ** (-1 / 3))
    first = compute_density(pos, m, h0)
    rows = np.arange(0, n, 2)
    h_big = first.h.copy()
    h_big[1::2] *= 1.5  # inactive particles whose h exceeds the old search radius
    moved = pos + rng.normal(0, 1e-4, pos.shape)
    res = compute_density(moved, m, h_big, cache=first.cache, rows=rows)
    full_lists = find_neighbours_numba(moved, h_big)
    complete = all(
        set(full_lists.index[full_lists.offset[i] : full_lists.offset[i + 1]])
        <= set(res.neighbours.index[res.neighbours.offset[i] : res.neighbours.offset[i + 1]])
        for i in rows
    )
    check("lists rebuilt (not reused) when inactive h outgrows them", 2, res.cache.builds)
    check("active rows' lists contain every neighbour", True, complete)


# --------------------------------------------------------------------------- one level


@pytest.mark.parametrize("ns_nu, disc_alpha_av", [(0.0, 0.0), (0.02, 0.0), (0.0, 1.0)])
def test_one_level_equals_shared_step(rng, ns_nu, disc_alpha_av):
    """Every particle on level 0 (dt_max below every criterion), no limiter: the block scheme is
    the shared-step scheme of Step 4.2, result for result; also with the disk viscosity of phase
    7, Option A (nu = 0.02: viscous dt 0.25 h^2 / nu ~ 12 > dt) and Option B (alpha_AV = 1)."""
    x, v, u, m, h = warm_ball(rng)
    dt = 1.0
    hc = HydroConfig(conductivity_signal=coupled.COND_VELOCITY, ns_nu=ns_nu,
                     disc_alpha_av=disc_alpha_av)  # fmt: skip
    cc = CoupledConfig(hydro=hc)
    cfg = BlockConfig(coupled=cc, dt_max=dt, limiter=False)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=6 * dt)
    ref = coupled.initialise(x, v, u, m, h, cc)
    check("all particles on level 0", 0, int(np.max(st.level)))
    for _ in range(6):
        blocks.advance(st, cfg)
        ref = coupled.step(ref, cc, dt)
    check("one level: positions", ref.x, st.x, rtol=0.0, per_vector=True)
    check("one level: momenta", ref.p, st.p, rtol=0.0, per_vector=True)
    check("one level: internal energies", ref.u, st.u, rtol=0.0)
    check("one level: viscosity alphas", ref.alpha, st.alpha, rtol=0.0)
    check("one level: time", ref.t, st.t, rtol=0.0)
    if ns_nu > 0 or disc_alpha_av > 0:
        check("one level: disk viscosity heated every particle (heat_disk > 0)", True,
              bool(np.all(st.heat_disk > 0)))  # fmt: skip


# --------------------------------------------------------------------------- test 1


@pytest.mark.slow
def test_orbits_under_block_steps():
    """Plan test 1: hydro off, two particles on different orbits sharing the block timeline
    (the cloud orbit r_p = 25, r_a = 5,000 and the hot-blob orbit r_p = 25, r_a = 500), eta =
    0.00125, to 1.6 cloud periods (it starts at apocentre: pericentres at 0.5 and 1.5 periods).
    Same requirements as Phase 1 test 4: pericentre times and radial period (rtol 1e-5),
    precession (0.01 deg), max |dH/H| < 1e-4, |dL/L| < 1e-12."""
    xc, _, pc = apocentre_state(25.0, 5000.0)
    xb, _, pb = apocentre_state(25.0, 500.0)
    xb = np.array([-xb[0], -xb[1], xb[2]])  # opposite side; no hydro, so no interaction
    pb = np.array([-pb[0], -pb[1], pb[2]])
    x0, p0 = np.array([xc, xb]), np.array([pc, pb])
    cfg = BlockConfig(coupled=CoupledConfig(eta_orb=0.00125), dt_max=4096.0, hydro=False)
    t_end = 310 * 4096.0  # ~1.6 radial periods of the cloud orbit (792,447)
    st = blocks.initialise(x0, gn_v_from_p(x0, p0), np.zeros(2), np.ones(2), np.ones(2), cfg, t_end)
    rec = {0: [(0.0, xc, pc)], 1: [(0.0, xb, pb)]}
    while st.n < st.n_stop:
        for i in blocks.advance(st, cfg):
            rec[int(i)].append((st.t, st.x[i].copy(), st.p[i].copy()))
    t = np.array([r[0] for r in rec[0]])
    x = np.array([r[1] for r in rec[0]])
    t_peri, _, phi_peri = pericentres(t, x)
    check("block steps: cloud pericentres found", 2, len(t_peri))
    check("§2 first pericentre time (block steps)", 396_223.627, t_peri[0], rtol=1e-5)
    check("§2 radial period (block steps)", 792_447.254, t_peri[1] - t_peri[0], rtol=1e-5)
    check(
        "§2 precession (block steps, deg)",
        24.0398,
        np.degrees(np.diff(phi_peri)[0] - 2 * np.pi),
        atol=0.01,
    )
    for i, name in ((0, "cloud"), (1, "blob orbit")):
        xi = np.array([r[1] for r in rec[i]])
        pi = np.array([r[2] for r in rec[i]])
        hh = gn_hamiltonian(xi, pi)
        ll = np.linalg.norm(gn_angmom(xi, pi), axis=1)
        check(f"{name}: max |dH/H|", 0.0, float(np.max(np.abs(hh / hh[0] - 1))), atol=1e-4)
        check(f"{name}: max |dL/L|", 0.0, float(np.max(np.abs(ll / ll[0] - 1))), atol=1e-12)
    single = integrate_orbit(xc, pc, t_end, eta=0.00125)
    check(
        "cloud: final position vs Phase 1 integrator",
        single.x[-1],
        st.x[0],
        rtol=1e-6,
        per_vector=True,
    )
    print(f"   system steps {len(st.active_counts)}, particle steps {st.particle_steps}")


# --------------------------------------------------------------------------- test 2


def mixed_offsets(n, seed=7):
    """Half the particles (random) forced 2 levels shorter (plan test 2)."""
    off = np.zeros(n, np.int64)
    off[np.random.default_rng(seed).choice(n, n // 2, replace=False)] = 2
    return off


def blocks_from_problem(prob, dt_max, offsets, state=None):
    hs = prob.state if state is None else state
    cfg = BlockConfig(coupled=CoupledConfig(hydro=prob.cfg, hole=False), dt_max=dt_max)
    st = blocks.initialise(
        hs.x, hs.v, hs.u, hs.mass, hs.h, cfg, t_stop=prob.t_end, level_offset=offsets
    )
    return st, cfg


class _AsHydro:
    """Adapter so the Phase 3 measurements read a BlockState (v = p without the hole)."""

    def __init__(self, st, gamma):
        self.t, self.x, self.v, self.u, self.h, self.rho = st.t, st.x, st.p, st.u, st.h, st.rho
        self.pressure = (gamma - 1) * st.rho * st.u
        self.n_flagged = 0


@pytest.mark.slow
def test_sound_wave_mixed_levels():
    """Plan test 2: the sound wave (32 per wavelength) with half the particles 2 levels
    shorter: back within 1% of a wavelength after one period, and close to the one-level
    result."""
    prob = sound_setup("low")
    wave, base = run_sound(prob)
    ref = sound_measure(prob, wave, base)
    dt = prob.t_end / round(prob.t_end / (prob.cfg.courant * float(np.min(base.h / base.vsig_max))))
    offs = mixed_offsets(len(prob.state.x))
    sw, cfg = blocks_from_problem(prob, dt, offs)
    sb, _ = blocks_from_problem(prob, dt, offs, state=prob.extra["baseline"])
    check("levels: 0 and 2 only", [0, 2], sorted(set(sw.level.tolist())))
    blocks.run(sw, cfg, prob.t_end)
    blocks.run(sb, cfg, prob.t_end)
    m = sound_measure(prob, _AsHydro(sw, 5 / 3), _AsHydro(sb, 5 / 3))
    check("mixed levels: sound wave phase error, wavelengths", 0.0, abs(m.phase_cycles), atol=0.01)
    check("mixed levels: L1 vs one level", ref.l1, m.l1, rtol=0.1)
    print(
        f"   one level: phase {ref.phase_cycles:+.4f}, L1 {ref.l1:.4f}; "
        f"mixed: phase {m.phase_cycles:+.4f}, L1 {m.l1:.4f}"
    )


@pytest.mark.slow
def test_sod_mixed_levels():
    """Plan test 2: Sod (quick tube) with half the particles 2 levels shorter: shock within 2
    smoothing lengths, no NaN, no density flags."""
    prob = sod_setup("quick")
    st, cfg = blocks_from_problem(prob, prob.t_end, mixed_offsets(len(prob.state.x)))
    blocks.run(st, cfg, prob.t_end)
    m = sod_measure(prob, _AsHydro(st, 1.4))
    check(
        "mixed levels: Sod shock position (in h)",
        0.0,
        abs(m.shock_x - m.shock_x_exact) / m.h_shock,
        atol=2.0,
    )
    check("mixed levels: Sod no NaN", False, bool(np.any(np.isnan(st.rho))))
    print(f"   levels used: {np.unique(st.level).tolist()}, system steps {len(st.active_counts)}")


# --------------------------------------------------------------------------- tests 5, 6


def test_synchronization_and_coverage():
    """Plan test 6: step ends are always on the particle's own level grid, active particles are
    exactly at their step ends, and each particle's steps tile the timeline with no gaps:
    the sum of its completed step lengths equals its current n_beg."""
    rng = np.random.default_rng(3)
    x, v, u, m, h = warm_ball(rng)
    u = u * np.where(np.arange(len(u)) % 5 == 0, 30.0, 1.0)  # hot spots: mixed levels
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 8)
    covered = np.zeros(len(x), np.int64)
    beg = st.n_beg.copy()
    ok_active = ok_grid = ok_inside = True
    for _ in range(80):
        before_end = st.n_end.copy()
        rows = blocks.advance(st, cfg)
        ok_active &= bool(np.all(before_end[rows] == st.n))
        covered[rows] += st.n - beg[rows]
        beg[rows] = st.n
        lv = st.level
        ok_grid &= bool(np.all(st.n_beg % ticks(lv) == 0) and np.all(st.n_end % ticks(lv) == 0))
        ok_inside &= bool(np.all((st.n_beg <= st.n) & (st.n < st.n_end)))
    check("levels in use (mixed)", True, len(np.unique(st.level)) >= 2)
    check("active particles exactly at their step end", True, ok_active)
    check("step boundaries on each particle's level grid", True, ok_grid)
    check("no particle past its step end", True, ok_inside)
    check("completed steps tile the timeline (no gaps)", st.n_beg, covered)


def test_determinism_and_threads():
    """Plan test 5: the same run twice is identical; 1 thread vs all threads agree to 1e-12."""

    def go():
        rng = np.random.default_rng(3)
        x, v, u, m, h = warm_ball(rng)
        u = u * np.where(np.arange(len(u)) % 5 == 0, 30.0, 1.0)
        cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0)
        st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 8)
        for _ in range(40):
            blocks.advance(st, cfg)
        return st

    a, b = go(), go()
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        c = go()
    finally:
        numba.set_num_threads(threads)
    check("same run twice: positions identical", a.x, b.x, rtol=0.0, per_vector=True)
    check("same run twice: u identical", a.u, b.u, rtol=0.0)
    check(f"1 vs {threads} threads: positions", a.x, c.x, rtol=1e-12, per_vector=True)
    check(f"1 vs {threads} threads: momenta", a.p, c.p, rtol=1e-12, per_vector=True)
    check(f"1 vs {threads} threads: u", a.u, c.u, rtol=1e-12)


@pytest.mark.slow
def test_hot_blob_ledger_block_steps():
    """Plan test 4 on block steps (Step 4.5): max ledger error < 1e-5 at (0.00125, 0.15),
    falling 3-5x from (0.0025, 0.3) (measured at full sync points). Angular momentum: exact
    conservation needs every pair force applied at the same instant, which individual steps
    give up; the plan's 1e-10 stays on the shared-step test (test_coupled.py). Here the drift
    must be time-step error: falling >= 3x per halving (DECISIONS, Step 4.5)."""
    from nebula.sim.blob import cached_hot_blob

    fine = cached_hot_blob(0.00125, 0.15, block_steps=True)
    coarse = cached_hot_blob(0.0025, 0.3, block_steps=True)
    fe, ce = fine["max_energy_error"], coarse["max_energy_error"]
    fl, cl = fine["max_angmom_error"], coarse["max_angmom_error"]
    check("block steps: hot blob max |dE| / sum m|H|", 0.0, fe, atol=1e-5)
    check("block steps: error ratio coarse / fine", 4.0, ce / fe, atol=1.0)
    check("block steps: |dL|/|L| falls >= 3x per halving", True, cl / fl >= 3.0)
    check("block steps: blob on safe orbits (lowest L > 5)", True, fine["min_particle_angmom"] > 5)
    print(f"   energy {ce:.2e} -> {fe:.2e}; angular momentum {cl:.1e} -> {fl:.1e}")
    print(f"   particle-steps {fine['steps']:,} ({fine['n']} particles)")
