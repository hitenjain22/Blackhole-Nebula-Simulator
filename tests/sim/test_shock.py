"""Step 3.5: shock capturing (artificial viscosity, conductivity, Cullen-Dehnen switch) and the
energy test (phases/phase-03-sph-core.md 3.1; 3.2 test 8)."""

import numpy as np
import pytest

from nebula.physics import viscosity as visc
from nebula.sim.glass import load_glass
from nebula.sim.hydro import HydroConfig, initialise, stable_dt, step, totals
from nebula.sim.ic import fcc_lattice, smoothing_length_uniform
from nebula.sim.sph import (
    compute_density,
    forces_numba,
    forces_reference,
    gradients_numba,
    gradients_reference,
)
from tests.helpers import check
from tests.sim.test_density import blob

GAMMA = 5 / 3


def _setup(rng):
    pos, mass = blob(rng)
    d = compute_density(pos, mass, np.full(len(pos), 0.05), h_max=0.5)
    u = rng.uniform(0.5, 1.5, len(pos))
    p = (GAMMA - 1) * d.rho * u
    cs = np.sqrt(GAMMA * (GAMMA - 1) * u)
    vel = rng.normal(0, 0.5, pos.shape)
    alpha = rng.uniform(0, 1, len(pos))
    return pos, vel, mass, d, p, u, cs, alpha


def test_switch_formulas():
    check(
        "v_sig = alpha c_s + beta |v.r|",
        0.5 * 2.0 + 2.0 * 0.3,
        visc.viscous_signal_speed(0.5, 2.0, -0.3),
    )
    check(
        "v_sig keeps the beta term when alpha = 0",
        0.6,
        visc.viscous_signal_speed(0.0, 2.0, -0.3),
        rtol=1e-15,
    )
    check("q = 0 for receding pairs", 0.0, float(visc.viscous_q(1.0, 1.0, 0.2)))
    check(
        "q > 0 for approaching pairs",
        0.5 * 2.0 * 1.0 * 0.2,
        float(visc.viscous_q(2.0, 1.0, -0.2)),
        rtol=1e-15,
    )
    check(
        "Balsara xi: pure compression 1, pure rotation 0",
        (1.0, 0.0),
        (float(visc.balsara_xi(-1.0, 0.0)), float(visc.balsara_xi(0.0, 2.0))),
    )
    check("alpha_loc capped at alpha_max", 1.0, float(visc.alpha_local(1.0, -1e9, 1.0, 1.0)))
    check(
        "alpha_loc = 0 when compression is easing",
        0.0,
        float(visc.alpha_local(1.0, +5.0, 1.0, 1.0)),
    )
    check("alpha jumps up to alpha_loc at once", 0.8, float(visc.alpha_update(0.1, 0.8, 1e-3, 1.0)))
    tau, dt = 2.0, 0.5
    check(
        "alpha decays by backward Euler",
        (1.0 + 0.0) / (1 + dt / tau),
        float(visc.alpha_update(1.0, 0.0, dt, tau)),
        rtol=1e-15,
    )


def test_gradients_exact_for_linear_fields(rng):
    pos, vel, mass, d, *_ = _setup(rng)
    mv, ma = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    vel, acc = pos @ mv.T + 0.3, pos @ ma.T - 1.0
    g = gradients_numba(pos, vel, acc, mass, d.h, d.neighbours)
    curl = np.linalg.norm([mv[2, 1] - mv[1, 2], mv[0, 2] - mv[2, 0], mv[1, 0] - mv[0, 1]])
    n = len(pos)
    check("linear field: div v exact", np.full(n, np.trace(mv)), g.div_v, atol=1e-12)
    check("linear field: |curl v| exact", np.full(n, curl), g.curl_v, atol=1e-12)
    check(
        "linear field: d(div v)/dt exact",
        np.full(n, np.trace(ma) - np.sum(mv * mv.T)),
        g.ddivv_dt,
        atol=1e-12,
    )
    ref = gradients_reference(pos, vel, acc, mass, d.h, d.neighbours)
    check("gradients: Numba vs reference", ref.ddivv_dt, g.ddivv_dt, atol=1e-12)


def test_rotation_does_not_trigger_switch():
    """Rigid rotation: no compression, so xi = 0 and the switch gives alpha_loc = 0."""
    pos = fcc_lattice(8) - 0.5
    n = len(pos)
    mass = np.full(n, 1 / n)
    d = compute_density(pos, mass, np.full(n, smoothing_length_uniform(n, 1.0)))
    vel = np.cross([0, 0, 1.0], pos)
    g = gradients_numba(pos, vel, np.cross([0, 0, 1.0], vel), mass, d.h, d.neighbours)
    inner = np.linalg.norm(pos, axis=1) < 0.3
    xi = visc.balsara_xi(g.div_v[inner], g.curl_v[inner])
    check("rigid rotation: Balsara xi (interior)", 0.0, xi.max(), atol=1e-20)


def test_full_scheme_numba_matches_reference(rng):
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    args = (pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    ref = forces_reference(*args, u=u, c_s=cs, alpha=alpha)
    fast = forces_numba(*args, u=u, c_s=cs, alpha=alpha)
    check(
        "full scheme: acceleration, Numba vs reference",
        ref.acc,
        fast.acc,
        rtol=1e-12,
        per_vector=True,
    )
    scale = np.abs(ref.dudt).max()
    check(
        "full scheme: du/dt, Numba vs reference (/max)",
        ref.dudt / scale,
        fast.dudt / scale,
        atol=1e-12,
    )
    check("full scheme: v_sig,max, Numba vs reference", ref.vsig_max, fast.vsig_max, rtol=1e-12)


def test_full_scheme_conserves_instantaneously(rng):
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    f = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours, u=u, c_s=cs, alpha=alpha)
    rate = np.sum(mass * (np.sum(vel * f.acc, axis=1) + f.dudt))
    scale = np.sum(mass * (np.abs(np.sum(vel * f.acc, axis=1)) + np.abs(f.dudt)))
    check("full scheme: d(total energy)/dt (relative)", 0.0, abs(rate) / scale, atol=1e-13)
    mom = np.sum(mass[:, None] * f.acc, axis=0)
    check(
        "full scheme: sum m a (relative)",
        0.0,
        np.linalg.norm(mom) / np.sum(mass * np.linalg.norm(f.acc, axis=1)),
        atol=1e-13,
    )


def test_viscosity_acts_only_on_approaching_pairs():
    """Uniform expansion: viscosity adds nothing. Uniform contraction: it heats, even at
    alpha = 0 (the beta term)."""
    pos = fcc_lattice(8) - 0.5
    n = len(pos)
    mass = np.full(n, 1 / n)
    d = compute_density(pos, mass, np.full(n, smoothing_length_uniform(n, 1.0)))
    p = np.full(n, 0.1)
    cs, zero = np.full(n, 0.3), np.zeros(n)
    inner = np.linalg.norm(pos, axis=1) < 0.25
    for name, vel in (("expansion", pos.copy()), ("contraction", -pos)):
        plain = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
        av = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours, c_s=cs, alpha=zero)
        extra_heat = (av.dudt - plain.dudt)[inner]
        if name == "expansion":
            check("expansion: viscous force = 0", plain.acc[inner], av.acc[inner], atol=1e-15)
            check("expansion: viscous heating = 0", 0.0, np.abs(extra_heat).max(), atol=1e-15)
        else:
            check("contraction, alpha = 0: beta term heats", True, bool(np.all(extra_heat > 0)))


def _sphere_run(n_steps: int, dt_frac: float, seed: int = 7):
    g = load_glass(4096)
    pos = g[np.linalg.norm(g - 0.5, axis=1) < 0.35] - 0.5
    n = len(pos)
    rng = np.random.default_rng(seed)
    vel, u = rng.normal(0, 0.3, (n, 3)), rng.uniform(0.5, 1.5, n)
    cfg = HydroConfig()  # full scheme: viscosity + switch + conductivity
    s = initialise(pos, vel, u, np.full(n, 1 / 4096), np.full(n, 1.2 * 4096 ** (-1 / 3)), cfg)
    e0 = totals(s)
    dt = dt_frac * stable_dt(s, cfg)
    worst = 0.0
    for _ in range(n_steps):
        s = step(s, cfg, dt)
        worst = max(worst, abs(totals(s)["energy"] - e0["energy"]) / e0["energy"])
    return s, e0, worst


def _report_conservation(label, s, e0, worst):
    e = totals(s)
    p_scale = np.sum(s.mass * np.linalg.norm(s.v, axis=1))
    l_scale = np.sum(s.mass * np.linalg.norm(np.cross(s.x, s.v), axis=1))
    check(f"{label}: max |dE/E|", 0.0, worst, atol=1e-6)
    check(
        f"{label}: momentum (relative)",
        0.0,
        np.linalg.norm(e["momentum"] - e0["momentum"]) / p_scale,
        atol=1e-13,
    )
    check(
        f"{label}: angular momentum (relative)",
        0.0,
        np.linalg.norm(e["angmom"] - e0["angmom"]) / l_scale,
        atol=1e-13,
    )


def test_energy_error_is_time_step_error():
    """The residual energy error falls ~4x per halving of dt (second order): it is time-step
    error, not a flaw in the equations (which conserve energy to round-off, test above)."""
    errs = [_sphere_run(20 * 2**k, 0.25 / 2**k)[2] for k in range(2)]
    check("energy error ratio for dt -> dt/2", 4.0, errs[0] / errs[1], atol=1.0)


def test_isolated_sphere_conservation_quick():
    """Everyday version of plan test 8: 100 steps at dt = Courant/64."""
    _report_conservation("100 steps, dt = C/64", *_sphere_run(100, 1 / 64))


@pytest.mark.slow
def test_isolated_sphere_conservation_full():
    """Plan test 8: an isolated gas sphere (cut from the glass; random velocities 0.3 c_s and
    internal energies 0.5-1.5) with the full scheme, 1,000 fixed global steps at Courant/64
    (step chosen to meet the requirement; docs/DECISIONS.md)."""
    _report_conservation("1,000 steps, dt = C/64", *_sphere_run(1000, 1 / 64))
