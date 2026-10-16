"""Newtonian control gravity for run R6.4 (phase-06 6.4): no relativistic precession."""

import numpy as np
import pytest

from nebula.analysis.orbit_metrics import pericentres
from nebula.physics.gravity import newton_hamiltonian
from nebula.physics.orbits import apocentre_state
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.integrator import kdk_newton_many, kdk_newton_step
from tests.helpers import check


def test_numba_matches_reference(rng):
    x = rng.uniform(50, 500, (200, 3)) * rng.choice([-1, 1], (200, 3))
    p = rng.normal(0, 0.05, (200, 3))
    dt = rng.uniform(0.1, 5.0, 200)
    xr, pr = zip(*(kdk_newton_step(x[i], p[i], dt[i]) for i in range(200)), strict=True)
    xn, pn = kdk_newton_many(x, p, dt)
    check("Newton KDK: Numba vs reference, positions", np.array(xr), xn, rtol=1e-14,
          per_vector=True)  # fmt: skip
    check("Newton KDK: Numba vs reference, momenta", np.array(pr), pn, rtol=1e-14, per_vector=True)


def _kepler(dt, n_orbits=3):
    a, e = 100.0, 0.8
    x0 = np.array([a * (1 + e), 0.0, 0.0])  # apocentre
    v0 = np.array([0.0, np.sqrt((1 - e) / (a * (1 + e))), 0.0])  # vis-viva at apocentre
    period = 2 * np.pi * a**1.5
    n = int(round(n_orbits * period / dt))
    x, p = x0[None].copy(), v0[None].copy()
    xs, h = [x0], []
    for _ in range(n):
        x, p = kdk_newton_many(x, p, dt)
        xs.append(x[0].copy())
        h.append(newton_hamiltonian(x, p)[0])
    h0 = newton_hamiltonian(x0[None], v0[None])[0]
    return np.array(xs), np.max(np.abs(np.array(h) / h0 - 1)), period, dt


def test_kepler_orbit_period_energy_and_no_precession():
    """Eccentric Kepler orbit (a = 100, e = 0.8): pericentre times one period 2 pi a^1.5 apart,
    the energy error falls 4x per halving (second order), and the orbit does not precess."""
    xs, de, period, dt = _kepler(0.5)
    _, de_f, _, _ = _kepler(0.25)
    t = np.arange(len(xs)) * dt
    t_peri, _, phi = pericentres(t, xs)
    check("Kepler period 2 pi a^1.5", period, float(np.mean(np.diff(t_peri))), rtol=1e-4)
    check("energy error falls ~4x per halving", 4.0, de / de_f, atol=0.3)
    prec = np.degrees(np.diff(phi) - 2 * np.pi)
    print(f"   max |dH/H| {de:.2e} -> {de_f:.2e}; precession per orbit {prec}")
    check("Newtonian: precession per orbit ~ 0 (deg)", np.zeros(len(prec)), prec, atol=0.05)


@pytest.mark.slow
def test_block_scheme_newton_vs_gn():
    """The cloud orbit's starting state on block steps with hydro off: GN precesses 24.04 deg
    per orbit (Phase 1); the Newtonian control does not (its slightly different orbit is
    started from the same position and velocity)."""
    x0, v0, _ = apocentre_state(25.0, 5000.0)
    out = {}
    for grav in ("gn", "newton"):
        cfg = BlockConfig(coupled=CoupledConfig(gravity=grav, eta_orb=0.00125), dt_max=4096.0,
                          hydro=False)  # fmt: skip
        t_end = 4096.0 * 300
        st = blocks.initialise(x0[None], v0[None], [1e-6], [1.0], [1.0], cfg, t_stop=t_end)
        ts, xs = [0.0], [x0]
        while st.n < st.n_stop:
            blocks.advance(st, cfg)
            ts.append(st.t)
            xs.append(st.x[0].copy())
        _, _, phi = pericentres(np.array(ts), np.array(xs))
        out[grav] = np.degrees(np.diff(phi) - 2 * np.pi)
    check("GN: precession 24.04 deg per orbit (Phase 1)", 24.0398, float(out["gn"][0]), atol=0.01)
    check("Newton control: no precession", 0.0, float(out["newton"][0]), atol=0.05)
