"""Phase 5 cloud diagnostics, each checked on a hand-made particle set with a known answer."""

import numpy as np

from nebula.analysis.cloud_metrics import (
    centre_of_mass_offset,
    energy_distribution,
    principal_axes,
    thermal_summary,
    thickness,
)
from nebula.physics.orbits import apocentre_state
from nebula.sim.cloud import cloud_ic
from nebula.sim.glass import load_glass, tile_glass
from nebula.sim.integrator import integrate_orbit
from nebula.units import gas_u_from_temperature
from tests.helpers import check


def test_energy_of_the_cloud_orbit():
    """A 'cloud' of identical particles on the cloud orbit: eps = -1.989189e-4 for all
    (expected-values §2), spread 0; the mass-weighted histogram integrates to the mass."""
    x0, _, p0 = apocentre_state(25.0, 5000.0)
    n = 50
    x, p = np.tile(x0, (n, 1)), np.tile(p0, (n, 1))
    mass = np.full(n, 1.0 / n)
    e = energy_distribution(x, p, mass)
    check("eps of the cloud orbit (expected-values §2)", -1.989189e-4, e.mean, rtol=1e-6)
    check("spread of identical particles", 0.0, e.half_range, atol=1e-20)
    rng = np.random.default_rng(1)
    eps_test = rng.normal(0, 1, 1000)
    from nebula.analysis.cloud_metrics import EnergyDistribution

    d = EnergyDistribution(eps_test, 0, 0, 0, 0, 0)
    centres, dmde = d.histogram(np.full(1000, 2e-3), bins=30)
    width = (eps_test.max() - eps_test.min()) / 30
    check("dM/d(eps) integrates to the total mass", 2.0, np.sum(dmde) * width, rtol=1e-12)


def test_energy_spread_scales_with_cloud_radius():
    """The glass cloud at the start: half-range of eps ~ R/r_a^2 (phase-05 5.1; expected-values
    §2b: 3.71e-6 for R = 100 from a 400-particle test-particle model, estimate 4.0e-6). Radius
    25 must give a quarter of radius 100 (the spread is linear in R)."""
    from nebula.physics.gravity import gn_p_from_v

    half = {}
    for radius in (25.0, 100.0):
        ic = cloud_ic(2000, radius)
        e = energy_distribution(ic.x, gn_p_from_v(ic.x, ic.v), ic.mass)
        half[radius] = e.half_range
    check("R = 100: half-range vs estimate R/r_a^2 = 4.0e-6", 4.0e-6, half[100.0], rtol=0.1)
    check("R = 25 / R = 100 (linear in R)", 0.25, half[25.0] / half[100.0], rtol=0.02)


def test_centre_of_mass_follows_test_particle():
    """Gravity-only identical particles: the centre of mass is exactly the test particle."""
    x0, _, p0 = apocentre_state(25.0, 5000.0)
    times = [0.0, 100_000.0, 300_000.0]
    centres = [integrate_orbit(x0, p0, t, eta=0.00125).x[-1] if t > 0 else x0 for t in times]
    off = centre_of_mass_offset(np.tile(x0, (3, 1)), np.tile(p0, (3, 1)), np.ones(3), times,
                                centres)  # fmt: skip
    check("centre-of-mass offset, identical particles", np.zeros(3), off, atol=1e-9)


def test_principal_axes_of_known_shapes():
    """Uniform ellipsoid with semi-axes (a, b, c): rms extents a/sqrt(5), b/sqrt(5), c/sqrt(5)
    (a glass ball stretched; rotated to check the axes are found in any orientation)."""
    g = tile_glass(load_glass(4096), (2, 2, 2), (1.0, 1.0, 1.0)) - 0.5
    ball = g[np.linalg.norm(g, axis=1) < 0.5] * 2.0  # unit ball
    a, b, c = 30.0, 4.0, 1.5
    ell = ball * np.array([a, b, c])
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    ell = ell @ rot.T + np.array([500.0, -20.0, 3.0])
    axes = principal_axes(ell, np.ones(len(ell)))
    check("principal axes of a uniform ellipsoid (a, b, c)/sqrt(5)",
          np.array([a, b, c]) / np.sqrt(5), axes, rtol=0.01)  # fmt: skip


def test_thermal_summary_known_values():
    temps = np.array([1e4, 2e5, 3e6, 5e4])
    u = gas_u_from_temperature(temps)
    s = thermal_summary(u, np.array([1.0, 4.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0, 4.0]))
    check("maximum temperature (K)", 3e6, s.t_max, rtol=1e-12)
    check("maximum density", 4.0, s.rho_max)
    check("mass above 1e5 K, above 1e6 K", (3 / 8, 2 / 8), (s.mass_above_1e5, s.mass_above_1e6))


def test_thickness_and_robust_spreads():
    """rms height above the plane of a slab uniform in |z| <= H: H/sqrt(3); percentile
    spreads and the unbound fraction of a known energy set."""
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.uniform(-100, 100, 20000), rng.uniform(-5, 5, 20000),
                         rng.uniform(-2.0, 2.0, 20000)])  # fmt: skip
    check("thickness of a slab |z| <= 2: 2/sqrt(3)", 2 / np.sqrt(3), thickness(x, np.ones(20000)),
          rtol=0.02)  # fmt: skip
    x0, _, p0 = apocentre_state(25.0, 5000.0)
    xs = np.array([x0, x0 * 0.5, x0 * 2.0, [3e4, 0, 0]])
    ps = np.array([p0, p0, p0, [0.0, 0.05, 0.0]])  # the last one is unbound
    e = energy_distribution(xs, ps, np.ones(4))
    check("unbound mass fraction", 0.25, e.unbound)


def test_offset_counts_from_the_first_snapshot():
    """Snapshots starting at t0 > 0: the test particle starts at t0, so the offset there is 0."""
    x0, _, p0 = apocentre_state(25.0, 5000.0)
    first = integrate_orbit(x0, p0, 256.0, eta=0.00125)
    later = integrate_orbit(first.x[-1], first.p[-1], 50_000.0, eta=0.00125)
    times, centres = [256.0, 50_256.0], [first.x[-1], later.x[-1]]
    off = centre_of_mass_offset(first.x[-1][None], first.p[-1][None], np.ones(1), times, centres,
                                t0=256.0)  # fmt: skip
    check("offset from the first snapshot's time", np.zeros(2), off, atol=1e-9)


def test_snapshot_temperature_uses_the_runs_eos():
    """Pictures and summaries use the run's EOS: ideal gas as before (Phase 5); with gas +
    radiation, the EOS's own T (hot thin gas is radiation-dominated, so far below ideal gas)."""
    from nebula.analysis.cloud_metrics import snapshot_temperature
    from nebula.sim.hydro import HydroConfig, temperature
    from nebula.units import gas_temperature_from_u

    rho, u = np.array([1e-7, 1e-5, 1e-3]), np.array([4.5e-6, 1e-7, 2.3e-9])
    t_id = snapshot_temperature("ideal", rho, u)
    check("ideal: unchanged formula", gas_temperature_from_u(u), t_id, rtol=0.0)
    t_gr = snapshot_temperature("gasrad", rho, u)
    check("gasrad: the EOS's temperature", temperature(HydroConfig(eos="gasrad"), rho, u), t_gr,
          rtol=0.0)  # fmt: skip
    check("hot thin gas: radiation makes T far below ideal gas", True,
          bool(t_gr[0] < 0.1 * gas_temperature_from_u(u)[0]))  # fmt: skip
