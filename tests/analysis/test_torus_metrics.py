"""Phase 6 diagnostics (phase-06 6.5), each on a hand-made case with a known answer."""

import numpy as np

from nebula import units
from nebula.analysis.torus_metrics import (
    accretion_rate,
    eccentricity_histogram,
    luminosity_watts,
    mass_in_shocks,
    optical_depth_fractions,
    orbit_elements,
    ring_measures,
    unbound_fraction,
)
from nebula.physics.gravity import gn_p_from_v
from nebula.physics.orbits import (
    apocentre_state,
    schwarzschild_EL_from_turning_points,
    turning_points,
)
from tests.helpers import check


def test_turning_points_round_trip_and_classes():
    """physics.orbits.turning_points inverts schwarzschild_EL_from_turning_points; unbound and
    plunging orbits are classified."""
    pairs = [(25, 5000), (25, 500), (10, 100), (6.5, 7.0), (51, 51.5)]
    e, ell = zip(*(schwarzschild_EL_from_turning_points(rp, ra) for rp, ra in pairs), strict=True)
    rp, ra, kind = turning_points(np.array(e), np.array(ell))
    check("pericentres recovered", np.array([p[0] for p in pairs], float), rp, rtol=1e-10)
    check("apocentres recovered", np.array([p[1] for p in pairs], float), ra, rtol=1e-10)
    check("all bound", np.zeros(5, int), kind)
    _, _, k2 = turning_points(np.array([1.01, 0.97]), np.array([5.0, 3.0]))
    check("E > 1: unbound; L = 3 < 4 (bound E): plunging", [1, 2], k2.tolist())


def test_orbit_elements_of_known_orbits():
    """Particles started at apocentre of known orbits: eccentricities (ra - rp)/(ra + rp)."""
    pairs = [(25.0, 5000.0), (51.0, 51.0 * 1.0001), (25.0, 500.0)]
    xs, ps = [], []
    for rp, ra in pairs:
        x0, _, p0 = apocentre_state(rp, ra)
        xs.append(x0)
        ps.append(p0)
    el = orbit_elements(np.array(xs), np.array(ps))
    # a nearly circular orbit has a nearly double root: e is then good to ~1e-10 absolute
    # (round-off of the root difference), far finer than any use (the ring cut is e < 0.3)
    check("eccentricities", np.array([(ra - rp) / (ra + rp) for rp, ra in pairs]), el.ecc,
          rtol=1e-8, atol=1e-9)  # fmt: skip
    xp = np.array([[30.0, 0, 0]])
    plunger = orbit_elements(xp, gn_p_from_v(xp, np.array([[0.0, 0.02, 0.0]])))
    check("low angular momentum: plunging (kind 2)", 2, int(plunger.kind[0]))


def test_ring_measures_of_a_constructed_ring():
    """A ring at r = 51, thickness rms(z) = 3 (uniform |z| <= 3 sqrt(3)), on circular orbits,
    plus an eccentric clump: ring mass fraction, radius, H/r and extent."""
    rng = np.random.default_rng(4)
    n = 4000
    phi = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(50.0, 52.0, n)
    z = rng.uniform(-3 * np.sqrt(3), 3 * np.sqrt(3), n)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    vt = (r - 2) / (r * np.sqrt(r - 3))
    v = np.column_stack([-vt * np.sin(phi), vt * np.cos(phi), np.zeros(n)])
    x_ecc, _, p_ecc = apocentre_state(25.0, 5000.0)
    x_all = np.vstack([x, np.tile(x_ecc, (1000, 1))])
    p_all = np.vstack([gn_p_from_v(x, v), np.tile(p_ecc, (1000, 1))])
    m = np.ones(len(x_all))
    el = orbit_elements(x_all, p_all)
    rm = ring_measures(x_all, m, el)
    check("ring mass fraction of the bound gas", 0.8, rm.mass_fraction, rtol=1e-12)
    check("ring mean radius", 51.0, rm.mean_radius, rtol=0.005)
    check("ring H/r = 3/51", 3.0 / 51.0, rm.thickness, rtol=0.03)
    r3d = np.sqrt(r**2 + z**2)  # the measure uses the 3-D distance from the hole
    check("radial extent 10-90%", tuple(np.percentile(r3d, [10, 90])), (rm.r10, rm.r90),
          rtol=1e-12)  # fmt: skip
    centres, hist = eccentricity_histogram(el, m)
    check("eccentricity histogram: ring in the lowest bin, total mass", (4000.0, 5000.0),
          (float(hist[0]), float(hist.sum())), rtol=0.01)  # fmt: skip
    check("unbound fraction 0", 0.0, unbound_fraction(el, m))


def test_luminosity_and_accretion_units():
    """1 code unit of luminosity and of mass rate, in watts and M_sun/yr (by hand)."""
    gu = units.GasUnits()
    one_w = gu.mass_kg_per_code * units.C**2 / gu.bh.time_s
    check("code luminosity 1 -> watts", one_w, float(luminosity_watts(1.0)), rtol=1e-14)
    t = np.array([10.0, 20.0, 30.0])
    _, rate, edd = accretion_rate(t, np.full(3, 0.01), np.array([0.0, 100.0]))
    by_hand = 0.03 / 100.0 * gu.mass_kg_per_code / units.M_SUN / gu.bh.time_s * units.YEAR_S
    check("accretion rate (M_sun/yr)", by_hand, float(rate[0]), rtol=1e-12)
    check("in Eddington units (0.0955 M_sun/yr)", by_hand / 0.0954515, float(edd[0]), rtol=1e-5)


def test_optical_depth_fractions():
    """tau_es,h = kappa_es rho h: one particle with tau << 1, one with tau >> 1."""
    gu = units.GasUnits()
    h_m = gu.bh.length_m  # h = 1 code length
    rho_si = np.array([0.01, 100.0]) / (0.034 * h_m)  # tau_es = 0.01 and 100
    f_es, f_eff = optical_depth_fractions(gu.density_from_si(rho_si), np.array([1e7, 1e7]),
                                          np.ones(2), np.ones(2))  # fmt: skip
    check("mass with tau_es,h < 1", 0.5, f_es)
    check("tau_eff <= tau_es fraction ordering (eff thinner)", True, f_eff >= f_es)


def test_mass_in_shocks_threshold():
    """Shocked = viscous heating rate > Omega_K(r) u: at r = 100 (Omega = 1e-3) with u = 1,
    rates 2e-3 and 5e-4 give one shocked particle of two (mass fraction 0.5, located there)."""
    x = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    frac, where = mass_in_shocks(x, np.ones(2), np.array([2e-3, 5e-4]), np.ones(2))
    check("mass fraction in shocks", 0.5, frac)
    check("location of the shocked gas", x[0], where, rtol=1e-15)
