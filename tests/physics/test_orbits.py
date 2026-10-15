"""Step 1.3: orbit constants, precession and special radii (expected-values §2, §8).

Tolerances: as stated in phase-01 Step 1.3 / §2 where given; otherwise from the printed
precision (at least 10x the rounding). Precession digits: phase-01 gives 0.419573409 rad; for
the rtol 1e-8 test the value was re-run with reference/scripts/01 (0.4195734093529) and
agrees with an independent factored-cubic quadrature (0.4195734093630) to 2.4e-11.
"""

import numpy as np
import pytest
from scipy.special import ellipk

from nebula.physics.gravity import gn_angmom, gn_hamiltonian, gn_v_from_p
from nebula.physics.orbits import (
    apocentre_state,
    circularization_radius,
    gn_energy_from_E,
    isco_radius,
    orbit_angle_from_pericentre,
    precession_exact,
    precession_weak_field,
    schwarzschild_EL_from_turning_points,
)
from tests.helpers import check

RP, RA = 25.0, 5000.0


def test_cloud_orbit_constants():
    e, ell = schwarzschild_EL_from_turning_points(RP, RA)
    check("§2 L", 7.35374434336, ell, rtol=1e-10)
    check("§2 E", 0.999801061330200, e, rtol=1e-11)
    check("§2 eps_GN = (E^2-1)/2", -1.98918881503e-4, gn_energy_from_E(e), rtol=1e-9)
    check("§2 circularization radius", 50.88962372, circularization_radius(ell), rtol=1e-8)


def test_apocentre_state():
    x0, v0, p0 = apocentre_state(RP, RA)
    check("§2 apocentre tangential speed (GN)", 1.4701605691e-3, v0[1], rtol=1e-8)
    check("start on +x axis, moving +y", True, bool(x0[0] == RA and v0[0] == 0 and v0[2] == 0))
    check("H(x0, p0) = eps_GN", -1.98918881503e-4, gn_hamiltonian(x0, p0), rtol=1e-9)
    check("|x0 cross p0| = L", 7.35374434336, np.linalg.norm(gn_angmom(x0, p0)), rtol=1e-10)
    check("p0 gives back v0", v0, gn_v_from_p(x0, p0), rtol=1e-15, per_vector=True)


def test_precession():
    prec = precession_exact(RP, RA)
    check("§2 precession per radial orbit (rad)", 0.41957340935, prec, rtol=1e-8)
    check("§2 precession per radial orbit (deg)", 24.03978555, np.degrees(prec), rtol=1e-8)
    angle = np.degrees(orbit_angle_from_pericentre(RP, RA, RA))
    check("§2 angle pericentre -> apocentre (deg)", 192.0198928, angle, rtol=1e-8)
    weak = np.degrees(precession_weak_field(RP, RA))
    check("§2 weak-field estimate (deg)", 21.708, weak, rtol=1e-4)
    check("exact precession > weak-field estimate", True, bool(np.degrees(prec) > weak))


def test_precession_vs_elliptic_integral():
    """Independent cross-check: the same integral in closed form,
    Delta_phi = 4 K(m) / sqrt(2 (u3 - ua)),  m = (up - ua)/(u3 - ua),  u3 = 1/2 - ua - up."""
    for rp, ra in ((25.0, 5000.0), (10.0, 100.0), (6.5, 50.0), (100.0, 1e4)):
        up, ua = 1 / rp, 1 / ra
        u3 = 0.5 - ua - up
        closed = 4 * ellipk((up - ua) / (u3 - ua)) / np.sqrt(2 * (u3 - ua)) - 2 * np.pi
        check(
            f"precession r_p={rp:g}, r_a={ra:g}: quad vs K(m)",
            closed,
            precession_exact(rp, ra),
            rtol=1e-12,
        )


def test_pericentre_variants():
    """§2 pericentre-variants table (r_a = 5,000), printed to 6-9 significant figures."""
    table = {  # r_p: (L, eps, precession deg, r_circ)
        15.0: (5.87468269, -1.993100e-4, 43.16715, 31.19266),
        25.0: (7.35374434, -1.989189e-4, 24.03979, 50.88962),
        40.0: (9.14015689, -1.983298e-4, 14.48218, 80.42623),
    }
    for rp, (ell_t, eps_t, prec_t, rc_t) in table.items():
        e, ell = schwarzschild_EL_from_turning_points(rp, RA)
        check(f"§2 r_p={rp:g}: L", ell_t, ell, rtol=1e-8)
        check(f"§2 r_p={rp:g}: eps", eps_t, gn_energy_from_E(e), rtol=5e-6)
        check(
            f"§2 r_p={rp:g}: precession (deg)",
            prec_t,
            np.degrees(precession_exact(rp, RA)),
            rtol=2e-6,
        )
        check(f"§2 r_p={rp:g}: r_circ", rc_t, circularization_radius(ell), rtol=2e-6)


def test_orbit_shape_is_symmetric_and_monotonic():
    r = np.geomspace(RP, RA, 50)
    phi = orbit_angle_from_pericentre(RP, RA, r)
    check("phi(r_peri) = 0", 0.0, phi[0], atol=1e-15)
    check("phi increases with r", True, bool(np.all(np.diff(phi) > 0)))


def test_isco():
    check("§8 ISCO a=0", 6.0, isco_radius(0.0), rtol=1e-14)
    for a, pro, retro in (
        (0.5, 4.23300253, 7.55458471),
        (0.9, 2.32088304, 8.71735228),
        (0.99, 1.45449794, 8.97186134),
    ):
        check(f"§8 ISCO a={a} prograde", pro, isco_radius(a, True), rtol=5e-8)
        check(f"§8 ISCO a={a} retrograde", retro, isco_radius(a, False), rtol=5e-8)


def test_impossible_orbits_refused():
    """Turning points that belong to no bound orbit are refused, not silently integrated."""
    for rp, ra in ((4.5, 6.0), (30.0, 20.0), (3.0, 1000.0)):
        with pytest.raises(ValueError, match="no bound orbit"):
            precession_exact(rp, ra)
    check("impossible turning points raise ValueError", True, True)
