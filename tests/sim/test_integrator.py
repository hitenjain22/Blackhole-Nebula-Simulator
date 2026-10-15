"""Step 1.4: the generalized-leapfrog integrator (phases/phase-01-orbits.md Step 1.4, tests 1-5;
expected-values §2 "Integrator reference behaviour").

Energy errors are the maximum over the run, not the end value: the leapfrog's energy error
swings up at each pericentre and comes back (expected-values §2, hot-blob note).
"""

import numpy as np
import pytest

from nebula.analysis.orbit_metrics import pericentres
from nebula.physics.gravity import gn_angmom, gn_dH_dp, gn_dH_dx, gn_hamiltonian, gn_p_from_v
from nebula.physics.orbits import apocentre_state
from nebula.sim.integrator import (
    GLFConvergenceError,
    _dh_dp,
    _dh_dx,
    glf_step,
    glf_step_many,
    integrate_orbit,
    orbit_dt,
)
from nebula.sim.reference import dop853_orbit
from tests.helpers import check
from tests.physics.test_gravity import random_states

ETA_CLOUD = 0.00125  # chosen to meet test 4 (docs/DECISIONS.md, Step 1.4)
ETA_CIRCULAR = 0.02  # the plan's own reference value for test 1


def max_rel_drift(values: np.ndarray) -> float:
    return float(np.max(np.abs(values - values[0])) / np.abs(values[0]))


def energy_and_angmom_drift(orbit) -> tuple[float, float]:
    h = gn_hamiltonian(orbit.x, orbit.p)
    ell = np.linalg.norm(gn_angmom(orbit.x, orbit.p), axis=1)
    return max_rel_drift(h), max_rel_drift(ell)


def test_numba_formula_copies_match_physics(rng):
    """CLAUDE.md rule 1: the inline copies of dH/dx and dH/dp in the Numba kernels."""
    x, v = random_states(rng, 1000)
    p = gn_p_from_v(x, v)
    dx = np.array([_dh_dx(*a, *b) for a, b in zip(x, p, strict=True)])
    dp = np.array([_dh_dp(*a, *b) for a, b in zip(x, p, strict=True)])
    check(
        "Numba _dh_dx vs gn_dH_dx (1,000 states)", gn_dH_dx(x, p), dx, rtol=1e-15, per_vector=True
    )
    check(
        "Numba _dh_dp vs gn_dH_dp (1,000 states)", gn_dH_dp(x, p), dp, rtol=1e-15, per_vector=True
    )


def test_circular_orbit_1000_orbits():
    """Test 1: circular r = 10, fixed step eta r^1.5, 1,000 orbits."""
    r = 10.0
    vt = (r - 2) / (r * np.sqrt(r - 3))
    x0 = np.array([r, 0.0, 0.0])
    p0 = gn_p_from_v(x0, np.array([0.0, vt, 0.0]))
    dt = ETA_CIRCULAR * r**1.5
    n_steps = int(np.ceil(1000 * 2 * np.pi * r / vt / dt))
    orbit = integrate_orbit(x0, p0, n_steps * dt, dt=dt)
    de, dl = energy_and_angmom_drift(orbit)
    check("circular r=10, 1,000 orbits: max |dH/H|", 0.0, de, atol=1e-8)
    check("circular r=10, 1,000 orbits: max |dL/L|", 0.0, dl, atol=1e-12)


def test_second_order_convergence():
    """Test 2: eccentric orbit r_p = 10, r_a = 100 from apocentre, fixed dt = 2, 1, 0.5; max
    |dH/H| over ~4 radial orbits (t = 4,800) falls by 4 +- 0.2 per halving; plan-time values
    (§2) reproduced within ~5%."""
    x0, _, p0 = apocentre_state(10.0, 100.0)
    plan = {2.0: 1.5698e-3, 1.0: 3.9235e-4, 0.5: 9.8087e-5}
    errors = {}
    for dt, expected in plan.items():
        errors[dt], _ = energy_and_angmom_drift(integrate_orbit(x0, p0, 4800.0, dt=dt))
        check(f"eccentric dt={dt}: max |dH/H| vs plan", expected, errors[dt], rtol=0.05)
    check("ratio dt 2 -> 1", 4.0, errors[2.0] / errors[1.0], atol=0.2)
    check("ratio dt 1 -> 0.5", 4.0, errors[1.0] / errors[0.5], atol=0.2)


def test_time_reversibility():
    """Test 3: 50 radial orbits forward (r_p = 10, r_a = 100, fixed dt = 0.5), flip p,
    integrate back: return to the start within 1e-8. Fixed steps: adaptive step choice is
    not itself time-symmetric, so it would test the step selection, not the integrator."""
    x0, _, p0 = apocentre_state(10.0, 100.0)
    t_end = 50 * 1200.0  # radial period of this orbit ~1,190
    fwd = integrate_orbit(x0, p0, t_end, dt=0.5)
    back = integrate_orbit(fwd.x[-1], -fwd.p[-1], t_end, dt=0.5)
    check("reversibility: position", x0, back.x[-1], rtol=1e-8, per_vector=True)
    check("reversibility: momentum", p0, -back.p[-1], rtol=1e-8, per_vector=True)


@pytest.fixture(scope="module")
def cloud_runs():
    x0, v0, p0 = apocentre_state(25.0, 5000.0)
    t_end = 3 * 792_447.254
    return integrate_orbit(x0, p0, t_end, eta=ETA_CLOUD), dop853_orbit(x0, v0, t_end)


def test_cloud_orbit_timing_and_precession(cloud_runs):
    """Test 4: cloud orbit r_p = 25, r_a = 5,000, quantized adaptive steps, 3 radial periods:
    pericentre times vs DOP853 and §2 (rtol 1e-5), precession 24.0398 deg (abs 0.01 deg)."""
    glf, ref = cloud_runs
    t_peri, _, phi_peri = pericentres(glf.t, glf.x)
    check("pericentres found (GLF, DOP853)", (3, 3), (len(t_peri), len(ref.t_peri)))
    check("§2 first pericentre time (DOP853)", 396_223.627, ref.t_peri[0], rtol=1e-5)
    check("§2 radial period (DOP853)", 792_447.254, ref.t_peri[1] - ref.t_peri[0], rtol=1e-5)
    check("pericentre times: GLF vs DOP853", ref.t_peri, t_peri, rtol=1e-5)
    check("§2 first pericentre time (GLF)", 396_223.627, t_peri[0], rtol=1e-5)
    periods = np.diff(t_peri)
    check("§2 radial period (GLF), passages 1-2, 2-3", np.full(2, 792_447.254), periods, rtol=1e-5)
    precession = np.degrees(np.diff(phi_peri) - 2 * np.pi)
    check("§2 precession per orbit (GLF, deg)", np.full(2, 24.0398), precession, atol=0.01)


def test_cloud_orbit_conservation(cloud_runs):
    glf, _ = cloud_runs
    de, dl = energy_and_angmom_drift(glf)
    check(f"cloud orbit eta={ETA_CLOUD}: max |dH/H|", 0.0, de, atol=1e-4)
    check(f"cloud orbit eta={ETA_CLOUD}: max |dL/L|", 0.0, dl, atol=1e-12)


def test_many_equals_scalar(rng):
    """Test 5: glf_step_many equals glf_step on 100 random particles (own dt each)."""
    x, v = random_states(rng, 100)
    p = gn_p_from_v(x, v)
    dt = 0.005 * np.linalg.norm(x, axis=1) ** 1.5
    ref = [glf_step(a, b, h) for a, b, h in zip(x, p, dt, strict=True)]
    x1, p1, iters = glf_step_many(x, p, dt)
    check(
        "glf_step_many vs glf_step: x",
        np.array([s[0] for s in ref]),
        x1,
        rtol=1e-15,
        per_vector=True,
    )
    check(
        "glf_step_many vs glf_step: p",
        np.array([s[1] for s in ref]),
        p1,
        rtol=1e-15,
        per_vector=True,
    )
    check("glf_step_many vs glf_step: iterations", np.array([s[2] for s in ref]), iters)


def test_orbit_dt_quantization():
    for r in (25.0, 77.7, 5000.0):
        dt = orbit_dt(r, ETA_CLOUD)
        k = np.log2(4096.0 / dt)
        ok = k == round(k) and dt <= ETA_CLOUD * r**1.5 < 2 * dt
        check(f"orbit_dt(r={r:g}) = 4096/2^k, largest <= eta r^1.5", True, bool(ok))


def test_failures_raise():
    """Non-convergence must raise, never continue silently (phase-01 Step 1.4)."""
    x = np.array([6.0, 0.0, 0.0])
    p = gn_p_from_v(x, np.array([0.0, 0.4, 0.0]))
    with pytest.raises(GLFConvergenceError, match="did not converge"):
        glf_step(x, p, 50.0)
    with pytest.raises(GLFConvergenceError, match="did not converge"):
        glf_step_many(x[None], p[None], 50.0)
    with pytest.raises(GLFConvergenceError, match="inside r"):
        glf_step(np.array([2.4, 0.0, 0.0]), p, 0.1)
    check("huge step / r < 2.5 raise GLFConvergenceError", True, True)
