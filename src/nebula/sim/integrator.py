"""Generalized leapfrog for the GN Hamiltonian (phases/phase-01-orbits.md Step 1.4).

The state is position x and canonical momentum p (not velocity): with this symplectic,
time-reversible, second-order method, angular momentum is conserved to round-off and the
energy error stays bounded (phase-01 "Why canonical variables"). One step of size dt
(Hairer, Lubich & Wanner, Geometric Numerical Integration, generalized leapfrog / Stormer-Verlet
for non-separable Hamiltonians):

    1. p_half = p - (dt/2) dH/dx(x, p_half)                               (implicit)
    2. x1     = x + (dt/2) [dH/dp(x, p_half) + dH/dp(x1, p_half)]          (implicit)
    3. p1     = p_half - (dt/2) dH/dx(x1, p_half)                         (explicit)

Implicit stages use fixed-point iteration until the change is <= 1e-15 of the value (vector
length), at most 10 iterations; failure raises GLFConvergenceError (never silently continue).

Two implementations that must agree (CLAUDE.md rule 2):
- ``glf_step``: readable reference built on nebula.physics.gravity.
- Numba kernels (``glf_step_many``, ``integrate_orbit``): the dH/dx and dH/dp formulas are
  copied inline for speed; each copy names the physics function it mirrors and is tested
  against it (tests/sim/test_integrator.py).
"""

from dataclasses import dataclass
from math import sqrt

import numba
import numpy as np

from nebula.physics.gravity import gn_dH_dp, gn_dH_dx

TOL = 1e-15  # fixed-point convergence, relative (vector length)
MAX_ITER = 10  # per implicit stage
R_MIN = 2.5  # refuse to step inside this radius (A(r), C(r) lose precision near r = 2)
K_MAX = 40  # deepest power-of-two level: dt = dt_max / 2^k, k <= K_MAX (phase-04 4.3)
DT_MAX_DEFAULT = 4096.0

# Failure codes returned by the Numba kernels (iteration count otherwise).
_FAIL_P_HALF = -1
_FAIL_X1 = -2
_FAIL_RADIUS = -3
_FAIL_LEVEL = -4
_MESSAGES = {
    _FAIL_P_HALF: "stage 1 (p_half) did not converge in 10 iterations: time step too large",
    _FAIL_X1: "stage 2 (x1) did not converge in 10 iterations: time step too large",
    _FAIL_RADIUS: f"particle inside r = {R_MIN}",
    _FAIL_LEVEL: f"time step would need more than {K_MAX} halvings of dt_max",
}


class GLFConvergenceError(RuntimeError):
    """An implicit stage of the generalized leapfrog failed, or the particle is too deep."""


# --------------------------------------------------------------------------- reference


def glf_step(x: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, int]:
    """One generalized-leapfrog step for one particle. Returns (x1, p1, iterations used).

    Readable reference: uses nebula.physics.gravity directly. Raises GLFConvergenceError.
    """
    if np.linalg.norm(x) <= R_MIN:
        raise GLFConvergenceError(_MESSAGES[_FAIL_RADIUS])
    half = 0.5 * dt
    iters = 0

    p_half = p.copy()
    for _ in range(MAX_ITER):
        new = p - half * gn_dH_dx(x, p_half)
        iters += 1
        done = np.linalg.norm(new - p_half) <= TOL * np.linalg.norm(new)
        p_half = new
        if done:
            break
    else:
        raise GLFConvergenceError(_MESSAGES[_FAIL_P_HALF])

    v0 = gn_dH_dp(x, p_half)
    x1 = x + dt * v0  # explicit first guess
    for _ in range(MAX_ITER):
        new = x + half * (v0 + gn_dH_dp(x1, p_half))
        iters += 1
        done = np.linalg.norm(new - x1) <= TOL * np.linalg.norm(new)
        x1 = new
        if done:
            break
    else:
        raise GLFConvergenceError(_MESSAGES[_FAIL_X1])

    p1 = p_half - half * gn_dH_dx(x1, p_half)
    return x1, p1, iters


def orbit_dt(r: float, eta: float, dt_max: float = DT_MAX_DEFAULT) -> float:
    """Orbit time step eta r^(3/2), rounded down to dt_max / 2^k (k = 0..K_MAX).

    The power-of-two quantization is the seed of Phase 4's block time steps. Raises
    GLFConvergenceError if more than K_MAX halvings would be needed.
    """
    return dt_max / 2.0 ** _level_for(r, eta, dt_max)


# --------------------------------------------------------------------------- Numba kernels


@numba.njit(cache=True)
def _dh_dx(x0, x1, x2, p0, p1, p2):
    """Mirrors nebula.physics.gravity.gn_dH_dx (tested against it)."""
    r = sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    c = -2.0 * (r - 2.0) / r**4
    da = 2.0 / (r * r)
    dc = (6.0 * r - 16.0) / r**5
    px = p0 * x0 + p1 * x1 + p2 * x2
    pp = p0 * p0 + p1 * p1 + p2 * p2
    radial = (0.5 * da * pp + 0.5 * dc * px * px) / r + 1.0 / r**3
    return (
        radial * x0 + c * px * p0,
        radial * x1 + c * px * p1,
        radial * x2 + c * px * p2,
    )


@numba.njit(cache=True)
def _dh_dp(x0, x1, x2, p0, p1, p2):
    """Mirrors nebula.physics.gravity.gn_dH_dp = gn_v_from_p (tested against it)."""
    r = sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    a = (r - 2.0) / r
    c = -2.0 * (r - 2.0) / r**4
    cpx = c * (p0 * x0 + p1 * x1 + p2 * x2)
    return a * p0 + cpx * x0, a * p1 + cpx * x1, a * p2 + cpx * x2


@numba.njit(cache=True)
def _converged(n0, n1, n2, o0, o1, o2):
    d = (n0 - o0) ** 2 + (n1 - o1) ** 2 + (n2 - o2) ** 2
    return d <= TOL * TOL * (n0 * n0 + n1 * n1 + n2 * n2)


@numba.njit(cache=True)
def _glf_one(x0, x1, x2, p0, p1, p2, dt):
    """One step for one particle; mirrors ``glf_step``. Returns (6 new values, iterations or
    a negative failure code)."""
    if x0 * x0 + x1 * x1 + x2 * x2 <= R_MIN * R_MIN:
        return x0, x1, x2, p0, p1, p2, _FAIL_RADIUS
    half = 0.5 * dt
    iters = 0

    h0, h1, h2 = p0, p1, p2
    ok = False
    for _ in range(MAX_ITER):
        g0, g1, g2 = _dh_dx(x0, x1, x2, h0, h1, h2)
        n0, n1, n2 = p0 - half * g0, p1 - half * g1, p2 - half * g2
        iters += 1
        ok = _converged(n0, n1, n2, h0, h1, h2)
        h0, h1, h2 = n0, n1, n2
        if ok:
            break
    if not ok:
        return x0, x1, x2, p0, p1, p2, _FAIL_P_HALF

    v0, v1, v2 = _dh_dp(x0, x1, x2, h0, h1, h2)
    y0, y1, y2 = x0 + dt * v0, x1 + dt * v1, x2 + dt * v2
    ok = False
    for _ in range(MAX_ITER):
        w0, w1, w2 = _dh_dp(y0, y1, y2, h0, h1, h2)
        n0, n1, n2 = x0 + half * (v0 + w0), x1 + half * (v1 + w1), x2 + half * (v2 + w2)
        iters += 1
        ok = _converged(n0, n1, n2, y0, y1, y2)
        y0, y1, y2 = n0, n1, n2
        if ok:
            break
    if not ok:
        return x0, x1, x2, p0, p1, p2, _FAIL_X1

    g0, g1, g2 = _dh_dx(y0, y1, y2, h0, h1, h2)
    return y0, y1, y2, h0 - half * g0, h1 - half * g1, h2 - half * g2, iters


@numba.njit(parallel=True, cache=True)
def _glf_many_kernel(x, p, dt, x_out, p_out, iters):
    for i in numba.prange(x.shape[0]):  # writes only to row i
        r = _glf_one(x[i, 0], x[i, 1], x[i, 2], p[i, 0], p[i, 1], p[i, 2], dt[i])
        x_out[i, 0], x_out[i, 1], x_out[i, 2] = r[0], r[1], r[2]
        p_out[i, 0], p_out[i, 1], p_out[i, 2] = r[3], r[4], r[5]
        iters[i] = r[6]


def glf_step_many(x: np.ndarray, p: np.ndarray, dt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``glf_step`` for N particles at once (Numba, parallel). x, p: (N, 3) float64; dt a
    scalar or (N,). Returns (x1, p1, iterations per particle). Raises GLFConvergenceError."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    dt = np.ascontiguousarray(np.broadcast_to(np.asarray(dt, dtype=np.float64), (len(x),)))
    x_out, p_out = np.empty_like(x), np.empty_like(p)
    iters = np.empty(len(x), dtype=np.int64)
    _glf_many_kernel(x, p, dt, x_out, p_out, iters)
    _raise_on_failure(iters)
    return x_out, p_out, iters


@numba.njit(cache=True)
def _level(r, eta, dt_max):
    """Smallest k with dt_max / 2^k <= eta r^(3/2); K_MAX + 1 if none (failure)."""
    target = eta * r**1.5
    dt = dt_max
    k = 0
    while dt > target and k <= K_MAX:
        dt *= 0.5
        k += 1
    return k


def _level_for(r: float, eta: float, dt_max: float) -> int:
    k = int(_level(r, eta, dt_max))
    if k > K_MAX:
        raise GLFConvergenceError(_MESSAGES[_FAIL_LEVEL])
    return k


@numba.njit(cache=True)
def _integrate_kernel(state0, dt_max, eta, n_end, t_out, s_out, it_out):
    """Integrate one particle on the integer tick timeline (tick = dt_max / 2^K_MAX) until
    tick n_end. eta > 0: adaptive quantized steps (``orbit_dt``); eta <= 0: fixed dt_max.
    Records every step. Returns the number of recorded states, or a negative failure code."""
    tick = dt_max / 2.0**K_MAX
    x0, x1, x2, p0, p1, p2 = state0[0], state0[1], state0[2], state0[3], state0[4], state0[5]
    n = np.int64(0)
    m = 0
    t_out[0] = 0.0
    s_out[0, 0], s_out[0, 1], s_out[0, 2] = x0, x1, x2
    s_out[0, 3], s_out[0, 4], s_out[0, 5] = p0, p1, p2
    it_out[0] = 0
    while n < n_end:
        if m + 1 >= t_out.shape[0]:
            return _FAIL_LEVEL - 1  # output arrays full
        k = 0
        if eta > 0.0:
            k = _level(sqrt(x0 * x0 + x1 * x1 + x2 * x2), eta, dt_max)
        while k <= K_MAX and (np.int64(1) << (K_MAX - k)) > n_end - n:
            k += 1  # never step past the end time
        if k > K_MAX:
            return _FAIL_LEVEL
        steps = np.int64(1) << (K_MAX - k)
        x0, x1, x2, p0, p1, p2, it = _glf_one(x0, x1, x2, p0, p1, p2, dt_max / 2.0**k)
        if it < 0:
            return it
        n += steps
        m += 1
        t_out[m] = n * tick
        s_out[m, 0], s_out[m, 1], s_out[m, 2] = x0, x1, x2
        s_out[m, 3], s_out[m, 4], s_out[m, 5] = p0, p1, p2
        it_out[m] = it
    return m + 1


@dataclass
class Orbit:
    """A recorded single-particle orbit: times t (M,), positions x and momenta p (M, 3), and
    the fixed-point iterations used by each step (M,; 0 for the initial state)."""

    t: np.ndarray
    x: np.ndarray
    p: np.ndarray
    iterations: np.ndarray


def integrate_orbit(
    x0: np.ndarray,
    p0: np.ndarray,
    t_end: float,
    *,
    eta: float | None = None,
    dt: float | None = None,
    dt_max: float = DT_MAX_DEFAULT,
    max_records: int = 5_000_000,
) -> Orbit:
    """Integrate one particle from t = 0 to t_end with the generalized leapfrog (Numba).

    Give either ``eta`` (adaptive steps ``orbit_dt(r, eta, dt_max)``) or ``dt`` (fixed step;
    t_end should then be a multiple of dt). Time is counted in integer ticks of
    dt_max / 2^K_MAX, so it never drifts. Raises GLFConvergenceError on failure.
    """
    if (eta is None) == (dt is None):
        raise ValueError("give exactly one of eta (adaptive) or dt (fixed)")
    if dt is not None:
        dt_max, eta = float(dt), 0.0
    tick = dt_max / 2.0**K_MAX
    n_end = np.int64(round(t_end / tick))
    state0 = np.concatenate([np.asarray(x0, float), np.asarray(p0, float)])
    t = np.empty(max_records)
    s = np.empty((max_records, 6))
    it = np.empty(max_records, dtype=np.int64)
    m = _integrate_kernel(state0, float(dt_max), float(eta), n_end, t, s, it)
    if m == _FAIL_LEVEL - 1:
        raise ValueError(f"more than max_records={max_records} steps; raise max_records")
    if m < 0:
        raise GLFConvergenceError(_MESSAGES[int(m)])
    return Orbit(t[:m].copy(), s[:m, :3].copy(), s[:m, 3:].copy(), it[:m].copy())


def _raise_on_failure(iters: np.ndarray) -> None:
    bad = iters < 0
    if np.any(bad):
        first = int(np.argmax(bad))
        raise GLFConvergenceError(f"particle {first}: {_MESSAGES[int(iters[first])]}")


# --------------------------------------------------------------------------- Newtonian control
#
# Kick-drift-kick leapfrog for H = |p|^2/2 - 1/r (physics.gravity.newton_hamiltonian): exact
# symplectic and time-reversible, second order (Phase 6 control run R6.4).


def kdk_newton_step(x: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Readable reference: p += dt/2 a(x); x += dt p; p += dt/2 a(x) (one particle or many)."""
    from nebula.physics.gravity import newton_accel

    p_half = p + 0.5 * dt * newton_accel(x)
    x1 = x + dt * p_half
    return x1, p_half + 0.5 * dt * newton_accel(x1)


@numba.njit(parallel=True, cache=True)
def _kdk_newton_kernel(x, p, dt, x_out, p_out):
    for i in numba.prange(x.shape[0]):  # writes only to row i
        # mirrors physics.gravity.newton_accel: a = -x / r^3
        r3 = (x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2) ** 1.5
        h = 0.5 * dt[i]
        q0 = p[i, 0] - h * x[i, 0] / r3
        q1 = p[i, 1] - h * x[i, 1] / r3
        q2 = p[i, 2] - h * x[i, 2] / r3
        y0 = x[i, 0] + dt[i] * q0
        y1 = x[i, 1] + dt[i] * q1
        y2 = x[i, 2] + dt[i] * q2
        s3 = (y0 * y0 + y1 * y1 + y2 * y2) ** 1.5
        x_out[i, 0], x_out[i, 1], x_out[i, 2] = y0, y1, y2
        p_out[i, 0] = q0 - h * y0 / s3
        p_out[i, 1] = q1 - h * y1 / s3
        p_out[i, 2] = q2 - h * y2 / s3


def kdk_newton_many(x: np.ndarray, p: np.ndarray, dt) -> tuple[np.ndarray, np.ndarray]:
    """``kdk_newton_step`` for N particles (Numba, parallel); dt scalar or (N,)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    dt = np.ascontiguousarray(np.broadcast_to(np.asarray(dt, dtype=np.float64), (len(x),)))
    x_out, p_out = np.empty_like(x), np.empty_like(p)
    _kdk_newton_kernel(x, p, dt, x_out, p_out)
    return x_out, p_out
