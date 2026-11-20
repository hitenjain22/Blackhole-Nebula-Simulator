"""Light rays around a spinning (Kerr) black hole: the float64 reference
(phases/phase-11-spin.md 11.2). Units G = c = M = 1; spin a = J / M along +z, 0 <= a < 1.

Cartesian Kerr-Schild coordinates (Kerr 1963; Visser 2007, arXiv:0706.0622, Eqs. 10-13), with
signature (-+++):

    g_mu_nu = eta_mu_nu + f l_mu l_nu,      g^mu_nu = eta^mu_nu - f l^mu l^nu
    f = 2 r^3 / (r^4 + a^2 z^2)
    l_mu = (1, (r x + a y) / (r^2 + a^2), (r y - a x) / (r^2 + a^2), z / r),
    l^mu = (-1, l_x, l_y, l_z)  (null for both metrics)
    r(x, y, z) > 0 from  r^4 - (x^2 + y^2 + z^2 - a^2) r^2 - a^2 z^2 = 0

r is the Boyer-Lindquist radius. Smooth through the horizon r_h = 1 + sqrt(1 - a^2) and on the
spin axis (Boyer-Lindquist coordinates have a pole singularity there). With a = 0 the space
part is the Schwarzschild geometry (ingoing Eddington-Finkelstein time).

Rays: H(x, p) = (1/2) g^mu_nu p_mu p_nu = 0 for light; along a photon's path
dx^i/dlambda = dH/dp_i = p_i - f s l_i,  dp_i/dlambda = -dH/dx^i,  s = l^mu p_mu = -p_t + l . p,
with p_t = -1 (a constant: the metric does not depend on t). Camera rays are the photons that
ARRIVE at the camera, followed backward in lambda (dx/dlambda' = -dH/dp, dp/dlambda' = +dH/dx).
Tracing instead a photon sent out from the camera gives the mirror image: reversing time also
reverses the spin (the shadow's flat side lands on the wrong side; caught by test 5).
dH/dx^i is written out analytically below (``hamiltonian_grad_x``) and tested against
finite differences (test 3).

The derivatives of r follow from differentiating the quartic (its r-derivative simplifies to
2 (r^4 + a^2 z^2) / r):
    dr/dx = x r^3 / D,  dr/dy = y r^3 / D,  dr/dz = z r (r^2 + a^2) / D,   D = r^4 + a^2 z^2.

The camera is a static observer (4-velocity along d/dt; exists outside the ergosphere, so
r_cam > 2 always works). With a = 0 it is Phase 2's static observer (test 1). Its local
directions are an orthonormal triad built by Gram-Schmidt from the flat-space radial,
polar and azimuthal directions at the camera (for a = 0 exactly Phase 2's static frame).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_ESCAPE = 1.0e4  # rays going outward past this are escaped (bending left there ~ 4/b r^-1)
HORIZON_MARGIN = 1.0e-3  # captured at r < r_h + HORIZON_MARGIN (phase-11 11.2)


def horizon_radius(a: float) -> float:
    """Outer horizon r_h = 1 + sqrt(1 - a^2) (Boyer-Lindquist r)."""
    return 1.0 + np.sqrt(1.0 - a * a)


def ks_radius(x, a: float):
    """Boyer-Lindquist r of Cartesian Kerr-Schild points x (..., 3): the positive root of
    r^4 - (rho^2 - a^2) r^2 - a^2 z^2 = 0, i.e.
    r^2 = (rho^2 - a^2) / 2 + sqrt((rho^2 - a^2)^2 / 4 + a^2 z^2)."""
    x = np.asarray(x, float)
    rho2 = np.sum(x * x, axis=-1)
    b = rho2 - a * a
    r2 = 0.5 * b + np.sqrt(0.25 * b * b + a * a * x[..., 2] ** 2)
    return np.sqrt(r2)


def metric_parts(x, a: float):
    """(r, f, l) at x: f = 2 r^3 / (r^4 + a^2 z^2) and the spatial l_i (..., 3)."""
    x = np.asarray(x, float)
    r = ks_radius(x, a)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    b = r * r + a * a
    f = 2.0 * r**3 / (r**4 + a * a * Z * Z)
    lv = np.stack([(r * X + a * Y) / b, (r * Y - a * X) / b, Z / r], axis=-1)
    return r, f, lv


def metric(x, a: float) -> np.ndarray:
    """g_mu_nu (4 x 4) at one point x (covariant; index 0 = t)."""
    _, f, lv = metric_parts(x, a)
    l4 = np.concatenate([[1.0], lv])
    return np.diag([-1.0, 1.0, 1.0, 1.0]) + f * np.outer(l4, l4)


def inverse_metric(x, a: float) -> np.ndarray:
    """g^mu_nu (4 x 4) at one point x: eta - f l^mu l^nu with l^mu = (-1, l_i)."""
    _, f, lv = metric_parts(x, a)
    lu = np.concatenate([[-1.0], lv])
    return np.diag([-1.0, 1.0, 1.0, 1.0]) - f * np.outer(lu, lu)


def hamiltonian(x, p, a: float, p_t: float = -1.0):
    """H = (1/2) g^mu_nu p_mu p_nu = (1/2) (-p_t^2 + |p|^2 - f s^2), s = -p_t + l . p."""
    _, f, lv = metric_parts(x, a)
    p = np.asarray(p, float)
    s = -p_t + np.sum(lv * p, axis=-1)
    return 0.5 * (-p_t * p_t + np.sum(p * p, axis=-1) - f * s * s)


def hamiltonian_grad_x(x, p, a: float, p_t: float = -1.0) -> np.ndarray:
    """dH/dx^i (3,) at one point (analytic; module doc): -(1/2) s^2 df/dx^i - f s d(l.p)/dx^i."""
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    X, Y, Z = x
    px, py, pz = p
    r, f, lv = metric_parts(x, a)
    D = r**4 + a * a * Z * Z
    dr = np.array([X * r**3, Y * r**3, Z * r * (r * r + a * a)]) / D
    # f = 2 r^3 / D(r, z)
    df_dr = (6.0 * r * r * D - 2.0 * r**3 * 4.0 * r**3) / (D * D)
    df = df_dr * dr
    df[2] += -2.0 * r**3 * 2.0 * a * a * Z / (D * D)
    # l . p = A / B + z pz / r,  A = r (x px + y py) + a (y px - x py),  B = r^2 + a^2
    B = r * r + a * a
    A = r * (X * px + Y * py) + a * (Y * px - X * py)
    dA = dr * (X * px + Y * py) + r * np.array([px, py, 0.0]) + a * np.array([-py, px, 0.0])
    dB = 2.0 * r * dr
    dlp = (dA * B - A * dB) / (B * B) - Z * pz * dr / (r * r)
    dlp[2] += pz / r
    s = -p_t + lv @ p
    return -0.5 * s * s * df - f * s * dlp


def ray_rhs(x, p, a: float, p_t: float = -1.0):
    """(dx/dlambda, dp/dlambda) along a photon's path, forward in time (module doc)."""
    _, f, lv = metric_parts(x, a)
    s = -p_t + lv @ np.asarray(p, float)
    return np.asarray(p, float) - f * s * lv, -hamiltonian_grad_x(x, p, a, p_t)


def backward_rhs(x, p, a: float, p_t: float = -1.0):
    """(dx/dlambda', dp/dlambda') with lambda' = -lambda: a camera ray, followed from the
    camera back toward where its light came from."""
    dx, dp = ray_rhs(x, p, a, p_t)
    return -dx, -dp


def static_observer_frame(x_cam, a: float):
    """(u^mu, e_r^mu, e_th^mu, e_ph^mu): the static observer at x_cam and its orthonormal triad
    by Gram-Schmidt (metric g) from the flat radial, polar and azimuthal coordinate directions,
    each first made orthogonal to u. Needs g_tt < 0 (outside the ergosphere)."""
    x_cam = np.asarray(x_cam, float)
    g = metric(x_cam, a)
    if g[0, 0] >= 0.0:
        raise ValueError("camera inside the ergosphere: no static observer there")
    u = np.array([1.0, 0.0, 0.0, 0.0]) / np.sqrt(-g[0, 0])
    rho = np.linalg.norm(x_cam)
    rh = x_cam / rho
    zh = np.array([0.0, 0.0, 1.0])
    ph = np.cross(zh, rh)
    if np.linalg.norm(ph) < 1e-12:  # on the axis: any perpendicular pair
        ph = np.cross(np.array([1.0, 0.0, 0.0]), rh)
    ph /= np.linalg.norm(ph)
    th = np.cross(ph, rh)
    triad = []
    for v3 in (rh, th, ph):
        v = np.concatenate([[0.0], v3])
        v = v + (v @ g @ u) * u  # orthogonal to u (g(u, u) = -1)
        for e in triad:
            v = v - (v @ g @ e) * e
        triad.append(v / np.sqrt(v @ g @ v))
    return (u, *triad)


def initial_momentum(x_cam, n_local, a: float) -> np.ndarray:
    """Covariant spatial p_i (with p_t = -1) of the photon that arrives at the camera from the
    local unit direction n_local = (n_r, n_th, n_ph) of the static observer's frame (n_r = -1:
    the camera looks straight at the hole). The photon itself moves along -n."""
    u, er, eth, eph = static_observer_frame(x_cam, a)
    n = np.asarray(n_local, float)
    k = u - (n[0] * er + n[1] * eth + n[2] * eph)  # the arriving photon, contravariant
    p_cov = metric(x_cam, a) @ k
    return p_cov[1:] / (-p_cov[0])  # scale so p_t = -1


def flat_to_local(x_cam, d) -> np.ndarray:
    """Local direction components (n_r, n_th, n_ph) of a flat-space unit direction d at the
    camera, on the same flat radial / polar / azimuthal axes the frame is built from."""
    x_cam = np.asarray(x_cam, float)
    rh = x_cam / np.linalg.norm(x_cam)
    ph = np.cross(np.array([0.0, 0.0, 1.0]), rh)
    if np.linalg.norm(ph) < 1e-12:
        ph = np.cross(np.array([1.0, 0.0, 0.0]), rh)
    ph /= np.linalg.norm(ph)
    th = np.cross(ph, rh)
    d = np.asarray(d, float)
    return np.array([d @ rh, d @ th, d @ ph])


def trace_ray(x_cam, n_local, a: float, r_escape: float = R_ESCAPE, rtol: float = 1e-11,
              dense: bool = False):  # fmt: skip
    """Follow the photon arriving at a static camera from local direction n_local back to its
    origin. Returns (status, x_end, p_end, sol):
    status "captured" (r < r_h + HORIZON_MARGIN), "escaped" (r > r_escape, moving out) or
    "orbiting" (neither within the affine-length cap). DOP853, rtol/atol ``rtol``."""
    x0 = np.asarray(x_cam, float)
    p0 = initial_momentum(x0, n_local, a)
    r_cap = horizon_radius(a) + HORIZON_MARGIN

    def rhs(_lam, s):
        dx, dp = backward_rhs(s[:3], s[3:], a)
        return np.concatenate([dx, dp])

    def captured(_lam, s):
        return ks_radius(s[:3], a) - r_cap

    def escaped(_lam, s):
        return ks_radius(s[:3], a) - r_escape

    captured.terminal = True
    escaped.terminal = True
    escaped.direction = 1.0
    lam_max = 50.0 * r_escape + 1.0e4
    sol = solve_ivp(rhs, (0.0, lam_max), np.concatenate([x0, p0]), method="DOP853",
                    rtol=rtol, atol=rtol, events=(captured, escaped),
                    dense_output=dense)  # fmt: skip
    end = sol.y[:, -1]
    if sol.t_events[0].size:
        status = "captured"
    elif sol.t_events[1].size:
        status = "escaped"
    else:
        status = "orbiting"
    return status, end[:3], end[3:], sol


def shadow_edge_alpha(a: float, incl_deg: float = 90.0):
    """Exact critical-curve abscissae alpha at beta = 0 for an observer at infinity at
    inclination ``incl_deg`` (90: equatorial): from Bardeen 1973, the spherical photon orbits
    r_p with xi(r) = (r^2 (3 - r) - a^2 (r + 1)) / (a (r - 1)) (prograde and retrograde roots
    of eta(r) = 0 in the equatorial plane, eta = r^3 (4 a^2 - r (r - 3)^2) / (a^2 (r - 1)^2)),
    alpha = -xi / sin(i). Returns (alpha_min, alpha_max). For a = 0.9: (-2.844421, 6.832319)
    (expected-values §8)."""
    from scipy.optimize import brentq

    def eta(r):
        return r**3 * (4.0 * a * a - r * (r - 3.0) ** 2)

    def xi(r):
        return (r * r * (3.0 - r) - a * a * (r + 1.0)) / (a * (r - 1.0))

    r_pro = brentq(eta, 1.0 + 1e-9, 3.0)
    r_ret = brentq(eta, 3.0, 4.0 + 1e-9)
    s = np.sin(np.radians(incl_deg))
    vals = sorted([-xi(r_pro) / s, -xi(r_ret) / s])
    return vals[0], vals[1]


# ---------------------------------------------------------------- gas colours (approximate)
#
# The gas moves under Schwarzschild-shaped (GN) gravity, so its 4-velocity around a spinning
# hole is only approximate anyway (phase-11 11.2: "use the Phase 9 static-frame velocity
# (gamma, beta) as the velocity relative to the local ZAMO"). The renderer keeps Phase 9's
# static-frame formulas and changes only the lapse: the static observer's redshift factor
# sqrt(-g_tt) = sqrt(1 - f) replaces sqrt(1 - 2/r) (exact for static emitters and cameras in
# Kerr; equal to it at a = 0, where f = 2/r). The direction of the light in the gas's frame is
# Phase 9's static-frame direction with that lapse. Corner cut (DECISIONS 2026-11-20): Doppler
# shifts near a spinning hole are approximate; the shadow, lensing and gravitational redshift
# are not.


def static_lapse(x, a: float):
    """sqrt(-g_tt) = sqrt(1 - f) at Kerr-Schild points x (..., 3): the redshift factor of a
    static emitter (Bardeen, Press & Teukolsky 1972); 0 at the ergosphere (clamped there)."""
    _, f, _ = metric_parts(x, a)
    return np.sqrt(np.clip(1.0 - f, 0.0, None))


def static_frame_direction_kerr(x, e, a: float):
    """physics.redshift.static_frame_direction with the Kerr lapse sqrt(1 - f) for
    sqrt(1 - 2/r) (module note above); the same function at a = 0."""
    x = np.asarray(x, float)
    e = np.asarray(e, float)
    rr = np.linalg.norm(x, axis=-1)
    r_hat = x / rr[..., None]
    er = np.sum(e * r_hat, axis=-1)
    n = e + (er / static_lapse(x, a) - er)[..., None] * r_hat
    return n / np.linalg.norm(n, axis=-1)[..., None]


def redshift_factor_kerr(x, gamma_beta, n_hat, a: float, f_cam: float):
    """g = [sqrt(1 - f(x)) / sqrt(1 - f_cam)] / [gamma - (gamma beta) . n_hat]: Phase 9's
    redshift_factor with the Kerr static lapse (module note); f_cam = f at the camera."""
    gb = np.asarray(gamma_beta, float)
    gamma = np.sqrt(1.0 + np.sum(gb * gb, axis=-1))
    grav = static_lapse(x, a) / np.sqrt(1.0 - f_cam)
    return grav / (gamma - np.sum(gb * np.asarray(n_hat, float), axis=-1))
