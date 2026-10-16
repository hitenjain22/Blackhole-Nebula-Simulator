"""Orbits around a Schwarzschild black hole: constants of motion, precession, special radii.

Plan: phases/phase-01-orbits.md Step 1.3; values in reference/expected-values.md §2, §8.
Units: G = c = M = 1. Because GN orbits have exactly the Schwarzschild shape r(phi), with
the same angular momentum L and GN energy eps = (E^2 - 1)/2 (Tejeda & Rosswog 2013,
arXiv:1303.4068, §2), every function here applies to both.

Schwarzschild bound orbits (any GR text, e.g. Misner, Thorne & Wheeler 1973, ch. 25;
Chandrasekhar 1983 ch. 3), with u = 1/r:

    (du/dphi)^2 = (E^2 - 1)/L^2 + 2u/L^2 - u^2 + 2u^3
    E^2 = (1 - 2u)(1 + L^2 u^2)  at the turning points u_a = 1/r_apo and u_p = 1/r_peri.

The right-hand side is the cubic 2 (u - u_a)(u_p - u)(u_3 - u) with u_3 = 1/2 - u_a - u_p
(the three roots sum to 1/2), which removes the cancellation of the raw form.
"""

import numpy as np
from scipy.integrate import quad


def schwarzschild_EL_from_turning_points(r_peri: float, r_apo: float) -> tuple[float, float]:  # noqa: N802
    """Energy at infinity E and angular momentum L (per unit mass) of the bound orbit with
    the given turning points, from E^2 = (1 - 2/r)(1 + L^2/r^2) at both radii:

        L^2 = [(1 - 2/r_p) - (1 - 2/r_a)] / [(1 - 2/r_a)/r_a^2 - (1 - 2/r_p)/r_p^2]

    E dimensionless, L in units of GM/c. Raises ValueError unless the turning points belong to
    a real bound orbit (``_turning_u``).
    """
    _turning_u(r_peri, r_apo)
    ap, aa = 1.0 - 2.0 / r_peri, 1.0 - 2.0 / r_apo
    l2 = (ap - aa) / (aa / r_apo**2 - ap / r_peri**2)
    e = np.sqrt(aa * (1.0 + l2 / r_apo**2))
    return float(e), float(np.sqrt(l2))


def gn_energy_from_E(e: float) -> float:  # noqa: N802
    """GN energy (Hamiltonian value) eps = (E^2 - 1)/2, units of c^2 (Tejeda & Rosswog 2013,
    Eq. 2.6)."""
    return 0.5 * (e * e - 1.0)


def apocentre_state(r_peri: float, r_apo: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start state at apocentre, on the +x axis moving in +y (PLAN.md §7): (x0, v0, p0).

    At a turning point the radial velocity is zero, so the GN velocity is purely tangential,
    v_t = L / (f(r_a) r_a) with f = r/(r - 2), and the canonical momentum is p = f v (|p| = L/r_a).
    Units: x in GM/c^2, v and p in c.
    """
    _, ell = schwarzschild_EL_from_turning_points(r_peri, r_apo)
    f = r_apo / (r_apo - 2.0)
    x0 = np.array([r_apo, 0.0, 0.0])
    v0 = np.array([0.0, ell / (f * r_apo), 0.0])
    return x0, v0, f * v0


def _turning_u(r_peri: float, r_apo: float) -> tuple[float, float, float]:
    """(u_p, u_a, u_3). A bound orbit between the turning points exists only if
    u_a <= u_p < u_3, i.e. 2/r_peri + 1/r_apo < 1/2 (otherwise the "pericentre" would be a
    maximum of the potential and the particle would plunge)."""
    u_p, u_a = 1.0 / r_peri, 1.0 / r_apo
    u_3 = 0.5 - u_a - u_p
    if not (r_apo >= r_peri > 0.0 and u_p < u_3):
        raise ValueError(f"no bound orbit with r_peri={r_peri}, r_apo={r_apo}")
    return u_p, u_a, u_3


def orbit_angle_from_pericentre(r_peri: float, r_apo: float, r):
    """Angle (rad) swept from pericentre to radius r along the orbit (either branch):

        phi(r) = integral_{u}^{u_p} du / sqrt(2 (u - u_a)(u_p - u)(u_3 - u)),
                 u = (u_p + u_a)/2 + (u_p - u_a)/2 cos(theta)  ->  dtheta / sqrt(2 (u_3 - u))

    The substitution removes the endpoint singularities (phase-01 Step 1.3). r must lie in
    [r_peri, r_apo]; it may be an array.
    """
    u_p, u_a, u_3 = _turning_u(r_peri, r_apo)
    mid, half = 0.5 * (u_p + u_a), 0.5 * (u_p - u_a)

    def one(rr: float) -> float:
        # theta = arccos((u - mid)/half), written as 2 atan2(sqrt(u_p - u), sqrt(u - u_a)) with
        # both differences formed from r directly: arccos loses ~sqrt(eps) = 1e-8 at theta = pi
        # (caught by the elliptic-integral cross-check in tests/physics/test_orbits.py).
        theta = 2.0 * np.arctan2(
            np.sqrt(max((rr - r_peri) / (rr * r_peri), 0.0)),
            np.sqrt(max((r_apo - rr) / (rr * r_apo), 0.0)),
        )
        val, _ = quad(
            lambda t: 1.0 / np.sqrt(2.0 * (u_3 - mid - half * np.cos(t))),
            0.0,
            theta,
            epsabs=0.0,
            epsrel=2e-14,  # quad's floor is 50 machine epsilons (1.1e-14)
            limit=200,
        )
        return val

    if np.ndim(r) == 0:
        return one(float(r))
    return np.array([one(float(rr)) for rr in np.ravel(r)]).reshape(np.shape(r))


def precession_exact(r_peri: float, r_apo: float) -> float:
    """Exact relativistic periapsis precession per radial orbit (rad): Delta_phi - 2 pi with

    Delta_phi = 2 integral (L/r^2) dr / sqrt(E^2 - (1 - 2/r)(1 + L^2/r^2))
              = 2 * orbit_angle_from_pericentre(r_peri, r_apo, r_apo).
    """
    return 2.0 * orbit_angle_from_pericentre(r_peri, r_apo, r_apo) - 2.0 * np.pi


def precession_weak_field(r_peri: float, r_apo: float) -> float:
    """Leading-order (weak-field) precession 6 pi / (a (1 - e^2)) = 6 pi / p, with the
    semi-latus rectum p = 2 r_p r_a / (r_p + r_a). Rad per orbit. Sanity reference only."""
    return 6.0 * np.pi / (2.0 * r_peri * r_apo / (r_peri + r_apo))


def circularization_radius(ell: float) -> float:
    """Radius of the circular orbit with angular momentum ell (the larger, stable root of
    ell^2 = r^2/(r - 3)):  r = (ell^2 + sqrt(ell^4 - 12 ell^2)) / 2.  Needs ell^2 >= 12."""
    l2 = ell * ell
    if l2 < 12.0:
        raise ValueError(f"no circular orbit for ell^2 = {l2} < 12")
    return 0.5 * (l2 + np.sqrt(l2 * l2 - 12.0 * l2))


def isco_radius(a: float = 0.0, prograde: bool = True) -> float:
    """Innermost stable circular orbit of a Kerr hole with spin a in [0, 1) (Bardeen, Press &
    Teukolsky 1972; expected-values §8):

        Z1 = 1 + (1 - a^2)^(1/3) [(1 + a)^(1/3) + (1 - a)^(1/3)],   Z2 = sqrt(3 a^2 + Z1^2)
        r_isco = 3 + Z2 -/+ sqrt((3 - Z1)(3 + Z1 + 2 Z2))     (- prograde, + retrograde)

    a = 0 gives 6 (Schwarzschild, and GN for any spin: phases/phase-11-spin.md).
    """
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = np.sqrt(3.0 * a * a + z1 * z1)
    root = np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return float(3.0 + z2 - root if prograde else 3.0 + z2 + root)


def turning_points(e_inf, ell):
    """Turning points (r_peri, r_apo) and orbit class of Schwarzschild geodesics with energy at
    infinity E and angular momentum L (per unit mass; phase-06 6.5 "per-particle orbit
    elements"). With u = 1/r, the turning points are the roots of

        E^2 = (1 - 2u)(1 + L^2 u^2)   <=>   2 L^2 u^3 - L^2 u^2 + 2u + (E^2 - 1) = 0,

    whose roots sum to 1/2 (as in ``_turning_u``). Bound orbits (E < 1) with three real roots
    u_a <= u_p <= u_3 move between r_apo = 1/u_a and r_peri = 1/u_p; with one real root there is
    no pericentre barrier and the particle plunges. Returns (r_peri, r_apo, kind) with kind
    0 = bound, 1 = unbound (E >= 1: r_apo = inf), 2 = plunging (r_peri = 0); arrays.
    """
    e_inf = np.atleast_1d(np.asarray(e_inf, float))
    ell = np.atleast_1d(np.asarray(ell, float))
    n = len(e_inf)
    l2 = ell * ell
    # monic cubic u^3 + a u^2 + b u + c = 0 (a = -1/2, b = 1/L^2, c = (E^2 - 1)/(2 L^2)).
    # Its largest real root u_3 is found from the companion-matrix eigenvalues; the other two
    # follow exactly from u_1 + u_2 = 1/2 - u_3 and u_1 u_2 = -c / u_3 (Vieta). This keeps
    # near-circular orbits (u_1 ~ u_2, which an eigen-solver can return as a complex pair)
    # bound: a discriminant that is negative only at round-off level is set to zero.
    a, b, c = -0.5 * np.ones(n), 1.0 / l2, (e_inf**2 - 1.0) / (2.0 * l2)
    comp = np.zeros((n, 3, 3))
    comp[:, 0, :] = np.column_stack([-a, -b, -c])
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    roots = np.linalg.eigvals(comp)
    real_part = np.where(np.abs(roots.imag) <= 1e-7 * np.abs(roots), roots.real, -np.inf)
    u3 = real_part.max(axis=1)  # the largest real root (a real cubic always has one)
    s12 = 0.5 - u3
    p12 = -c / u3
    disc = s12 * s12 - 4.0 * p12
    disc = np.where(np.abs(disc) <= 1e-9 * s12 * s12, 0.0, disc)
    ok = disc >= 0.0
    root = np.sqrt(np.where(ok, disc, 0.0))
    u_p, u_a = 0.5 * (s12 + root), 0.5 * (s12 - root)
    unbound = e_inf >= 1.0
    bound = ~unbound & ok & (u_a > 0.0)
    kind = np.where(unbound, 1, np.where(bound, 0, 2)).astype(np.int64)
    with np.errstate(divide="ignore"):
        r_peri = np.where(ok & (u_p > 0.0), 1.0 / u_p, 0.0)
        r_apo = np.where(
            bound, 1.0 / np.where(u_a > 0, u_a, 1.0), np.where(unbound, np.inf, np.nan)
        )
    kind = np.where(unbound & ~(ok & (u_p > 0.0)), 2, kind)  # unbound but no barrier: plunges
    return r_peri, r_apo, kind
