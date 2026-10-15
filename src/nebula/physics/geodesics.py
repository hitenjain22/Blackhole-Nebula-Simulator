"""Light rays around a non-spinning black hole: float64 reference integrators.

Plan: phases/phase-02-blackhole-render.md "Background" and Step 2.1; values in
reference/expected-values.md §3. Units G = c = M = 1; the horizon is at r = 2, the photon
sphere at r = 3, and rays with impact parameter b < 3 sqrt(3) are captured.

A light ray stays in a plane through the hole. With u = 1/r and the polar angle phi in that
plane (any GR text, e.g. Misner, Thorne & Wheeler 1973, ch. 25; Chandrasekhar 1983, ch. 3):

    d^2u/dphi^2 = 3u^2 - u                          (integrated: no stall at turning points)
    (du/dphi)^2 = 1/b^2 - u^2 (1 - 2u)              (first integral; used for initial slope)

Camera: a static observer at distance D. For a viewing direction at angle psi from the
direction *toward* the hole, measured in the observer's own frame,
    b = D sin(psi) / sqrt(1 - 2/D),   u0 = 1/D,   du/dphi(0) = +-sqrt(1/b^2 - u0^2 (1 - 2 u0))
(+ for psi < 90 deg, looking inward). Ray-plane basis: e1 = x_cam/|x_cam| (outward), e2 =
the part of the viewing direction perpendicular to e1. A ray escapes when u returns to 0 at
phi_inf; the sky direction it sees is cos(phi_inf) e1 + sin(phi_inf) e2. It is captured
when u reaches 1/2 (r = 2).

These SciPy (DOP853) integrators are the slow, exact references for the GPU ray tracer
(render/shaders) and its tests. Not for rendering.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

B_CRIT = 3.0 * np.sqrt(3.0)  # critical impact parameter (photon sphere r = 3)
PHI_MAX = 100.0  # rad; a ray still orbiting after this is reported as captured (NaN)
RTOL, ATOL = 1e-12, 1e-15


def _rhs(_phi, s):
    u, du = s
    return [du, 3.0 * u * u - u]


def _event_escape(_phi, s):
    return s[0]


def _event_capture(_phi, s):
    return s[0] - 0.5


_event_escape.terminal = True
_event_escape.direction = -1.0
_event_capture.terminal = True


def _trace(u0: float, du0: float, dense: bool = False):
    """Integrate a ray from (u0, du0) at phi = 0. Returns (phi_inf or NaN, solution)."""
    sol = solve_ivp(
        _rhs,
        (0.0, PHI_MAX),
        [u0, du0],
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        events=[_event_escape, _event_capture],
        dense_output=dense,
    )
    if sol.t_events[0].size:
        return float(sol.t_events[0][0]), sol
    return float("nan"), sol  # captured, or still circling the photon sphere at PHI_MAX


def schw_impact_parameter(D: float, psi: float) -> float:
    """Impact parameter b = D sin(psi) / sqrt(1 - 2/D) of the ray a static observer at D sees
    at angle psi from the hole (aberration-free static frame). Units GM/c^2."""
    return D * np.sin(psi) / np.sqrt(1.0 - 2.0 / D)


def _initial_state(D: float, psi: float) -> tuple[float, float]:
    u0 = 1.0 / D
    b = schw_impact_parameter(D, psi)
    slope2 = 1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * u0)
    du0 = np.sqrt(max(slope2, 0.0))  # exactly 0 at psi = 90 deg (turning point at the camera)
    return u0, (du0 if psi < 0.5 * np.pi else -du0)


def schw_ray_escape_angle(D: float, psi: float) -> float:
    """phi_inf (rad) at which the ray seen at angle psi (rad, 0..pi) from the hole by a static
    observer at r = D escapes to infinity, or NaN if captured. The ray sees the sky in the
    direction cos(phi_inf) e1 + sin(phi_inf) e2 (module docstring). Exactly radial rays:
    psi = 0 is captured; psi = pi sees e1 (phi_inf = 0)."""
    if D <= 3.0:
        raise ValueError(f"camera must be outside the photon sphere, got D = {D}")
    if psi <= 0.0:
        return float("nan")
    if psi >= np.pi:
        return 0.0
    u0, du0 = _initial_state(D, psi)
    return _trace(u0, du0)[0]


def schw_deflection(b: float) -> float:
    """Total deflection (rad) of a ray from infinity with impact parameter b: phi_inf - pi.
    NaN if captured (b < 3 sqrt(3))."""
    phi_inf, _ = _trace(0.0, 1.0 / b)
    return phi_inf - np.pi


def schw_shadow_angle(D: float) -> float:
    """Angular radius (rad) of the shadow seen by a static observer at D >= 3 (Synge 1966):
    sin(psi_sh) = (3 sqrt(3) / D) sqrt(1 - 2/D)."""
    if D < 3.0:
        raise ValueError(f"formula valid for D >= 3, got D = {D}")
    return float(np.arcsin(B_CRIT / D * np.sqrt(1.0 - 2.0 / D)))


def schw_einstein_ring_angle(D: float, order: int = 0) -> float:
    """Viewing angle psi (rad) of the Einstein ring of a source at infinity exactly behind the
    hole (sky direction -e1): the ray with phi_inf = (2 order + 1) pi. order 0 = primary ring,
    1 = secondary (one extra loop around the hole)."""
    target = (2 * order + 1) * np.pi
    psi_sh = schw_shadow_angle(D)
    return brentq(
        lambda psi: schw_ray_escape_angle(D, psi) - target,
        psi_sh * (1.0 + 1e-9),
        0.5 * np.pi,
        xtol=1e-15,
        rtol=1e-15,
    )


def schw_ray_path(D: float, psi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points along the ray seen at angle psi from a camera at D, uniformly spaced in phi
    from the camera to escape (until r = 10 D) or capture (r = 2). Returns (phi, positions)
    with positions (n, 2) in the ray plane: r (cos phi, sin phi) on the (e1, e2) basis
    (3D position = pos[:, 0] e1 + pos[:, 1] e2). Used for plots and Phase 8 tests."""
    u0, du0 = _initial_state(D, psi)
    phi_inf, sol = _trace(u0, du0, dense=True)
    phi_end = sol.t[-1]
    phi = np.linspace(0.0, phi_end, n)
    u = sol.sol(phi)[0]
    if np.isfinite(phi_inf):  # stop where r = 10 D on the way out, not at infinity
        far = u < 0.1 / D
        far[: int(np.argmax(u))] = False
        if np.any(far):
            phi = np.linspace(0.0, phi[np.argmax(far)], n)
            u = sol.sol(phi)[0]
    r = 1.0 / u
    return phi, np.column_stack([r * np.cos(phi), r * np.sin(phi)])
