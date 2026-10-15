"""Exact Riemann solver for an ideal gas (phases/phase-03-sph-core.md 3.2 test 5; Phase 6 test 4).

Two uniform states (rho, v, P) meeting at x0 at t = 0 in 1D. The solution consists of a left
wave (rarefaction or shock), a contact discontinuity moving at u*, and a right wave. The star
pressure p* solves f_L(p) + f_R(p) + (v_R - v_L) = 0 with (Toro 2009, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, ch. 4; same method as reference/scripts/13 and 14):

    f_K(p) = (p - P_K) sqrt(A_K / (p + B_K))                       p > P_K (shock)
           = 2 c_K / (gamma - 1) [ (p / P_K)^((gamma-1)/(2 gamma)) - 1 ]   p <= P_K (rarefaction)
    A_K = 2 / ((gamma + 1) rho_K),   B_K = (gamma - 1)/(gamma + 1) P_K
    u* = (v_L + v_R)/2 + (f_R(p*) - f_L(p*))/2

Dimensionless (any consistent units). Pure functions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class RiemannSolution:
    gamma: float
    rho_l: float
    v_l: float
    p_l: float
    rho_r: float
    v_r: float
    p_r: float
    p_star: float
    u_star: float
    rho_star_l: float
    rho_star_r: float
    # wave speeds: left wave (head, tail; equal for a shock), right wave (tail, head)
    s_l_head: float
    s_l_tail: float
    s_r_tail: float
    s_r_head: float


def _f(p, rho, pk, ck, g):
    if p > pk:
        a = 2.0 / ((g + 1.0) * rho)
        b = (g - 1.0) / (g + 1.0) * pk
        return (p - pk) * np.sqrt(a / (p + b))
    return 2.0 * ck / (g - 1.0) * ((p / pk) ** ((g - 1.0) / (2.0 * g)) - 1.0)


def solve_riemann(rho_l, v_l, p_l, rho_r, v_r, p_r, gamma) -> RiemannSolution:
    """Star state and wave speeds of the Riemann problem (no vacuum generation assumed)."""
    g = gamma
    c_l, c_r = np.sqrt(g * p_l / rho_l), np.sqrt(g * p_r / rho_r)
    p_star = brentq(
        lambda p: _f(p, rho_l, p_l, c_l, g) + _f(p, rho_r, p_r, c_r, g) + (v_r - v_l),
        1e-14 * min(p_l, p_r), 100.0 * max(p_l, p_r), xtol=1e-15, rtol=1e-15,
    )  # fmt: skip
    u_star = 0.5 * (v_l + v_r) + 0.5 * (
        _f(p_star, rho_r, p_r, c_r, g) - _f(p_star, rho_l, p_l, c_l, g)
    )
    gm = (g - 1.0) / (g + 1.0)

    def star_density(rho, pk, ck, side):
        if p_star > pk:  # shock
            rho_s = rho * (p_star / pk + gm) / (gm * p_star / pk + 1.0)
            s = np.sqrt((g + 1.0) / (2.0 * g) * p_star / pk + (g - 1.0) / (2.0 * g))
            speed = (v_l - ck * s) if side == "l" else (v_r + ck * s)
            return rho_s, speed, speed
        rho_s = rho * (p_star / pk) ** (1.0 / g)  # rarefaction
        c_s = ck * (p_star / pk) ** ((g - 1.0) / (2.0 * g))
        if side == "l":
            return rho_s, v_l - ck, u_star - c_s
        return rho_s, v_r + ck, u_star + c_s

    rho_sl, l_head, l_tail = star_density(rho_l, p_l, c_l, "l")
    rho_sr, r_head, r_tail = star_density(rho_r, p_r, c_r, "r")
    return RiemannSolution(
        g, rho_l, v_l, p_l, rho_r, v_r, p_r, p_star, u_star, rho_sl, rho_sr,
        l_head, l_tail, r_tail, r_head,
    )  # fmt: skip


def sample(sol: RiemannSolution, x, t: float, x0: float = 0.0):
    """(rho, v, P) at positions x and time t > 0 (diaphragm at x0)."""
    g = sol.gamma
    xi = (np.asarray(x, float) - x0) / t
    rho, v, p = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
    c_l = np.sqrt(g * sol.p_l / sol.rho_l)
    c_r = np.sqrt(g * sol.p_r / sol.rho_r)
    for k, s in enumerate(xi):
        if s < sol.u_star:  # left of the contact
            if sol.p_star > sol.p_l:  # left shock
                state = (
                    (sol.rho_l, sol.v_l, sol.p_l)
                    if s < sol.s_l_head
                    else (sol.rho_star_l, sol.u_star, sol.p_star)
                )
            elif s < sol.s_l_head:
                state = (sol.rho_l, sol.v_l, sol.p_l)
            elif s > sol.s_l_tail:
                state = (sol.rho_star_l, sol.u_star, sol.p_star)
            else:  # inside the left rarefaction fan
                vv = 2.0 / (g + 1.0) * (c_l + (g - 1.0) / 2.0 * sol.v_l + s)
                cc = vv - s
                rr = sol.rho_l * (cc / c_l) ** (2.0 / (g - 1.0))
                state = (rr, vv, sol.p_l * (rr / sol.rho_l) ** g)
        else:  # right of the contact
            if sol.p_star > sol.p_r:  # right shock
                state = (
                    (sol.rho_r, sol.v_r, sol.p_r)
                    if s > sol.s_r_head
                    else (sol.rho_star_r, sol.u_star, sol.p_star)
                )
            elif s > sol.s_r_head:
                state = (sol.rho_r, sol.v_r, sol.p_r)
            elif s < sol.s_r_tail:
                state = (sol.rho_star_r, sol.u_star, sol.p_star)
            else:
                vv = 2.0 / (g + 1.0) * (-c_r + (g - 1.0) / 2.0 * sol.v_r + s)
                cc = s - vv
                rr = sol.rho_r * (cc / c_r) ** (2.0 / (g - 1.0))
                state = (rr, vv, sol.p_r * (rr / sol.rho_r) ** g)
        rho[k], v[k], p[k] = state
    return rho, v, p
