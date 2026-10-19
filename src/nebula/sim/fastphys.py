"""Numba copies of physics formulas evaluated for many particles every system step (CLAUDE.md
rule 1 exception: each mirrors a physics/ function, named below, and is tested against it in
tests/sim/test_fastphys.py). Added 2026-10-18: as whole-array NumPy code these ran on one core
and limited how much a many-core machine could speed up a run.
"""

import math

import numba
import numpy as np

from nebula import units
from nebula.physics.eos import (
    EOS_TOL,
    T_MAX_SI,
    T_MIN_SI,
    gamma1,
    gasrad_beta_si,
    gasrad_pressure_si,
    gasrad_u_si,
)


@numba.njit(parallel=True, cache=True)
def _gn_v_from_p_kernel(x, p, out):
    for i in numba.prange(x.shape[0]):
        r = math.sqrt(x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1] + x[i, 2] * x[i, 2])
        a = (r - 2.0) / r
        c = -2.0 * (r - 2.0) / r**4
        px = p[i, 0] * x[i, 0] + p[i, 1] * x[i, 1] + p[i, 2] * x[i, 2]
        for k in range(3):
            out[i, k] = a * p[i, k] + c * px * x[i, k]


def gn_v_from_p_nb(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mirrors physics.gravity.gn_v_from_p for (N, 3) arrays: v = A p + C (p.x) x,
    A = (r - 2)/r, C = -2 (r - 2)/r^4 (Tejeda & Rosswog 2013)."""
    x = np.ascontiguousarray(x, np.float64)
    p = np.ascontiguousarray(p, np.float64)
    out = np.empty_like(p)
    _gn_v_from_p_kernel(x, p, out)
    return out


@numba.njit(parallel=True, cache=True)
def _gasrad_t_kernel(rho, u, a1, a_rad, lt_min, lt_max, tol, max_iter, out, ok):
    t_min, t_max = math.exp(lt_min), math.exp(lt_max)
    for i in numba.prange(rho.shape[0]):
        r, uu = rho[i], u[i]
        lo, hi = lt_min, lt_max
        t0 = min(uu / a1, (uu * r / a_rad) ** 0.25)
        lt = math.log(min(max(t0, t_min), t_max))
        ok[i] = False
        for _ in range(max_iter):
            t = math.exp(lt)
            t4 = t**4
            f = a1 * t + a_rad * t4 / r - uu
            dfdl = a1 * t + 4.0 * a_rad * t4 / r
            if f < 0:
                lo = lt
            if f > 0:
                hi = lt
            new = lt - f / dfdl
            if new <= lo or new >= hi or not math.isfinite(new):
                new = 0.5 * (lo + hi)
            step = abs(new - lt)
            lt = new
            if step < tol:
                ok[i] = True
                break
        out[i] = math.exp(lt)


def gasrad_temperature_nb(rho_si, u_si, tol: float = EOS_TOL, max_iter: int = 200) -> np.ndarray:
    """Mirrors physics.eos.gasrad_temperature_si (same Newton-on-log-T iteration inside the
    narrowing bracket [1, 1e13] K, element by element): T [K] from rho [kg/m^3], u [J/kg]."""
    rho = np.ascontiguousarray(np.broadcast_to(np.asarray(rho_si, float), np.shape(u_si)))
    u = np.ascontiguousarray(np.asarray(u_si, float))
    if np.any(u < gasrad_u_si(rho, T_MIN_SI)) or np.any(u > gasrad_u_si(rho, T_MAX_SI)):
        raise ValueError("u outside the EOS bracket [1, 1e13] K")
    a1 = 1.5 * units.K_B / (units.MU * units.M_P)
    out = np.empty_like(u)
    ok = np.empty(u.shape, np.bool_)
    _gasrad_t_kernel(rho.ravel(), u.ravel(), a1, units.A_RAD, np.log(T_MIN_SI), np.log(T_MAX_SI),
                     tol, max_iter, out.ravel(), ok.ravel())  # fmt: skip
    if not np.all(ok):
        raise RuntimeError("gas + radiation EOS: temperature did not converge")
    return out


def gasrad_state_nb(rho_code, u_code, gas_units=None):
    """Mirrors physics.eos.gasrad_state with the temperature from ``gasrad_temperature_nb``:
    (P [code], c_s [c], T [K], beta_gas)."""
    gu = units.GasUnits() if gas_units is None else gas_units
    rho_si = gu.density_to_si(np.asarray(rho_code, float))
    u_si = units.specific_energy_to_si(np.asarray(u_code, float))
    t = gasrad_temperature_nb(rho_si, u_si)
    p_si = gasrad_pressure_si(rho_si, t)
    beta = gasrad_beta_si(rho_si, t)
    p_code = p_si / (gu.density_kg_m3 * units.C**2)
    cs = np.sqrt(gamma1(beta) * p_code / np.asarray(rho_code, float))
    return p_code, cs, t, beta
