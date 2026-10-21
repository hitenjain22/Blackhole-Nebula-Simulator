"""Equations of state (phases/phase-03-sph-core.md 3.1; Phase 6 adds gas + radiation).

Code units throughout: rho in code density, u (specific internal energy) in c^2,
P in code pressure (rho c^2), sound speeds in c.

Ideal gas (adiabatic index gamma; Phantom, Price et al. 2018, Eq. 27-28):
    P = (gamma - 1) rho u,        c_s = sqrt(gamma P / rho) = sqrt(gamma (gamma - 1) u)
Isothermal:
    P = c_s^2 rho
Locally isothermal (phase-07 7.4 ring test; Lodato & Price 2010 §3; Phantom ieos = 3):
    c_s(r) = c_s0 (r / r_0)^(-q),  r = distance from the hole, q = 3/4 so that
    nu = alpha c_s^2 / Omega_K is constant in r (the Lynden-Bell & Pringle case)
Radiation limit (Phase 6 test 4: a gamma = 4/3 gas):
    P = rho u / 3,                c_s = sqrt(4/3 P / rho) = sqrt(4 u / 9)
Gas + radiation: see below.
"""

import numpy as np


def ideal_gas_pressure(rho, u, gamma: float):
    """P = (gamma - 1) rho u."""
    return (gamma - 1.0) * np.asarray(rho) * np.asarray(u)


def ideal_gas_sound_speed(u, gamma: float):
    """Adiabatic sound speed sqrt(gamma (gamma - 1) u)."""
    return np.sqrt(gamma * (gamma - 1.0) * np.asarray(u))


def radiation_limit_pressure(rho, u):
    """P = rho u / 3 (radiation-dominated gas: gamma = 4/3)."""
    return np.asarray(rho) * np.asarray(u) / 3.0


def radiation_limit_sound_speed(u):
    """c_s = sqrt(4/3 P/rho) = sqrt(4 u / 9)."""
    return np.sqrt(4.0 * np.asarray(u) / 9.0)


def ideal_gas_u_si(t_k, gamma: float = 5.0 / 3.0):
    """u = k T / ((gamma - 1) mu m_p) [J/kg] (3kT/(2 mu m_p) for gamma = 5/3)."""
    from nebula import units

    return units.K_B * np.asarray(t_k, float) / ((gamma - 1.0) * units.MU * units.M_P)


def isothermal_pressure(rho, c_s):
    """P = c_s^2 rho (c_s a number, or one per particle)."""
    return np.asarray(c_s) ** 2 * np.asarray(rho)


def locally_isothermal_cs(r, c_s0: float, r0: float, q: float = 0.75):
    """c_s = c_s0 (r / r0)^(-q) (phase-07 7.4; Lodato & Price 2010): with q = 3/4,
    c_s^2 / Omega_K = c_s0^2 r0^(3/2) is the same at every r."""
    return c_s0 * (np.asarray(r, float) / r0) ** (-q)


# --------------------------------------------------------------------------- gas + radiation
#
# Phase 6 (phases/phase-06-shocks-cooling-ring.md 6.2; expected-values §9), local thermodynamic
# equilibrium (gas and radiation at one temperature T):
#     u = 3 k T / (2 mu m_p) + a T^4 / rho                  (specific internal energy, J/kg)
#     P = rho k T / (mu m_p) + a T^4 / 3                    (Pa)
#     beta_gas = P_gas / P
#     Gamma_1 = beta + (4 - 3 beta)^2 (gamma - 1) / (beta + 12 (gamma - 1)(1 - beta)),  gamma = 5/3
#     c_s = sqrt(Gamma_1 P / rho)
# Gamma_1 (Chandrasekhar 1939; Clayton 1968 Eq. 2-145 for gamma = 5/3:
# (32 - 24 beta - 3 beta^2) / (24 - 21 beta), the same expression) is the true adiabatic index;
# Phantom's eos_idealplusrad uses the simpler 1 + P/(rho u), equal to it only in the two
# limits (DECISIONS, Phase 6.2). T from (rho, u): Newton on log T with a bisection bracket
# [1, 1e13] K, relative tolerance 1e-12 (u(T) is monotonic, so the bracket always holds the
# root); never a closed-form quartic (cancellation). Physical units inside (*_si, rule 4);
# code-unit wrappers convert through nebula.units.GasUnits.

T_MIN_SI, T_MAX_SI = 1.0, 1.0e13  # K, solver bracket
EOS_TOL = 1e-12


def gasrad_u_si(rho_si, t_k):
    """u = 3kT/(2 mu m_p) + a T^4 / rho  [J/kg] (rho kg/m^3, T K; units.MU, units.A_RAD)."""
    from nebula import units

    t = np.asarray(t_k, float)
    return 1.5 * units.K_B * t / (units.MU * units.M_P) + units.A_RAD * t**4 / np.asarray(rho_si)


def gasrad_pressure_si(rho_si, t_k):
    """P = rho k T/(mu m_p) + a T^4 / 3  [Pa]."""
    from nebula import units

    t = np.asarray(t_k, float)
    return np.asarray(rho_si) * units.K_B * t / (units.MU * units.M_P) + units.A_RAD * t**4 / 3.0


def gasrad_beta_si(rho_si, t_k):
    """beta_gas = P_gas / P (dimensionless)."""
    from nebula import units

    t = np.asarray(t_k, float)
    p_gas = np.asarray(rho_si) * units.K_B * t / (units.MU * units.M_P)
    return p_gas / (p_gas + units.A_RAD * t**4 / 3.0)


def gamma1(beta, gamma: float = 5.0 / 3.0):
    """Gamma_1 = beta + (4 - 3 beta)^2 (gamma - 1) / (beta + 12 (gamma - 1)(1 - beta)): 5/3 for
    beta = 1 (gas), 4/3 for beta = 0 (radiation) (Chandrasekhar 1939; Clayton 1968 Eq. 2-145)."""
    b = np.asarray(beta, float)
    g = gamma - 1.0
    return b + (4.0 - 3.0 * b) ** 2 * g / (b + 12.0 * g * (1.0 - b))


def gasrad_temperature_si(rho_si, u_si, tol: float = EOS_TOL, max_iter: int = 200):
    """T [K] from (rho [kg/m^3], u [J/kg]): Newton on log T, kept inside a bisection bracket
    [1, 1e13] K that is narrowed every iteration. Raises if any element does not converge or
    u lies outside u(T_MIN)..u(T_MAX)."""
    rho = np.broadcast_to(np.asarray(rho_si, float), np.broadcast(rho_si, u_si).shape).copy()
    u = np.broadcast_to(np.asarray(u_si, float), rho.shape).copy()
    lo = np.full(rho.shape, np.log(T_MIN_SI))
    hi = np.full(rho.shape, np.log(T_MAX_SI))
    if np.any(u < gasrad_u_si(rho, T_MIN_SI)) or np.any(u > gasrad_u_si(rho, T_MAX_SI)):
        raise ValueError("u outside the EOS bracket [1, 1e13] K")
    from nebula import units

    a1 = 1.5 * units.K_B / (units.MU * units.M_P)
    # start from the smaller of the gas-only and radiation-only temperatures (both overshoot)
    lt = np.log(np.clip(np.minimum(u / a1, (u * rho / units.A_RAD) ** 0.25), T_MIN_SI, T_MAX_SI))
    done = np.zeros(rho.shape, bool)
    for _ in range(max_iter):
        t = np.exp(lt)
        f = a1 * t + units.A_RAD * t**4 / rho - u
        dfdl = a1 * t + 4.0 * units.A_RAD * t**4 / rho  # df / d(log T)
        lo = np.where(f < 0, lt, lo)
        hi = np.where(f > 0, lt, hi)
        new = lt - f / dfdl
        bad = (new <= lo) | (new >= hi) | ~np.isfinite(new)
        new = np.where(bad, 0.5 * (lo + hi), new)
        step = np.abs(new - lt)
        lt = np.where(done, lt, new)
        done |= step < tol
        if np.all(done):
            return np.exp(lt)
    raise RuntimeError("gas + radiation EOS: temperature did not converge")


def gasrad_state(rho_code, u_code, gas_units=None):
    """Code-unit wrapper: (P [code], c_s [c], T [K], beta_gas) from code density and u (c^2).
    The cloud's mass scale enters only here (units.GasUnits; PLAN.md §3)."""
    from nebula import units

    gu = units.GasUnits() if gas_units is None else gas_units
    rho_si = gu.density_to_si(np.asarray(rho_code, float))
    u_si = units.specific_energy_to_si(np.asarray(u_code, float))
    t = gasrad_temperature_si(rho_si, u_si)
    p_si = gasrad_pressure_si(rho_si, t)
    beta = gasrad_beta_si(rho_si, t)
    p_code = p_si / (gu.density_kg_m3 * units.C**2)
    cs = np.sqrt(gamma1(beta) * p_code / np.asarray(rho_code, float))
    return p_code, cs, t, beta
