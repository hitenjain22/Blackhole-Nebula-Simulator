"""Radiative cooling and the temperature floor (phases/phase-06-shocks-cooling-ring.md 6.2).

beta-cooling (Gammie 2001, ApJ 553, 174: t_cool = beta / Omega), relaxing u toward the floor
u_floor(rho), the specific energy at T_floor = 10^4 K from the equation of state:

    t_cool = beta / Omega_K(r),   Omega_K = r^(-3/2)                 (code units, G = M = 1)
    u_new = u_floor + (u - u_floor) exp(-dt / t_cool)                (exact for fixed t_cool)
    E_radiated += m (u - u_new)

The update is exact, so there is no time-step limit and splitting an interval into any number of
steps gives the same u (for fixed r and rho). Gas below the floor relaxes *up* to it with the
same law: the floor stands for photo-ionization heating (G2 is kept at ~10^4 K by the radiation
around it), and the energy put in is booked as negative radiated energy (DECISIONS, Phase 6.4).
Nothing else may clip u (phase-06 6.2). beta = inf: no cooling (u unchanged).
"""

import numpy as np

T_FLOOR = 1.0e4  # K (phase-06 6.2)
BETA_DEFAULT = 230.0  # expected-values §9 scenario estimate


def omega_kepler(r):
    """Omega_K = r^(-3/2) (code units: G = M = 1, r in GM/c^2, Omega in c^3/GM)."""
    return np.asarray(r, float) ** -1.5


def cooling_time(r, beta: float):
    """t_cool = beta / Omega_K(r) (Gammie 2001). Code time units."""
    return beta / omega_kepler(r)


def beta_cool(u, u_floor, dt, t_cool):
    """u after cooling for dt: u_floor + (u - u_floor) exp(-dt / t_cool) (exact). t_cool = inf
    leaves u unchanged."""
    u = np.asarray(u, float)
    u_floor = np.asarray(u_floor, float)
    with np.errstate(divide="ignore"):
        decay = np.exp(-np.asarray(dt, float) / np.asarray(t_cool, float))
    return u_floor + (u - u_floor) * decay
