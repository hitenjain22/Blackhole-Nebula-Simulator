"""Shock capturing in SPH: artificial viscosity, conductivity and the viscosity switch
(phases/phase-03-sph-core.md 3.1). Pure formulas; the pair sums live in sim/sph.py.

Source: Price et al. 2018 (Phantom, PASA 35, e031, arXiv:1702.03930), §2.2.7-2.2.9, checked
2026-10-14; following Monaghan 1997 (signal-velocity viscosity), Price 2008 (conductivity),
Cullen & Dehnen 2010 (switch; Phantom's variant), Balsara 1995 (shear limiter).

Viscosity (Phantom Eqs. 39-41), for an approaching pair (v_ab . r_hat_ab < 0) only:
    q^a_ab = -1/2 rho_a v_sig,a (v_ab . r_hat_ab),
    v_sig,a = alpha_a c_s,a + beta |v_ab . r_hat_ab|
    with beta = 2. alpha multiplies ONLY the sound-speed part: the beta term stays on when the
    switch sets alpha ~ 0, which stops particles streaming through each other in shocks.
Shock heating (Eq. 42, first term), approaching pairs only, energy-consistent with the force:
    du_a/dt += -1/(Omega_a rho_a) sum_b m_b v_sig,a 1/2 (v_ab . r_hat)^2 F_ab(h_a),  F = dW/dr
Conductivity (Eq. 42, second term; Eq. 43), every pair:
    du_a/dt += sum_b m_b alpha_u v^u_sig (u_a - u_b)
               x 1/2 [F_ab(h_a)/(Omega_a rho_a) + F_ab(h_b)/(Omega_b rho_b)]
    v^u_sig = sqrt(|P_a - P_b| / rho_ab), rho_ab = (rho_a + rho_b)/2   (Price 2008; Phantom's
    choice without external forces; with gravity Phantom uses |v_ab . r_hat_ab|, Eq. 44)
Switch (Eqs. 46-50):
    A_a = xi_a max(-d(div v_a)/dt, 0),   xi = |div v|^2 / (|div v|^2 + |curl v|^2)
    alpha_loc = min(10 h^2 A / c_s^2, alpha_max),  alpha_max = 1
    alpha <- alpha_loc if alpha_loc > alpha, else backward Euler decay toward alpha_loc:
    alpha_new = (alpha + alpha_loc dt/tau) / (1 + dt/tau),  tau = h / (sigma_decay c_s), sigma = 0.1
Units: code units (velocities in c, u in c^2).
"""

import numpy as np

BETA_AV = 2.0
ALPHA_MAX = 1.0
ALPHA_U = 1.0
SIGMA_DECAY = 0.1


def viscous_signal_speed(alpha, c_s, vdotr, beta: float = BETA_AV):
    """v_sig = alpha c_s + beta |v_ab . r_hat_ab| (Phantom Eq. 41)."""
    return alpha * c_s + beta * np.abs(vdotr)


def viscous_q(rho, v_sig, vdotr):
    """q_ab = -1/2 rho v_sig (v_ab . r_hat) for approaching pairs, else 0 (Eq. 40)."""
    vdotr = np.asarray(vdotr, float)
    return np.where(vdotr < 0.0, -0.5 * rho * v_sig * vdotr, 0.0)


def conductivity_signal_speed(p_a, p_b, rho_a, rho_b):
    """v^u_sig = sqrt(|P_a - P_b| / rho_ab), rho_ab = (rho_a + rho_b) / 2 (Price 2008; Eq. 43)."""
    return np.sqrt(np.abs(p_a - p_b) / (0.5 * (rho_a + rho_b)))


def balsara_xi(div_v, curl_v_magnitude):
    """xi = |div v|^2 / (|div v|^2 + |curl v|^2) (Eq. 47); 0 for a flow at rest."""
    d2 = np.asarray(div_v, float) ** 2
    c2 = np.asarray(curl_v_magnitude, float) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d2 + c2 > 0, d2 / (d2 + c2), 0.0)


def alpha_local(h, ddivv_dt, xi, c_s, alpha_max: float = ALPHA_MAX):
    """alpha_loc = min(10 h^2 xi max(-d(div v)/dt, 0) / c_s^2, alpha_max) (Eqs. 46, 48)."""
    a = xi * np.maximum(-np.asarray(ddivv_dt, float), 0.0)
    return np.minimum(10.0 * h * h * a / (c_s * c_s), alpha_max)


def decay_time(h, c_s, sigma: float = SIGMA_DECAY):
    """tau = h / (sigma_decay c_s) (Morris & Monaghan 1997; Phantom uses c_s here)."""
    return h / (sigma * c_s)


def alpha_update(alpha, alpha_loc, dt, tau):
    """Phantom Eq. 49-50: jump up to alpha_loc at once; otherwise decay toward it with a stable
    backward-Euler step."""
    alpha = np.asarray(alpha, float)
    decayed = (alpha + alpha_loc * dt / tau) / (1.0 + dt / tau)
    return np.where(alpha_loc > alpha, alpha_loc, decayed)
