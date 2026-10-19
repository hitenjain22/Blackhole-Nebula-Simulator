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
Velocity reconstruction (Hiten's decision D2, 2026-10-15; review docs/review-2026-10-15-plunge.md):
the viscosity uses the pair's velocity jump *reconstructed to the midpoint* instead of the raw
jump, so that smooth (linear) flow, e.g. cold gas squeezed supersonically on its way to the
hole, is not mistaken for a shock. Frontiere, Raskin & Owen 2017 (J. Comput. Phys. 332, 160,
arXiv:1605.00725) Eqs. 48-53; the same linear part as Rosswog 2020 (MAGMA2, MNRAS 498, 4230,
arXiv:1911.13093) Eqs. 17, 21-23; both checked 2026-10-15. With x_ab = x_a - x_b and
G = dv/dx (G_ij = d v^i / d x^j, the corrected gradient of the switch pass):
    v*_a = v_a - (Phi_ab / 2) G_a x_ab,   v*_b = v_b + (Phi_ab / 2) G_b x_ab   (Eqs. 48-50)
    Phi_ab = max(0, min(1, 4 A / (1 + A)^2)) x closeness           (van Leer limiter, Eq. 51)
    A_ab = (x_ab . G_a . x_ab) / (x_ab . G_b . x_ab)               (Eq. 52)
    closeness = exp(-((eta_ab - eta_crit)/eta_fold)^2) if eta_ab < eta_crit, else 1
    eta_ab = min(|x_ab|/h_a, |x_ab|/h_b)   (Eq. 53);  eta_crit = 1/hfact, eta_fold = 0.2
The limiter is symmetric (A -> 1/A leaves it unchanged), so pair forces stay antisymmetric.
The jump that drives q is mu = max(v*_ab . r_hat, v_ab . r_hat) when both are negative, else 0
("min-mod gate", ``viscous_jump``): the less compressive of the two. It replaces v_ab . r_hat
in q and in v_sig of q; the shock heating keeps the actual v_ab . r_hat (so the energy
equation still matches the work exactly). This gate is this project's choice (DECISIONS,
D2): it is continuous (needed on block steps, where pair forces act at different times: the
block-step angular-momentum drift then falls 5.7x per halving, against 1.9x when q switches
off abruptly as v_ab . r_hat crosses 0), it never heats negatively (MAGMA2's gate on v*_ab
alone allows that, and drove one particle's u below 0 in a test), and it never gives more
viscosity than the plain form. Linear flow: v*_ab = 0, no viscosity. At a
shock the gradients on the two sides disagree, Phi -> 0, and the standard viscosity returns.
Units: code units (velocities in c, u in c^2).
"""

import numpy as np

BETA_AV = 2.0
ALPHA_MAX = 1.0
ALPHA_U = 1.0  # Phantom's conductivity strength (Price et al. 2018); the test problems use it
# Conductivity strength for runs with the black hole (DECISIONS 2026-10-19, Hiten's decision):
# MAGMA2's "conservative" default, Rosswog 2020 (MNRAS 498, 4230) §2.2.2, Eqs. 24-26: "we choose
# a low value of alpha_u = 0.05" (checked in the paper 2026-10-19; also PySPH magma2.py's default).
# Only the coefficient is adopted; MAGMA2 also reconstructs the u difference to the pair midpoint.
# With alpha_u = 1 the conductivity pumped the heat of a few over-heated particles into the
# whole stream at pericentre (heat budget, runs/heat10k_r60); 0.05 halves the unbound gas.
ALPHA_U_MAGMA2 = 0.05
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


ETA_FOLD = 0.2  # Frontiere et al. 2017 (eta_crit, eta_fold) = (1/n_h, 0.2)


def van_leer_limiter(ratio):
    """Phi = max(0, min(1, 4 A / (1 + A)^2)) (van Leer; Frontiere et al. 2017 Eq. 51 first
    factor, MAGMA2 Eq. 21). 1 for A = 1 (both sides agree), 0 for A <= 0. Dimensionless."""
    a = np.asarray(ratio, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = 4.0 * a / ((1.0 + a) * (1.0 + a))
    return np.where(a > 0.0, np.clip(phi, 0.0, 1.0), 0.0)


def limiter_ratio(grad_a, grad_b, x_ab):
    """A_ab = (x_ab . G_a . x_ab) / (x_ab . G_b . x_ab), G_ij = d v^i / d x^j (Frontiere et
    al. 2017 Eq. 52; MAGMA2 Eq. 22). Returns 0 where the denominator is 0 (no reconstruction).
    grad_*: (..., 3, 3); x_ab: (..., 3)."""
    x = np.asarray(x_ab, float)
    pa = np.einsum("...i,...ij,...j->...", x, np.asarray(grad_a, float), x)
    pb = np.einsum("...i,...ij,...j->...", x, np.asarray(grad_b, float), x)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pb != 0.0, pa / np.where(pb != 0.0, pb, 1.0), 0.0)


def closeness_factor(eta_ab, eta_crit, eta_fold: float = ETA_FOLD):
    """exp(-((eta_ab - eta_crit)/eta_fold)^2) for eta_ab < eta_crit, else 1 (Frontiere et al.
    2017 Eq. 51 second factor; MAGMA2 Eq. 21): switches reconstruction off for pairs closer than
    the typical spacing. eta_ab = min(r/h_a, r/h_b) (Eq. 53); eta_crit = 1/hfact."""
    e = np.asarray(eta_ab, float)
    return np.where(e < eta_crit, np.exp(-(((e - eta_crit) / eta_fold) ** 2)), 1.0)


def reconstructed_jump(v_a, v_b, grad_a, grad_b, x_ab, phi):
    """v*_ab = (v_a - Phi/2 G_a x_ab) - (v_b + Phi/2 G_b x_ab): both velocities extrapolated
    to the pair midpoint (Frontiere et al. 2017 Eqs. 48-50; MAGMA2 Eq. 17, linear part).
    Shapes (..., 3), gradients (..., 3, 3), phi (...). Units of velocity (c)."""
    x = np.asarray(x_ab, float)
    ph = np.asarray(phi, float)[..., None]
    ga = np.einsum("...ij,...j->...i", np.asarray(grad_a, float), x)
    gb = np.einsum("...ij,...j->...i", np.asarray(grad_b, float), x)
    return (np.asarray(v_a, float) - 0.5 * ph * ga) - (np.asarray(v_b, float) + 0.5 * ph * gb)


def viscous_jump(vdr_reconstructed, vdr_actual):
    """mu = max(v*_ab . r_hat, v_ab . r_hat) if both < 0, else 0: the velocity jump that drives
    the reconstructed viscosity ("min-mod gate"; module docstring, DECISIONS D2). Velocity (c)."""
    a = np.asarray(vdr_reconstructed, float)
    b = np.asarray(vdr_actual, float)
    return np.where((a < 0.0) & (b < 0.0), np.maximum(a, b), 0.0)
