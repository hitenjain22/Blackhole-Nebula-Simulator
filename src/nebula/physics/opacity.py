"""Opacities for the gas (phases/phase-06-shocks-cooling-ring.md 6.2 "LTE" diagnostics; reused by
Phase 9). Physical units (*_si): kappa in m^2/kg, rho in kg/m^3, T in K.

    kappa_es  = 0.02 (1 + X) m^2/kg  (electron scattering, fully ionized; units.KAPPA_ES = 0.034)
    kappa_ff  = 3.68e22 g_ff (1 - Z)(1 + X) rho[g/cm^3] T^(-7/2)  cm^2/g    (Kramers free-free)
              = 3.68e18 g_ff (1 - Z)(1 + X) rho[kg/m^3] T^(-7/2)  m^2/kg
    kappa_eff = sqrt(kappa_abs (kappa_abs + kappa_es))  (effective, thermalizing opacity)
    tau = kappa rho L  over a length L (e.g. tau_h with L = h)

Sources: Kramers' free-free Rosseland-mean form as in Carroll & Ostlie, An Introduction to Modern
Astrophysics, §9.2 (Eq. 9.22), and standard lecture notes (checked 2026-10-16); the plan
(phase-06 6.2) asked for the coefficient to be checked, as it was written from memory there.
Gaunt factor g_ff = 1 (O(1) correction, as the plan's estimates). Metal bound-free absorption
(up to ~20x more, phase-06 6.2) is not included: kappa_abs here is a lower bound, so optical
depths from it are lower bounds too.
"""

import numpy as np

from nebula import units

KFF_SI = 3.68e18  # m^2/kg per (kg/m^3) per K^(-7/2): 3.68e22 cm^2/g x 0.1 (cm^2/g -> m^2/kg)
# x 1e-3 (rho in g/cm^3 -> kg/m^3)


def kappa_es_si(x_h: float = units.X_HYDROGEN) -> float:
    """Electron-scattering opacity 0.02 (1 + X) m^2/kg (fully ionized gas)."""
    return 0.02 * (1.0 + x_h)


def kappa_ff_si(rho_si, t_k, x_h: float = units.X_HYDROGEN, z: float = units.Z_METALS,
                gaunt: float = 1.0):  # fmt: skip
    """Kramers free-free (Rosseland mean) opacity 3.68e18 g (1-Z)(1+X) rho T^(-7/2) m^2/kg."""
    return (
        KFF_SI
        * gaunt
        * (1.0 - z)
        * (1.0 + x_h)
        * np.asarray(rho_si, float)
        * np.asarray(t_k, float) ** (-3.5)
    )


def kappa_eff_si(kappa_abs, kappa_es):
    """Effective (thermalization) opacity sqrt(kappa_abs (kappa_abs + kappa_es)): kappa_abs when
    absorption dominates, the geometric mean sqrt(kappa_abs kappa_es) when scattering does."""
    ka = np.asarray(kappa_abs, float)
    return np.sqrt(ka * (ka + np.asarray(kappa_es, float)))


def optical_depth(kappa, rho_si, length_m):
    """tau = kappa rho L (dimensionless)."""
    return np.asarray(kappa) * np.asarray(rho_si) * np.asarray(length_m)


def thermal_absorption_si(rho_si, t_k, x_h: float = units.X_HYDROGEN, z: float = units.Z_METALS):
    """Thermalizing absorption coefficient of hot gas [1/m] (phase-09 9.4 option 1):

        alpha = kappa_eff rho,  kappa_eff = sqrt(kappa_ff (kappa_ff + kappa_es)),
        kappa_ff = Kramers free-free (kappa_ff_si), kappa_es = 0.02 (1 + X) m^2/kg

    rho in kg/m^3, T in K (the gas's own frame). kappa_ff grows as rho T^(-7/2), so thin hot
    gas is nearly transparent while dense gas stays opaque. Sources as the module docstring
    (Carroll & Ostlie §9.2; Rybicki & Lightman 1979 §1.7 for the effective opacity)."""
    rho = np.asarray(rho_si, float)
    return kappa_eff_si(kappa_ff_si(rho, t_k, x_h, z), kappa_es_si(x_h)) * rho


def thermal_absorption_code(rho_code, t_k, gas_units: units.GasUnits | None = None):
    """thermal_absorption_si in code units: rho in code density, T in K, alpha in 1 / (GM/c^2).
    The rule-4 wrapper (units.GasUnits converts density in and length out; default: the runs'
    gas units, units.GasUnits())."""
    gas_units = gas_units or units.GasUnits()
    return thermal_absorption_si(gas_units.density_to_si(rho_code), t_k) * gas_units.bh.length_m


def thermal_absorption_constants(gas_units: units.GasUnits | None = None):
    """(c_es, q_ff) of the same alpha in the form the renderer evaluates (march.py and the WGSL
    colour shader mirror thermal_absorption_code with them):

        alpha = c_es rho sqrt(q (1 + q)),  q = kappa_ff / kappa_es = q_ff rho T^(-7/2)

    with rho in code density and alpha in 1 / (GM/c^2). Both constants fit float32."""
    gas_units = gas_units or units.GasUnits()
    d, length = gas_units.density_kg_m3, gas_units.bh.length_m
    k_es = kappa_es_si()
    k_ff = float(kappa_ff_si(d, 1.0))  # kappa_ff at rho = 1 code density, T = 1 K
    return k_es * d * length, k_ff / k_es


def thin_emission_fraction(rho_code, t_k, gas_units: units.GasUnits | None = None):
    """sqrt(q / (1 + q)), q = kappa_ff / kappa_es (thermal_absorption_constants): the source
    function factor S = sqrt(q / (1 + q)) B that makes gas emit kappa_ff B per length when thin
    (alpha_eff S = c_es rho q B) and sqrt(epsilon) B when thick (epsilon = kappa_abs /
    kappa_total; the scattering-dominated "modified blackbody" up to an order-unity factor,
    Rybicki & Lightman 1979 §1.8). The renderer's default source S = B is exact only when
    absorption dominates (q >> 1); in thin, scattering-dominated gas it overstates the
    continuum by 1 / sqrt(q). Phase 14 N1 uses this factor for the Act 1 nebula only, where
    real nebulae are line-dominated (DECISIONS 2026-11-23); the simulated close-up keeps S = B
    (the approved look). Dimensionless; rho in code density, T in K."""
    c_es, q_ff = thermal_absorption_constants(gas_units)
    q = q_ff * np.asarray(rho_code, float) * np.asarray(t_k, float) ** -3.5
    return np.sqrt(q / (1.0 + q))
