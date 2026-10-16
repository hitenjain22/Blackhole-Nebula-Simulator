"""Opacity formulas (phase-06 6.2), checked against hand-computed values and limits."""

import numpy as np

from nebula import units
from nebula.physics.opacity import kappa_eff_si, kappa_es_si, kappa_ff_si, optical_depth
from tests.helpers import check


def test_electron_scattering_matches_units():
    check("kappa_es = 0.02 (1 + X) = units.KAPPA_ES", units.KAPPA_ES, kappa_es_si(), rtol=1e-12)


def test_free_free_hand_value_and_scaling():
    """rho = 1 g/cm^3 = 1000 kg/m^3, T = 1e7 K: 3.68e22 x 0.98 x 1.7 x 10^-24.5 cm^2/g
    = 0.01939 cm^2/g = 0.001939 m^2/kg (by hand)."""
    hand_cgs = 3.68e22 * 0.98 * 1.70 * 1.0 * (1e7) ** -3.5
    check("kappa_ff at 1 g/cm^3, 1e7 K (m^2/kg) vs hand value", 0.1 * hand_cgs,
          kappa_ff_si(1000.0, 1e7), rtol=1e-12)  # fmt: skip
    k1 = kappa_ff_si(1e-6, 1e6)
    check("kappa_ff ~ rho (x10)", 10 * k1, kappa_ff_si(1e-5, 1e6), rtol=1e-12)
    check("kappa_ff ~ T^-3.5 (x2 in T)", k1 * 2**-3.5, kappa_ff_si(1e-6, 2e6), rtol=1e-12)


def test_effective_opacity_limits_and_torus_estimate():
    check("kappa_eff -> kappa_abs when absorption dominates", 10.0,
          float(kappa_eff_si(10.0, 1e-8)), rtol=1e-9)  # fmt: skip
    check("kappa_eff -> sqrt(ka ks) when scattering dominates", np.sqrt(1e-8 * 0.034),
          float(kappa_eff_si(1e-8, 0.034)), rtol=1e-6)  # fmt: skip
    # plan-time torus (expected-values §9): rho 1.66e-6 kg/m^3, T 9.92e5 K, H/r = 0.3 at r = 50.89
    rho, t = 1.6586853e-6, 9.9188303029e5
    keff = kappa_eff_si(kappa_ff_si(rho, t), kappa_es_si())
    h = 0.3 * 50.89 * units.BlackHoleUnits().length_m
    tau = float(optical_depth(keff, rho, h))
    print(
        f"   torus: kappa_ff {float(kappa_ff_si(rho, t)):.2e}, kappa_eff {float(keff):.2e} m^2/kg, "
        f"tau_eff over H = {tau:.1f}"
    )
    check("torus tau_eff: 'a few to ~15' (phase-06 6.2, free-free only)", True, 1.0 < tau < 15.0)
