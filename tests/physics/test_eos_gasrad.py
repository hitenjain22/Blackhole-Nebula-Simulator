"""Phase 6 tests 1-3: the gas + radiation equation of state (phases/phase-06-shocks-cooling-ring.md
6.2-6.3; expected-values §9, script 16/21 values to 10 significant figures)."""

import numpy as np

from nebula.physics.eos import (
    gamma1,
    gasrad_beta_si,
    gasrad_pressure_si,
    gasrad_state,
    gasrad_temperature_si,
    gasrad_u_si,
)
from nebula.units import GasUnits, specific_energy_from_si
from tests.helpers import check

# expected-values §9: (rho kg/m^3, T K, u J/kg, P Pa, beta_gas, Gamma_1)
POINTS = [
    (1e-3, 1e4, 2.0636755965e8, 1.3757585119e5, 9.9998166894e-1, 1.6666361185),
    (1e-6, 1e6, 7.5659396100e14, 2.5220486567e8, 5.4548245497e-5, 1.3333424248),
    (1e-9, 1e5, 7.5659396100e13, 2.5220486567e4, 5.4548245497e-5, 1.3333424248),
    (1e-9, 1e7, 7.5657332502e21, 2.5219110835e12, 5.4551221167e-11, 1.3333333333),
]
TORUS = (1.6586853e-6, 9.9188303029e5, 4.415186e14, 2.4412478743e8, 9.2714321654e-5, 1.3333487859)
EXTREMES = [  # (rho, T, u, P, Gamma_1) for test 2
    (1e3, 1e3, 2.0635999392e7, 1.3757332928e10, 1.6666666667),
    (1e-15, 1e8, 7.5657332500e31, 2.5219110833e16, 1.3333333333),
]


def test_eos_points():
    """Test 1: u, P, beta_gas, Gamma_1 to rtol 1e-8 at the five points; T recovered from
    (rho, u) to 1e-10."""
    for rho, t, u, p, beta, g1 in POINTS:
        tag = f"rho={rho:g}, T={t:g}"
        check(f"{tag}: u (J/kg)", u, gasrad_u_si(rho, t), rtol=1e-8)
        check(f"{tag}: P (Pa)", p, gasrad_pressure_si(rho, t), rtol=1e-8)
        check(f"{tag}: beta_gas", beta, gasrad_beta_si(rho, t), rtol=1e-8)
        check(f"{tag}: Gamma_1", g1, gamma1(gasrad_beta_si(rho, t)), rtol=1e-8)
        check(f"{tag}: T from (rho, u)", t, gasrad_temperature_si(rho, u), rtol=1e-10)
    rho, t, u, p, beta, g1 = TORUS
    t_rec = gasrad_temperature_si(rho, u)
    check("torus point: T recovered from u", t, t_rec, rtol=1e-10)
    check("torus point: P", p, gasrad_pressure_si(rho, t_rec), rtol=1e-8)
    check("torus point: beta_gas", beta, gasrad_beta_si(rho, t_rec), rtol=1e-8)
    check("torus point: Gamma_1", g1, gamma1(gasrad_beta_si(rho, t_rec)), rtol=1e-8)


def test_eos_limits():
    """Test 2: Gamma_1 -> 5/3 (gas) and 4/3 (radiation) within 1e-6 at the two extreme points,
    whose u and P match to rtol 1e-8; the formula's exact limits."""
    for rho, t, u, p, g1 in EXTREMES:
        tag = f"rho={rho:g}, T={t:g}"
        check(f"{tag}: u", u, gasrad_u_si(rho, t), rtol=1e-8)
        check(f"{tag}: P", p, gasrad_pressure_si(rho, t), rtol=1e-8)
        check(f"{tag}: Gamma_1", g1, gamma1(gasrad_beta_si(rho, t)), atol=1e-6)
    check("Gamma_1(beta = 1) = 5/3, (beta = 0) = 4/3", [5 / 3, 4 / 3], gamma1(np.array([1.0, 0.0])),
          rtol=1e-15)  # fmt: skip


def test_eos_round_trip():
    """Test 3: T -> u -> T on a 200 x 200 grid, rho in [1e-15, 1e3] kg/m^3, T in [1e3, 1e10] K:
    max relative error 1e-10."""
    rho, t = np.meshgrid(np.logspace(-15, 3, 200), np.logspace(3, 10, 200))
    t_back = gasrad_temperature_si(rho, gasrad_u_si(rho, t))
    check("round trip T -> u -> T, max relative error", 0.0, float(np.max(np.abs(t_back / t - 1))),
          atol=1e-10)  # fmt: skip


def test_code_unit_wrapper():
    """The code-unit wrapper agrees with the SI functions (cloud mass scale 0.1 M_sun)."""
    gu = GasUnits()
    rho_si, t = 1e-6, 1e6
    rho_code = gu.density_from_si(rho_si)
    u_code = specific_energy_from_si(gasrad_u_si(rho_si, t))
    p, cs, t_out, beta = gasrad_state(rho_code, u_code, gu)
    from nebula import units

    check("wrapper: T", t, t_out, rtol=1e-10)
    check("wrapper: P in code units back to Pa", gasrad_pressure_si(rho_si, t),
          p * gu.density_kg_m3 * units.C**2, rtol=1e-10)  # fmt: skip
    check("wrapper: c_s^2 = Gamma_1 P / rho", gamma1(beta) * p / rho_code, cs**2, rtol=1e-12)
