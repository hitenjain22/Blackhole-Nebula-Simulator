"""Step 1.1: unit conversions (reference/expected-values.md §1, plus §2b and §9 gas numbers).

Tolerances: those stated in §1 where given; otherwise set from the printed precision of the
expected value (at least 10x its rounding, expected-values §3 rule).
"""

import numpy as np

from nebula import units
from nebula.units import BlackHoleUnits, GasUnits
from tests.helpers import check

BH = BlackHoleUnits(4.3e6)


def test_code_units_si():
    check("§1 time unit GM/c^3 (s)", 21.180252, BH.time_s, rtol=1e-5)
    check("§1 length unit GM/c^2 (m)", 6.349680e9, BH.length_m, rtol=1e-5)
    check("§1 length unit (km)", 6.349680e6, BH.length_m / units.KM, rtol=1e-5)
    check("§1 length unit (AU)", 0.042445, BH.length_m / units.AU, rtol=1e-4)
    check("radiation constant a = 4 sigma/c", 7.565733e-16, units.A_RAD, rtol=1e-6)


def test_key_times():
    days = BH.time_to_s(1.0) / units.DAY
    check("§1 first pericentre t=396,223.627 (days)", 97.1310, 396_223.627 * days, rtol=1e-5)
    check("§1 radial period 792,447.254 (days)", 194.2619, 792_447.254 * days, rtol=1e-5)
    hours = BH.time_to_s(2_303.257) / units.HOUR
    check("§1 ring orbit period 2,303.257 (hours)", 13.551, hours, rtol=1e-4)


def test_gas_conversions():
    cs = units.gas_sound_speed_from_temperature(1e4)
    check("§9 sound speed at 1e4 K (c)", 5.0509e-5, cs, rtol=1e-4)
    check("§9 sound speed at 1e4 K (km/s)", 15.14, cs * units.C / units.KM, rtol=1e-3)
    u = units.gas_u_from_temperature(1e4)
    check("§9 u at 1e4 K (c^2)", 2.2961e-9, u, rtol=1e-4)
    check("T -> u -> T round trip (K)", 1e4, units.gas_temperature_from_u(u), rtol=1e-14)


def test_cloud_density():
    """0.1 M_sun (code mass 1) uniform sphere of radius 100 code lengths (§2b)."""
    gas = GasUnits(BH, cloud_mass_msun=0.1, cloud_mass_code=1.0)
    radius = 100.0
    rho_code = 1.0 / (4.0 / 3.0 * np.pi * radius**3)
    rho_si = gas.density_to_si(rho_code)
    check("§2b cloud density (kg/m^3)", 1.8543e-7, rho_si, rtol=3e-4)
    tau = units.KAPPA_ES * rho_si * BH.length_to_m(radius)
    check("§2b electron-scattering depth across R", 4003.2, tau, rtol=2e-4)
    check("density round trip (code)", rho_code, gas.density_from_si(rho_si), rtol=1e-14)


def test_formatting():
    check("format_time(97.131 days)", "97.13 days", units.format_time(97.131 * units.DAY))
    check("format_time(13.551 h)", "13.55 hours", units.format_time(13.551 * units.HOUR))
    check("format_time(21.18 s)", "21.18 s", units.format_time(21.180252))
    check("format_time(2.5 yr)", "2.5 years", units.format_time(2.5 * units.YEAR))
    check("format_length(1 code length)", "6,349,680 km", units.format_length(BH.length_m))
    check("format_length(r=5000)", "212.2 AU", units.format_length(BH.length_to_m(5000.0)))
