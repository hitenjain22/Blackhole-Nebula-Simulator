"""Physical constants and unit conversions: the only place physical units live (CLAUDE.md rule 4).

The simulation uses G = c = M_bh = 1 ("code units"). Then
    length unit   GM/c^2   (half the Schwarzschild radius)
    time unit     GM/c^3
    velocity unit c,   specific energy unit c^2.
The gas mass scale is separate: the cloud's code mass (default 1) stands for its physical
mass (default 0.1 M_sun, PLAN.md §3). It enters only through density -> radiation physics.

Constants: CODATA 2018 / SI exact values, as in reference/expected-values.md (header).
"""

from dataclasses import dataclass

import numpy as np

# --- SI constants (CODATA 2018; c, h, k_B exact by SI definition) ---
G = 6.67430e-11  # m^3 kg^-1 s^-2
C = 299_792_458.0  # m/s
M_SUN = 1.98847e30  # kg
AU = 1.495978707e11  # m (exact, IAU 2012)
K_B = 1.380649e-23  # J/K
M_P = 1.67262192369e-27  # kg
H_PLANCK = 6.62607015e-34  # J s
SIGMA_SB = 5.670374419e-8  # W m^-2 K^-4
A_RAD = 4.0 * SIGMA_SB / C  # J m^-3 K^-4, radiation constant
SIGMA_T = 6.6524587321e-29  # m^2, Thomson cross-section (CODATA 2018)
YEAR_S = 365.25 * 86_400.0  # Julian year (s), for accretion rates in M_sun/yr

# --- conventions (expected-values header; PLAN.md §3) ---
MU = 0.6  # mean molecular weight, fully ionized solar-composition gas
KAPPA_ES = 0.034  # m^2/kg, electron-scattering opacity, solar composition
X_HYDROGEN = 0.70  # hydrogen mass fraction (solar); KAPPA_ES = 0.02 (1 + X) m^2/kg
Z_METALS = 0.02  # metal mass fraction (solar)
GAMMA_GAS = 5.0 / 3.0  # ideal monatomic gas

# --- time and length for display ---
MINUTE = 60.0
HOUR = 3600.0
DAY = 86_400.0
YEAR = 365.25 * DAY  # Julian year
KM = 1e3

SGR_A_MASS_MSUN = 4.3e6  # PLAN.md §3
CLOUD_MASS_MSUN = 0.1  # PLAN.md §3


@dataclass(frozen=True)
class BlackHoleUnits:
    """Code units for a black hole of ``mass_msun`` solar masses (G = c = M = 1)."""

    mass_msun: float = SGR_A_MASS_MSUN

    @property
    def mass_kg(self) -> float:
        return self.mass_msun * M_SUN

    @property
    def length_m(self) -> float:
        """GM/c^2 in metres."""
        return G * self.mass_kg / C**2

    @property
    def time_s(self) -> float:
        """GM/c^3 in seconds."""
        return G * self.mass_kg / C**3

    @property
    def velocity_m_s(self) -> float:
        return C

    @property
    def specific_energy_j_kg(self) -> float:
        """c^2 in J/kg: code specific energy (e.g. u or H) times this gives J/kg."""
        return C**2

    def time_to_s(self, t):
        return t * self.time_s

    def length_to_m(self, r):
        return r * self.length_m


@dataclass(frozen=True)
class GasUnits:
    """Mass scale for the gas: ``cloud_mass_code`` code units of mass are ``cloud_mass_msun``.

    Needed only where the gas's physical density matters (radiation pressure, opacity,
    luminosity). Gravity from the gas is ignored, so this does not affect orbits.
    """

    bh: BlackHoleUnits = BlackHoleUnits()
    cloud_mass_msun: float = CLOUD_MASS_MSUN
    cloud_mass_code: float = 1.0

    @property
    def mass_kg_per_code(self) -> float:
        return self.cloud_mass_msun * M_SUN / self.cloud_mass_code

    @property
    def density_kg_m3(self) -> float:
        """One code density unit (code mass / (GM/c^2)^3) in kg/m^3."""
        return self.mass_kg_per_code / self.bh.length_m**3

    def density_to_si(self, rho_code):
        return rho_code * self.density_kg_m3

    def density_from_si(self, rho_si):
        return rho_si / self.density_kg_m3


def specific_energy_to_si(u_code):
    """Code specific energy (units of c^2) to J/kg."""
    return u_code * C**2


def specific_energy_from_si(u_si):
    return u_si / C**2


def gas_temperature_from_u(u_code):
    """Ideal-gas temperature (K) from code specific internal energy, gas only:
    u = 3 k T / (2 mu m_p)  =>  T = 2 mu m_p u c^2 / (3 k).  Phase 6 adds radiation (eos.py)."""
    return 2.0 * MU * M_P * specific_energy_to_si(u_code) / (3.0 * K_B)


def gas_u_from_temperature(t_kelvin):
    """Code specific internal energy of an ideal gas at T (inverse of gas_temperature_from_u)."""
    return specific_energy_from_si(1.5 * K_B * t_kelvin / (MU * M_P))


def gas_sound_speed_from_temperature(t_kelvin, gamma: float = GAMMA_GAS):
    """Ideal-gas adiabatic sound speed sqrt(gamma k T / (mu m_p)), in units of c."""
    return np.sqrt(gamma * K_B * t_kelvin / (MU * M_P)) / C


# --- display helpers (on-screen and report text use km, s, days, K; CLAUDE.md rule 4) ---


def format_time(seconds: float) -> str:
    """A time in the most readable unit: s, min, hours, days or years."""
    a = abs(seconds)
    for unit, size, limit in (
        ("s", 1.0, MINUTE),
        ("min", MINUTE, HOUR),
        ("hours", HOUR, 2 * DAY),
        ("days", DAY, 2 * YEAR),
    ):
        if a < limit:
            return f"{seconds / size:.4g} {unit}"
    return f"{seconds / YEAR:.4g} years"


def format_length(metres: float) -> str:
    """A length in km below 0.1 AU, otherwise in AU."""
    if abs(metres) < 0.1 * AU:
        km = metres / KM
        return f"{km:,.0f} km" if abs(km) >= 1e4 else f"{km:.4g} km"
    return f"{metres / AU:.4g} AU"


def eddington_luminosity_w(mass_msun: float = SGR_A_MASS_MSUN) -> float:
    """L_Edd = 4 pi G M m_p c / sigma_T (pure hydrogen, the convention for Eddington ratios;
    expected-values §9: 1.257e31 W per M_sun, 5.41e37 W for 4.3e6 M_sun)."""
    return 4.0 * np.pi * G * mass_msun * M_SUN * M_P * C / SIGMA_T


def eddington_accretion_msun_per_yr(mass_msun: float = SGR_A_MASS_MSUN,
                                    efficiency: float = 0.1) -> float:  # fmt: skip
    """Mdot_Edd = L_Edd / (efficiency c^2) in M_sun/yr (expected-values §9: 0.0955 for 10%)."""
    return eddington_luminosity_w(mass_msun) / (efficiency * C**2) / M_SUN * YEAR_S
