"""Phase 6 diagnostics (phases/phase-06-shocks-cooling-ring.md 6.5): orbit elements of every
particle, the eccentricity histogram, ring/torus measures, luminosity, accretion rate,
temperature histogram, optical-depth fractions and the unbound fraction. Pure functions of
particle arrays; physics from nebula.physics, unit conversions from nebula.units.
"""

from dataclasses import dataclass

import numpy as np

from nebula import units
from nebula.physics.gravity import gn_angmom, gn_hamiltonian
from nebula.physics.opacity import kappa_eff_si, kappa_es_si, kappa_ff_si
from nebula.physics.orbits import turning_points

RING_ECC_MAX = 0.3  # phase-06 6.4: "torus formed" = e < 0.3 and r < 200
RING_R_MAX = 200.0


@dataclass
class OrbitElements:
    r_peri: np.ndarray
    r_apo: np.ndarray
    ecc: np.ndarray  # (r_apo - r_peri)/(r_apo + r_peri) for bound orbits, NaN otherwise
    kind: np.ndarray  # 0 bound, 1 unbound, 2 plunging (no turning point, or r_peri < 4)


def orbit_elements(x: np.ndarray, p: np.ndarray, r_plunge: float = 4.0) -> OrbitElements:
    """Exact Schwarzschild turning points from each particle's GN energy H = (E^2 - 1)/2 and
    angular momentum L = |x cross p| (physics.orbits.turning_points); orbits whose pericentre
    lies below r_plunge (r_acc) count as plunging (phase-06 6.5)."""
    h = gn_hamiltonian(np.asarray(x, float), np.asarray(p, float))
    ell = np.linalg.norm(gn_angmom(np.asarray(x, float), np.asarray(p, float)), axis=1)
    e_inf = np.sqrt(np.maximum(1.0 + 2.0 * h, 0.0))
    r_peri, r_apo, kind = turning_points(e_inf, ell)
    kind = np.where((kind == 0) & (r_peri < r_plunge), 2, kind)
    with np.errstate(invalid="ignore"):
        ecc = np.where(kind == 0, (r_apo - r_peri) / (r_apo + r_peri), np.nan)
    return OrbitElements(r_peri, r_apo, ecc, kind)


@dataclass
class RingMeasures:
    mass_fraction: float  # of the bound mass present: e < 0.3 and r < 200
    mean_radius: float  # mass-weighted, of the ring gas
    thickness: float  # H/r = rms(z) / mean radius
    r10: float  # radial extent: 10th and 90th percentiles of r
    r90: float


def ring_measures(x: np.ndarray, mass: np.ndarray, el: OrbitElements) -> RingMeasures:
    r = np.linalg.norm(x, axis=1)
    bound = el.kind == 0
    ring = bound & (el.ecc < RING_ECC_MAX) & (r < RING_R_MAX)
    m = np.asarray(mass, float)
    frac = float(np.sum(m[ring]) / np.sum(m[bound])) if np.any(bound) else 0.0
    if not np.any(ring):
        return RingMeasures(frac, np.nan, np.nan, np.nan, np.nan)
    rr, zz, mm = r[ring], x[ring, 2], m[ring]
    mean_r = float(np.sum(mm * rr) / np.sum(mm))
    h_over_r = float(np.sqrt(np.sum(mm * zz**2) / np.sum(mm)) / mean_r)
    r10, r90 = np.percentile(rr, [10, 90])
    return RingMeasures(frac, mean_r, h_over_r, float(r10), float(r90))


def eccentricity_histogram(el: OrbitElements, mass: np.ndarray, bins=20):
    """Mass-weighted histogram of e over [0, 1] (bound orbits): (bin centres, mass per bin)."""
    sel = el.kind == 0
    counts, edges = np.histogram(el.ecc[sel], bins=bins, range=(0.0, 1.0), weights=mass[sel])
    return 0.5 * (edges[1:] + edges[:-1]), counts


def unbound_fraction(el: OrbitElements, mass: np.ndarray) -> float:
    m = np.asarray(mass, float)
    return float(np.sum(m[el.kind == 1]) / np.sum(m))


def luminosity_watts(luminosity_code, gas_units: units.GasUnits | None = None):
    """Code luminosity (code mass x c^2 per GM/c^3) to watts."""
    gu = units.GasUnits() if gas_units is None else gas_units
    return np.asarray(luminosity_code) * gu.mass_kg_per_code * units.C**2 / gu.bh.time_s


def accretion_rate(times, masses, t_edges, gas_units: units.GasUnits | None = None):
    """Binned accretion rate from removals (times, code masses): (bin centres, Mdot in M_sun/yr,
    Mdot / Mdot_Edd)."""
    gu = units.GasUnits() if gas_units is None else gas_units
    m, edges = np.histogram(np.asarray(times), bins=t_edges, weights=np.asarray(masses))
    rate_code = m / np.diff(edges)  # code mass per code time
    rate_msun_yr = rate_code * gu.mass_kg_per_code / units.M_SUN / gu.bh.time_s * units.YEAR_S
    return (0.5 * (edges[1:] + edges[:-1]), rate_msun_yr,
            rate_msun_yr / units.eddington_accretion_msun_per_yr(gu.bh.mass_msun))  # fmt: skip


def temperature_histogram(temps_k, mass, bins=40, range_log=(3.5, 10.0)):
    """Mass-weighted histogram of log10 T: (bin centres, mass per bin)."""
    counts, edges = np.histogram(np.log10(temps_k), bins=bins, range=range_log, weights=mass)
    return 0.5 * (edges[1:] + edges[:-1]), counts


def optical_depth_fractions(rho_code, temps_k, h_code, mass, gas_units=None):
    """Mass fractions with tau_es,h = kappa_es rho h < 1 and tau_eff,h = kappa_eff rho h < 1
    (free-free absorption only, so tau_eff is a lower bound; physics/opacity.py)."""
    gu = units.GasUnits() if gas_units is None else gas_units
    rho_si = gu.density_to_si(np.asarray(rho_code, float))
    h_m = np.asarray(h_code, float) * gu.bh.length_m
    k_es = kappa_es_si()
    k_eff = kappa_eff_si(kappa_ff_si(rho_si, temps_k), k_es)
    m = np.asarray(mass, float)
    return (float(np.sum(m[k_es * rho_si * h_m < 1.0]) / np.sum(m)),
            float(np.sum(m[k_eff * rho_si * h_m < 1.0]) / np.sum(m)))  # fmt: skip


def mass_in_shocks(x, u, visc_heating, mass):
    """Shock marker (Hiten, 2026-10-15): the viscous heating rate after velocity reconstruction.
    A particle is "in a shock" when that heating would double its internal energy within one
    orbital time unit at its radius: (du/dt)_visc > Omega_K(r) u, Omega_K = r^(-3/2) (a
    scale-free threshold; DECISIONS, Phase 6.7). Returns (mass fraction in shocks, mass-weighted
    mean position of the shocked gas, or NaN if none)."""
    from nebula.physics.cooling import omega_kepler

    r = np.linalg.norm(x, axis=1)
    shocked = np.asarray(visc_heating) > omega_kepler(r) * np.asarray(u)
    m = np.asarray(mass, float)
    frac = float(np.sum(m[shocked]) / np.sum(m))
    where = (np.sum(m[shocked, None] * x[shocked], axis=0) / np.sum(m[shocked])
             if np.any(shocked) else np.full(3, np.nan))  # fmt: skip
    return frac, where
