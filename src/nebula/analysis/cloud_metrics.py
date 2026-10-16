"""Measurements on cloud snapshots (phases/phase-05-cloud-infall.md 5.2 "Diagnostics per
snapshot"). Pure functions of particle arrays; physics formulas come from nebula.physics and
nebula.units.

- ``energy_distribution``: specific orbital energy eps = H_GN(x, p) of every particle
  (= (E^2 - 1)/2, expected-values §2: -1.989189e-4 for the cloud's orbit) and its spread; the
  mass-weighted histogram dM/d(eps). The spread across the cloud sets how fast the stream
  lengthens and when it first crashes into itself (Phase 6; review §7).
- ``centre_of_mass_offset``: the cloud's centre of mass against a single test particle started
  from the cloud's initial mean position and momentum (phase-05 5.3: with pressure negligible
  it must follow the point-mass orbit).
- ``principal_axes``: rms extents along the principal directions of the mass-weighted position
  covariance (for a uniform ball of radius R each is R/sqrt(5)). The largest is the stream's
  length; for a stream bent around the hole the other two mostly measure its curvature, so
  ``thickness`` (rms height above the orbital plane z = 0) is the vertical measure.
- ``thermal_summary``: maximum density and temperature, mass above 10^5 K and 10^6 K.
"""

from dataclasses import dataclass

import numpy as np

from nebula.physics.gravity import gn_hamiltonian
from nebula.sim.hydro import HydroConfig, temperature
from nebula.sim.integrator import integrate_orbit
from nebula.units import gas_temperature_from_u


@dataclass
class EnergyDistribution:
    eps: np.ndarray  # per particle, units of c^2
    mean: float
    half_range: float  # (max - min) / 2: dominated by a few outliers after pericentre
    p01: float  # percentiles of eps
    p99: float
    std: float
    p05: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    unbound: float = 0.0  # mass fraction with eps > 0

    def histogram(self, mass, bins=60, edges=None):
        """Mass-weighted dM/d(eps): (bin centres, dM/d(eps)). Its integral is the total mass."""
        edges = np.linspace(self.eps.min(), self.eps.max(), bins + 1) if edges is None else edges
        counts, edges = np.histogram(self.eps, bins=edges, weights=mass)
        return 0.5 * (edges[1:] + edges[:-1]), counts / np.diff(edges)


def energy_distribution(x: np.ndarray, p: np.ndarray, mass: np.ndarray) -> EnergyDistribution:
    """Specific orbital energy of every particle (GN Hamiltonian per unit mass, c^2) and its
    mass-weighted mean and spread. Gas pressure and heat are not included: this is the energy
    that sets each particle's orbit."""
    eps = gn_hamiltonian(np.asarray(x, float), np.asarray(p, float))
    m = np.asarray(mass, float)
    mean = float(np.sum(m * eps) / np.sum(m))
    std = float(np.sqrt(np.sum(m * (eps - mean) ** 2) / np.sum(m)))
    p01, p05, p25, p75, p95, p99 = np.percentile(eps, [1, 5, 25, 75, 95, 99])
    return EnergyDistribution(
        eps, mean, 0.5 * float(eps.max() - eps.min()), float(p01), float(p99), std,
        float(p05), float(p25), float(p75), float(p95), float(np.sum(m[eps > 0]) / np.sum(m)),
    )  # fmt: skip


def centre_of_mass_offset(
    x0: np.ndarray,
    p0: np.ndarray,
    mass: np.ndarray,
    times,
    centres,
    eta: float = 0.00125,
    t0: float = 0.0,
) -> np.ndarray:
    """Distance between the cloud's centre of mass ``centres[k]`` at ``times[k]`` and a test
    particle started at time t0 from the mass-weighted mean position and momentum (x0, p0) of
    the cloud at t0, integrated with the Phase 1 generalized leapfrog (step eta r^(3/2))."""
    m = np.asarray(mass, float)[:, None]
    xc = np.sum(m * x0, axis=0) / np.sum(m)
    pc = np.sum(m * p0, axis=0) / np.sum(m)
    out = []
    for t, c in zip(times, centres, strict=True):
        if t <= t0:
            out.append(float(np.linalg.norm(np.asarray(c) - xc)))
            continue
        orb = integrate_orbit(xc, pc, float(t) - t0, eta=eta)
        out.append(float(np.linalg.norm(np.asarray(c) - orb.x[-1])))
    return np.array(out)


def thickness(x: np.ndarray, mass: np.ndarray) -> float:
    """rms height above the orbital plane z = 0 (the cloud orbits in the x-y plane)."""
    m = np.asarray(mass, float)
    return float(np.sqrt(np.sum(m * np.asarray(x, float)[:, 2] ** 2) / np.sum(m)))


def principal_axes(x: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """rms extents along the principal directions, largest first: (length, width, thickness)."""
    m = np.asarray(mass, float)
    xc = np.asarray(x, float) - np.sum(m[:, None] * x, axis=0) / np.sum(m)
    cov = (m[:, None, None] * xc[:, :, None] * xc[:, None, :]).sum(axis=0) / np.sum(m)
    return np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(cov))[::-1], 0.0, None))


@dataclass
class ThermalSummary:
    rho_max: float  # code units
    t_max: float  # K
    mass_above_1e5: float  # fraction of the mass present
    mass_above_1e6: float


def snapshot_temperature(eos: str, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Temperature (K) of each particle for the run's equation of state (manifest "eos"; "ideal"
    for Phase 5 runs). With gas + radiation, T is far below the ideal-gas value once radiation
    holds most of the energy, so pictures must not use the ideal-gas formula for those runs."""
    if eos == "ideal":
        return gas_temperature_from_u(np.asarray(u, float))
    return temperature(HydroConfig(eos=eos), np.asarray(rho, float), np.asarray(u, float))


def thermal_summary(
    u: np.ndarray, rho: np.ndarray, mass: np.ndarray, eos: str = "ideal"
) -> ThermalSummary:
    """Maximum density and temperature (for the run's EOS), mass fractions above 10^5, 10^6 K."""
    temp = snapshot_temperature(eos, rho, u)
    m = np.asarray(mass, float)
    total = np.sum(m)
    return ThermalSummary(float(np.max(rho)), float(np.max(temp)),
                          float(np.sum(m[temp > 1e5]) / total),
                          float(np.sum(m[temp > 1e6]) / total))  # fmt: skip
