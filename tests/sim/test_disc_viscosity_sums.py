"""Disc viscosity pair sums (sim/sph.py, Phantom Eq. 120; phase-07 Option B): Numba vs
reference, exact energy, exact linear and angular momentum (unlike Option A's 2.7% torque),
heating never negative."""

import numpy as np

from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import compute_density, disc_forces_numba, disc_forces_reference
from tests.helpers import check
from tests.sim.test_coupled import glass_sphere

ALPHA_AV = 0.5


def _ball():
    pos = glass_sphere(1.0, 900)
    mass = np.full(len(pos), 1.0 / len(pos))
    d = compute_density(pos, mass, np.full(len(pos), 0.2))
    return pos, mass, d, find_neighbours_numba(pos, d.h)


def test_numba_matches_reference(rng):
    pos, mass, d, nb = _ball()
    vel = rng.normal(0.0, 0.1, pos.shape)
    cs = rng.uniform(0.05, 0.2, len(pos))
    a_ref, u_ref = disc_forces_reference(pos, vel, mass, d.h, d.rho, d.omega, cs, ALPHA_AV, nb)
    rows = np.sort(rng.choice(len(pos), 300, replace=False))
    a_nb, u_nb = disc_forces_numba(pos, vel, mass, d.h, d.rho, d.omega, cs, ALPHA_AV, nb,
                                   rows=rows)  # fmt: skip
    check("disc viscosity force: Numba vs reference (300 rows)", a_ref[rows], a_nb, rtol=1e-12,
          per_vector=True)  # fmt: skip
    check("disc viscosity heating: Numba vs reference", u_ref[rows], u_nb, rtol=1e-12)


def test_conservation(rng):
    """Rotating ball with velocity noise: sum m v.a + sum m du/dt = 0, total force 0, total
    torque 0 (pair forces along r_ab), each to round-off; du/dt >= 0 everywhere."""
    pos, mass, d, nb = _ball()
    vel = np.cross([0.0, 0.0, 0.3], pos) + rng.normal(0.0, 0.01, pos.shape)
    cs = rng.uniform(0.05, 0.2, len(pos))
    acc, dudt = disc_forces_numba(pos, vel, mass, d.h, d.rho, d.omega, cs, ALPHA_AV, nb)
    work = float(np.sum(mass[:, None] * vel * acc))
    heat = float(np.sum(mass * dudt))
    check("kinetic energy lost = heat made (relative)", 0.0, (work + heat) / heat, atol=1e-12)
    check("heating >= 0 for every particle", True, bool(np.all(dudt >= 0.0)))
    scale = float(np.sum(mass * np.linalg.norm(acc, axis=1)))
    check("total force / sum |m a|", np.zeros(3), np.sum(mass[:, None] * acc, axis=0) / scale,
          atol=1e-12)  # fmt: skip
    torque = np.sum(mass[:, None] * np.cross(pos, acc), axis=0)
    t_scale = float(np.sum(mass * np.linalg.norm(np.cross(pos, acc), axis=1)))
    print(f"   net torque / sum |torques| = {np.linalg.norm(torque) / t_scale:.2e}")
    check("total torque / sum |torques| (angular momentum exact)", np.zeros(3), torque / t_scale,
          atol=1e-12)  # fmt: skip
