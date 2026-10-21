"""Navier-Stokes viscosity pair sums (sim/sph.py, Phantom Eqs. 130-131): Numba vs reference,
the gradient of a linear field, exact energy conservation, and the angular-momentum error the
plan asks to measure (phase-07 7.2, Option A)."""

import numpy as np

from nebula.physics.navier_stokes import viscous_heating, viscous_stress
from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import (
    compute_density,
    ns_forces_numba,
    ns_forces_reference,
    ns_gradients_numba,
    ns_gradients_reference,
)
from tests.helpers import check
from tests.sim.test_coupled import glass_sphere


def _ball(rng, n_target=900):
    pos = glass_sphere(1.0, n_target)
    mass = np.full(len(pos), 1.0 / len(pos))
    d = compute_density(pos, mass, np.full(len(pos), 0.2))
    nb = find_neighbours_numba(pos, d.h)
    return pos, mass, d, nb


def test_numba_matches_reference(rng):
    pos, mass, d, nb = _ball(rng)
    vel = rng.normal(0.0, 0.1, pos.shape)
    g_ref = ns_gradients_reference(pos, vel, mass, d.h, d.rho, d.omega, nb)
    g_nb = ns_gradients_numba(pos, vel, mass, d.h, d.rho, d.omega, nb)
    check("NS gradient (Eq. 131): Numba vs reference", g_ref, g_nb, rtol=1e-12, per_vector=True)
    nu = rng.uniform(0.01, 0.05, len(pos))
    stress = viscous_stress(g_ref, d.rho, nu)
    a_ref = ns_forces_reference(pos, mass, d.h, d.rho, d.omega, stress, nb)
    rows = np.arange(len(pos))
    a_nb = ns_forces_numba(pos, mass, d.h, d.rho, d.omega, stress, nb, rows=rows)
    check("NS force (Eq. 130): Numba vs reference", a_ref, a_nb, rtol=1e-12, per_vector=True)


def test_gradient_of_a_linear_field(rng):
    """v = A x: the difference operator recovers A in the well-sampled interior of a glass (it
    is not exact, unlike the corrected gradient; within a few per cent)."""
    pos, mass, d, nb = _ball(rng)
    a_mat = rng.normal(0.0, 1.0, (3, 3))
    g = ns_gradients_numba(pos, pos @ a_mat.T, mass, d.h, d.rho, d.omega, nb)
    inner = np.linalg.norm(pos, axis=1) < 0.6
    err = np.linalg.norm(g[inner] - a_mat, axis=(1, 2)) / np.linalg.norm(a_mat)
    print(f"   linear field: median relative error {np.median(err):.3%}, worst {err.max():.3%}")
    check("difference-operator gradient of v = A x (interior median)", 0.0, float(np.median(err)),
          atol=0.03)  # fmt: skip


def test_energy_conserved_exactly(rng):
    """sum m (v . a_visc) + sum m Lambda = 0 (Eq. 132) for random velocities and viscosities."""
    pos, mass, d, nb = _ball(rng)
    vel = rng.normal(0.0, 0.1, pos.shape)
    nu = rng.uniform(0.01, 0.05, len(pos))
    g = ns_gradients_numba(pos, vel, mass, d.h, d.rho, d.omega, nb)
    acc = ns_forces_numba(pos, mass, d.h, d.rho, d.omega, viscous_stress(g, d.rho, nu), nb,
                          rows=np.arange(len(pos)))  # fmt: skip
    work = float(np.sum(mass[:, None] * vel * acc))
    heat = float(np.sum(mass * viscous_heating(g, nu)))
    print(f"   kinetic energy change {work:.6e}, heat {heat:.6e}")
    check("viscosity: kinetic energy lost = heat made (relative)", 0.0, (work + heat) / heat,
          atol=1e-12)  # fmt: skip
    check("viscosity heats (positive)", True, heat > 0)


def test_momentum_conserved_and_angular_momentum_measured(rng):
    """Linear momentum is conserved exactly (pair terms antisymmetric); angular momentum is not
    guaranteed for the two-first-derivatives form (pair force not along r_ab): measured for a
    rotating ball and reported (phase-07 7.2)."""
    pos, mass, d, nb = _ball(rng)
    omega_vec = np.array([0.0, 0.0, 0.3])
    vel = np.cross(omega_vec, pos) + rng.normal(0.0, 0.01, pos.shape)
    nu = np.full(len(pos), 0.02)
    g = ns_gradients_numba(pos, vel, mass, d.h, d.rho, d.omega, nb)
    acc = ns_forces_numba(pos, mass, d.h, d.rho, d.omega, viscous_stress(g, d.rho, nu), nb,
                          rows=np.arange(len(pos)))  # fmt: skip
    scale = float(np.sum(mass * np.linalg.norm(acc, axis=1)))
    check("viscosity: total force / sum |m a|", np.zeros(3),
          np.sum(mass[:, None] * acc, axis=0) / scale, atol=1e-12)  # fmt: skip
    torque = np.sum(mass[:, None] * np.cross(pos, acc), axis=0)
    t_scale = float(np.sum(mass * np.linalg.norm(np.cross(pos, acc), axis=1)))
    print(f"   net viscous torque / sum |torques| = {np.linalg.norm(torque) / t_scale:.2e}")
