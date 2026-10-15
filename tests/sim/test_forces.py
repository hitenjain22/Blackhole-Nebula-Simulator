"""Step 3.4: pressure force and P dV heating (phases/phase-03-sph-core.md 3.2 tests 3 and 8;
conservation by antisymmetric pair forces)."""

import numba
import numpy as np

from nebula.physics.eos import ideal_gas_pressure
from nebula.sim.ic import fcc_lattice, smoothing_length_uniform
from nebula.sim.sph import compute_density, forces_numba, forces_reference
from tests.helpers import check
from tests.sim.test_density import blob

GAMMA = 5.0 / 3.0


def hydro_state(pos, mass, u, h0, box=None, h_max=np.inf):
    d = compute_density(pos, mass, h0, box=box, h_max=h_max)
    return d, ideal_gas_pressure(d.rho, u, GAMMA)


def uneven_moving(rng):
    pos, mass = blob(rng)
    u = rng.uniform(0.5, 1.5, len(pos))
    vel = rng.normal(0.0, 0.3, size=pos.shape)
    d, p = hydro_state(pos, mass, u, np.full(len(pos), 0.05), h_max=0.5)
    return pos, vel, mass, d, p


def test_uniform_lattice_no_net_force():
    """Plan test 3: uniform pressure on a lattice, periodic box: zero acceleration to round-off."""
    pos = fcc_lattice(8)
    n = len(pos)
    mass = np.full(n, 1.0 / n)
    d, p = hydro_state(
        pos, mass, np.full(n, 1.0), np.full(n, smoothing_length_uniform(n, 1.0)), box=1.0
    )
    f = forces_numba(pos, np.zeros_like(pos), mass, d.h, d.rho, d.omega, p, d.neighbours)
    scale = p[0] / (d.rho[0] * d.h[0])  # size of one neighbour's push
    check("lattice: max |a| / (P / rho h)", 0.0, np.abs(f.acc).max() / scale, atol=1e-12)
    check("lattice at rest: du/dt = 0", 0.0, np.abs(f.dudt).max(), atol=1e-14)


def test_conservation_of_pair_forces(rng):
    """Uneven, moving gas: total momentum, angular momentum and energy rates vanish to
    round-off (relative to the sum of magnitudes)."""
    pos, vel, mass, d, p = uneven_moving(rng)
    f = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    mom = np.sum(mass[:, None] * f.acc, axis=0)
    check(
        "sum m a = 0 (relative)",
        0.0,
        np.linalg.norm(mom) / np.sum(mass * np.linalg.norm(f.acc, axis=1)),
        atol=1e-13,
    )
    torque = np.sum(mass[:, None] * np.cross(pos, f.acc), axis=0)
    t_scale = np.sum(mass * np.linalg.norm(pos, axis=1) * np.linalg.norm(f.acc, axis=1))
    check("sum m x cross a = 0 (relative)", 0.0, np.linalg.norm(torque) / t_scale, atol=1e-13)
    power = np.sum(mass * (np.sum(vel * f.acc, axis=1) + f.dudt))
    p_scale = np.sum(mass * (np.abs(np.sum(vel * f.acc, axis=1)) + np.abs(f.dudt)))
    check("d/dt sum m (v^2/2 + u) = 0 (relative)", 0.0, abs(power) / p_scale, atol=1e-13)


def test_numba_matches_reference(rng):
    pos, vel, mass, d, p = uneven_moving(rng)
    ref = forces_reference(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    fast = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    check("Numba vs reference: acceleration", ref.acc, fast.acc, rtol=1e-12, per_vector=True)
    scale = np.abs(ref.dudt).max()
    check("Numba vs reference: du/dt (/ max)", ref.dudt / scale, fast.dudt / scale, atol=1e-12)


def test_threads_identical(rng):
    pos, vel, mass, d, p = uneven_moving(rng)
    numba.set_num_threads(1)
    one = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    many = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    check(
        "1 thread vs all threads: identical forces",
        True,
        bool(np.array_equal(one.acc, many.acc) and np.array_equal(one.dudt, many.dudt)),
    )


def test_compression_heats():
    """Uniform contraction v = -(x - c): du/dt = -(P/rho) div v = 3 P / rho (interior particles of
    an open lattice; SPH estimates div v on a lattice to within ~1%)."""
    pos = fcc_lattice(10)
    n = len(pos)
    mass = np.full(n, 1.0 / n)
    d, p = hydro_state(pos, mass, np.full(n, 1.0), np.full(n, smoothing_length_uniform(n, 1.0)))
    vel = -(pos - 0.5)
    f = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    inner = np.linalg.norm(pos - 0.5, axis=1) < 0.25
    check(
        "contraction: du/dt = 3 P / rho (interior)",
        3 * p[inner] / d.rho[inner],
        f.dudt[inner],
        rtol=1e-2,
    )
    expansion = forces_numba(pos, -vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    check("expansion cools (du/dt < 0, interior)", True, bool(np.all(expansion.dudt[inner] < 0)))
