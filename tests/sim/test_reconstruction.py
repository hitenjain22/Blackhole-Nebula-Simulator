"""Velocity reconstruction in the artificial viscosity (DECISIONS D2; Frontiere, Raskin & Owen
2017 Eqs. 48-53; Rosswog 2020 MAGMA2 Eqs. 17, 21-23) and the near-singular gradient fallback
(DECISIONS D11)."""

import numpy as np
import pytest

from nebula.physics import viscosity as visc
from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import (
    GATE_BOTH,
    GATE_PUBLISHED,
    ShockCapturing,
    compute_density,
    forces_numba,
    forces_reference,
    gradients_numba,
    gradients_reference,
    velocity_gradients_numba,
    velocity_gradients_reference,
)
from tests.helpers import check
from tests.sim.test_coupled import glass_sphere
from tests.sim.test_shock import _setup

RECON = ShockCapturing(reconstruction=True)
RECON_NO_CLOSENESS = ShockCapturing(reconstruction=True, closeness=False)


# --------------------------------------------------------------------------- formulas


def test_limiter_formulas():
    a = np.array([-1.0, 0.0, 1.0, 3.0, 1 / 3, 1e9])
    check("van Leer: 0 for A <= 0, 1 at A = 1, 0.75 at A = 3 and 1/3", [0, 0, 1, 0.75, 0.75, 0],
          visc.van_leer_limiter(a), atol=1e-8)  # fmt: skip
    check("van Leer symmetric under A -> 1/A", visc.van_leer_limiter(a[2:]),
          visc.van_leer_limiter(1 / a[2:]), rtol=1e-12)  # fmt: skip
    eta_c = 1 / 1.6
    check("closeness: 1 at and above eta_crit, exp(-(d/0.2)^2) below",
          [1.0, 1.0, np.exp(-1.0)], visc.closeness_factor([eta_c, 1.0, eta_c - 0.2], eta_c),
          rtol=1e-14)  # fmt: skip
    rng = np.random.default_rng(1)
    g = rng.normal(size=(3, 3))
    x = rng.normal(size=3)
    va, vb = rng.normal(size=3), rng.normal(size=3)
    jump = visc.reconstructed_jump(va, vb, g, g, x, 1.0)
    check("linear field (same gradient, v_a - v_b = G x_ab): jump 0", np.zeros(3),
          visc.reconstructed_jump(g @ x, np.zeros(3), g, g, x, 1.0), atol=1e-14)  # fmt: skip
    check("phi = 0: raw jump", va - vb, visc.reconstructed_jump(va, vb, g, -g, x, 0.0), atol=0)
    check("ratio of equal gradients is 1", 1.0, float(visc.limiter_ratio(g, g, x)), rtol=1e-14)
    mu = visc.viscous_jump([-1.0, -1.0, 0.5, -0.2], [-0.5, 0.1, -1.0, -0.3])
    check("min-mod gate: less compressive of the two, 0 unless both approach",
          [-0.5, 0.0, 0.0, -0.2], mu, atol=0)  # fmt: skip
    assert np.all(np.isfinite(jump))


# --------------------------------------------------------------------------- gradients


def test_velocity_gradient_numba_matches_reference_and_is_exact_for_linear_fields(rng):
    pos = glass_sphere(0.4, 900)
    n = len(pos)
    m = np.full(n, 1 / 4096)
    d = compute_density(pos, m, np.full(n, 1.6 * 4096 ** (-1 / 3)))
    vel = rng.normal(0, 1, (n, 3))
    ref = velocity_gradients_reference(pos, vel, m, d.h, d.neighbours)
    fast = velocity_gradients_numba(pos, vel, m, d.h, d.neighbours)
    check("velocity gradient: Numba vs reference", ref.reshape(n, 9), fast.reshape(n, 9),
          rtol=1e-11, per_vector=True)  # fmt: skip
    rows = np.arange(0, n, 3)
    part = velocity_gradients_numba(pos, vel, m, d.h, d.neighbours, rows=rows)
    check("velocity gradient: rows identical to full pass", fast[rows], part[rows], rtol=0.0)
    g = np.array([[0.3, -1.2, 0.5], [0.7, 0.1, -0.4], [-0.2, 0.9, 0.6]])
    lin = pos @ g.T + np.array([1.0, -2.0, 0.5])
    gl = velocity_gradients_numba(pos, lin, m, d.h, d.neighbours)
    check("linear field: gradient exact (all particles)", np.tile(g.ravel(), (n, 1)),
          gl.reshape(n, 9), rtol=1e-10, per_vector=True)  # fmt: skip
    sw = gradients_numba(pos, lin, np.zeros_like(pos), m, d.h, d.neighbours)
    check("same estimate as the switch's: div v", np.full(n, np.trace(g)), sw.div_v, rtol=1e-10)


def test_near_singular_gradient_falls_back_instead_of_crashing():
    """D11: particles on a line (correction matrix singular) or with too few neighbours."""
    pos = np.zeros((6, 3))
    pos[:, 0] = np.arange(6) * 0.1
    m = np.full(6, 1.0)
    h = np.full(6, 0.12)
    nb = find_neighbours_numba(pos, h)
    vel = np.zeros((6, 3))
    vel[:, 0] = pos[:, 0] * 2.0
    for name, fn in (("reference", gradients_reference), ("numba", gradients_numba)):
        g = fn(pos, vel, np.zeros_like(pos), m, h, nb)
        check(f"singular R ({name}): finite div v", True, bool(np.all(np.isfinite(g.div_v))))
    gv = velocity_gradients_numba(pos, vel, m, h, nb)
    gr = velocity_gradients_reference(pos, vel, m, h, nb)
    check("singular R: velocity gradient finite, Numba = reference", gr.reshape(6, 9),
          gv.reshape(6, 9), rtol=1e-12, atol=1e-14)  # fmt: skip


# --------------------------------------------------------------------------- forces


@pytest.mark.parametrize(
    "shock",
    [
        RECON,
        RECON_NO_CLOSENESS,
        ShockCapturing(reconstruction=True, recon_gate=GATE_BOTH),
        ShockCapturing(reconstruction=True, recon_gate=GATE_PUBLISHED),
    ],
    ids=["minmod", "minmod-no-closeness", "gate-both", "gate-published"],
)
def test_reconstructed_forces_numba_matches_reference(rng, shock):
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    gv = velocity_gradients_numba(pos, vel, mass, d.h, d.neighbours)
    args = (pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    ref = forces_reference(*args, u=u, c_s=cs, alpha=alpha, shock=shock, grad_v=gv)
    fast = forces_numba(*args, u=u, c_s=cs, alpha=alpha, shock=shock, grad_v=gv)
    check("recon: acceleration, Numba vs reference", ref.acc, fast.acc, rtol=1e-12,
          per_vector=True)  # fmt: skip
    scale = np.abs(ref.dudt).max()
    check("recon: du/dt, Numba vs reference (/max)", ref.dudt / scale, fast.dudt / scale,
          atol=1e-12)  # fmt: skip
    check("recon: v_sig,max, Numba vs reference", ref.vsig_max, fast.vsig_max, rtol=1e-12)
    plain = forces_numba(*args, u=u, c_s=cs, alpha=alpha)
    check("recon differs from plain viscosity for random velocities", True,
          bool(np.abs(plain.dudt - fast.dudt).max() > 1e-6 * scale))  # fmt: skip


def test_reconstructed_forces_conserve_instantaneously(rng):
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    gv = velocity_gradients_numba(pos, vel, mass, d.h, d.neighbours)
    f = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours, u=u, c_s=cs,
                     alpha=alpha, shock=RECON, grad_v=gv)  # fmt: skip
    rate = np.sum(mass * (np.sum(vel * f.acc, axis=1) + f.dudt))
    scale = np.sum(mass * (np.abs(np.sum(vel * f.acc, axis=1)) + np.abs(f.dudt)))
    check("recon: d(total energy)/dt (relative)", 0.0, abs(rate) / scale, atol=1e-13)
    mom = np.sum(mass[:, None] * f.acc, axis=0)
    rel_mom = np.linalg.norm(mom) / np.sum(mass * np.linalg.norm(f.acc, axis=1))
    check("recon: sum m a (relative)", 0.0, rel_mom, atol=1e-13)
    am = np.sum(mass[:, None] * np.cross(pos, f.acc), axis=0)
    am_scale = np.sum(mass * np.linalg.norm(np.cross(pos, f.acc), axis=1))
    check("recon: sum m x cross a (relative)", 0.0, np.linalg.norm(am) / am_scale, atol=1e-13)


def test_linear_flow_has_no_viscosity():
    """The point of D2: a smooth compression (linear velocity field, e.g. gas squeezed on its
    way in) gets no artificial viscosity. Without the closeness factor exactly none (interior
    particles, whose neighbourhoods see the exact gradient); plain viscosity heats strongly."""
    pos = glass_sphere(0.45, 2000)
    n = len(pos)
    m = np.full(n, 1 / 4096)
    d = compute_density(pos, m, np.full(n, 1.6 * 4096 ** (-1 / 3)))
    vel = -1.0 * pos  # uniform contraction: div v = -3, supersonic for c_s = 0.01
    u = np.full(n, 1e-4)
    p = (2 / 3) * d.rho * u
    cs = np.sqrt(10 / 9 * u)
    alpha = np.ones(n)
    gv = velocity_gradients_numba(pos, vel, m, d.h, d.neighbours)
    args = (pos, vel, m, d.h, d.rho, d.omega, p, d.neighbours)
    base = forces_numba(*args)  # pressure only
    plain = forces_numba(*args, c_s=cs, alpha=alpha)
    recon = forces_numba(*args, c_s=cs, alpha=alpha, shock=RECON_NO_CLOSENESS, grad_v=gv)
    close = forces_numba(*args, c_s=cs, alpha=alpha, shock=RECON, grad_v=gv)
    inner = np.linalg.norm(pos, axis=1) < 0.45 - 4 * np.median(d.h)
    heat_plain = np.abs(plain.dudt - base.dudt)[inner].max()
    heat_recon = np.abs(recon.dudt - base.dudt)[inner].max()
    heat_close = np.abs(close.dudt - base.dudt)[inner].max()
    print(f"   viscous heating, interior: plain {heat_plain:.3e}, recon {heat_recon:.3e}, "
          f"recon + closeness {heat_close:.3e}")  # fmt: skip
    check("linear flow: reconstruction (no closeness) removes the viscous heating", 0.0,
          heat_recon / heat_plain, atol=1e-10)  # fmt: skip
    check("linear flow: with the closeness factor, heating < 10% of plain", True,
          bool(heat_close < 0.1 * heat_plain))  # fmt: skip


def test_opposite_gradients_give_plain_viscosity(rng):
    """Limiter path in the kernel: when every pair's two gradients disagree in sign
    (x_ab . G_a . x_ab and x_ab . G_b . x_ab of opposite signs, A < 0, Phi = 0), the viscosity
    is exactly the plain one. (At a sharp velocity step the two sides see the same smeared
    gradient, so reconstruction still lowers the interface heating, to ~40% in a test; that
    shocks are still captured is checked by the Sod and Sedov tests.)"""
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    n = len(pos)
    gv = np.where((np.arange(n) % 2 == 0)[:, None, None], 1.0, -1.0) * np.eye(3)[None]
    nb = d.neighbours
    keep = [
        np.array([b for b in nb.of(a) if b == a or (a + b) % 2 == 1], np.int64) for a in range(n)
    ]  # only pairs of opposite parity: x.G_a.x = +|x|^2, x.G_b.x = -|x|^2
    off = np.concatenate([[0], np.cumsum([len(k) for k in keep])]).astype(np.int64)
    pairs = type(nb)(off, np.concatenate(keep), nb.box)
    args = (pos, vel, mass, d.h, d.rho, d.omega, p, pairs)
    plain = forces_numba(*args, u=u, c_s=cs, alpha=alpha)
    recon = forces_numba(*args, u=u, c_s=cs, alpha=alpha, shock=RECON, grad_v=gv)
    check("A < 0 for every pair: acceleration = plain", plain.acc, recon.acc, rtol=1e-14,
          per_vector=True)  # fmt: skip
    check("A < 0 for every pair: du/dt = plain", plain.dudt, recon.dudt, rtol=1e-13)


def test_viscous_heating_marker(rng):
    """The shock marker (Hiten, 2026-10-15): the viscous part of du/dt, returned separately.
    Numba = reference; with reconstruction (no closeness factor) it is zero in a smooth
    squeeze; in a head-on collision it is positive at the interface."""
    pos, vel, mass, d, p, u, cs, alpha = _setup(rng)
    gv = velocity_gradients_numba(pos, vel, mass, d.h, d.neighbours)
    args = (pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours)
    ref = forces_reference(*args, u=u, c_s=cs, alpha=alpha, shock=RECON, grad_v=gv)
    fast = forces_numba(*args, u=u, c_s=cs, alpha=alpha, shock=RECON, grad_v=gv)
    scale = np.abs(ref.dudt_visc).max()
    check("viscous heating: Numba vs reference (/max)", ref.dudt_visc / scale,
          fast.dudt_visc / scale, atol=1e-12)  # fmt: skip
    check("viscous heating is never negative (min-mod gate)", True, bool(fast.dudt_visc.min() >= 0))
    pos = glass_sphere(0.45, 2000)
    n = len(pos)
    m = np.full(n, 1 / 4096)
    d = compute_density(pos, m, np.full(n, 1.6 * 4096 ** (-1 / 3)))
    u = np.full(n, 1e-4)
    p = (2 / 3) * d.rho * u
    cs = np.sqrt(10 / 9 * u)
    inner = np.linalg.norm(pos, axis=1) < 0.45 - 4 * np.median(d.h)
    for label, vel, expect_zero in (("smooth squeeze", -1.0 * pos, True),
                                    ("head-on collision", None, False)):  # fmt: skip
        if vel is None:
            vel = np.zeros((n, 3))
            vel[:, 0] = np.where(pos[:, 0] < 0, 1.0, -1.0)
        gv = velocity_gradients_numba(pos, vel, m, d.h, d.neighbours)
        f = forces_numba(pos, vel, m, d.h, d.rho, d.omega, p, d.neighbours, c_s=cs,
                         alpha=np.ones(n), shock=RECON_NO_CLOSENESS, grad_v=gv)  # fmt: skip
        if expect_zero:
            check(f"{label}: viscous heating (interior) ~ 0", 0.0,
                  float(np.abs(f.dudt_visc[inner]).max()), atol=1e-15)  # fmt: skip
        else:
            check(f"{label}: viscous heating at the interface > 0", True,
                  bool(f.dudt_visc[np.abs(pos[:, 0]) < 0.02].mean() > 0))  # fmt: skip
