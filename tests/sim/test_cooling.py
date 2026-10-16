"""Phase 6 tests 6, 7, 9: beta-cooling toward the 10^4 K floor (phase-06 6.2-6.3), and the
energy ledger with cooling on."""

import math

import numpy as np

from nebula.physics.cooling import beta_cool, cooling_time, omega_kepler
from nebula.sim import blocks, coupled
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.hydro import HydroConfig, floor_energy, temperature
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def test_static_particle_exact_decay_and_ledger():
    """Test 6: u(t) = u_floor + (u0 - u_floor) exp(-t/t_cool) to 1e-12; sum m du = E_radiated."""
    r, beta, u0, uf, m = 100.0, 10.0, 1e-5, 2.3e-9, 0.3
    tc = cooling_time(r, beta)
    check("t_cool = beta / Omega_K, Omega_K = r^-1.5", beta * r**1.5, tc, rtol=1e-15)
    check("Omega_K(1) = 1", 1.0, float(omega_kepler(1.0)))
    for t in (0.1 * tc, tc, 3.7 * tc):
        check(f"u at t = {t / tc:.1f} t_cool", uf + (u0 - uf) * math.exp(-t / tc),
              float(beta_cool(u0, uf, t, tc)), rtol=1e-12)  # fmt: skip
    u1 = float(beta_cool(u0, uf, tc, tc))
    check("radiated energy m (u0 - u1)", m * (u0 - uf) * (1 - math.exp(-1)), m * (u0 - u1),
          rtol=1e-12)  # fmt: skip
    check("beta = inf: u unchanged", u0, float(beta_cool(u0, uf, tc, math.inf)))


def test_floor_is_reached_and_never_crossed():
    """Test 9: after 100 t_cool u sits at u_floor exactly, from above and from below, and the
    floor is the EOS energy at 10^4 K (ideal and gas + radiation)."""
    uf, tc = 2.3e-9, 5.0
    above = float(beta_cool(1e-4, uf, 100 * tc, tc))
    below = float(beta_cool(1e-10, uf, 100 * tc, tc))
    check("from above: u = u_floor after 100 t_cool", uf, above, rtol=1e-15)
    check("from below (photo-ionization floor): u = u_floor", uf, below, rtol=1e-15)
    path = beta_cool(1e-4, uf, np.linspace(0, 100 * tc, 1000), tc)
    check("never below the floor on the way down", True, bool(np.all(path >= uf)))
    for eos in ("ideal", "gasrad"):
        hc = HydroConfig(eos=eos)
        rho = np.array([1e-7, 1e-5])
        check(f"{eos}: T(u_floor) = 10^4 K", np.full(2, 1e4),
              temperature(hc, rho, floor_energy(hc, rho, 1e4)), rtol=1e-10)  # fmt: skip


def _static_cooling_particle(level_offset):
    """A particle at rest (no hole, no hydro) at r = 1000: its cooling time is exactly fixed.
    Its step there is level 1 of dt_max 64 (32), so t_end = 32 is one coarse step."""
    r, t_end = 1000.0, 32.0
    cc = CoupledConfig(hole=False, cooling_beta=40.0 / r**1.5)  # t_cool = 40: e^-0.8 over t_end
    cfg = BlockConfig(coupled=cc, dt_max=64.0, hydro=False)
    st = blocks.initialise([[r, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [1e-4], [1.0], [1.0], cfg,
                           t_stop=t_end, level_offset=[level_offset])  # fmt: skip
    blocks.run(st, cfg, t_end)
    return st, t_end


def test_cooling_independent_of_step_count():
    """Test 7: the same (static) particle cooled over the same interval in 1 step or 1,024 steps
    (block level forced 10 deeper): identical u to 1e-12; it equals the exact decay."""
    one, t_end = _static_cooling_particle(0)
    many, _ = _static_cooling_particle(10)
    check("steps taken: 1 and 1,024", (1, 1024), (one.particle_steps, many.particle_steps))
    check("u identical, 1 vs 1,024 steps", one.u, many.u, rtol=1e-12)
    uf = float(floor_energy(HydroConfig(), np.zeros(1), 1e4)[0])
    exact = uf + (1e-4 - uf) * math.exp(-t_end / 40.0)
    check("u = exact decay", exact, float(one.u[0]), rtol=1e-12)
    check("E_radiated = m (u0 - u)", 1e-4 - one.u[0], one.e_radiated, rtol=1e-12)


def _ledger_error(cooling_beta, frac, block_steps=False):
    rng = np.random.default_rng(20261014)
    x, v, u, m, h = warm_ball(rng, u=1e-3)
    cc = CoupledConfig(cooling_beta=cooling_beta)
    t_end = 128.0
    if block_steps:
        hc = HydroConfig(courant=0.3 * frac / 0.5, c_force=0.25 * frac / 0.5,
                         conductivity_signal=coupled.COND_VELOCITY)  # fmt: skip
        cfg = BlockConfig(coupled=CoupledConfig(hydro=hc, cooling_beta=cooling_beta), dt_max=64.0)
        st = blocks.initialise(x, v, u, m, h, cfg, t_stop=t_end)
        led0 = blocks.ledger(st, cfg)
        blocks.run(st, cfg, t_end)
        led = blocks.ledger(st, cfg)
    else:
        st = coupled.initialise(x, v, u, m, h, cc)
        led0 = coupled.ledger(st, cc)
        while st.t < t_end * (1 - 1e-14):
            st = coupled.step(st, cc, min(frac * coupled.stable_dt(st, cc), t_end - st.t))
        led = coupled.ledger(st, cc)
    return abs(led["energy"] - led0["energy"]) / led0["abs_h"], led["e_radiated"], np.sum(m * u)


def test_ledger_with_cooling_and_hydro():
    """A hot (u = 1e-3), stirred ball at r = 500 with hydro and strong cooling (beta = 0.01;
    ~40% of its heat radiated): the ledger sum m (H + u) + E_radiated has the same error as
    with cooling off (cooling adds no error of its own; the rest is the hydro time-step error
    of this violent ball), and that error falls ~4x when the step is halved."""
    for label, blk in (("shared step", False), ("block steps", True)):
        e_off, _, _ = _ledger_error(math.inf, 0.5, blk)
        e_on, rad, heat = _ledger_error(0.01, 0.5, blk)
        e_on_fine, _, _ = _ledger_error(0.01, 0.25, blk)
        print(f"   {label}: radiated {rad / heat:.0%} of the heat; ledger error cooling off "
              f"{e_off:.2e}, on {e_on:.2e}, on with half the step {e_on_fine:.2e}")  # fmt: skip
        check(f"{label}: heat radiated > 10%", True, rad > 0.1 * heat)
        check(f"{label}: ledger error with cooling / without", 1.0, e_on / e_off, atol=0.25)
        check(f"{label}: ledger error falls >= 3x per halving", True, e_on / e_on_fine >= 3.0)
