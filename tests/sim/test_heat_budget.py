"""Per-particle heat budget on block steps (sim/blocks.py heat_visc, heat_cond, heat_pdv,
heat_cool): at a sync point every particle's u - u_0 equals the sum of its parts, and the
conductivity split agrees between the reference and Numba force passes."""

from dataclasses import replace

import numpy as np
import pytest

from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.hydro import HydroConfig
from nebula.sim.sph import forces_numba, forces_reference
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def test_heat_budget_adds_up_with_cooling(rng):
    """Warm stirred ball on its orbit, gas + radiation EOS and beta-cooling, block steps:
    every kick is booked, including the opening half-kick of the step each particle has just
    begun, so u_half - u_0 = visc + cond + pdv - cool for every particle (round-off)."""
    x, v, u, m, h = warm_ball(rng)
    cc = CoupledConfig(cooling_beta=2.0)
    cc = replace(cc, hydro=replace(cc.hydro, eos="gasrad"))
    cfg = BlockConfig(coupled=cc, dt_max=64.0, r_acc=4.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 6)
    u0 = st.u.copy()
    blocks.run(st, cfg, 64.0 * 6)
    parts = st.heat_visc + st.heat_cond + st.heat_pdv - st.heat_cool
    scale = float(np.max(np.abs(u0)))
    print(f"   per-particle changes: visc {np.median(st.heat_visc):.2e}, cond "
          f"{np.median(np.abs(st.heat_cond)):.2e}, pdv {np.median(st.heat_pdv):.2e}, cool "
          f"{np.median(st.heat_cool):.2e}; woken {st.woken}")  # fmt: skip
    check("heat budget: u_half - u0 = visc + cond + pdv - cool (per particle, / max u0)", 0.0,
          np.max(np.abs(st.u_half - u0 - parts)) / scale, atol=1e-12)  # fmt: skip
    check("every part acted", True, bool(np.all([np.any(a != 0) for a in (
        st.heat_visc, st.heat_cond, st.heat_pdv, st.heat_cool)])))  # fmt: skip
    check("cooling budget equals the radiated energy", st.e_radiated,
          float(np.sum(st.mass * st.heat_cool)), rtol=1e-12)  # fmt: skip


def test_conductivity_part_reference_vs_numba(rng):
    """The new dudt_cond output: Numba pass vs reference (rule 2), with reconstruction on."""
    from nebula.sim.neighbours import find_neighbours
    from nebula.sim.sph import compute_density, velocity_gradients_reference

    x, v, u, m, h = warm_ball(rng)
    u = u * rng.uniform(0.5, 2.0, len(u))  # uneven u, so conductivity acts
    d = compute_density(x, m, h)
    nb = find_neighbours(x, d.h)
    hc = HydroConfig()
    p = (hc.gamma - 1.0) * d.rho * u
    cs = np.sqrt(hc.gamma * (hc.gamma - 1.0) * u)
    alpha = np.ones(len(u))
    g = velocity_gradients_reference(x, v, m, d.h, nb)
    kw = dict(u=u, c_s=cs, alpha=alpha, shock=hc.shock(), grad_v=g)
    ref = forces_reference(x, v, m, d.h, d.rho, d.omega, p, nb, **kw)
    fast = forces_numba(x, v, m, d.h, d.rho, d.omega, p, nb, **kw)
    scale = float(np.max(np.abs(ref.dudt)))
    check("conductivity part: Numba vs reference (/ max |du/dt|)", 0.0,
          float(np.max(np.abs(ref.dudt_cond - fast.dudt_cond))) / scale, atol=1e-12)  # fmt: skip
    check("conductivity part is not zero", True, bool(np.any(ref.dudt_cond != 0)))


def test_heat_budget_adds_up_with_wake_ups(rng):
    """As above with forced fine levels on a few particles, so the limiter wakes neighbours up
    (their steps are cut short and the half-kick corrected): the budget still adds up."""
    x, v, u, m, h = warm_ball(rng)
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0, r_acc=4.0)
    offset = np.where(rng.random(len(u)) < 0.05, 5, 0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 4, level_offset=offset)
    u0 = st.u.copy()
    for _ in range(400):  # until neighbours of the forced particles have been woken
        blocks.advance(st, cfg)
        if st.woken > 20 and st.synced():
            break
    parts = st.heat_visc + st.heat_cond + st.heat_pdv - st.heat_cool
    print(f"   woken {st.woken}, synced {st.synced()}")
    check("wake-ups exercised", True, st.woken > 0)
    check("heat budget with wake-ups (per particle, / max u0)", 0.0,
          np.max(np.abs(st.u_half - u0 - parts)) / float(np.max(u0)), atol=1e-12)  # fmt: skip


@pytest.mark.parametrize("option", ["A", "B"])
def test_heat_budget_with_disk_viscosity(rng, option):
    """Phase 7 disk viscosity on block steps with wake-ups (Option A: Navier-Stokes nu = 0.02;
    Option B: disc viscosity alpha_AV = 1): its heating is booked in heat_disk, so u_half - u_0 =
    visc + cond + disk + pdv - cool still holds per particle; and the viscosity changes the
    motion (compared with the same run without it)."""
    x, v, u, m, h = warm_ball(rng)
    offset = np.where(rng.random(len(u)) < 0.05, 5, 0)
    runs = {}
    for nu in (0.0, 0.02):
        cc = CoupledConfig()
        on = {"ns_nu": nu} if option == "A" else {"disc_alpha_av": 50.0 * nu}
        cc = replace(cc, hydro=replace(cc.hydro, **on))
        cfg = BlockConfig(coupled=cc, dt_max=64.0, r_acc=4.0)
        st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 2, level_offset=offset)
        u0 = st.u.copy()
        blocks.run(st, cfg, 64.0 * 2)
        runs[nu] = st
    st = runs[0.02]
    parts = st.heat_visc + st.heat_cond + st.heat_disk + st.heat_pdv - st.heat_cool
    print(f"   woken {st.woken}; median heat_disk {np.median(st.heat_disk):.2e}, heat_visc "
          f"{np.median(st.heat_visc):.2e}")  # fmt: skip
    check("wake-ups exercised", True, st.woken > 0)
    check("heat budget with disk viscosity (per particle, / max u0)", 0.0,
          np.max(np.abs(st.u_half - u0 - parts)) / float(np.max(u0)), atol=1e-12)  # fmt: skip
    check("disk viscosity heated every particle", True, bool(np.all(st.heat_disk > 0)))
    check("without it, heat_disk stays zero", True, bool(np.all(runs[0.0].heat_disk == 0)))
    dv = np.linalg.norm(st.p - runs[0.0].p, axis=1).max()
    check("disk viscosity changed the motion", True, bool(dv > 0))
