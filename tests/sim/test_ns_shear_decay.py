"""Navier-Stokes viscosity in the shared-step hydro (sim/hydro.py, ns_nu): a shear wave
v_x = A sin(k y) in a periodic isothermal box decays as exp(-nu k^2 t) (phase-07 7.3, Option A).

The SPH run without viscosity also decays slowly (particle disorder in the glass; 0.075 per unit
time at 4,096 particles, 0.048 at 32,768), so the viscous rate is the measured rate minus that
baseline. Measured 2026-10-21: viscous part / (nu k^2) = 0.949 at 4,096 (k h = 0.63) and 0.987 at
32,768 (k h = 0.31); the error falls ~4x per halving of h, the second-order error of the
difference-operator gradient (Phantom Eq. 131). Tolerances follow from that: 8% at k h = 0.63,
2% at k h = 0.31.
"""

import numpy as np
import pytest

from nebula.sim import blocks, hydro
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.glass import load_glass, tile_glass
from nebula.sim.hydro import HydroConfig
from tests.helpers import check

K = 2.0 * np.pi
NU = 0.01


def _decay_rate(pos, nu, t_end=1.0, disc_alpha_av=0.0, mean_h=None):
    n = len(pos)
    v = np.zeros((n, 3))
    v[:, 0] = 0.01 * np.sin(K * pos[:, 1])
    cfg = HydroConfig(box=(1.0, 1.0, 1.0), isothermal_cs=1.0, viscosity=False, conductivity=False,
                      ns_nu=nu, disc_alpha_av=disc_alpha_av)  # fmt: skip
    st = hydro.initialise(pos, v, np.ones(n), np.full(n, 1.0 / n),
                          np.full(n, 1.6 * (1.0 / n) ** (1 / 3)), cfg)  # fmt: skip

    def amp(s):
        return 2.0 * np.mean(s.v[:, 0] * np.sin(K * s.x[:, 1]))

    ts, a, hs = [0.0], [amp(st)], [np.mean(st.h)]
    while st.t < t_end - 1e-12:
        st = hydro.step(st, cfg, min(hydro.stable_dt(st, cfg), t_end - st.t))
        ts.append(st.t)
        a.append(amp(st))
        hs.append(np.mean(st.h))
    if mean_h is not None:
        mean_h.append(float(np.mean(hs)))
    return -np.polyfit(ts, np.log(a), 1)[0]


def _viscous_ratio(pos):
    base = _decay_rate(pos, 0.0)
    visc = _decay_rate(pos, NU)
    ratio = (visc - base) / (NU * K**2)
    print(f"   N={len(pos)}: baseline {base:.4f}, with nu {visc:.4f}, viscous part / nu k^2 = "
          f"{ratio:.4f}")  # fmt: skip
    return ratio


def test_shear_wave_decays_at_nu_k2():
    ratio = _viscous_ratio(load_glass(4096))
    check("shear decay rate / nu k^2 (4,096 particles, k h = 0.63)", 1.0, ratio, rtol=0.08)


@pytest.mark.slow
def test_shear_wave_decay_converges():
    coarse = _viscous_ratio(load_glass(4096))
    fine = _viscous_ratio(tile_glass(load_glass(4096), (2, 2, 2), (1.0, 1.0, 1.0)))
    check("shear decay rate / nu k^2 (32,768 particles, k h = 0.31)", 1.0, fine, rtol=0.02)
    order = np.log2(abs(1.0 - coarse) / abs(1.0 - fine))
    check("convergence order of the decay rate (second order: 2)", 2.0, order, atol=0.5)


def _block_decay_rate(pos, nu, offsets, disc_alpha_av=0.0):
    n = len(pos)
    v = np.zeros((n, 3))
    v[:, 0] = 0.01 * np.sin(K * pos[:, 1])
    hc = HydroConfig(box=(1.0, 1.0, 1.0), isothermal_cs=1.0, viscosity=False, conductivity=False,
                     ns_nu=nu, disc_alpha_av=disc_alpha_av)  # fmt: skip
    cfg = BlockConfig(coupled=CoupledConfig(hydro=hc, hole=False), dt_max=1.0 / 16)
    st = blocks.initialise(pos, v, np.ones(n), np.full(n, 1.0 / n),
                           np.full(n, 1.6 * (1.0 / n) ** (1 / 3)), cfg, t_stop=1.0 / 16,
                           level_offset=offsets)  # fmt: skip
    ts, a = [0.0], [0.01]
    for j in range(1, 17):
        blocks.run(st, cfg, j / 16)
        ts.append(st.t)
        a.append(2.0 * np.mean(st.p[:, 0] * np.sin(K * st.x[:, 1])))
    return -np.polyfit(ts, np.log(a), 1)[0], st


@pytest.mark.slow
@pytest.mark.parametrize("option", ["A", "B"])
def test_shear_wave_decay_on_mixed_block_levels(rng, option):
    """Block steps with half the particles (random) forced 2 levels finer: the decay with the
    disk viscosity equals the one-level decay (Option A: each particle's viscous force needs its
    neighbours' stresses at the current time: sim/blocks.py). Option A measured 2026-10-21:
    0.4482 vs 0.4479, i.e. 0.07% of nu k^2; tolerance 1% (a time-integration difference, small
    next to the 5% spatial error at this resolution). Option B: alpha_AV = 1 (nu_AV ~ 0.01)."""
    pos = load_glass(4096)
    nu, a_av = (NU, 0.0) if option == "A" else (0.0, 1.0)
    one, _ = _block_decay_rate(pos, nu, np.zeros(len(pos), np.int64), a_av)
    offsets = np.where(rng.random(len(pos)) < 0.5, 2, 0)
    mixed, st = _block_decay_rate(pos, nu, offsets, a_av)
    print(f"   levels {np.bincount(st.level)}; decay one level {one:.5f}, mixed {mixed:.5f}")
    check("two levels in use", True, int(np.count_nonzero(np.bincount(st.level))) == 2)
    check("decay rate: mixed levels - one level (/ nu k^2)", 0.0, (mixed - one) / (NU * K**2),
          atol=0.01)  # fmt: skip


def test_disc_viscosity_shear_wave_decays_at_nu_av_k2():
    """Option B (disc viscosity through the AV form, alpha_AV = 1, c_s = 1): the decay rate
    (minus the no-viscosity baseline) against nu_AV k^2 with nu_AV = alpha_AV c_s <h> / 10
    (Phantom Eq. 121, a continuum-limit estimate). Measured 2026-10-21: 0.960 at 4,096 particles
    and 1.063 at 32,768. Unlike Option A this does not converge to 1 at a fixed neighbour number
    (the AV-type pair sum keeps an error set by the neighbour sampling), which is why the plan
    measures Option B's effective alpha in the ring test (7.5); tolerance 10% here."""
    pos = load_glass(4096)
    hm = []
    base = _decay_rate(pos, 0.0)
    rate = _decay_rate(pos, 0.0, disc_alpha_av=1.0, mean_h=hm)
    ratio = (rate - base) / (1.0 * 1.0 * hm[0] / 10.0 * K**2)
    print(f"   baseline {base:.4f}, alpha_AV = 1: {rate:.4f}, <h> {hm[0]:.4f}, ratio {ratio:.4f}")
    check("Option B shear decay / (alpha_AV c_s <h>/10 k^2) (4,096 particles)", 1.0, ratio,
          rtol=0.10)  # fmt: skip
