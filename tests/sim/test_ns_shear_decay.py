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

from nebula.sim import hydro
from nebula.sim.glass import load_glass, tile_glass
from nebula.sim.hydro import HydroConfig
from tests.helpers import check

K = 2.0 * np.pi
NU = 0.01


def _decay_rate(pos, nu, t_end=1.0):
    n = len(pos)
    v = np.zeros((n, 3))
    v[:, 0] = 0.01 * np.sin(K * pos[:, 1])
    cfg = HydroConfig(box=(1.0, 1.0, 1.0), isothermal_cs=1.0, viscosity=False, conductivity=False,
                      ns_nu=nu)  # fmt: skip
    st = hydro.initialise(pos, v, np.ones(n), np.full(n, 1.0 / n),
                          np.full(n, 1.6 * (1.0 / n) ** (1 / 3)), cfg)  # fmt: skip

    def amp(s):
        return 2.0 * np.mean(s.v[:, 0] * np.sin(K * s.x[:, 1]))

    ts, a = [0.0], [amp(st)]
    while st.t < t_end - 1e-12:
        st = hydro.step(st, cfg, min(hydro.stable_dt(st, cfg), t_end - st.t))
        ts.append(st.t)
        a.append(amp(st))
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
