"""Locally isothermal sound speed (physics/eos.py; phase-07 7.4 ring test) and its use in the
hydro EOS dispatch (sim/hydro.py eos_state)."""

import numpy as np

from nebula.physics.eos import isothermal_pressure, locally_isothermal_cs
from nebula.sim.hydro import HydroConfig, eos_state
from tests.helpers import check


def test_locally_isothermal_hand_values():
    """c_s0 = 0.01414 at r0 = 50; at r = 800 (16 r0): c_s = c_s0 16^(-3/4) = c_s0 / 8."""
    check("c_s at r0", 0.01414, float(locally_isothermal_cs(50.0, 0.01414, 50.0)), rtol=1e-15)
    cs = float(locally_isothermal_cs(800.0, 0.01414, 50.0))
    check("c_s at 16 r0 = c_s0 / 8", 0.01414 / 8, cs, rtol=1e-14)


def test_nu_constant_in_r_for_q_three_quarters():
    """c_s^2 / Omega_K = c_s^2 r^(3/2) is the same at every radius (so nu = alpha c_s^2/Omega_K
    is constant: the Lynden-Bell & Pringle case)."""
    r = np.array([10.0, 50.0, 200.0, 3000.0])
    cs = locally_isothermal_cs(r, 0.02, 50.0)
    want = np.full(4, 0.02**2 * 50.0**1.5)
    check("c_s^2 r^(3/2) constant", want, cs**2 * r**1.5, rtol=1e-13)


def test_eos_state_dispatch():
    rho = np.array([1.0, 2.0, 3.0])
    x = np.array([[50.0, 0.0, 0.0], [0.0, 800.0, 0.0], [0.0, 0.0, 12.5]])
    cfg = HydroConfig(isothermal_cs=0.02, iso_r0=50.0)
    p, cs = eos_state(cfg, rho, np.ones(3), x)
    want_cs = np.array([0.02, 0.02 / 8, 0.02 * 4**0.75])
    check("eos_state: local c_s from |x|", want_cs, cs, rtol=1e-14)
    check("eos_state: P = c_s^2 rho", want_cs**2 * rho, p, rtol=1e-14)
    p0, cs0 = eos_state(HydroConfig(isothermal_cs=0.02), rho, np.ones(3))
    check("plain isothermal unchanged", (isothermal_pressure(rho, 0.02), np.full(3, 0.02)),
          (p0, cs0), rtol=0.0)  # fmt: skip
