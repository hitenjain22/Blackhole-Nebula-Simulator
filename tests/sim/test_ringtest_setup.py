"""Ring-spreading test initial conditions (sim/ringtest.py; phase-07 7.4): the plan's ring
(r0 = 50, Gaussian width 5, H/R = 0.1 at r0) as measured from the particles."""

import numpy as np

from nebula.analysis.disk_profile import disk_profile
from nebula.physics.thin_disk import vphi_pressure_supported
from nebula.sim.ringtest import ring_setup
from tests.helpers import check


def test_ring_matches_the_plan():
    ring = ring_setup(20_000)
    rr = np.hypot(ring.x[:, 0], ring.x[:, 1])
    check("c_s at r0 = 0.1 / sqrt(50)", 0.1 / np.sqrt(50.0), ring.c_s0, rtol=1e-15)
    check("total mass", 1.0, float(ring.mass.sum()), rtol=1e-12)
    # mass-weighted radius of Sigma R exp(-(R-r0)^2/2w^2): r0 + w^2/r0 = 50.5 (to O(w^4))
    check("mean radius (r0 + w^2/r0)", 50.5, float(np.mean(rr)), rtol=2e-3)
    check("radial spread (width 5)", 5.0, float(np.std(rr)), rtol=0.02)
    cs = ring.c_s0 * (rr / 50.0) ** -0.75
    prof = disk_profile(ring.x, ring.v, ring.mass, cs, np.array([48.0, 52.0]))
    print(f"   H/R in 48-52: {prof.height[0] / prof.centres[0]:.4f}")
    check("H/R at r0 (rms z in 48 < R < 52)", 0.1, float(prof.height[0] / prof.centres[0]),
          rtol=0.03)  # fmt: skip
    # v^2 / v_K^2 = 1 + (H/R)^2 [dlnP/dlnR - 3/2 z^2/H^2] (radial pressure gradient; weaker
    # radial gravity at height z): near r0, dlnP/dlnR ~ -2.25 and <z^2/H^2> = 1, so the core mean
    # is ~ 1 - 0.01 x 3.75 = 0.9625 (ignoring the Sigma slope and H/R varying across the core,
    # ~0.5% effects; measured 0.9587). The dynamic check is the ring holding its shape (7.4).
    v2 = (ring.v[:, 0] ** 2 + ring.v[:, 1] ** 2) * rr
    core = np.abs(rr - 50.0) < 5.0
    check("core mean v_phi^2 / v_K^2 (estimate 1 - 3.75 (H/R)^2)", 0.9625, float(v2[core].mean()),
          rtol=0.01)  # fmt: skip


def test_pressure_supported_speed_hand_value():
    """R = 50, c_s = 0.01, z = 0, flat Sigma (dlnSigma = 0), q = 3/4: dlnP/dlnR = -1.5 - 0.75
    = -2.25, v^2 = 1/50 - 2.25e-4 = 0.019775."""
    check("v_phi", np.sqrt(0.019775), float(vphi_pressure_supported(50.0, 0.01, 0.0)), rtol=1e-14)
