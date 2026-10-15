"""Independent reference integrator for tests and plots (phases/phase-01-orbits.md Step 1.4).

SciPy DOP853 (8th-order Runge-Kutta, tight tolerances) on the Lagrangian-form GN acceleration
``gn_accel`` in (x, v): a different equation form and a different method from the
generalized leapfrog, so agreement between the two is meaningful. Too slow for gas; never
used by the simulator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from nebula.physics.gravity import gn_accel


@dataclass
class ReferenceOrbit:
    """Dense solution plus pericentre and apocentre events (times and states (x, v))."""

    sol: object
    t_peri: np.ndarray
    state_peri: np.ndarray
    t_apo: np.ndarray
    state_apo: np.ndarray


def dop853_orbit(
    x0: np.ndarray, v0: np.ndarray, t_end: float, rtol: float = 1e-12, atol: float = 1e-14
) -> ReferenceOrbit:
    """Integrate x'' = gn_accel(x, x') from (x0, v0) to t_end. Pericentres are where
    x.v crosses zero from below, apocentres from above."""

    def rhs(_t, s):
        return np.concatenate([s[3:], gn_accel(s[:3], s[3:])])

    def radial(_t, s):
        return s[:3] @ s[3:]

    def peri(t, s):
        return radial(t, s)

    def apo(t, s):
        return radial(t, s)

    peri.direction = 1.0
    apo.direction = -1.0
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        np.concatenate([x0, v0]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=[peri, apo],
        dense_output=True,
    )
    if sol.status != 0:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    return ReferenceOrbit(sol, sol.t_events[0], sol.y_events[0], sol.t_events[1], sol.y_events[1])
