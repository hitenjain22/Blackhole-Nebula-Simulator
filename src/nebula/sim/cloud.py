"""The gas cloud's initial conditions (phases/phase-05-cloud-infall.md 5.1; a minimal version is
used by Phase 4's speed-up test).

A uniform sphere carved from the relaxed glass (random points would give density noise and
spurious pressure kicks), radius 100, total mass 1, temperature 10^4 K (ideal gas), every
particle moving with the cloud centre's apocentre velocity on the r_p = 25, r_a = 5,000 orbit.
"""

from dataclasses import dataclass

import numpy as np

from nebula.physics.kernels import DEFAULT_ETA
from nebula.physics.orbits import apocentre_state
from nebula.sim.glass import load_glass, tile_glass
from nebula.units import gas_u_from_temperature

CLOUD_RADIUS = 100.0
CLOUD_MASS = 1.0
CLOUD_TEMPERATURE = 1e4  # K
ORBIT_PERI = 25.0
ORBIT_APO = 5000.0


@dataclass
class CloudIC:
    x: np.ndarray
    v: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    h: np.ndarray  # starting guess, eta (m / rho)^(1/3)


def glass_ball(n_target: int, radius: float) -> np.ndarray:
    """About n_target points of the stored glass inside a ball of the given radius, centred
    on the origin (the glass is tiled first when one copy has too few points)."""
    glass = load_glass(4096)
    fraction = np.pi / 6  # ball volume / cube volume for a ball touching the cube's faces
    reps = max(1, int(np.ceil((n_target / (4096 * fraction)) ** (1 / 3))))
    pts = tile_glass(glass, (reps, reps, reps), (1.0, 1.0, 1.0)) - 0.5
    r_unit = (n_target / (4096 * reps**3) / (4 / 3 * np.pi)) ** (1 / 3)
    return pts[np.linalg.norm(pts, axis=1) < r_unit] * (radius / r_unit)


def cloud_ic(n_target: int = 1000, radius: float = CLOUD_RADIUS) -> CloudIC:
    x0, v0, _ = apocentre_state(ORBIT_PERI, ORBIT_APO)
    pos = glass_ball(n_target, radius)
    n = len(pos)
    rho = CLOUD_MASS / (4 / 3 * np.pi * radius**3)
    mass = np.full(n, CLOUD_MASS / n)
    return CloudIC(
        x=pos + x0,
        v=np.tile(v0, (n, 1)),
        u=np.full(n, gas_u_from_temperature(CLOUD_TEMPERATURE)),
        mass=mass,
        h=DEFAULT_ETA * (mass / rho) ** (1 / 3),
    )
