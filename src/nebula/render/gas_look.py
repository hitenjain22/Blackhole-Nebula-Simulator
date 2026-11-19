"""Smoother-looking gas for the final render (Hiten 2026-11-19: "a flowing smooth stream"):

- ``h_scale``: each particle drawn with its smoothing length times this factor (a standard
  trick in SPH visualisation: overlapping kernels hide the particle graininess; mass and
  hence total brightness of thin gas are unchanged, the light is spread wider);
- motion blur: each particle drawn as ``n`` copies of mass m / n spread evenly along its
  path over the shutter time (a camera shutter open for ``shutter`` of a frame interval), so
  gas reads as streaks along the flow. The path is the straight line x + v t, so each streak
  is capped at ``max_frac`` of the particle's distance from the hole (an orbit curves; a long
  straight streak would cut across it).
"""

import numpy as np

from nebula.physics.gravity import gn_v_from_p


def motion_blur_copies(x, p, dt: float, n: int, max_frac: float = 0.2):
    """(n N, 3) positions: for each particle, n points at times (j + 1/2)/n - 1/2 of dt around
    its position along v = gn_v_from_p(x, p) (code units), copy j of particle i at row j N + i.
    Streak length |v| dt capped at max_frac r. Their mean is x exactly."""
    x = np.asarray(x, float)
    v = gn_v_from_p(x, np.asarray(p, float))
    r = np.linalg.norm(x, axis=1)
    length = np.linalg.norm(v, axis=1) * abs(dt)
    scale = np.where(length > max_frac * r, max_frac * r / np.maximum(length, 1e-300), 1.0)
    step = v * (dt * scale)[:, None]
    frac = (np.arange(n) + 0.5) / n - 0.5
    return (x[None, :, :] + frac[:, None, None] * step[None, :, :]).reshape(n * len(x), 3)
