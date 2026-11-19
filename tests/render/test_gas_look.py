"""Smoother gas for the final render (render/gas_look.py; DECISIONS 2026-11-19)."""

import numpy as np

from nebula.physics.gravity import gn_v_from_p
from nebula.render.gas_look import motion_blur_copies
from tests.helpers import check


def test_motion_blur_streak():
    """n copies spread evenly over the path |v| dt, centred on the particle; the streak is
    capped at 0.2 r where the straight line would cut across a curved orbit."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)) * 100.0 + 300.0
    p = rng.normal(size=(50, 3)) * 0.02
    n, dt = 6, 40.0
    c = motion_blur_copies(x, p, dt, n).reshape(n, 50, 3)
    check("blur: copies centred on the particle", x, c.mean(axis=0), atol=1e-9)
    v = gn_v_from_p(x, p)
    span = np.linalg.norm(c[-1] - c[0], axis=1)
    check("blur: streak = |v| dt (n - 1) / n", np.linalg.norm(v, axis=1) * dt * (n - 1) / n,
          span, rtol=1e-12)  # fmt: skip
    far = motion_blur_copies(x, p, 1e6, n).reshape(n, 50, 3)
    r = np.linalg.norm(x, axis=1)
    check("blur: long streaks capped at 0.2 r", 0.2 * r * (n - 1) / n,
          np.linalg.norm(far[-1] - far[0], axis=1), rtol=1e-12)  # fmt: skip
    check("blur: one copy = no blur", x, motion_blur_copies(x, p, dt, 1), atol=0.0)
