"""Over-the-pole camera paths (render/camera_path.py)."""

import numpy as np
import pytest

from nebula.render.camera_path import direction_curve, rotate_onto, transported_up, unit
from tests.helpers import check


def _loop(n=2000, tilt_deg=60.0):
    """A diagonal loop: a great circle through the poles' side, tilted tilt_deg from the
    vertical, sampled densely (camera directions from the hole), and its forward vectors."""
    th = np.linspace(0.0, 2.0 * np.pi * 0.9, n)
    tilt = np.radians(tilt_deg)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    pos = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b
    return pos, -pos  # looking at the hole


def test_keys_are_hit_and_the_curve_is_unit():
    key_t = [0.0, 1.0, 2.0, 3.0, 4.0]
    keys = unit(np.array([[1, 0, 0.2], [0.3, 0.6, 0.9], [-0.4, 0.2, 1.0], [-1, -0.3, 0.4],
                          [-0.5, -1, -0.2]]))  # fmt: skip
    d = direction_curve(key_t, keys, key_t)
    check("direction curve passes through the keys", keys, d, atol=1e-12)
    t = np.linspace(0, 4, 401)
    check("direction curve is unit length", 1.0, np.linalg.norm(direction_curve(key_t, keys, t),
          axis=1), atol=1e-12)  # fmt: skip
    with pytest.raises(ValueError):
        direction_curve([0, 1], [[1, 0, 0], [-1, 0.01, 0]], [0.5])


def test_up_over_the_pole_stays_smooth_and_perpendicular():
    """A loop passing straight over the pole (tilt 90): up stays perpendicular to the view,
    unit, finite, and turns by at most ~ the view's own turn per frame (no snap at the pole)."""
    pos, fwd = _loop(tilt_deg=90.0)
    up = transported_up(fwd, up0=(0.0, 0.0, 1.0))
    check("up is unit and finite", 1.0, np.linalg.norm(up, axis=1), atol=1e-12)
    check("up perpendicular to the view", 0.0, np.abs(np.sum(up * fwd, axis=1)).max(), atol=1e-12)
    turn_view = np.arccos(np.clip(np.sum(fwd[1:] * fwd[:-1], 1), -1, 1))
    turn_up = np.arccos(np.clip(np.sum(up[1:] * up[:-1], 1), -1, 1))
    check("up never turns more than the view per frame (no snap)", True,
          bool(np.all(turn_up <= turn_view + 1e-9)))  # fmt: skip


def test_great_circle_transport_keeps_the_roll():
    """Along a great circle, parallel transport is a pure rotation about the circle's axis:
    the up vector keeps its angle to that axis (no roll drift)."""
    pos, fwd = _loop(tilt_deg=60.0)
    axis = unit(np.cross(pos[0], pos[10]))
    up = transported_up(fwd, up0=(0.0, 0.0, 1.0))
    check("angle between up and the loop's axis constant", 0.0,
          float(np.ptp(up @ axis)), atol=1e-9)  # fmt: skip


def test_rotate_onto():
    a, b = unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])
    check("rotate a onto b takes a to b", b, rotate_onto(a, b, a), atol=1e-15)
    check("the rotation axis is left alone", [0, 0, 1.0], rotate_onto(a, b, [0, 0, 1.0]),
          atol=1e-15)  # fmt: skip
