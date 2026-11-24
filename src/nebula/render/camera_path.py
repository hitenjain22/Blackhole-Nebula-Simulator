"""Camera paths that can pass over the hole's poles (Hiten 2026-11-23: a diagonal loop around
and over the hole, so the disc's lensed halo opens into a ring from above and the disc seems to
flip). Pure geometry, no rendering.

Today's [[camera]] keys use elevation and azimuth with |elev| <= 80 and a world-z "up": at a
pole the azimuth is undefined and the picture would spin. Here a path is given by directions
(unit vectors from the target to the camera) at keyframe times instead:

- ``direction_curve``: the directions at any times, from a monotone cubic (PCHIP) through each
  component of the keyframe vectors, normalised. Smooth (continuous slope) and passes through
  the keys exactly. Consecutive keys must be < 120 degrees apart (otherwise the interpolated
  vector can pass near zero).
- ``transported_up``: the camera's up vector carried along the path by parallel transport:
  each frame's up is the previous one turned by the rotation that takes the previous view
  direction to the new one (Rodrigues' formula), then made exactly perpendicular. The picture
  never snaps, also straight above the hole; the horizon rolls smoothly as the camera goes
  over, which is what makes the disc seem to flip. Over a closed loop that is not a great
  circle the up vector comes back turned (holonomy); paths here are open shots.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

MAX_KEY_ANGLE_DEG = 120.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def direction_curve(key_t, key_dirs, t) -> np.ndarray:
    """Unit directions (len(t), 3) through the keyframe directions (normalised first)."""
    key_t = np.asarray(key_t, float)
    d = unit(np.asarray(key_dirs, float))
    if len(key_t) != len(d) or len(d) < 2:
        raise ValueError("need matching times and at least two direction keys")
    ang = np.degrees(np.arccos(np.clip(np.sum(d[1:] * d[:-1], axis=1), -1.0, 1.0)))
    if np.any(ang >= MAX_KEY_ANGLE_DEG):
        k = int(np.argmax(ang))
        raise ValueError(f"direction keys {k} and {k + 1} are {ang[k]:.0f} deg apart "
                         f"(< {MAX_KEY_ANGLE_DEG:.0f}: add a key between them)")  # fmt: skip
    return unit(PchipInterpolator(key_t, d, axis=0, extrapolate=False)(np.asarray(t, float)))


def rotate_onto(a, b, v):
    """v turned by the smallest rotation taking unit vector a onto unit vector b (Rodrigues);
    a == b leaves v unchanged; a == -b is undefined (raises)."""
    a, b, v = (np.asarray(x, float) for x in (a, b, v))
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return v.copy()
        raise ValueError("view direction reversed in one frame: path too fast")
    k = axis / s
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1.0 - c)


def transported_up(forward, up0=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Up vectors (n, 3), perpendicular to each forward (n, 3) view direction, carried along
    by parallel transport from up0 (projected off the first forward direction)."""
    f = unit(np.asarray(forward, float))
    up = np.asarray(up0, float) - (np.asarray(up0, float) @ f[0]) * f[0]
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("up0 is parallel to the first view direction")
    out = np.empty_like(f)
    out[0] = up / np.linalg.norm(up)
    for k in range(1, len(f)):
        u = rotate_onto(f[k - 1], f[k], out[k - 1])
        u = u - (u @ f[k]) * f[k]
        out[k] = u / np.linalg.norm(u)
    return out
