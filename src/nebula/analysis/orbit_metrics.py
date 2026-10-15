"""Measurements on recorded single-particle orbits (Phase 1): pericentres and precession."""

import numpy as np


def _vertex(t: np.ndarray, y: np.ndarray) -> float:
    """Time of the extremum of the parabola through three points (t, y)."""
    a, b, _ = np.polyfit(t - t[1], y, 2)
    return t[1] - b / (2.0 * a)


def pericentres(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pericentre passages of a recorded orbit, refined between steps.

    Finds the local minima of r, fits a parabola through the three samples around each minimum
    for its time, and evaluates a parabola through the (unwrapped) polar angle at that time.
    Returns (times, radii, polar angles in rad, unwrapped so successive passages differ by
    2 pi + precession). Orbits are assumed to lie in the x-y plane.
    """
    r = np.linalg.norm(x, axis=1)
    phi = np.unwrap(np.arctan2(x[:, 1], x[:, 0]))
    idx = np.where((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:]))[0] + 1
    times, radii, angles = [], [], []
    for i in idx:
        sl = slice(i - 1, i + 2)
        tp = _vertex(t[sl], r[sl])
        times.append(tp)
        radii.append(np.polyval(np.polyfit(t[sl] - tp, r[sl], 2), 0.0))
        angles.append(np.polyval(np.polyfit(t[sl] - tp, phi[sl], 2), 0.0))
    return np.array(times), np.array(radii), np.array(angles)
