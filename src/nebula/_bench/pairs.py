"""Stand-in for an SPH neighbour sum, used to check Numba and calibrate speed (Phase 0, Step 0.3).

For N points x_i, computes S_i = sum over j with |x_i - x_j| < R_CUT of exp(-|x_i - x_j|^2)
(j = i included, contributing 1). Two implementations that must agree:

- ``pair_sum_reference``: plain NumPy, all pairs, obviously correct, O(N^2).
- ``pair_sum_numba``: cell grid + parallel Numba loop, O(N * neighbours).

The parallel loop follows the rule every SPH kernel will follow: thread i writes only to
out[i], never to j, so there are no race conditions.
"""

import numba
import numpy as np

R_CUT = 2.0


def pair_sum_reference(x: np.ndarray, chunk: int = 512) -> np.ndarray:
    """All-pairs reference. x: (N, 3) float64. Chunked to keep memory at chunk * N."""
    n = len(x)
    out = np.empty(n)
    for start in range(0, n, chunk):
        xi = x[start : start + chunk]
        d2 = np.sum((xi[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        out[start : start + chunk] = np.sum(np.where(d2 < R_CUT**2, np.exp(-d2), 0.0), axis=1)
    return out


@numba.njit(cache=True)
def build_cells(x: np.ndarray, cell_size: float):
    """Counting sort of points into a cubic grid of cells (serial; cheap next to the sum).

    Returns the grid origin, cells per side, point order sorted by cell, and the start
    offset of each cell in that order (length n_cells + 1).
    """
    n = x.shape[0]
    lo = np.empty(3)
    hi = np.empty(3)
    for d in range(3):
        lo[d] = x[:, d].min()
        hi[d] = x[:, d].max()
    nc = np.empty(3, dtype=np.int64)
    for d in range(3):
        nc[d] = max(1, int((hi[d] - lo[d]) / cell_size) + 1)
    cell_of = np.empty(n, dtype=np.int64)
    for i in range(n):
        cx = int((x[i, 0] - lo[0]) / cell_size)
        cy = int((x[i, 1] - lo[1]) / cell_size)
        cz = int((x[i, 2] - lo[2]) / cell_size)
        cell_of[i] = (cx * nc[1] + cy) * nc[2] + cz
    n_cells = nc[0] * nc[1] * nc[2]
    start = np.zeros(n_cells + 1, dtype=np.int64)
    for i in range(n):
        start[cell_of[i] + 1] += 1
    for c in range(n_cells):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = cell_of[i]
        order[fill[c]] = i
        fill[c] += 1
    return lo, nc, order, start


@numba.njit(parallel=True, cache=True, fastmath=False)
def _pair_sum_cells(x, lo, nc, order, start, cell_size, r_cut):
    n = x.shape[0]
    out = np.zeros(n)
    r2 = r_cut * r_cut
    for i in numba.prange(n):
        cx = int((x[i, 0] - lo[0]) / cell_size)
        cy = int((x[i, 1] - lo[1]) / cell_size)
        cz = int((x[i, 2] - lo[2]) / cell_size)
        s = 0.0
        for ax in range(max(cx - 1, 0), min(cx + 2, nc[0])):
            for ay in range(max(cy - 1, 0), min(cy + 2, nc[1])):
                for az in range(max(cz - 1, 0), min(cz + 2, nc[2])):
                    c = (ax * nc[1] + ay) * nc[2] + az
                    for k in range(start[c], start[c + 1]):
                        j = order[k]
                        dx = x[i, 0] - x[j, 0]
                        dy = x[i, 1] - x[j, 1]
                        dz = x[i, 2] - x[j, 2]
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 < r2:
                            s += np.exp(-d2)
        out[i] = s  # write only to i
    return out


def pair_sum_numba(x: np.ndarray) -> np.ndarray:
    """Fast version of ``pair_sum_reference``. Cell size = R_CUT, so 27 cells cover the sphere."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    lo, nc, order, start = build_cells(x, R_CUT)
    return _pair_sum_cells(x, lo, nc, order, start, R_CUT, R_CUT)


def uniform_cube(n: int, neighbours: float, rng: np.random.Generator) -> np.ndarray:
    """n random points in a cube sized so each has about `neighbours` points within R_CUT."""
    density = neighbours / (4.0 / 3.0 * np.pi * R_CUT**3)
    side = (n / density) ** (1.0 / 3.0)
    return rng.uniform(0.0, side, size=(n, 3))
