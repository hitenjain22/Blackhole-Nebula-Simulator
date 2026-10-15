"""Neighbour lists for SPH (phases/phase-03-sph-core.md 3.1 "Neighbour search").

Particle i interacts with every j (including itself) with |r_ij| < 2 max(h_i, h_j): the
density sum needs r < 2 h_i ("gather") and the symmetric force sum also needs the j whose
own kernel reaches i ("scatter"). Lists are therefore symmetric: j in N(i) <=> i in N(j);
an asymmetric list breaks momentum conservation (phase-03 3.3).

Built with scipy.spatial.cKDTree: gather sets {j : r_ij < 2 h_i} as a sparse matrix G, then
N = G OR G^T. Periodic boxes (tests) via cKDTree's ``boxsize``; separations then use the
minimum image (``minimum_image``). Stored as CSR arrays for the Numba SPH loops.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

R_KERN = 2.0  # kernel support in units of h (physics.kernels.R_KERN)


@dataclass
class Neighbours:
    """CSR neighbour lists: the neighbours of i are index[offset[i]:offset[i + 1]] (sorted,
    including i itself). ``box``: periodic box lengths (3,) or None."""

    offset: np.ndarray  # (N + 1,) int64
    index: np.ndarray  # (M,) int64
    box: np.ndarray | None

    def of(self, i: int) -> np.ndarray:
        return self.index[self.offset[i] : self.offset[i + 1]]

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offset)


def minimum_image(dx: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Separation vectors folded into (-L/2, L/2] for a periodic box (unchanged if None)."""
    if box is None:
        return dx
    return dx - box * np.round(dx / box)


def wrap(pos: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Positions folded into [0, L) for a periodic box."""
    if box is None:
        return pos
    return np.mod(pos, box)


def find_neighbours(pos: np.ndarray, h: np.ndarray, box=None, workers: int = -1) -> Neighbours:
    """Symmetric neighbour lists for positions (N, 3) and smoothing lengths (N,)."""
    pos = np.asarray(pos, np.float64)
    h = np.asarray(h, np.float64)
    n = len(pos)
    box_arr = None if box is None else np.broadcast_to(np.asarray(box, np.float64), (3,)).copy()
    if box_arr is not None and np.any(R_KERN * h.max() >= 0.5 * box_arr):
        raise ValueError("kernel support exceeds half the periodic box: neighbours ambiguous")
    tree = cKDTree(wrap(pos, box_arr), boxsize=box_arr)
    lists = tree.query_ball_point(
        wrap(pos, box_arr), r=R_KERN * h, workers=workers, return_sorted=False
    )
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=n)
    cols = np.fromiter((j for x in lists for j in x), dtype=np.int64, count=int(lengths.sum()))
    rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    g = sp.csr_matrix((np.ones(len(cols), dtype=np.int8), (rows, cols)), shape=(n, n))
    sym = (g + g.T).tocsr()
    sym.sort_indices()
    return Neighbours(sym.indptr.astype(np.int64), sym.indices.astype(np.int64), box_arr)


def brute_force_neighbours(pos: np.ndarray, h: np.ndarray, box=None) -> list[np.ndarray]:
    """O(N^2) reference for tests: sorted {j : |r_ij| < 2 max(h_i, h_j)} for each i."""
    box_arr = None if box is None else np.broadcast_to(np.asarray(box, float), (3,))
    out = []
    for i in range(len(pos)):
        dx = minimum_image(pos - pos[i], box_arr)
        r = np.linalg.norm(dx, axis=1)
        out.append(np.flatnonzero(r < R_KERN * np.maximum(h[i], h)))
    return out
