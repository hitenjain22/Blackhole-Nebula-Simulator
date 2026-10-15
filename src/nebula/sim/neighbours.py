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


# --------------------------------------------------------------------------- Numba cell grid
#
# Fast twin of ``find_neighbours`` (Phase 3, Step 3.8; the cKDTree version above is its
# reference). Particles are counting-sorted into a cubic grid of cells; each particle gathers
# {j : r_ij < 2 h_i} from the cells its search sphere overlaps (periodic boxes wrap the cell
# indices); the symmetric lists are then N(i) = G(i) U {j : i in G(j)}, built by transposing
# the gather lists and merging sorted rows. Parallel loops write only to their own row.

import numba  # noqa: E402


@numba.njit(cache=True)
def _cell_of(x, lo, inv_cell, ncell, c):
    k = int(np.floor((x - lo[c]) * inv_cell[c]))
    if k < 0:
        k = 0
    if k >= ncell[c]:
        k = ncell[c] - 1
    return k


@numba.njit(cache=True)
def _build_grid(pos, lo, inv_cell, ncell):
    n = pos.shape[0]
    total = ncell[0] * ncell[1] * ncell[2]
    cell = np.empty(n, np.int64)
    for i in range(n):
        cx = _cell_of(pos[i, 0], lo, inv_cell, ncell, 0)
        cy = _cell_of(pos[i, 1], lo, inv_cell, ncell, 1)
        cz = _cell_of(pos[i, 2], lo, inv_cell, ncell, 2)
        cell[i] = (cx * ncell[1] + cy) * ncell[2] + cz
    start = np.zeros(total + 1, np.int64)
    for i in range(n):
        start[cell[i] + 1] += 1
    for c in range(total):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        order[fill[cell[i]]] = i
        fill[cell[i]] += 1
    return start, order


@numba.njit(cache=True)
def _cell_range(xc, r, lo, inv_cell, ncell, c, periodic):
    """(first cell index, number of cells, wraps-whole-axis flag) covering [xc - r, xc + r]
    along axis c. If a periodic search spans the whole axis, every cell is visited once and
    the per-cell image shift is ambiguous: the caller then uses the minimum image per pair."""
    a = int(np.floor((xc - r - lo[c]) * inv_cell[c]))
    b = int(np.floor((xc + r - lo[c]) * inv_cell[c]))
    if periodic:
        count = b - a + 1
        if count >= ncell[c]:
            return a, ncell[c], True
        return a, count, False
    if a < 0:
        a = 0
    if b > ncell[c] - 1:
        b = ncell[c] - 1
    return a, max(b - a + 1, 0), False


@numba.njit(cache=True)
def _image_delta(d, box_c, raw, cell, ncell_c, whole):
    """Separation component with the right periodic image: per-cell shift normally,
    minimum image when the search wraps the whole axis (box_c = 0: not periodic)."""
    if box_c <= 0.0:
        return d
    if whole:
        return d - box_c * np.round(d / box_c)
    return d + box_c * ((raw - cell) // ncell_c)


@numba.njit(parallel=True, cache=True)
def _gather(pos, h, box, lo, inv_cell, ncell, start, order, counts, out_off, out_idx, fill_pass):
    """Gather {j : r_ij < 2 h_i} per particle; periodic images handled per cell."""
    n = pos.shape[0]
    for i in numba.prange(n):
        r = R_KERN * h[i]
        r2 = r * r
        m = 0
        ax, nx, wx = _cell_range(pos[i, 0], r, lo, inv_cell, ncell, 0, box[0] > 0)
        ay, ny, wy = _cell_range(pos[i, 1], r, lo, inv_cell, ncell, 1, box[1] > 0)
        az, nz, wz = _cell_range(pos[i, 2], r, lo, inv_cell, ncell, 2, box[2] > 0)
        for di in range(nx):
            rx = ax + di
            cx = rx % ncell[0]
            for dj in range(ny):
                ry = ay + dj
                cy = ry % ncell[1]
                for dk in range(nz):
                    rz = az + dk
                    cz = rz % ncell[2]
                    c = (cx * ncell[1] + cy) * ncell[2] + cz
                    for k in range(start[c], start[c + 1]):
                        j = order[k]
                        dx = _image_delta(pos[j, 0] - pos[i, 0], box[0], rx, cx, ncell[0], wx)
                        dy = _image_delta(pos[j, 1] - pos[i, 1], box[1], ry, cy, ncell[1], wy)
                        dz = _image_delta(pos[j, 2] - pos[i, 2], box[2], rz, cz, ncell[2], wz)
                        if dx * dx + dy * dy + dz * dz < r2:
                            if fill_pass:
                                out_idx[out_off[i] + m] = j
                            m += 1
        if not fill_pass:
            counts[i] = m


@numba.njit(cache=True)
def _transpose(off, idx, n):
    t_count = np.zeros(n + 1, np.int64)
    for k in range(idx.shape[0]):
        t_count[idx[k] + 1] += 1
    for i in range(n):
        t_count[i + 1] += t_count[i]
    t_idx = np.empty(idx.shape[0], np.int64)
    fill = t_count[:-1].copy()
    for i in range(n):
        for k in range(off[i], off[i + 1]):
            j = idx[k]
            t_idx[fill[j]] = i
            fill[j] += 1
    return t_count, t_idx


@numba.njit(parallel=True, cache=True)
def _merge_rows(off_a, idx_a, off_b, idx_b, out_off, out_idx, counts, fill_pass):
    """Union of two sorted CSR row sets (duplicates removed)."""
    for i in numba.prange(off_a.shape[0] - 1):
        p, pe = off_a[i], off_a[i + 1]
        q, qe = off_b[i], off_b[i + 1]
        m = 0
        last = -1
        while p < pe or q < qe:
            if q >= qe or (p < pe and idx_a[p] <= idx_b[q]):
                v = idx_a[p]
                p += 1
            else:
                v = idx_b[q]
                q += 1
            if v != last:
                if fill_pass:
                    out_idx[out_off[i] + m] = v
                m += 1
                last = v
        if not fill_pass:
            counts[i] = m


@numba.njit(parallel=True, cache=True)
def _sort_rows(off, idx):
    for i in numba.prange(off.shape[0] - 1):
        idx[off[i] : off[i + 1]] = np.sort(idx[off[i] : off[i + 1]])


def find_neighbours_numba(pos: np.ndarray, h: np.ndarray, box=None) -> Neighbours:
    """Fast twin of ``find_neighbours`` (identical lists; tested against it)."""
    pos = np.ascontiguousarray(pos, np.float64)
    h = np.ascontiguousarray(h, np.float64)
    n = len(pos)
    box_arr = None if box is None else np.broadcast_to(np.asarray(box, np.float64), (3,)).copy()
    if box_arr is not None and np.any(R_KERN * h.max() >= 0.5 * box_arr):
        raise ValueError("kernel support exceeds half the periodic box: neighbours ambiguous")
    boxv = np.zeros(3) if box_arr is None else box_arr
    wrapped = wrap(pos, box_arr)
    if box_arr is None:
        lo = wrapped.min(axis=0)
        span = wrapped.max(axis=0) - lo
    else:
        lo = np.zeros(3)
        span = box_arr
    # Cell size: half the median search radius (most particles look into ~5^3 small cells,
    # which wastes less volume than 3^3 large ones); particles with larger h look into more
    # cells. At most 160^3 cells.
    cell = max(0.5 * R_KERN * float(np.median(h)), float(np.max(span)) / 160.0, 1e-300)
    if box_arr is not None:  # cells must tile each periodic side exactly: per-axis size
        ncell = np.maximum(np.floor(box_arr / cell).astype(np.int64), 1)
        inv_cell = ncell / box_arr
    else:
        ncell = np.maximum(np.floor(span / cell).astype(np.int64) + 1, 1)
        inv_cell = np.full(3, 1.0 / cell)
    start, order = _build_grid(wrapped, lo, inv_cell, ncell)
    counts = np.empty(n, np.int64)
    dummy = np.empty(0, np.int64)
    _gather(wrapped, h, boxv, lo, inv_cell, ncell, start, order, counts, dummy, dummy, False)
    g_off = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=g_off[1:])
    g_idx = np.empty(g_off[-1], np.int64)
    _gather(wrapped, h, boxv, lo, inv_cell, ncell, start, order, counts, g_off, g_idx, True)
    _sort_rows(g_off, g_idx)
    t_off, t_idx = _transpose(g_off, g_idx, n)  # rows of the transpose come out sorted
    _merge_rows(g_off, g_idx, t_off, t_idx, dummy, dummy, counts, False)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    idx = np.empty(off[-1], np.int64)
    _merge_rows(g_off, g_idx, t_off, t_idx, off, idx, counts, True)
    return Neighbours(off, idx, box_arr)


@numba.njit(parallel=True, cache=True)
def _tighten(pos, h, box, offset, index, rows, out_off, out_idx, counts, fill_pass):
    for q in numba.prange(rows.shape[0]):
        i = rows[q]
        m = 0
        for k in range(offset[i], offset[i + 1]):
            j = index[k]
            d2 = 0.0
            for c in range(3):
                dx = pos[j, c] - pos[i, c]
                if box[c] > 0.0:
                    dx -= box[c] * np.round(dx / box[c])
                d2 += dx * dx
            rr = R_KERN * max(h[i], h[j])
            if d2 < rr * rr:
                if fill_pass:
                    out_idx[out_off[i] + m] = j
                m += 1
        if not fill_pass:
            counts[i] = m


def tighten(nb: Neighbours, pos: np.ndarray, h: np.ndarray, rows=None) -> Neighbours:
    """Keep only the pairs that interact at the current smoothing lengths, r < 2 max(h_i, h_j)
    (still symmetric and sorted). Lists built with a search margin, or reused between steps,
    carry extra entries; the force and gradient passes only need these. ``rows`` (block time
    steps): lists only for these particles, the others' left empty."""
    pos = np.ascontiguousarray(pos, np.float64)
    h = np.ascontiguousarray(h, np.float64)
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, np.float64)
    n = len(pos)
    rows = np.arange(n, dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    counts = np.zeros(n, np.int64)
    dummy = np.empty(0, np.int64)
    _tighten(pos, h, box, nb.offset, nb.index, rows, dummy, dummy, counts, False)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    idx = np.empty(off[-1], np.int64)
    _tighten(pos, h, box, nb.offset, nb.index, rows, off, idx, counts, True)
    return Neighbours(off, idx, nb.box)
