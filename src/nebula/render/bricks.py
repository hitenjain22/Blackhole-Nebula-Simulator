"""Sparse brick grid for thin gas (DECISIONS 2026-11-16 "sparse brick grid"; extends
phases/phase-08-gas-rendering.md 8.3).

The log-spherical grid's cells grow with r (r dln r), so a stream far from the hole sits in
cells several times its smoothing length and renders as blocks. Here gas is deposited on a
fine uniform lattice instead, stored only where there is gas:

- Lattice: cubic cells of size dx with centres at (i + 1/2) dx (i any integer), fixed in space
  (the same for every frame, so nothing shimmers as the gas moves). Cells are grouped in
  bricks of BRICK^3 cells; brick (I, J, K) holds cells BRICK I .. BRICK I + BRICK - 1 on each
  axis. Only bricks that hold gas are stored.
- Which particles go here ("fine" particles): kernel support 2h no wider than
  MAX_SUPPORT_BRICKS bricks, and r_in <= r <= r_fine, where r_in is the radius inside which the
  log grid's cells are already finer than dx. Every other particle goes to the log grid. Each
  particle is deposited exactly once, so the two grids' fields simply add (rho and
  rho (T/T_REF)^4 are additive): no blending, no seams, mass conserved by construction.
- Deposition rule: exactly gasgrid's (kernel weights f(q) over the cell centres inside the
  support, normalised to 1; trilinear fallback when fewer than MIN_KERNEL_CELLS centres are
  inside). A brick grid is therefore the same as deposit_box on any aligned box holding the
  particles (tests/render/test_bricks.py checks this).
- Allocation: bricks are allocated over each fine particle's support grown by one cell on
  every side. Sampling is trilinear between cell centres; a sample at p reads cells
  floor(f) and floor(f) + 1 with f = p / dx - 1/2 ("home cell" floor(f)). A non-zero sample
  needs a gas cell among those corners, and the home cell is at most one cell below it, so
  the home cell's brick is always allocated where the sample is non-zero. The marcher uses
  that: in an allocated home brick the step is <= STEP_FRACTION dx; in an empty one the step
  ends just past the home brick's far face, so no brick is ever jumped over.

``deposit_bricks_reference`` and ``sample_bricks`` are the readable versions (CLAUDE.md rule
2); ``deposit_bricks`` (Numba) must match the reference, and the WGSL sampler must match
``sample_bricks``.
"""

import math
from bisect import bisect_left
from dataclasses import dataclass

import numba
import numpy as np

from nebula.physics.kernels import get_kernel
from nebula.render.gasgrid import MIN_KERNEL_CELLS, N_CHUNKS, R_SUPPORT, LogSphericalGrid
from nebula.sim.kernels_nb import KERNEL_IDS, kernel_f

BRICK = 8  # cells per brick side
MAX_SUPPORT_BRICKS = 4  # particles with a wider support (2h) stay on the log grid
KEY_OFFSET = 1 << 20  # brick coordinates are packed in 21 bits each (|I| < 2^20)
KEY_BITS = 21
EXIT_EPS = 1e-4  # in cells: a step out of an empty brick ends this far past its face


@dataclass(frozen=True)
class BrickLattice:
    """Settings: cell size dx [code lengths] and the outer radius of the fine region."""

    dx: float = 1.0
    r_fine: float = 1000.0

    def r_in(self, grid: LogSphericalGrid) -> float:
        """Radius inside which the log grid's cells (r min(dln r, dth)) are finer than dx."""
        return self.dx / min(grid.dlnr, grid.dth)


@dataclass
class BrickGrid:
    """Deposited fields on the allocated bricks (sorted keys; arrays (n_bricks, B, B, B))."""

    lattice: BrickLattice
    keys: np.ndarray  # int64, sorted
    mass: np.ndarray  # float64
    rho: np.ndarray  # float32
    t_em: np.ndarray  # float32, K: (mass-weighted mean of T^4)^(1/4)
    n_trilinear: int = 0
    vel: np.ndarray | None = None  # float32 (n_bricks, B, B, B, 3): mass-weighted velocity


@dataclass
class BrickTexture:
    """What the renderer samples (float32): rho and rho (T_em/T_REF)^4 per brick cell, and
    optionally rho x mass-weighted velocity (Phase 9: rho gamma beta, (n, B, B, B, 3))."""

    lattice: BrickLattice
    keys: np.ndarray
    rho: np.ndarray
    rho_t4: np.ndarray
    rho_gb: np.ndarray | None = None

    def __post_init__(self):
        self.keys_list = self.keys.tolist()  # for bisect in the readable samplers
        # box holding every allocated brick's home-brick region (on each axis brick B's region
        # is [(B BRICK + 1/2) dx, ((B + 1) BRICK + 1/2) dx)); the marcher's brick rule applies
        # inside it, and a ray from outside first steps to its face
        dx = self.lattice.dx
        if len(self.keys):
            b = np.stack(unpack_keys(self.keys), axis=1)
            self.box_lo = (b.min(axis=0) * BRICK + 0.5) * dx
            self.box_hi = ((b.max(axis=0) + 1) * BRICK + 0.5) * dx
        else:
            self.box_lo = self.box_hi = np.zeros(3)


# ---------------------------------------------------------------- keys and selection


def pack_keys(bi, bj, bk):
    """int64 key of brick (I, J, K); increasing in I, then J, then K."""
    bi, bj, bk = (np.asarray(a, np.int64) + KEY_OFFSET for a in (bi, bj, bk))
    return (bi << (2 * KEY_BITS)) | (bj << KEY_BITS) | bk


def unpack_keys(keys):
    keys = np.asarray(keys, np.int64)
    mask = (1 << KEY_BITS) - 1
    return (
        ((keys >> (2 * KEY_BITS)) & mask) - KEY_OFFSET,
        ((keys >> KEY_BITS) & mask) - KEY_OFFSET,
        (keys & mask) - KEY_OFFSET,
    )


def fine_particles(x, h, lattice: BrickLattice, grid: LogSphericalGrid) -> np.ndarray:
    """Mask of the particles deposited on the bricks (the rest go to the log grid)."""
    r = np.linalg.norm(np.asarray(x, float), axis=1)
    small = R_SUPPORT * np.asarray(h, float) <= MAX_SUPPORT_BRICKS * BRICK * lattice.dx
    return small & (r >= lattice.r_in(grid)) & (r <= lattice.r_fine)


def brick_keys(x, h, dx) -> np.ndarray:
    """Sorted keys of the bricks covering every particle's support grown by one cell."""
    x, h = np.asarray(x, float), np.asarray(h, float)
    if len(h) == 0:
        return np.zeros(0, np.int64)
    reach = (R_SUPPORT * h + dx)[:, None]
    lo = np.floor(np.floor((x - reach) / dx) / BRICK).astype(np.int64)
    hi = np.floor(np.floor((x + reach) / dx) / BRICK).astype(np.int64)
    span = hi - lo
    found = []
    for oi in range(int(span[:, 0].max()) + 1):
        for oj in range(int(span[:, 1].max()) + 1):
            for ok in range(int(span[:, 2].max()) + 1):
                sel = (span[:, 0] >= oi) & (span[:, 1] >= oj) & (span[:, 2] >= ok)
                b = lo[sel]
                found.append(np.unique(pack_keys(b[:, 0] + oi, b[:, 1] + oj, b[:, 2] + ok)))
    return np.unique(np.concatenate(found))


# ---------------------------------------------------------------- readable reference


def _cell_slot(keys_list, ci, cj, ck):
    """(slot, local i, j, k) of lattice cell (ci, cj, ck), slot -1 if its brick is absent."""
    key = int(pack_keys(ci // BRICK, cj // BRICK, ck // BRICK))
    s = bisect_left(keys_list, key)
    if s == len(keys_list) or keys_list[s] != key:
        return -1, 0, 0, 0
    return s, ci % BRICK, cj % BRICK, ck % BRICK


def deposit_bricks_reference(
    x, h, m, t, lattice: BrickLattice, kernel: str = "wendland_c4", v=None
):
    """Readable deposition of (fine) particles onto the brick lattice (module docstring).
    x [code lengths], h, m [code masses] (scalar or per particle), t [K]; v (N, 3) optional
    velocities (Phase 9: gamma beta), deposited mass-weighted like t (zero if not given)."""
    f = get_kernel(kernel).f
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    m = np.broadcast_to(np.asarray(m, float), h.shape)
    t = np.asarray(t, float)
    nf = 2 if v is None else 5
    v = np.zeros((len(h), 3)) if v is None else np.asarray(v, float)
    dx = lattice.dx
    keys = brick_keys(x, h, dx)
    keys_list = keys.tolist()
    acc = np.zeros((len(keys), BRICK, BRICK, BRICK, nf))  # m, m T^4 (, m v)
    n_tri = 0

    def add(ci, cj, ck, w, p):
        s, a, b, c = _cell_slot(keys_list, ci, cj, ck)
        if s < 0:
            raise AssertionError("deposit outside the allocated bricks")  # allocation bug
        acc[s, a, b, c, 0] += w * m[p]
        acc[s, a, b, c, 1] += w * m[p] * t[p] ** 4
        if nf == 5:
            acc[s, a, b, c, 2:5] += w * m[p] * v[p]

    for p in range(len(h)):
        big = R_SUPPORT * h[p]
        rng = [range(math.floor((x[p, d] - big) / dx), math.floor((x[p, d] + big) / dx) + 1)
               for d in range(3)]  # fmt: skip
        cells, weights = [], []
        for ci in rng[0]:
            for cj in rng[1]:
                for ck in rng[2]:
                    c = (np.array([ci, cj, ck]) + 0.5) * dx
                    q = float(np.linalg.norm(c - x[p])) / h[p]
                    if q < R_SUPPORT:
                        cells.append((ci, cj, ck))
                        weights.append(f(q))
        if len(cells) >= MIN_KERNEL_CELLS:
            total = sum(weights)
            for (ci, cj, ck), w in zip(cells, weights, strict=True):
                if w > 0.0:
                    add(ci, cj, ck, w / total, p)
            continue
        n_tri += 1
        fi = x[p] / dx - 0.5
        i0 = np.floor(fi).astype(int)
        tt = fi - i0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((tt[0] if di else 1 - tt[0]) * (tt[1] if dj else 1 - tt[1])
                         * (tt[2] if dk else 1 - tt[2]))  # fmt: skip
                    add(i0[0] + di, i0[1] + dj, i0[2] + dk, w, p)
    return _finish(lattice, keys, acc, n_tri)


def _finish(lattice, keys, acc, n_tri) -> BrickGrid:
    mass = acc[..., 0]
    has = mass > 0
    safe = np.where(has, mass, 1.0)
    t_em = np.where(has, np.maximum(acc[..., 1] / safe, 0.0) ** 0.25, 0.0)
    vel = None
    if acc.shape[-1] == 5:  # velocities deposited
        vel = np.where(has[..., None], acc[..., 2:5] / safe[..., None], 0.0).astype(np.float32)
    rho = mass / lattice.dx**3
    return BrickGrid(lattice, keys, mass, rho.astype(np.float32), t_em.astype(np.float32),
                     int(n_tri), vel)  # fmt: skip


# ---------------------------------------------------------------- fast (Numba)


@numba.njit(cache=True)
def _nb_slot(keys, ci, cj, ck):
    """Mirrors _cell_slot: binary search of the cell's brick key (-1 if absent)."""
    bi, bj, bk = ci >> 3, cj >> 3, ck >> 3  # BRICK = 8: floor division for negatives too
    key = (
        ((bi + KEY_OFFSET) << (2 * KEY_BITS)) | ((bj + KEY_OFFSET) << KEY_BITS) | (bk + KEY_OFFSET)
    )
    lo, hi = 0, keys.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < keys.shape[0] and keys[lo] == key:
        return lo
    return -1


@numba.njit(cache=True)
def _nb_brick_chunk(x, h, vals, lo, hi, dx, kid, keys, acc):
    """Brick deposition of particles lo..hi-1; mirrors deposit_bricks_reference."""
    n_tri = 0
    bad = 0
    for p in range(lo, hi):
        xp, yp, zp = x[p, 0], x[p, 1], x[p, 2]
        hp = h[p]
        big = 2.0 * hp
        a0 = int(np.floor((xp - big) / dx))
        a1 = int(np.floor((xp + big) / dx))
        b0 = int(np.floor((yp - big) / dx))
        b1 = int(np.floor((yp + big) / dx))
        c0 = int(np.floor((zp - big) / dx))
        c1 = int(np.floor((zp + big) / dx))
        wsum = 0.0
        n_in = 0
        for a in range(a0, a1 + 1):
            dxc = (a + 0.5) * dx - xp
            for b in range(b0, b1 + 1):
                dyc = (b + 0.5) * dx - yp
                for c in range(c0, c1 + 1):
                    dzc = (c + 0.5) * dx - zp
                    q = np.sqrt(dxc * dxc + dyc * dyc + dzc * dzc) / hp
                    if q < 2.0:
                        n_in += 1
                        wsum += kernel_f(q, kid)
        if n_in >= MIN_KERNEL_CELLS:
            inv = 1.0 / wsum
            for a in range(a0, a1 + 1):
                dxc = (a + 0.5) * dx - xp
                for b in range(b0, b1 + 1):
                    dyc = (b + 0.5) * dx - yp
                    for c in range(c0, c1 + 1):
                        dzc = (c + 0.5) * dx - zp
                        q = np.sqrt(dxc * dxc + dyc * dyc + dzc * dzc) / hp
                        if q < 2.0:
                            w = kernel_f(q, kid) * inv
                            if w > 0.0:
                                s = _nb_slot(keys, a, b, c)
                                if s < 0:
                                    bad += 1
                                else:
                                    for ch in range(vals.shape[1]):
                                        acc[s, a & 7, b & 7, c & 7, ch] += w * vals[p, ch]
            continue
        n_tri += 1
        fa, fb, fc = xp / dx - 0.5, yp / dx - 0.5, zp / dx - 0.5
        ia, ib, ic = int(np.floor(fa)), int(np.floor(fb)), int(np.floor(fc))
        ta, tb, tc = fa - ia, fb - ib, fc - ic
        for da in range(2):
            wa = ta if da == 1 else 1.0 - ta
            for db in range(2):
                wb = tb if db == 1 else 1.0 - tb
                for dc in range(2):
                    wc = tc if dc == 1 else 1.0 - tc
                    w = wa * wb * wc
                    a, b, c = ia + da, ib + db, ic + dc
                    s = _nb_slot(keys, a, b, c)
                    if s < 0:
                        bad += 1
                    else:
                        for ch in range(vals.shape[1]):
                            acc[s, a & 7, b & 7, c & 7, ch] += w * vals[p, ch]
    return n_tri, bad


@numba.njit(parallel=True, cache=True)
def _nb_bricks(x, h, vals, bounds, dx, kid, keys, acc, res):
    for c in numba.prange(bounds.shape[0] - 1):
        nt, bad = _nb_brick_chunk(x, h, vals, bounds[c], bounds[c + 1], dx, kid, keys, acc[c])
        res[c, 0] = nt
        res[c, 1] = bad


PASS_BYTES = 1.2e9  # accumulator memory per deposition pass (CLAUDE.md: stay under ~4 GB)


def deposit_bricks(
    x, h, m, t, lattice: BrickLattice, kernel: str = "wendland_c4", v=None
) -> BrickGrid:
    """Fast (Numba) brick deposition; matches deposit_bricks_reference (same chunking rule as
    gasgrid: fixed chunks summed in a fixed order, so the result is deterministic).

    Without v only m and m T^4 are deposited (vel None). The accumulated fields (m, m T^4,
    m v) are deposited in passes of as many fields as
    fit in PASS_BYTES of per-chunk float64 accumulators (each pass repeats the same kernel
    sums for its fields, so the result does not depend on the grouping)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    m = np.broadcast_to(np.asarray(m, float), h.shape)
    t = np.asarray(t, float)
    nf = 2 if v is None else 5
    keys = brick_keys(x, h, lattice.dx)
    per_field = N_CHUNKS * len(keys) * BRICK**3 * 8
    if per_field > PASS_BYTES:
        raise MemoryError(f"brick buffers would need {per_field / 1e9:.1f} GB; raise dx")
    group = max(1, min(nf, int(PASS_BYTES // max(per_field, 1))))
    vals = np.stack([m, m * t**4], axis=1)
    if v is not None:
        vals = np.concatenate([vals, m[:, None] * np.asarray(v, float)], axis=1)
    order = np.concatenate([np.arange(c, len(h), N_CHUNKS) for c in range(N_CHUNKS)])
    bounds = np.cumsum([0] + [len(range(c, len(h), N_CHUNKS)) for c in range(N_CHUNKS)])
    xo, ho, vo = x[order], h[order], vals[order]
    total = np.zeros((len(keys), BRICK, BRICK, BRICK, nf))
    n_tri = 0
    for f0 in range(0, nf, group):
        f1 = min(f0 + group, nf)
        acc = np.zeros((N_CHUNKS, len(keys), BRICK, BRICK, BRICK, f1 - f0))
        res = np.zeros((N_CHUNKS, 2), np.int64)
        _nb_bricks(xo, ho, np.ascontiguousarray(vo[:, f0:f1]), bounds.astype(np.int64),
                   lattice.dx, KERNEL_IDS[kernel], keys, acc, res)  # fmt: skip
        if res[:, 1].sum():
            raise AssertionError("deposit outside the allocated bricks")  # allocation bug
        part = acc[0].copy()
        for c in range(1, N_CHUNKS):  # fixed order: deterministic
            part += acc[c]
        del acc
        total[..., f0:f1] = part
        n_tri = int(res[:, 0].sum())
    return _finish(lattice, keys, total, n_tri)


# ---------------------------------------------------------------- sampling (readable)


def texture_from_bricks(bg: BrickGrid, t_ref: float) -> BrickTexture:
    """rho, rho (T_em / t_ref)^4 and rho v as float32 (march.T_REF is passed in: no import
    cycle)."""
    rho = bg.rho.astype(float)
    rt4 = rho * (bg.t_em.astype(float) / t_ref) ** 4
    if not (np.all(np.isfinite(rt4)) and rt4.max(initial=0.0) < 1e37):
        raise OverflowError("brick texture values would overflow float32")
    rgb = None if bg.vel is None else (rho[..., None] * bg.vel).astype(np.float32)
    return BrickTexture(bg.lattice, bg.keys, bg.rho.astype(np.float32), rt4.astype(np.float32),
                        rgb)  # fmt: skip


def _cell_value(tex: BrickTexture, keys_list, ci, cj, ck):
    s, a, b, c = _cell_slot(keys_list, ci, cj, ck)
    if s < 0:
        return 0.0, 0.0
    return float(tex.rho[s, a, b, c]), float(tex.rho_t4[s, a, b, c])


def sample_bricks(tex: BrickTexture, p, keys_list=None):
    """(rho, rho (T/T_REF)^4) at p: trilinear between lattice cell centres; absent cells are
    empty (0)."""
    keys_list = tex.keys_list if keys_list is None else keys_list
    f = np.asarray(p, float) / tex.lattice.dx - 0.5
    i0 = np.floor(f).astype(int)
    tt = f - i0
    out_r = out_t = 0.0
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((tt[0] if di else 1 - tt[0]) * (tt[1] if dj else 1 - tt[1])
                     * (tt[2] if dk else 1 - tt[2]))  # fmt: skip
                r_, t_ = _cell_value(tex, keys_list, i0[0] + di, i0[1] + dj, i0[2] + dk)
                out_r += w * r_
                out_t += w * t_
    return out_r, out_t


def sample_bricks_rho_gb(tex: BrickTexture, p, keys_list=None):
    """rho x velocity (3,) at p, trilinear like sample_bricks (zero if the texture has none)."""
    if tex.rho_gb is None:
        return np.zeros(3)
    keys_list = tex.keys_list if keys_list is None else keys_list
    f = np.asarray(p, float) / tex.lattice.dx - 0.5
    i0 = np.floor(f).astype(int)
    tt = f - i0
    out = np.zeros(3)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((tt[0] if di else 1 - tt[0]) * (tt[1] if dj else 1 - tt[1])
                     * (tt[2] if dk else 1 - tt[2]))  # fmt: skip
                s, a, b, c = _cell_slot(keys_list, i0[0] + di, i0[1] + dj, i0[2] + dk)
                if s >= 0:
                    out += w * tex.rho_gb[s, a, b, c].astype(float)
    return out


def home_brick(p, dx):
    """Brick of the home cell floor(p / dx - 1/2) (the cell sampling at p starts from)."""
    f = np.floor(np.asarray(p, float) / dx - 0.5).astype(np.int64)
    return f // BRICK


def home_allocated(tex: BrickTexture, p, keys_list=None) -> bool:
    keys_list = tex.keys_list if keys_list is None else keys_list
    b = home_brick(p, tex.lattice.dx)
    key = int(pack_keys(b[0], b[1], b[2]))
    s = bisect_left(keys_list, key)
    return s < len(keys_list) and keys_list[s] == key


def entry_distance(p, e, lo, hi) -> float:
    """Distance along unit vector e from p (outside the box [lo, hi)) to just inside the box,
    or inf if the ray misses it (slab method)."""
    t0, t1 = -math.inf, math.inf
    for a in range(3):
        if e[a] != 0.0:
            u, v = (lo[a] - p[a]) / e[a], (hi[a] - p[a]) / e[a]
            t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
        elif not lo[a] <= p[a] < hi[a]:
            return math.inf
    if t1 < max(t0, 0.0):
        return math.inf
    return max(t0, 0.0)


def exit_distance(p, e, dx):
    """Distance along unit vector e from p to just past the far face of p's home brick (the
    home-brick region on each axis is [(B BRICK + 1/2) dx, ((B + 1) BRICK + 1/2) dx))."""
    p, e = np.asarray(p, float), np.asarray(e, float)
    b = home_brick(p, dx)
    best = math.inf
    for a in range(3):
        if e[a] > 0.0:
            face = ((b[a] + 1) * BRICK + 0.5) * dx
            best = min(best, (face - p[a]) / e[a])
        elif e[a] < 0.0:
            face = (b[a] * BRICK + 0.5) * dx
            best = min(best, (face - p[a]) / e[a])
    return max(best, 0.0) + EXIT_EPS * dx
