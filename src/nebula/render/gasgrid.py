"""Depositing SPH particles onto render grids (phases/phase-08-gas-rendering.md 8.3).

Two grid kinds:
- ``LogSphericalGrid``: centred on the hole, logarithmic in r, uniform in theta and phi
  (phi periodic). Cell (i, j, k) spans ln r in [ln r_min + i dlnr, + dlnr], theta in
  [j dth, (j+1) dth], phi in [k dph, (k+1) dph]; volume (r2^3 - r1^3)/3 (cos th1 - cos th2) dph.
  Cell centre: r_c = r_min exp((i + 1/2) dlnr), theta and phi at their mid values.
- ``CartesianBox``: a cube of n^3 equal cells (detail boxes, 8.3 item 2).

Deposition rule (8.3): a particle's mass goes to the cells whose centres lie inside its
kernel support (|x_c - x| < 2h), with weights f(|x_c - x| / h) * V_c normalised to sum to 1
(f: the SPH kernel shape, physics.kernels). If fewer than ``MIN_KERNEL_CELLS`` centres lie
inside the support (h small compared with a cell), it falls back to trilinear (cloud-in-cell)
weights in index space over the 8 nearest cell centres. Either way each particle deposits
exactly its mass.

Grid edges: cells beyond the grid (r > r_max or r < r_min, or outside a box) are evaluated as
"virtual" cells with the same geometry; their share of the weight goes to an ``outside`` total
instead of being squeezed into edge cells (which would draw a bright rim). So
``grid mass + outside = sum m`` exactly, for every particle. theta and phi have no edges
(the sphere closes; phi wraps across the 0 / 2 pi seam).

Fields per cell: mass M, rho = M / V, mass-weighted T, emission-weighted
T_em = (sum m T^4 / sum m)^(1/4) (so hot thin gas mixed with cool dense gas is not dimmed,
8.3), and mass-weighted velocity.

Determinism (8.3 parallel rule): particles are split into ``N_CHUNKS`` fixed contiguous
chunks, each deposited into its own float64 grid, summed in a fixed order. The chunking does
not depend on the thread count, so 1 and 2 threads give identical grids (test 2).

``deposit_*_reference`` are the plain readable versions (CLAUDE.md rule 2); the Numba versions
must match them (tests/render/test_gasgrid.py).
"""

from dataclasses import dataclass, field

import numba
import numpy as np

from nebula.physics.kernels import get_kernel
from nebula.sim.kernels_nb import KERNEL_IDS, kernel_f

R_SUPPORT = 2.0  # kernel support in units of h (physics.kernels.R_KERN)
MIN_KERNEL_CELLS = 8  # fewer cell centres inside the support -> trilinear weights
N_CHUNKS = 2  # fixed deposition chunks (determinism; 2 = this Mac's cores)
N_FIELDS = 6  # accumulated per cell: m, m T, m T^4, m vx, m vy, m vz
INNER_VIRTUAL = 64  # virtual log cells below r_min (inside the horizon; their weight is just
# left out of the normalisation, so the mass ledger stays exact)


@dataclass(frozen=True)
class LogSphericalGrid:
    """Global grid around the hole (8.3 item 1). Default 192 x 96 x 192 = 3.5 M cells."""

    r_min: float = 2.0
    r_max: float = 2000.0
    nr: int = 192
    nth: int = 96
    nph: int = 192

    @property
    def dlnr(self) -> float:
        return float(np.log(self.r_max / self.r_min) / self.nr)

    @property
    def dth(self) -> float:
        return float(np.pi / self.nth)

    @property
    def dph(self) -> float:
        return float(2.0 * np.pi / self.nph)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nr, self.nth, self.nph)

    def r_centres(self) -> np.ndarray:
        return self.r_min * np.exp((np.arange(self.nr) + 0.5) * self.dlnr)

    def volumes(self) -> np.ndarray:
        """Cell volumes, shape (nr, nth) (independent of phi)."""
        return _log_volume(np.arange(self.nr)[:, None], np.arange(self.nth)[None, :], self)


@dataclass(frozen=True)
class CartesianBox:
    """A detail box (8.3 item 2): cube of side ``size`` centred on ``centre``, n^3 cells."""

    centre: tuple[float, float, float]
    size: float
    n: int = 160

    @property
    def dx(self) -> float:
        return self.size / self.n

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.n, self.n)

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.centre, float) - 0.5 * self.size


@dataclass
class GasGrid:
    """Deposited fields on one grid (float32 for rendering, 8.3) plus the mass ledger."""

    grid: object
    mass: np.ndarray  # float64, per cell
    rho: np.ndarray  # float32
    t_mass: np.ndarray  # float32, K, mass-weighted (0 in empty cells)
    t_em: np.ndarray  # float32, K, (mass-weighted mean of T^4)^(1/4)
    vel: np.ndarray  # float32, (..., 3), mass-weighted
    outside: float  # mass deposited beyond the grid
    n_trilinear: int = 0  # particles that used the trilinear fallback
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry (shared)


def _log_volume(i, j, g):
    """Volume of log-spherical cell(s) (i, j); i may be a virtual index beyond [0, nr)."""
    r1 = g.r_min * np.exp(i * g.dlnr)
    r2 = r1 * np.exp(g.dlnr)
    return (r2**3 - r1**3) / 3.0 * (np.cos(j * g.dth) - np.cos((j + 1) * g.dth)) * g.dph


def _log_centre(i, j, k, g):
    """Cartesian centre(s) of log-spherical cell(s); stacked on the last axis."""
    r = g.r_min * np.exp((np.asarray(i) + 0.5) * g.dlnr)
    th = (np.asarray(j) + 0.5) * g.dth
    ph = (np.asarray(k) + 0.5) * g.dph
    return np.stack(
        np.broadcast_arrays(
            r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)
        ),
        -1,
    )


def _log_candidates(x, h, g):
    """Index ranges (virtual r indices allowed) covering every cell centre within 2h of x.

    Returns (i_lo, i_hi, j_lo, j_hi, k_list); k_list is a sorted array of phi indices (wrapped,
    no repeats).
    """
    rr = float(np.sqrt(x @ x))
    big = R_SUPPORT * h
    lnr0 = np.log(g.r_min)
    i_hi = int(np.floor((np.log(rr + big) - lnr0) / g.dlnr))
    i_lo = -INNER_VIRTUAL if big >= rr else int(np.floor((np.log(rr - big) - lnr0) / g.dlnr))
    i_lo = max(i_lo, -INNER_VIRTUAL)
    th = np.arccos(np.clip(x[2] / rr, -1.0, 1.0))
    ph = np.arctan2(x[1], x[0]) % (2.0 * np.pi)
    if big >= rr:
        return i_lo, i_hi, 0, g.nth - 1, np.arange(g.nph)
    a = np.arcsin(big / rr)  # largest angle from the particle's direction
    j_lo = max(0, int(np.floor((th - a) / g.dth)))
    j_hi = min(g.nth - 1, int(np.floor((th + a) / g.dth)))
    s_min = min(np.sin(max(th - a, 0.0)), np.sin(min(th + a, np.pi)))
    if th - a <= 0.0 or th + a >= np.pi or np.sin(a) >= s_min:
        return i_lo, i_hi, j_lo, j_hi, np.arange(g.nph)
    dphi = np.arcsin(np.sin(a) / s_min)
    k0 = int(np.floor((ph - dphi) / g.dph))
    k1 = int(np.floor((ph + dphi) / g.dph))
    if k1 - k0 + 1 >= g.nph:
        return i_lo, i_hi, j_lo, j_hi, np.arange(g.nph)
    return i_lo, i_hi, j_lo, j_hi, np.unique(np.arange(k0, k1 + 1) % g.nph)


# ---------------------------------------------------------------- reference (readable)


def _particle_temps(t):
    t = np.asarray(t, float)
    return t, t**4


def deposit_log_reference(x, h, m, t, v, g: LogSphericalGrid, kernel: str = "wendland_c4"):
    """Readable reference deposition onto a log-spherical grid (see module docstring)."""
    f = get_kernel(kernel).f
    acc = np.zeros((*g.shape, N_FIELDS))
    outside = 0.0
    n_tri = 0
    x = np.asarray(x, float)
    h, m = np.asarray(h, float), np.broadcast_to(np.asarray(m, float), h.shape)
    t, t4 = _particle_temps(t)
    v = np.asarray(v, float)
    for p in range(len(h)):
        vals = np.array([1.0, t[p], t4[p], *v[p]]) * m[p]
        i_lo, i_hi, j_lo, j_hi, ks = _log_candidates(x[p], h[p], g)
        ii, jj, kk = np.meshgrid(
            np.arange(i_lo, i_hi + 1), np.arange(j_lo, j_hi + 1), ks, indexing="ij"
        )
        q = np.linalg.norm(_log_centre(ii, jj, kk, g) - x[p], axis=-1) / h[p]
        inside = q < R_SUPPORT
        if inside.sum() >= MIN_KERNEL_CELLS:
            w = np.where(inside, f(q) * _log_volume(ii, jj, g), 0.0)
            w = w / w.sum()
            ing = (ii >= 0) & (ii < g.nr) & (w > 0)
            np.add.at(acc, (ii[ing], jj[ing], kk[ing]), w[ing][:, None] * vals)
            outside += m[p] * w[~((ii >= 0) & (ii < g.nr))].sum()
            continue
        n_tri += 1
        for (a, b, c), w in _log_trilinear(x[p], g):
            if 0 <= a < g.nr:
                acc[a, b, c] += w * vals
            else:
                outside += w * m[p]
    return _finish(g, acc, g.volumes()[:, :, None], outside, n_tri)


def _log_trilinear(x, g):
    """The 8 (cell, weight) pairs of cloud-in-cell weights in (ln r, theta, phi) index space.

    theta is clamped at the poles, phi wraps; r beyond the grid gives virtual indices.
    """
    rr = float(np.sqrt(x @ x))
    fr = (np.log(rr / g.r_min)) / g.dlnr - 0.5
    fth = np.arccos(np.clip(x[2] / rr, -1.0, 1.0)) / g.dth - 0.5
    fph = (np.arctan2(x[1], x[0]) % (2.0 * np.pi)) / g.dph - 0.5
    i0, j0, k0 = int(np.floor(fr)), int(np.floor(fth)), int(np.floor(fph))
    ti, tj, tk = fr - i0, fth - j0, fph - k0
    if j0 < 0:
        j0, tj = 0, 0.0
    if j0 > g.nth - 2:
        j0, tj = g.nth - 2, 1.0
    out = []
    for di, wi in ((0, 1 - ti), (1, ti)):
        for dj, wj in ((0, 1 - tj), (1, tj)):
            for dk, wk in ((0, 1 - tk), (1, tk)):
                out.append(((i0 + di, j0 + dj, (k0 + dk) % g.nph), wi * wj * wk))
    return out


def deposit_box_reference(x, h, m, t, v, b: CartesianBox, kernel: str = "wendland_c4"):
    """Readable reference deposition onto a Cartesian box (see module docstring)."""
    f = get_kernel(kernel).f
    acc = np.zeros((*b.shape, N_FIELDS))
    outside = 0.0
    n_tri = 0
    x = np.asarray(x, float)
    h, m = np.asarray(h, float), np.broadcast_to(np.asarray(m, float), h.shape)
    t, t4 = _particle_temps(t)
    v = np.asarray(v, float)
    lo, dx = b.lo, b.dx
    for p in range(len(h)):
        vals = np.array([1.0, t[p], t4[p], *v[p]]) * m[p]
        big = R_SUPPORT * h[p]
        rng = [
            np.arange(
                int(np.floor((x[p, d] - big - lo[d]) / dx)),
                int(np.floor((x[p, d] + big - lo[d]) / dx)) + 1,
            )
            for d in range(3)
        ]
        ii, jj, kk = np.meshgrid(*rng, indexing="ij")
        c = lo + (np.stack([ii, jj, kk], -1) + 0.5) * dx
        q = np.linalg.norm(c - x[p], axis=-1) / h[p]
        inside = q < R_SUPPORT
        ing_all = (ii >= 0) & (ii < b.n) & (jj >= 0) & (jj < b.n) & (kk >= 0) & (kk < b.n)
        if inside.sum() >= MIN_KERNEL_CELLS:
            w = np.where(inside, f(q), 0.0)  # equal cell volumes
            w = w / w.sum()
            ing = ing_all & (w > 0)
            np.add.at(acc, (ii[ing], jj[ing], kk[ing]), w[ing][:, None] * vals)
            outside += m[p] * w[~ing_all].sum()
            continue
        n_tri += 1
        fi = (x[p] - lo) / dx - 0.5
        i0 = np.floor(fi).astype(int)
        tt = fi - i0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (
                        (tt[0] if di else 1 - tt[0])
                        * (tt[1] if dj else 1 - tt[1])
                        * (tt[2] if dk else 1 - tt[2])
                    )
                    a, bb, c = i0[0] + di, i0[1] + dj, i0[2] + dk
                    if 0 <= a < b.n and 0 <= bb < b.n and 0 <= c < b.n:
                        acc[a, bb, c] += w * vals
                    else:
                        outside += w * m[p]
    return _finish(b, acc, np.full((1, 1, 1), b.dx**3), outside, n_tri)


def _finish(g, acc, vol, outside, n_tri) -> GasGrid:
    """Cell fields from the accumulated sums (empty cells: rho = T = v = 0)."""
    mass = acc[..., 0]
    safe = np.where(mass > 0, mass, 1.0)
    has = mass > 0
    t_mass = np.where(has, acc[..., 1] / safe, 0.0)
    t_em = np.where(has, np.maximum(acc[..., 2] / safe, 0.0) ** 0.25, 0.0)
    vel = np.where(has[..., None], acc[..., 3:6] / safe[..., None], 0.0)
    return GasGrid(
        g,
        mass,
        (mass / vol).astype(np.float32),
        t_mass.astype(np.float32),
        t_em.astype(np.float32),
        vel.astype(np.float32),
        float(outside),
        int(n_tri),
    )


# ---------------------------------------------------------------- fast (Numba)


@numba.njit(cache=True)
def _nb_log_kernel_pass(
    p,
    xp,
    yp,
    zp,
    rr,
    th,
    ph,
    hp,
    i_lo,
    i_hi,
    r_min,
    dlnr,
    dth,
    dph,
    nr,
    nth,
    nph,
    kid,
    sth,
    cth,
    dcth,
    cph,
    sph,
    mode,
    inv,
    vals,
    m,
    acc,
):
    """One pass over the cell centres inside particle p's kernel support.

    mode 0: returns (sum of f(q) V_c, number of centres inside, 0). mode 1: adds the normalised
    weights f(q) V_c * inv to acc and returns (0, 0, mass beyond the grid).
    Candidate centres come from exact sphere geometry: on the shell of radius r_c, centres
    inside the support have angular distance <= alpha with
    cos alpha = (r_c^2 + r^2 - (2h)^2) / (2 r_c r); on a theta row, cos dphi >=
    (cos alpha - cos th cos th_j) / (sin th sin th_j). One cell of margin on each side; the test
    q < 2 decides. (The reference uses a plain bounding block; tests compare the two.)
    """
    big = 2.0 * hp
    ct0, st0 = np.cos(th), np.sin(th)
    wsum = 0.0
    n_in = 0
    outside = 0.0
    for i in range(i_lo, i_hi + 1):
        rc = r_min * np.exp((i + 0.5) * dlnr)
        cos_a = (rc * rc + rr * rr - big * big) / (2.0 * rc * rr)
        if cos_a > 1.0:
            continue
        if cos_a <= -1.0:
            j0, j1 = 0, nth - 1
        else:
            a = np.arccos(cos_a)
            j0 = max(0, int(np.floor((th - a) / dth)) - 1)
            j1 = min(nth - 1, int(np.floor((th + a) / dth)) + 1)
        r1 = r_min * np.exp(i * dlnr)
        r2 = r1 * np.exp(dlnr)
        vr = (r2 * r2 * r2 - r1 * r1 * r1) / 3.0
        ing = 0 <= i < nr
        for j in range(j0, j1 + 1):
            ss = st0 * sth[j]
            full = True
            k0, k1 = 0, nph - 1
            if ss > 1e-300 and cos_a > -1.0:
                cd = (cos_a - ct0 * cth[j]) / ss
                if cd > 1.0 + 1e-12:
                    continue
                if cd > -1.0:
                    dp = np.arccos(min(cd, 1.0))
                    k0 = int(np.ceil((ph - dp) / dph - 0.5)) - 1
                    k1 = int(np.floor((ph + dp) / dph - 0.5)) + 1
                    if k1 - k0 + 1 < nph:
                        full = False
            if full:
                k0, k1 = 0, nph - 1
            rst = rc * sth[j]
            dzc = rc * cth[j] - zp
            vrt = vr * dcth[j] * dph
            for kk in range(k0, k1 + 1):
                k = kk % nph
                dxc = rst * cph[k] - xp
                dyc = rst * sph[k] - yp
                q = np.sqrt(dxc * dxc + dyc * dyc + dzc * dzc) / hp
                if q < 2.0:
                    if mode == 0:
                        n_in += 1
                        wsum += kernel_f(q, kid) * vrt
                    else:
                        w = kernel_f(q, kid) * vrt * inv
                        if ing:
                            for f_ in range(N_FIELDS):
                                acc[i, j, k, f_] += w * vals[p, f_]
                        else:
                            outside += w * m[p]
    return wsum, n_in, outside


@numba.njit(cache=True)
def _nb_log_deposit_chunk(
    x, h, m, vals, lo, hi, r_min, dlnr, dth, dph, nr, nth, nph, kid, acc, sth, cth, dcth, cph, sph
):
    """Deposit particles lo..hi-1 into acc (nr, nth, nph, N_FIELDS); returns (outside, n_tri).

    sth, cth: sin, cos of the theta cell centres; dcth: cos th1 - cos th2 per theta row; cph, sph:
    cos, sin of the phi cell centres (precomputed: no trigonometry in the inner loops).
    Mirrors deposit_log_reference / _log_trilinear (tested against them).
    """
    outside = 0.0
    n_tri = 0
    lnr0 = np.log(r_min)
    two_pi = 2.0 * np.pi
    for p in range(lo, hi):
        xp, yp, zp = x[p, 0], x[p, 1], x[p, 2]
        rr = np.sqrt(xp * xp + yp * yp + zp * zp)
        hp = h[p]
        big = 2.0 * hp
        i_hi = int(np.floor((np.log(rr + big) - lnr0) / dlnr))
        if big >= rr:
            i_lo = -INNER_VIRTUAL
        else:
            i_lo = max(int(np.floor((np.log(rr - big) - lnr0) / dlnr)), -INNER_VIRTUAL)
        th = np.arccos(min(1.0, max(-1.0, zp / rr)))
        ph = np.arctan2(yp, xp) % two_pi
        wsum, n_in, _ = _nb_log_kernel_pass(
            p,
            xp,
            yp,
            zp,
            rr,
            th,
            ph,
            hp,
            i_lo,
            i_hi,
            r_min,
            dlnr,
            dth,
            dph,
            nr,
            nth,
            nph,
            kid,
            sth,
            cth,
            dcth,
            cph,
            sph,
            0,
            0.0,
            vals,
            m,
            acc,
        )
        if n_in >= MIN_KERNEL_CELLS:
            _, _, o = _nb_log_kernel_pass(
                p,
                xp,
                yp,
                zp,
                rr,
                th,
                ph,
                hp,
                i_lo,
                i_hi,
                r_min,
                dlnr,
                dth,
                dph,
                nr,
                nth,
                nph,
                kid,
                sth,
                cth,
                dcth,
                cph,
                sph,
                1,
                1.0 / wsum,
                vals,
                m,
                acc,
            )
            outside += o
            continue
        # trilinear fallback (as _log_trilinear)
        n_tri += 1
        fr = np.log(rr / r_min) / dlnr - 0.5
        fth = th / dth - 0.5
        fph = ph / dph - 0.5
        i0 = int(np.floor(fr))
        j0 = int(np.floor(fth))
        k0 = int(np.floor(fph))
        ti, tj, tk = fr - i0, fth - j0, fph - k0
        if j0 < 0:
            j0, tj = 0, 0.0
        if j0 > nth - 2:
            j0, tj = nth - 2, 1.0
        for di in range(2):
            wi = ti if di == 1 else 1.0 - ti
            for dj in range(2):
                wj = tj if dj == 1 else 1.0 - tj
                for dk in range(2):
                    wk = tk if dk == 1 else 1.0 - tk
                    w = wi * wj * wk
                    i = i0 + di
                    if 0 <= i < nr:
                        k = (k0 + dk) % nph
                        for f_ in range(N_FIELDS):
                            acc[i, j0 + dj, k, f_] += w * vals[p, f_]
                    else:
                        outside += w * m[p]
    return outside, n_tri


@numba.njit(cache=True)
def _nb_box_deposit_chunk(x, h, m, vals, lo, hi, x0, y0, z0, dx, n, kid, acc):
    """Box deposition of particles lo..hi-1; mirrors deposit_box_reference."""
    outside = 0.0
    n_tri = 0
    for p in range(lo, hi):
        xp, yp, zp = x[p, 0], x[p, 1], x[p, 2]
        hp = h[p]
        big = 2.0 * hp
        a0 = int(np.floor((xp - big - x0) / dx))
        a1 = int(np.floor((xp + big - x0) / dx))
        b0 = int(np.floor((yp - big - y0) / dx))
        b1 = int(np.floor((yp + big - y0) / dx))
        c0 = int(np.floor((zp - big - z0) / dx))
        c1 = int(np.floor((zp + big - z0) / dx))
        wsum = 0.0
        n_in = 0
        for a in range(a0, a1 + 1):
            dxc = x0 + (a + 0.5) * dx - xp
            for b in range(b0, b1 + 1):
                dyc = y0 + (b + 0.5) * dx - yp
                for c in range(c0, c1 + 1):
                    dzc = z0 + (c + 0.5) * dx - zp
                    q = np.sqrt(dxc * dxc + dyc * dyc + dzc * dzc) / hp
                    if q < 2.0:
                        n_in += 1
                        wsum += kernel_f(q, kid)
        if n_in >= MIN_KERNEL_CELLS:
            inv = 1.0 / wsum
            for a in range(a0, a1 + 1):
                dxc = x0 + (a + 0.5) * dx - xp
                for b in range(b0, b1 + 1):
                    dyc = y0 + (b + 0.5) * dx - yp
                    for c in range(c0, c1 + 1):
                        dzc = z0 + (c + 0.5) * dx - zp
                        q = np.sqrt(dxc * dxc + dyc * dyc + dzc * dzc) / hp
                        if q < 2.0:
                            w = kernel_f(q, kid) * inv
                            if 0 <= a < n and 0 <= b < n and 0 <= c < n:
                                for f_ in range(N_FIELDS):
                                    acc[a, b, c, f_] += w * vals[p, f_]
                            else:
                                outside += w * m[p]
            continue
        n_tri += 1
        fa = (xp - x0) / dx - 0.5
        fb = (yp - y0) / dx - 0.5
        fc = (zp - z0) / dx - 0.5
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
                    if 0 <= a < n and 0 <= b < n and 0 <= c < n:
                        for f_ in range(N_FIELDS):
                            acc[a, b, c, f_] += w * vals[p, f_]
                    else:
                        outside += w * m[p]
    return outside, n_tri


@numba.njit(parallel=True, cache=True)
def _nb_log_deposit(
    x,
    h,
    m,
    vals,
    bounds,
    r_min,
    dlnr,
    dth,
    dph,
    nr,
    nth,
    nph,
    kid,
    acc,
    res,
    sth,
    cth,
    dcth,
    cph,
    sph,
):
    for c in numba.prange(bounds.shape[0] - 1):
        o, nt = _nb_log_deposit_chunk(
            x,
            h,
            m,
            vals,
            bounds[c],
            bounds[c + 1],
            r_min,
            dlnr,
            dth,
            dph,
            nr,
            nth,
            nph,
            kid,
            acc[c],
            sth,
            cth,
            dcth,
            cph,
            sph,
        )
        res[c, 0] = o
        res[c, 1] = nt


@numba.njit(parallel=True, cache=True)
def _nb_box_deposit(x, h, m, vals, bounds, x0, y0, z0, dx, n, kid, acc, res):
    for c in numba.prange(bounds.shape[0] - 1):
        o, nt = _nb_box_deposit_chunk(
            x, h, m, vals, bounds[c], bounds[c + 1], x0, y0, z0, dx, n, kid, acc[c]
        )
        res[c, 0] = o
        res[c, 1] = nt


def _prepare(x, h, m, t, v):
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    m = np.ascontiguousarray(np.broadcast_to(np.asarray(m, float), h.shape), dtype=np.float64)
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    vals = np.empty((len(h), N_FIELDS))
    vals[:, 0] = m
    vals[:, 1] = m * t
    vals[:, 2] = m * t**4
    vals[:, 3:6] = m[:, None] * v
    # Deal particles to the chunks in turn (p -> chunk p mod N_CHUNKS), so expensive neighbours
    # in the list (e.g. a run of large-h halo particles) are shared out; still a fixed order.
    order = np.concatenate([np.arange(c, len(h), N_CHUNKS) for c in range(N_CHUNKS)])
    bounds = np.cumsum([0] + [len(range(c, len(h), N_CHUNKS)) for c in range(N_CHUNKS)])
    return x[order], h[order], m[order], vals[order], bounds.astype(np.int64)


def _check_memory(shape):
    need = N_CHUNKS * int(np.prod(shape)) * N_FIELDS * 8
    if need > 2.5e9:  # CLAUDE.md: keep memory under ~4 GB
        raise MemoryError(f"deposition buffers would need {need / 1e9:.1f} GB; use a smaller grid")


def deposit_log(x, h, m, t, v, g: LogSphericalGrid, kernel: str = "wendland_c4") -> GasGrid:
    """Fast (Numba) deposition onto a log-spherical grid; matches deposit_log_reference.

    x: (N, 3) positions [code lengths]; h: (N,) smoothing lengths; m: particle mass(es)
    [code masses]; t: (N,) temperatures [K]; v: (N, 3) velocities [code units].
    """
    _check_memory(g.shape)
    x, h, m, vals, bounds = _prepare(x, h, m, t, v)
    acc = np.zeros((N_CHUNKS, *g.shape, N_FIELDS))
    res = np.zeros((N_CHUNKS, 2))
    thc = (np.arange(g.nth) + 0.5) * g.dth
    phc = (np.arange(g.nph) + 0.5) * g.dph
    dcth = np.cos(np.arange(g.nth) * g.dth) - np.cos((np.arange(g.nth) + 1) * g.dth)
    _nb_log_deposit(
        x,
        h,
        m,
        vals,
        bounds,
        g.r_min,
        g.dlnr,
        g.dth,
        g.dph,
        g.nr,
        g.nth,
        g.nph,
        KERNEL_IDS[kernel],
        acc,
        res,
        np.sin(thc),
        np.cos(thc),
        dcth,
        np.cos(phc),
        np.sin(phc),
    )
    total = acc[0].copy()
    for c in range(1, N_CHUNKS):  # fixed order: deterministic
        total += acc[c]
    del acc
    return _finish(g, total, g.volumes()[:, :, None], res[:, 0].sum(), int(res[:, 1].sum()))


def deposit_box(x, h, m, t, v, b: CartesianBox, kernel: str = "wendland_c4") -> GasGrid:
    """Fast (Numba) deposition onto a Cartesian box; matches deposit_box_reference."""
    _check_memory(b.shape)
    x, h, m, vals, bounds = _prepare(x, h, m, t, v)
    acc = np.zeros((N_CHUNKS, *b.shape, N_FIELDS))
    res = np.zeros((N_CHUNKS, 2))
    lo = b.lo
    _nb_box_deposit(
        x, h, m, vals, bounds, lo[0], lo[1], lo[2], b.dx, b.n, KERNEL_IDS[kernel], acc, res
    )
    total = acc[0].copy()
    for c in range(1, N_CHUNKS):
        total += acc[c]
    del acc
    return _finish(b, total, np.full((1, 1, 1), b.dx**3), res[:, 0].sum(), int(res[:, 1].sum()))
