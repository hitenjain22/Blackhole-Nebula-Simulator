"""SPH passes over particles and their neighbours (phases/phase-03-sph-core.md 3.1).

Density and smoothing length (Price 2012, J. Comput. Phys. 231, 759, §2-3; Price et al.
2018, Phantom, §2.1.3-2.1.4, Eqs. 3-12; checked 2026-10-14), solved together per particle:

    rho_i = sum_j m_j W(|r_ij|, h_i)                       (gather, includes j = i)
    h_i   = eta (m_i / rho_i)^(1/3)
    f(h)  = rho_sum(h) - m_i (eta / h)^3 = 0,   f'(h) = sum_j m_j dW/dh(h) + 3 m_i eta^3 / h^4
    Omega_i = 1 + h_i / (3 rho_i) sum_j m_j dW_ij/dh_i      (grad-h term; dh/drho = -h/(3 rho))

Newton-Raphson from the current h, safeguarded by a bisection bracket (lo, hi); converged
when |h_new - h| / h_start < tol (Phantom: 1e-4). h is capped at h_max: a particle with no
root below it (too few neighbours, e.g. isolated) gets h = h_max and a flag, never a NaN:
whenever h ends within 0.1% of its limit, f(limit) is evaluated to decide whether a root
exists below it (the iteration only creeps toward a limit it cannot cross).

Two implementations that must agree (CLAUDE.md rule 2): ``density_reference`` (plain
Python/NumPy, readable) and the parallel Numba kernel behind ``compute_density``. Parallel
rule: the loop over i writes only to i.
"""

from dataclasses import dataclass

import numba
import numpy as np

from nebula.physics.kernels import get_kernel
from nebula.sim.kernels_nb import C_NORM, KERNEL_IDS, kernel_df, kernel_f
from nebula.sim.neighbours import Neighbours, find_neighbours, minimum_image

FLAG_OK = 0
FLAG_AT_H_MAX = 1  # no root below h_max: h capped (too few neighbours)
FLAG_SEARCH_TOO_SMALL = 2  # root lies beyond the neighbour-search radius: redo with larger lists
FLAG_NO_CONVERGENCE = 3  # iteration limit reached

MAX_ITER = 50


@dataclass
class DensityResult:
    h: np.ndarray
    rho: np.ndarray
    omega: np.ndarray
    iterations: np.ndarray
    flags: np.ndarray
    neighbours: Neighbours


# --------------------------------------------------------------------------- reference


def _rho_sum_reference(r: np.ndarray, m_j: np.ndarray, h: float, kernel):
    """rho_sum(h) and sum_j m_j dW/dh for one particle with neighbour distances r."""
    return np.sum(m_j * kernel.w(r, h)), np.sum(m_j * kernel.dw_dh(r, h))


def density_reference(
    pos,
    mass,
    h_start,
    nb: Neighbours,
    h_search,
    kernel_name="cubic",
    eta=1.2,
    tol=1e-4,
    h_max=np.inf,
) -> DensityResult:
    """Readable reference: same algorithm as the Numba kernel, one particle at a time."""
    kernel = get_kernel(kernel_name)
    n = len(pos)
    h_out, rho, omega = np.empty(n), np.empty(n), np.empty(n)
    iters, flags = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for i in range(n):
        j = nb.of(i)
        r = np.linalg.norm(minimum_image(pos[j] - pos[i], nb.box), axis=1)
        m_j, m_i = mass[j], mass[i]
        hi = min(h_search[i], h_max)
        lo = 0.0
        h = min(h_start[i], hi)
        h0 = h
        flag = FLAG_NO_CONVERGENCE
        for it in range(1, MAX_ITER + 1):
            rs, dws = _rho_sum_reference(r, m_j, h, kernel)
            f = rs - m_i * (eta / h) ** 3
            df = dws + 3.0 * m_i * eta**3 / h**4
            if f > 0:
                hi = h
            else:
                lo = h
            h_new = h - f / df if df > 0 else 0.5 * (lo + hi)
            if not lo <= h_new <= hi:  # inclusive: a step landing on the bracket edge is valid
                h_new = 0.5 * (lo + hi)
            iters[i] = it
            if abs(h_new - h) < tol * h0:
                h = h_new
                flag = FLAG_OK
                break
            h = h_new
        h_lim = min(h_search[i], h_max)
        if h >= h_lim * (1 - 1e-3):  # near the limit: is there a root below it at all?
            rs_lim = _rho_sum_reference(r, m_j, h_lim, kernel)[0]
            if rs_lim - m_i * (eta / h_lim) ** 3 < 0:  # no root below the limit
                h = h_lim
                flag = FLAG_AT_H_MAX if h_lim >= h_max else FLAG_SEARCH_TOO_SMALL
        rs, dws = _rho_sum_reference(r, m_j, h, kernel)
        h_out[i], rho[i], flags[i] = h, rs, flag
        omega[i] = 1.0 + h / (3.0 * rs) * dws
    return DensityResult(h_out, rho, omega, iters, flags, nb)


# --------------------------------------------------------------------------- Numba


@numba.njit(cache=True)
def _sums(i, h, pos, mass, offset, index, box, kid, cnorm):
    """rho_sum(h) and sum_j m_j dW/dh for particle i (mirrors _rho_sum_reference)."""
    rs = 0.0
    dws = 0.0
    inv_h = 1.0 / h
    for k in range(offset[i], offset[i + 1]):
        j = index[k]
        d2 = 0.0
        for c in range(3):
            dx = pos[j, c] - pos[i, c]
            if box[c] > 0.0:
                dx -= box[c] * np.round(dx / box[c])
            d2 += dx * dx
        q = np.sqrt(d2) * inv_h
        if q < 2.0:
            fq = kernel_f(q, kid)
            rs += mass[j] * cnorm * inv_h**3 * fq
            dws += mass[j] * (-cnorm * inv_h**4 * (3.0 * fq + q * kernel_df(q, kid)))
    return rs, dws


@numba.njit(parallel=True, cache=True)
def _density_kernel(
    pos, mass, h_start, h_search, offset, index, box, kid, cnorm, eta, tol, h_max,
    h_out, rho, omega, iters, flags,
):  # fmt: skip
    for i in numba.prange(pos.shape[0]):
        m_i = mass[i]
        hi = min(h_search[i], h_max)
        lo = 0.0
        h = min(h_start[i], hi)
        h0 = h
        flag = 3
        it = 0
        while it < 50:
            it += 1
            rs, dws = _sums(i, h, pos, mass, offset, index, box, kid, cnorm)
            f = rs - m_i * (eta / h) ** 3
            df = dws + 3.0 * m_i * eta**3 / h**4
            if f > 0:
                hi = h
            else:
                lo = h
            h_new = h - f / df if df > 0 else 0.5 * (lo + hi)
            if not (lo <= h_new <= hi):  # inclusive (see density_reference)
                h_new = 0.5 * (lo + hi)
            if abs(h_new - h) < tol * h0:
                h = h_new
                flag = 0
                break
            h = h_new
        h_lim = min(h_search[i], h_max)
        if h >= h_lim * (1 - 1e-3):  # near the limit: is there a root below it at all?
            rs_lim, _ = _sums(i, h_lim, pos, mass, offset, index, box, kid, cnorm)
            if rs_lim - m_i * (eta / h_lim) ** 3 < 0:
                h = h_lim
                flag = 1 if h_lim >= h_max else 2
        rs, dws = _sums(i, h, pos, mass, offset, index, box, kid, cnorm)
        h_out[i] = h
        rho[i] = rs
        omega[i] = 1.0 + h / (3.0 * rs) * dws
        iters[i] = it
        flags[i] = flag


def density_numba(
    pos,
    mass,
    h_start,
    nb: Neighbours,
    h_search,
    kernel_name="cubic",
    eta=1.2,
    tol=1e-4,
    h_max=np.inf,
) -> DensityResult:
    """The Numba density pass on given neighbour lists (no list rebuilding)."""
    n = len(pos)
    h_out, rho, omega = np.empty(n), np.empty(n), np.empty(n)
    iters, flags = np.empty(n, np.int64), np.empty(n, np.int64)
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, float)
    kid = KERNEL_IDS[kernel_name]
    _density_kernel(
        np.ascontiguousarray(pos, float), np.ascontiguousarray(mass, float),
        np.ascontiguousarray(h_start, float), np.ascontiguousarray(h_search, float),
        nb.offset, nb.index, box, kid, C_NORM[kid], float(eta), float(tol), float(h_max),
        h_out, rho, omega, iters, flags,
    )  # fmt: skip
    return DensityResult(h_out, rho, omega, iters, flags, nb)


def compute_density(
    pos, mass, h_guess, box=None, kernel_name="cubic", eta=1.2, tol=1e-4, h_max=np.inf, skin=1.25
) -> DensityResult:
    """Density pass with automatic neighbour lists: search radius 2 h_guess x skin; particles
    whose h grows past it are redone with larger lists (at most 8 rounds)."""
    h_guess = np.asarray(h_guess, float)
    h_search = np.minimum(h_guess * skin, h_max)
    if box is not None:
        h_search = np.minimum(h_search, 0.2499 * np.min(np.broadcast_to(box, (3,))))
    for _ in range(8):
        nb = find_neighbours(pos, h_search, box=box)
        res = density_numba(pos, mass, h_guess, nb, h_search, kernel_name, eta, tol, h_max)
        grow = res.flags == FLAG_SEARCH_TOO_SMALL
        if not np.any(grow):
            return res
        h_search = h_search.copy()
        h_search[grow] = np.minimum(h_search[grow] * 1.5, h_max)
        h_guess = res.h
    raise RuntimeError("smoothing lengths kept outgrowing the neighbour search")


# --------------------------------------------------------------------------- forces
#
# Pressure force and P dV heating, no dissipation yet (Phantom Eqs. 34-35; Price 2012 §3):
#     dv_i/dt = - sum_j m_j [ P_i/(Omega_i rho_i^2) grad_i W_ij(h_i)
#                           + P_j/(Omega_j rho_j^2) grad_i W_ij(h_j) ]
#     du_i/dt =   P_i/(Omega_i rho_i^2) sum_j m_j (v_i - v_j) . grad_i W_ij(h_i)
# with grad_i W_ij(h) = r_hat_ij dW/dr(|r_ij|, h), r_ij = x_i - x_j. Pair terms are
# antisymmetric, so momentum, angular momentum and total energy sum(m (v^2/2 + u)) are
# conserved to round-off by the sums themselves.


@dataclass
class ForceResult:
    acc: np.ndarray  # (N, 3) hydro acceleration
    dudt: np.ndarray  # (N,)


def forces_reference(pos, vel, mass, h, rho, omega, pressure, nb: Neighbours, kernel_name="cubic"):
    """Readable reference of the pressure force and heating (one particle at a time)."""
    kernel = get_kernel(kernel_name)
    n = len(pos)
    acc, dudt = np.zeros((n, 3)), np.zeros(n)
    for i in range(n):
        j = nb.of(i)
        j = j[j != i]
        dx = minimum_image(pos[i] - pos[j], nb.box)
        r = np.linalg.norm(dx, axis=1)
        rhat = dx / r[:, None]
        gi = kernel.dw_dr(r, h[i])
        gj = kernel.dw_dr(r, h[j])
        pi = pressure[i] / (omega[i] * rho[i] ** 2)
        pj = pressure[j] / (omega[j] * rho[j] ** 2)
        acc[i] = -np.sum((mass[j] * (pi * gi + pj * gj))[:, None] * rhat, axis=0)
        vdotr = np.sum((vel[i] - vel[j]) * rhat, axis=1)
        dudt[i] = pi * np.sum(mass[j] * vdotr * gi)
    return ForceResult(acc, dudt)


@numba.njit(cache=True)
def _dwdr(r, h, kid, cnorm):
    """dW/dr; mirrors physics.kernels.Kernel.dw_dr."""
    return cnorm / h**4 * kernel_df(r / h, kid)


@numba.njit(parallel=True, cache=True)
def _forces_kernel(
    pos, vel, mass, h, rho, omega, pressure, offset, index, box, kid, cnorm, acc, dudt
):
    for i in numba.prange(pos.shape[0]):  # writes only to row i
        pi = pressure[i] / (omega[i] * rho[i] * rho[i])
        ax = 0.0
        ay = 0.0
        az = 0.0
        du = 0.0
        for k in range(offset[i], offset[i + 1]):
            j = index[k]
            if j == i:
                continue
            dx0 = pos[i, 0] - pos[j, 0]
            dx1 = pos[i, 1] - pos[j, 1]
            dx2 = pos[i, 2] - pos[j, 2]
            if box[0] > 0.0:
                dx0 -= box[0] * np.round(dx0 / box[0])
            if box[1] > 0.0:
                dx1 -= box[1] * np.round(dx1 / box[1])
            if box[2] > 0.0:
                dx2 -= box[2] * np.round(dx2 / box[2])
            r = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            rx = dx0 / r
            ry = dx1 / r
            rz = dx2 / r
            gi = _dwdr(r, h[i], kid, cnorm)
            gj = _dwdr(r, h[j], kid, cnorm)
            pj = pressure[j] / (omega[j] * rho[j] * rho[j])
            f = mass[j] * (pi * gi + pj * gj)
            ax -= f * rx
            ay -= f * ry
            az -= f * rz
            vdotr = (
                (vel[i, 0] - vel[j, 0]) * rx
                + (vel[i, 1] - vel[j, 1]) * ry
                + (vel[i, 2] - vel[j, 2]) * rz
            )
            du += mass[j] * vdotr * gi
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
        dudt[i] = pi * du


def forces_numba(
    pos, vel, mass, h, rho, omega, pressure, nb: Neighbours, kernel_name="cubic"
) -> ForceResult:
    """Fast (parallel Numba) pressure force and heating; twin of ``forces_reference``."""
    n = len(pos)
    acc, dudt = np.empty((n, 3)), np.empty(n)
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, float)
    kid = KERNEL_IDS[kernel_name]
    _forces_kernel(
        np.ascontiguousarray(pos, float), np.ascontiguousarray(vel, float),
        np.ascontiguousarray(mass, float), np.ascontiguousarray(h, float),
        np.ascontiguousarray(rho, float), np.ascontiguousarray(omega, float),
        np.ascontiguousarray(pressure, float), nb.offset, nb.index, box, kid, C_NORM[kid],
        acc, dudt,
    )  # fmt: skip
    return ForceResult(acc, dudt)
