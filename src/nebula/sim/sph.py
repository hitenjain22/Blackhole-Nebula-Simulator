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

from nebula.physics.kernels import DEFAULT_ETA, DEFAULT_KERNEL, get_kernel
from nebula.sim.kernels_nb import C_NORM, KERNEL_IDS, kernel_df, kernel_f
from nebula.sim.neighbours import Neighbours, find_neighbours_numba, minimum_image

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
    cache: "NeighbourCache | None" = None


@dataclass
class NeighbourCache:
    """Neighbour lists kept between steps ("Verlet lists"): built at positions x_ref with search
    smoothing lengths h_search, so they hold every pair with r < 2 max(hs_i, hs_j) then.

    Validity (exact): if no particle has moved more than d_max since, any pair that now has
    r < 2 max(h_i, h_j) had r_build < 2 max(h_i + d_max, h_j + d_max); so the lists are
    complete for every particle with h_i <= hs_i - d_max. That bound is passed to the density
    solver as its search limit; a particle that needs more is flagged and the lists rebuilt.
    """

    neighbours: Neighbours
    x_ref: np.ndarray
    h_search: np.ndarray
    builds: int = 1
    reuses: int = 0


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
    kernel_name=DEFAULT_KERNEL,
    eta=DEFAULT_ETA,
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
def _sums(r, m, cnt, h, kid, cnorm):
    """rho_sum(h) and sum_j m_j dW/dh from precomputed neighbour distances r[:cnt] and masses
    m[:cnt] (mirrors _rho_sum_reference)."""
    rs = 0.0
    dws = 0.0
    inv_h = 1.0 / h
    for k in range(cnt):
        q = r[k] * inv_h
        if q < 2.0:
            fq = kernel_f(q, kid)
            rs += m[k] * cnorm * inv_h**3 * fq
            dws += m[k] * (-cnorm * inv_h**4 * (3.0 * fq + q * kernel_df(q, kid)))
    return rs, dws


@numba.njit(cache=True)
def _distances(i, r_max, pos, mass, offset, index, box, r, m):
    """Distances (minimum image) and masses of i's list entries closer than r_max, into r, m;
    returns how many. Done once per particle: the Newton iterations only re-evaluate W."""
    cnt = 0
    for k in range(offset[i], offset[i + 1]):
        j = index[k]
        d2 = 0.0
        for c in range(3):
            dx = pos[j, c] - pos[i, c]
            if box[c] > 0.0:
                dx -= box[c] * np.round(dx / box[c])
            d2 += dx * dx
        if d2 < r_max * r_max:
            r[cnt] = np.sqrt(d2)
            m[cnt] = mass[j]
            cnt += 1
    return cnt


@numba.njit(parallel=True, cache=True)
def _density_kernel(
    pos, mass, h_start, h_search, offset, index, box, kid, cnorm, eta, tol, h_max,
    h_out, rho, omega, iters, flags,
):  # fmt: skip
    for i in numba.prange(pos.shape[0]):
        m_i = mass[i]
        row = offset[i + 1] - offset[i]
        r = np.empty(row)
        m = np.empty(row)
        cnt = _distances(i, 2.0 * min(h_search[i], h_max), pos, mass, offset, index, box, r, m)
        hi = min(h_search[i], h_max)
        lo = 0.0
        h = min(h_start[i], hi)
        h0 = h
        flag = 3
        it = 0
        while it < 50:
            it += 1
            rs, dws = _sums(r, m, cnt, h, kid, cnorm)
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
            rs_lim, _ = _sums(r, m, cnt, h_lim, kid, cnorm)
            if rs_lim - m_i * (eta / h_lim) ** 3 < 0:
                h = h_lim
                flag = 1 if h_lim >= h_max else 2
        rs, dws = _sums(r, m, cnt, h, kid, cnorm)
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
    kernel_name=DEFAULT_KERNEL,
    eta=DEFAULT_ETA,
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
    pos,
    mass,
    h_guess,
    box=None,
    kernel_name=DEFAULT_KERNEL,
    eta=DEFAULT_ETA,
    tol=1e-4,
    h_max=np.inf,
    skin=1.4,
    cache: NeighbourCache | None = None,
) -> DensityResult:
    """Density pass with neighbour lists. Lists are built with search radius 2 h_guess x skin,
    or reused from ``cache`` while still provably complete (see NeighbourCache); particles whose
    h outgrows the search are redone with rebuilt lists, doubling their search (up to 12
    rounds). The result carries the cache to pass to the next call."""
    h_guess = np.asarray(h_guess, float)
    box_arr = None if box is None else np.broadcast_to(np.asarray(box, float), (3,))
    cap = h_max if box_arr is None else min(h_max, 0.2499 * float(np.min(box_arr)))
    if cache is not None and len(cache.x_ref) == len(pos):
        d = minimum_image(np.asarray(pos, float) - cache.x_ref, box_arr)
        d_max = float(np.sqrt(np.max(np.sum(d * d, axis=1)))) if len(pos) else 0.0
        h_eff = cache.h_search - d_max
        if np.all(h_eff > 0.5 * h_guess):
            res = density_numba(
                pos, mass, h_guess, cache.neighbours, h_eff, kernel_name, eta, tol, h_max
            )
            if not np.any(res.flags == FLAG_SEARCH_TOO_SMALL):
                cache.reuses += 1
                res.cache = cache
                return res
            h_guess = np.maximum(h_guess, res.h)
    builds = 0 if cache is None else cache.builds
    h_search = np.minimum(h_guess * skin, cap)
    for _ in range(12):
        nb = find_neighbours_numba(pos, h_search, box=box_arr)
        builds += 1
        res = density_numba(pos, mass, h_guess, nb, h_search, kernel_name, eta, tol, h_max)
        grow = res.flags == FLAG_SEARCH_TOO_SMALL
        if not np.any(grow):
            reuses = 0 if cache is None else cache.reuses
            res.cache = NeighbourCache(nb, np.array(pos, float), h_search.copy(), builds, reuses)
            return res
        h_search = h_search.copy()
        h_search[grow] = np.minimum(h_search[grow] * 2.0, cap)
        h_guess = res.h
    raise RuntimeError("smoothing lengths kept outgrowing the neighbour search")


# --------------------------------------------------------------------------- forces
#
# Pressure force, P dV heating (Phantom Eqs. 34-35; Price 2012 §3) and, optionally, shock
# capturing (Eqs. 39-43; formulas in physics/viscosity.py):
#     dv_a/dt = - sum_b m_b [ (P_a + q^a_ab)/(Omega_a rho_a^2) F_ab(h_a)
#                           + (P_b + q^b_ab)/(Omega_b rho_b^2) F_ab(h_b) ] r_hat_ab
#     du_a/dt =   P_a/(Omega_a rho_a^2) sum_b m_b (v_ab . r_hat_ab) F_ab(h_a)
#               - 1/(Omega_a rho_a) sum_b m_b v_sig,a 1/2 (v_ab . r_hat)^2 F_ab(h_a)   [approaching]
#               + sum_b m_b alpha_u v^u_sig (u_a - u_b) 1/2 [F_ab(h_a)/(Omega_a rho_a)
#                                                          + F_ab(h_b)/(Omega_b rho_b)]
# with F_ab(h) = dW/dr(|r_ab|, h) and r_ab = x_a - x_b. Pair terms are antisymmetric (and the
# shock heating matches the viscous work), so momentum, angular momentum and the total
# energy sum(m (v^2/2 + u)) are conserved to round-off by the sums themselves.

COND_PRESSURE = 0  # v^u_sig = sqrt(|P_a - P_b| / rho_ab)   (Price 2008; Phantom Eq. 43)
COND_VELOCITY = 1  # v^u_sig = |v_ab . r_hat_ab|          (Phantom Eq. 44, with external forces)


@dataclass(frozen=True)
class ShockCapturing:
    """Shock-capturing settings. ``None`` arrays passed to the force pass switch terms off."""

    beta: float = 2.0  # physics.viscosity.BETA_AV
    alpha_u: float = 1.0  # physics.viscosity.ALPHA_U
    conductivity_signal: int = COND_PRESSURE


DEFAULT_SHOCK = ShockCapturing()


@dataclass
class ForceResult:
    acc: np.ndarray  # (N, 3) hydro acceleration
    dudt: np.ndarray  # (N,)
    vsig_max: np.ndarray  # (N,) largest signal speed with any neighbour (Courant condition)


def forces_reference(
    pos, vel, mass, h, rho, omega, pressure, nb: Neighbours, kernel_name=DEFAULT_KERNEL,
    u=None, c_s=None, alpha=None, shock: ShockCapturing = DEFAULT_SHOCK,
) -> ForceResult:  # fmt: skip
    """Readable reference (one particle at a time). Viscosity is on when ``alpha`` and ``c_s``
    are given; conductivity when ``u`` is given as well."""
    from nebula.physics import viscosity as visc

    kernel = get_kernel(kernel_name)
    n = len(pos)
    acc, dudt, vsig_max = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    use_visc = alpha is not None and c_s is not None
    use_cond = use_visc and u is not None
    for a in range(n):
        b = nb.of(a)
        b = b[b != a]
        dx = minimum_image(pos[a] - pos[b], nb.box)
        r = np.linalg.norm(dx, axis=1)
        rhat = dx / r[:, None]
        fa = kernel.dw_dr(r, h[a])
        fb = kernel.dw_dr(r, h[b])
        vdotr = np.sum((vel[a] - vel[b]) * rhat, axis=1)
        num_a, num_b = np.full(len(b), pressure[a]), pressure[b].astype(float)
        heat = pressure[a] / (omega[a] * rho[a] ** 2) * np.sum(mass[b] * vdotr * fa)
        if use_visc:
            vsig_a = visc.viscous_signal_speed(alpha[a], c_s[a], vdotr, shock.beta)
            vsig_b = visc.viscous_signal_speed(alpha[b], c_s[b], vdotr, shock.beta)
            num_a = num_a + visc.viscous_q(rho[a], vsig_a, vdotr)
            num_b = num_b + visc.viscous_q(rho[b], vsig_b, vdotr)
            approaching = vdotr < 0
            heat -= np.sum((mass[b] * vsig_a * 0.5 * vdotr**2 * fa)[approaching]) / (
                omega[a] * rho[a]
            )
            vsig_max[a] = max(np.max(vsig_a), c_s[a]) if len(b) else c_s[a]
            if use_cond:
                if shock.conductivity_signal == COND_PRESSURE:
                    vu = visc.conductivity_signal_speed(pressure[a], pressure[b], rho[a], rho[b])
                else:
                    vu = np.abs(vdotr)
                heat += np.sum(
                    mass[b] * shock.alpha_u * vu * (u[a] - u[b])
                    * 0.5 * (fa / (omega[a] * rho[a]) + fb / (omega[b] * rho[b]))
                )  # fmt: skip
        pair = mass[b] * (
            num_a / (omega[a] * rho[a] ** 2) * fa + num_b / (omega[b] * rho[b] ** 2) * fb
        )
        acc[a] = -np.sum(pair[:, None] * rhat, axis=0)
        dudt[a] = heat
    return ForceResult(acc, dudt, vsig_max)


@numba.njit(cache=True)
def _dwdr(r, h, kid, cnorm):
    """dW/dr; mirrors physics.kernels.Kernel.dw_dr."""
    return cnorm / h**4 * kernel_df(r / h, kid)


@numba.njit(parallel=True, cache=True)
def _forces_kernel(
    pos, vel, mass, h, rho, omega, pressure, u, c_s, alpha, offset, index, box, kid, cnorm,
    use_visc, use_cond, beta, alpha_u, cond_signal, acc, dudt, vsig_max,
):  # fmt: skip
    for a in numba.prange(pos.shape[0]):  # writes only to row a
        inv_or_a = 1.0 / (omega[a] * rho[a])
        ax = 0.0
        ay = 0.0
        az = 0.0
        work = 0.0
        heat = 0.0
        vmax = c_s[a] if use_visc else 0.0
        for k in range(offset[a], offset[a + 1]):
            b = index[k]
            if b == a:
                continue
            dx0 = pos[a, 0] - pos[b, 0]
            dx1 = pos[a, 1] - pos[b, 1]
            dx2 = pos[a, 2] - pos[b, 2]
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
            fa = _dwdr(r, h[a], kid, cnorm)
            fb = _dwdr(r, h[b], kid, cnorm)
            vdotr = (
                (vel[a, 0] - vel[b, 0]) * rx
                + (vel[a, 1] - vel[b, 1]) * ry
                + (vel[a, 2] - vel[b, 2]) * rz
            )
            num_a = pressure[a]
            num_b = pressure[b]
            if use_visc:
                # mirrors physics.viscosity.viscous_signal_speed and viscous_q
                vsig_a = alpha[a] * c_s[a] + beta * abs(vdotr)
                vsig_b = alpha[b] * c_s[b] + beta * abs(vdotr)
                if vsig_a > vmax:
                    vmax = vsig_a
                if vdotr < 0.0:
                    num_a += -0.5 * rho[a] * vsig_a * vdotr
                    num_b += -0.5 * rho[b] * vsig_b * vdotr
                    heat -= mass[b] * vsig_a * 0.5 * vdotr * vdotr * fa * inv_or_a
                if use_cond:
                    # mirrors physics.viscosity.conductivity_signal_speed (or |v_ab . r_hat|)
                    if cond_signal == 0:
                        vu = np.sqrt(abs(pressure[a] - pressure[b]) / (0.5 * (rho[a] + rho[b])))
                    else:
                        vu = abs(vdotr)
                    heat += (
                        mass[b]
                        * alpha_u
                        * vu
                        * (u[a] - u[b])
                        * 0.5
                        * (fa * inv_or_a + fb / (omega[b] * rho[b]))
                    )
            f = mass[b] * (
                num_a / (omega[a] * rho[a] * rho[a]) * fa
                + num_b / (omega[b] * rho[b] * rho[b]) * fb
            )
            ax -= f * rx
            ay -= f * ry
            az -= f * rz
            work += mass[b] * vdotr * fa
        acc[a, 0] = ax
        acc[a, 1] = ay
        acc[a, 2] = az
        dudt[a] = pressure[a] * inv_or_a / rho[a] * work + heat
        vsig_max[a] = vmax


def forces_numba(
    pos, vel, mass, h, rho, omega, pressure, nb: Neighbours, kernel_name=DEFAULT_KERNEL,
    u=None, c_s=None, alpha=None, shock: ShockCapturing = DEFAULT_SHOCK,
) -> ForceResult:  # fmt: skip
    """Fast (parallel Numba) twin of ``forces_reference``."""
    n = len(pos)
    acc, dudt, vsig_max = np.empty((n, 3)), np.empty(n), np.empty(n)
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, float)
    kid = KERNEL_IDS[kernel_name]
    use_visc = alpha is not None and c_s is not None
    use_cond = use_visc and u is not None
    zeros = np.zeros(n)

    def arr(x):
        return zeros if x is None else np.ascontiguousarray(x, float)

    _forces_kernel(
        np.ascontiguousarray(pos, float), np.ascontiguousarray(vel, float),
        np.ascontiguousarray(mass, float), np.ascontiguousarray(h, float),
        np.ascontiguousarray(rho, float), np.ascontiguousarray(omega, float),
        np.ascontiguousarray(pressure, float), arr(u), arr(c_s), arr(alpha),
        nb.offset, nb.index, box, kid, C_NORM[kid], use_visc, use_cond,
        float(shock.beta), float(shock.alpha_u), int(shock.conductivity_signal),
        acc, dudt, vsig_max,
    )  # fmt: skip
    return ForceResult(acc, dudt, vsig_max)


# --------------------------------------------------------------------------- gradients
#
# Velocity and acceleration gradients for the shock switch (Phantom Eqs. 51-54, after
# Cullen & Dehnen 2010 App. B3), exact for linear fields thanks to the matrix correction:
#     R_lj = sum_b m_b (x_b - x_a)^l dW_ab(h_a)/dx_a^j          (~ identity)
#     G_kj = sum_b m_b (v_b - v_a)^k dW_ab(h_a)/dx_a^j  ->  dv^k/dx^j = (G R^-1)_kj
# with dW_ab/dx_a = r_hat_ab F_ab(h_a). Then div v = trace, curl v from the antisymmetric
# part, and d(div v)/dt = div a - sum_ij (dv^i/dx^j)(dv^j/dx^i) using the previous step's
# accelerations a.


@dataclass
class Gradients:
    div_v: np.ndarray
    curl_v: np.ndarray  # magnitude |curl v|
    ddivv_dt: np.ndarray


def _gradient_scalars(dv: np.ndarray, da: np.ndarray) -> tuple[float, float, float]:
    """(div v, |curl v|, d(div v)/dt) from gradient matrices dv[k, j] = dv^k/dx^j and da."""
    curl = np.array([dv[2, 1] - dv[1, 2], dv[0, 2] - dv[2, 0], dv[1, 0] - dv[0, 1]])
    return float(np.trace(dv)), float(np.linalg.norm(curl)), float(np.trace(da) - np.sum(dv * dv.T))


def gradients_reference(
    pos, vel, acc, mass, h, nb: Neighbours, kernel_name=DEFAULT_KERNEL
) -> Gradients:
    """Readable reference of the corrected velocity / acceleration gradients."""
    kernel = get_kernel(kernel_name)
    n = len(pos)
    div, curl, ddiv = np.empty(n), np.empty(n), np.empty(n)
    for a in range(n):
        b = nb.of(a)
        b = b[b != a]
        dx = minimum_image(pos[a] - pos[b], nb.box)  # x_a - x_b
        r = np.linalg.norm(dx, axis=1)
        grad = (kernel.dw_dr(r, h[a]) / r)[:, None] * dx  # dW_ab(h_a)/dx_a
        m = mass[b][:, None, None]
        rmat = np.sum(m * (-dx)[:, :, None] * grad[:, None, :], axis=0)
        gv = np.sum(m * (vel[b] - vel[a])[:, :, None] * grad[:, None, :], axis=0)
        ga = np.sum(m * (acc[b] - acc[a])[:, :, None] * grad[:, None, :], axis=0)
        rinv = np.linalg.inv(rmat)
        div[a], curl[a], ddiv[a] = _gradient_scalars(gv @ rinv, ga @ rinv)
    return Gradients(div, curl, ddiv)


@numba.njit(parallel=True, cache=True)
def _gradients_kernel(pos, vel, acc, mass, h, offset, index, box, kid, cnorm, div, curl, ddiv):
    for a in numba.prange(pos.shape[0]):  # writes only to row a
        rmat = np.zeros((3, 3))
        gv = np.zeros((3, 3))
        ga = np.zeros((3, 3))
        dx = np.empty(3)
        for k in range(offset[a], offset[a + 1]):
            b = index[k]
            if b == a:
                continue
            for c in range(3):
                d = pos[a, c] - pos[b, c]
                if box[c] > 0.0:
                    d -= box[c] * np.round(d / box[c])
                dx[c] = d
            r = np.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
            g = _dwdr(r, h[a], kid, cnorm) / r
            for i in range(3):
                for j in range(3):
                    gj = mass[b] * g * dx[j]
                    rmat[i, j] -= dx[i] * gj
                    gv[i, j] += (vel[b, i] - vel[a, i]) * gj
                    ga[i, j] += (acc[b, i] - acc[a, i]) * gj
        rinv = np.linalg.inv(rmat)
        dv = gv @ rinv
        da = ga @ rinv
        div[a] = dv[0, 0] + dv[1, 1] + dv[2, 2]
        c0 = dv[2, 1] - dv[1, 2]
        c1 = dv[0, 2] - dv[2, 0]
        c2 = dv[1, 0] - dv[0, 1]
        curl[a] = np.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
        s = 0.0
        for i in range(3):
            for j in range(3):
                s += dv[i, j] * dv[j, i]
        ddiv[a] = da[0, 0] + da[1, 1] + da[2, 2] - s


def gradients_numba(
    pos, vel, acc, mass, h, nb: Neighbours, kernel_name=DEFAULT_KERNEL
) -> Gradients:
    """Fast twin of ``gradients_reference``."""
    n = len(pos)
    div, curl, ddiv = np.empty(n), np.empty(n), np.empty(n)
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, float)
    kid = KERNEL_IDS[kernel_name]
    _gradients_kernel(
        np.ascontiguousarray(pos, float), np.ascontiguousarray(vel, float),
        np.ascontiguousarray(acc, float), np.ascontiguousarray(mass, float),
        np.ascontiguousarray(h, float), nb.offset, nb.index, box, kid, C_NORM[kid], div, curl, ddiv,
    )  # fmt: skip
    return Gradients(div, curl, ddiv)
