"""PROTOTYPE (review 2026-10-15): artificial viscosity with slope-limited linear velocity
reconstruction to the pair midpoint (Frontiere, Raskin & Owen 2017, J. Comput. Phys. 332, 160,
after Christensen 1990; Rosswog 2020, MNRAS 498, 4230 "MAGMA2", Eqs. 17-18, which adds a
second-order term and a closeness factor). Not part of the repo: installed by monkey-patching
``forces_numba`` in nebula.sim.blocks, nebula.sim.coupled and nebula.sim.hydro (and wrapping
blocks._density_touched so that, on block steps, the neighbours' gradients are recomputed at the
current time, like their densities).

STORE["mode"] (tests of the block-step angular-momentum check, review section D2):
  0 = limiter + closeness factor (as published)   1 = closeness factor only (Phi = 1)
  2 = neither (Phi = 1 always; not shock-safe)    3 = limiter only
  4 = limiter squared (smooth at A = 0), no closeness factor
  5 = limiter only, plus a continuous gate: q is scaled by min(1, v_ab.r_hat / v*_ab.r_hat), so it
      fades to zero as the actual approach speed does instead of switching off at once
  6 = 4 and 5 together
  Block-step blob, angular-momentum fall per halving (needs >= 3): mode 0: 1.9x, 1: 2.3x, 2: 6.7x,
  3: 2.6x, 4: 1.4x; mode 5 ran about ten times slower and was stopped; mode 6 was not run.

For a pair a, b with dx = x_a - x_b (midpoint x_a - dx/2):
    v_a* = v_a - (Phi/2) (dv_a . dx),   v_b* = v_b + (Phi/2) (dv_b . dx)
    dv_a = corrected velocity gradient (same matrix-corrected estimate as the switch, exact for
    linear fields); Phi = van Leer limiter max(0, min(1, 4A/(1+A)^2)),
    A = (dx . dv_a . dx) / (dx . dv_b . dx); and, for pairs closer than the typical particle
    spacing (eta_ab = r/h < ETA_CRIT = 1/hfact), Phi *= exp(-((eta_ab - ETA_CRIT)/0.2)^2).
The viscous pressure q uses (v_a* - v_b*) . r_hat in place of v_ab . r_hat (signal speed too);
the viscous heating is q_a/(Omega_a rho_a^2) v_ab . r_hat F_ab (actual v_ab), so the energy
ledger stays exact; q acts only when both v_ab . r_hat and the reconstructed one are < 0, so
the heating is never negative. Smooth (linear) flow: v_a* - v_b* = 0 -> no dissipation. Shock or
crossing: the gradients disagree, Phi -> 0 -> standard Phantom viscosity.
Courant signal speed and conductivity use the actual v_ab . r_hat (unchanged)."""

import numba
import numpy as np

from nebula.sim import blocks, coupled, hydro
from nebula.sim.kernels_nb import C_NORM, KERNEL_IDS, kernel_df
from nebula.sim.sph import DEFAULT_SHOCK, ForceResult, _rows

STORE = {"dv": None, "eta_crit": 1.0 / 1.6, "eta_fold": 0.2, "calls": 0, "st": None, "fresh": True, "mode": 0}


@numba.njit(cache=False)
def _dwdr(r, h, kid, cnorm):
    return cnorm / h**4 * kernel_df(r / h, kid)


@numba.njit(parallel=True, cache=False)
def _dv_kernel(pos, vel, mass, h, offset, index, box, kid, cnorm, rows, dvout):
    for q in numba.prange(rows.shape[0]):
        a = rows[q]
        rmat = np.zeros((3, 3))
        gv = np.zeros((3, 3))
        dx = np.empty(3)
        cnt = 0
        for k in range(offset[a], offset[a + 1]):
            b = index[k]
            if b == a:
                continue
            cnt += 1
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
        det = np.linalg.det(rmat)
        scale = (rmat[0, 0] + rmat[1, 1] + rmat[2, 2]) / 3.0
        if cnt < 4 or not np.isfinite(det) or abs(det) < 1e-8 * abs(scale) ** 3:
            dvout[a] = 0.0  # no reliable gradient: Phi -> 0, plain viscosity
        else:
            dvout[a] = gv @ np.linalg.inv(rmat)


@numba.njit(parallel=True, cache=False)
def _forces_recon_kernel(
    pos, vel, mass, h, rho, omega, pressure, u, c_s, alpha, offset, index, box, kid, cnorm,
    use_visc, use_cond, beta, alpha_u, cond_signal, rows, dv, eta_crit, eta_fold, mode,
    acc, dudt, vsig_max,
):  # fmt: skip
    for q in numba.prange(rows.shape[0]):
        a = rows[q]
        inv_or_a = 1.0 / (omega[a] * rho[a])
        ax = 0.0
        ay = 0.0
        az = 0.0
        work = 0.0
        heat = 0.0
        vmax = c_s[a] if use_visc else 0.0
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
            rx = dx[0] / r
            ry = dx[1] / r
            rz = dx[2] / r
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
                # reconstruction to the midpoint
                da0 = dv[a, 0, 0] * dx[0] + dv[a, 0, 1] * dx[1] + dv[a, 0, 2] * dx[2]
                da1 = dv[a, 1, 0] * dx[0] + dv[a, 1, 1] * dx[1] + dv[a, 1, 2] * dx[2]
                da2 = dv[a, 2, 0] * dx[0] + dv[a, 2, 1] * dx[1] + dv[a, 2, 2] * dx[2]
                db0 = dv[b, 0, 0] * dx[0] + dv[b, 0, 1] * dx[1] + dv[b, 0, 2] * dx[2]
                db1 = dv[b, 1, 0] * dx[0] + dv[b, 1, 1] * dx[1] + dv[b, 1, 2] * dx[2]
                db2 = dv[b, 2, 0] * dx[0] + dv[b, 2, 1] * dx[1] + dv[b, 2, 2] * dx[2]
                pa = da0 * dx[0] + da1 * dx[1] + da2 * dx[2]
                pb = db0 * dx[0] + db1 * dx[1] + db2 * dx[2]
                phi = 0.0
                if pb != 0.0:
                    aa = pa / pb
                    if aa > 0.0:
                        phi = min(1.0, 4.0 * aa / ((1.0 + aa) * (1.0 + aa)))
                        if mode == 4 or mode == 6:  # smooth limiter (C1 at A = 0)
                            phi = phi * phi
                if mode == 1 or mode == 2:  # tests: no limiter
                    phi = 1.0
                eta_ab = min(r / h[a], r / h[b])
                if eta_ab < eta_crit and mode < 2:  # mode 2, 3: no closeness factor
                    t = (eta_ab - eta_crit) / eta_fold
                    phi *= np.exp(-t * t)
                vdr = (
                    (vel[a, 0] - 0.5 * phi * da0 - vel[b, 0] - 0.5 * phi * db0) * rx
                    + (vel[a, 1] - 0.5 * phi * da1 - vel[b, 1] - 0.5 * phi * db1) * ry
                    + (vel[a, 2] - 0.5 * phi * da2 - vel[b, 2] - 0.5 * phi * db2) * rz
                )
                vsig_a = alpha[a] * c_s[a] + beta * abs(vdr)
                vsig_b = alpha[b] * c_s[b] + beta * abs(vdr)
                vs_cfl = alpha[a] * c_s[a] + beta * abs(vdotr)
                if vs_cfl > vmax:
                    vmax = vs_cfl
                if vdr < 0.0 and vdotr < 0.0:  # approaching both ways: heating >= 0
                    g = 1.0
                    if mode >= 5 and vdotr > vdr:  # continuous gate: -> 0 as v_ab.r_hat -> 0
                        g = vdotr / vdr
                    num_a += -0.5 * rho[a] * vsig_a * vdr * g
                    num_b += -0.5 * rho[b] * vsig_b * vdr * g
                    heat += mass[b] * (-0.5 * vsig_a * vdr * g) * vdotr * fa * inv_or_a
                if use_cond:
                    if cond_signal == 0:
                        vu = np.sqrt(abs(pressure[a] - pressure[b]) / (0.5 * (rho[a] + rho[b])))
                    else:
                        vu = abs(vdotr)
                    heat += (
                        mass[b] * alpha_u * vu * (u[a] - u[b]) * 0.5
                        * (fa * inv_or_a + fb / (omega[b] * rho[b]))
                    )  # fmt: skip
            f = mass[b] * (
                num_a / (omega[a] * rho[a] * rho[a]) * fa
                + num_b / (omega[b] * rho[b] * rho[b]) * fb
            )
            ax -= f * rx
            ay -= f * ry
            az -= f * rz
            work += mass[b] * vdotr * fa
        acc[q, 0] = ax
        acc[q, 1] = ay
        acc[q, 2] = az
        dudt[q] = pressure[a] * inv_or_a / rho[a] * work + heat
        vsig_max[q] = vmax


def forces_recon(
    pos, vel, mass, h, rho, omega, pressure, nb, kernel_name="wendland_c4",
    u=None, c_s=None, alpha=None, shock=DEFAULT_SHOCK, rows=None,
):  # fmt: skip
    n = len(pos)
    rows = _rows(rows, n)
    if STORE["dv"] is None or len(STORE["dv"]) != n:
        STORE["dv"] = np.zeros((n, 3, 3))
    box = np.zeros(3) if nb.box is None else np.asarray(nb.box, float)
    kid = KERNEL_IDS[kernel_name]
    pos = np.ascontiguousarray(pos, float)
    vel = np.ascontiguousarray(vel, float)
    mass = np.ascontiguousarray(mass, float)
    h = np.ascontiguousarray(h, float)
    st = STORE["st"]
    if STORE["fresh"] and st is not None and len(st.x) == n and st.neighbour_cache is not None:
        # block steps: gradients of the rows AND their neighbours at the current time (stale
        # neighbour gradients make the pair reconstruction differ between a's and b's steps)
        full = st.neighbour_cache.neighbours
        touched = blocks._with_neighbours(nb, rows, n)
        _dv_kernel(pos, vel, mass, h, full.offset, full.index, box, kid, C_NORM[kid], touched, STORE["dv"])
    else:
        _dv_kernel(pos, vel, mass, h, nb.offset, nb.index, box, kid, C_NORM[kid], rows, STORE["dv"])
    acc, dudt, vsig_max = np.empty((len(rows), 3)), np.empty(len(rows)), np.empty(len(rows))
    use_visc = alpha is not None and c_s is not None
    use_cond = use_visc and u is not None
    zeros = np.zeros(n)

    def arr(x):
        return zeros if x is None else np.ascontiguousarray(x, float)

    _forces_recon_kernel(
        pos, vel, mass, h, np.ascontiguousarray(rho, float), np.ascontiguousarray(omega, float),
        np.ascontiguousarray(pressure, float), arr(u), arr(c_s), arr(alpha),
        nb.offset, nb.index, box, kid, C_NORM[kid], use_visc, use_cond,
        float(shock.beta), float(shock.alpha_u), int(shock.conductivity_signal), rows,
        STORE["dv"], STORE["eta_crit"], STORE["eta_fold"], STORE["mode"], acc, dudt, vsig_max,
    )  # fmt: skip
    STORE["calls"] += 1
    return ForceResult(acc, dudt, vsig_max)


_density_touched_orig = blocks._density_touched


def _density_touched_tracking(st, hc, rows, box):
    STORE["st"] = st
    return _density_touched_orig(st, hc, rows, box)


def install(eta_crit=1.0 / 1.6, fresh=True):
    STORE["eta_crit"] = eta_crit
    STORE["fresh"] = fresh
    blocks._density_touched = _density_touched_tracking
    blocks.forces_numba = forces_recon
    hydro.forces_numba = forces_recon
    coupled.forces_numba = forces_recon


def prune(keep):
    if STORE["dv"] is not None and len(STORE["dv"]) == len(keep):
        STORE["dv"] = STORE["dv"][keep]
