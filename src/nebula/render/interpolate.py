"""Gas state at frame times between snapshots (phases/phase-08-gas-rendering.md 8.2).

Linear blending of positions cuts across the curve of gas swinging round the hole (at r = 6
gas sweeps 60 deg in a sixth of an orbit), so each particle is moved along its orbit instead:

- forward: from the snapshot before the frame time, with the gravity-only generalized
  leapfrog of Phase 1 (sim/integrator.py: the same Numba step ``_glf_one`` and the same
  quantized orbit steps eta r^(3/2) on the integer tick timeline as ``integrate_orbit``);
- backward: from the snapshot after, by time reversal (H(x, p) is even in p, so moving back by
  dt is moving (x, -p) forward by dt and flipping p back);
- positions follow the forward state and cross-fade to the backward state over the last
  ``FADE`` (20%) of each interval (smoothstep weight), so there is no jump at the next
  snapshot; the cross-fade blends length and direction separately (blend_vectors), so it
  never cuts through the hole; scalars (h, rho, u) are blended linearly between the two snapshots;
- particles missing from the next snapshot (swallowed) follow their orbit until they cross
  ``r_drop`` (the run's accretion radius) and are then dropped; a particle whose backward path
  crosses r_drop keeps its forward position.

Pressure is ignored between snapshots: where it matters (hot gas near the hole), forward and
backward disagree and the cross-fade hides the difference.
"""

from dataclasses import dataclass

import numba
import numpy as np

from nebula.sim.integrator import (
    _FAIL_LEVEL,
    _MESSAGES,
    DT_MAX_DEFAULT,
    K_MAX,
    GLFConvergenceError,
    _glf_one,
    _level,
)

ETA_ORB = 0.00125  # the runs' orbit criterion (sim/coupled.py, DECISIONS Step 1.4)
FADE = 0.2  # cross-fade over the last 20% of each snapshot interval (8.2)


@numba.njit(parallel=True, cache=True)
def _advance_kernel(x, p, n_end, dt_max, eta, r_drop, x_out, p_out, status):
    """Each particle forward by n_end ticks (tick = dt_max / 2^K_MAX) with the stepping of
    integrator._integrate_kernel (copied loop, no records). status: 0 ok, 1 dropped (crossed
    r_drop; state frozen there), < 0 a failure code."""
    for i in numba.prange(x.shape[0]):  # writes only to row i
        x0, x1, x2 = x[i, 0], x[i, 1], x[i, 2]
        p0, p1, p2 = p[i, 0], p[i, 1], p[i, 2]
        n = np.int64(0)
        st = 0
        while n < n_end:
            r = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
            if r < r_drop:
                st = 1
                break
            k = _level(r, eta, dt_max)
            while k <= K_MAX and (np.int64(1) << (K_MAX - k)) > n_end - n:
                k += 1  # never step past the end time
            if k > K_MAX:
                st = _FAIL_LEVEL
                break
            x0, x1, x2, p0, p1, p2, it = _glf_one(x0, x1, x2, p0, p1, p2, dt_max / 2.0**k)
            if it < 0:
                st = it
                break
            n += np.int64(1) << (K_MAX - k)
        if st == 0 and np.sqrt(x0 * x0 + x1 * x1 + x2 * x2) < r_drop:
            st = 1
        x_out[i, 0], x_out[i, 1], x_out[i, 2] = x0, x1, x2
        p_out[i, 0], p_out[i, 1], p_out[i, 2] = p0, p1, p2
        status[i] = st


def advance(x, p, dt: float, eta: float = ETA_ORB, r_drop: float = 4.0,
            dt_max: float = DT_MAX_DEFAULT):  # fmt: skip
    """Move particles along their gravity-only GN orbits by dt (code time; negative = back in
    time). Returns (x, p, alive): alive is False for particles that crossed r_drop. float64.

    Forward, this equals sim.integrator.integrate_orbit(x, p, dt, eta=eta) for each particle
    (test 8a)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    sign = 1.0 if dt >= 0 else -1.0
    tick = dt_max / 2.0**K_MAX
    n_end = np.int64(round(abs(dt) / tick))
    x_out, p_out = np.empty_like(x), np.empty_like(p)
    status = np.empty(len(x), dtype=np.int64)
    _advance_kernel(x, sign * p, n_end, dt_max, eta, r_drop, x_out, p_out, status)
    bad = status < 0
    if np.any(bad):
        raise GLFConvergenceError(f"{bad.sum()} particles: {_MESSAGES[int(status[bad][0])]}")
    return x_out, sign * p_out, status == 0


def smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def fade_weight(a: float) -> float:
    """Weight of the backward state at fraction a of the interval: 0 until 1 - FADE, then a
    smoothstep to 1 at a = 1."""
    return float(smoothstep((a - (1.0 - FADE)) / FADE))


def blend_vectors(u, v, w):
    """Cross-fade of 3-vectors u -> v (rows) by weight w (scalar or per row): the length
    blended linearly and the direction turned along the great circle (slerp). A straight blend
    of two positions on opposite sides of the hole cuts through it (R6.6 day 288.6: forward
    r = 35, backward r = 16, blend r = 1.28, inside the horizon; AI_LOG 2026-11-18); this one
    never comes closer than the nearer of the two. w = 0 gives u and w = 1 gives v exactly."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    w = np.broadcast_to(np.asarray(w, float).reshape(-1, 1) if np.ndim(w) else w, (len(u), 1))
    lu, lv = np.linalg.norm(u, axis=1, keepdims=True), np.linalg.norm(v, axis=1, keepdims=True)
    du, dv = u / np.maximum(lu, 1e-300), v / np.maximum(lv, 1e-300)
    om = np.arccos(np.clip(np.sum(du * dv, axis=1, keepdims=True), -1.0, 1.0))
    so = np.sin(om)
    ok = so > 1e-9
    ca = np.where(ok, np.sin((1.0 - w) * om) / np.where(ok, so, 1.0), 1.0 - w)
    cb = np.where(ok, np.sin(w * om) / np.where(ok, so, 1.0), w)
    d = ca * du + cb * dv
    flip = ~ok & (om > 1.0)  # exactly opposite: no unique great circle, take the nearer end
    d = np.where(flip, np.where(w < 0.5, du, dv), d)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    out = ((1.0 - w) * lu + w * lv) * d
    out = np.where(w <= 0.0, u, out)
    return np.where(w >= 1.0, v, out)


@dataclass
class Snap:
    """One snapshot's particles (float64): time t, ids, x, p, and scalars (dict of arrays)."""

    t: float
    ids: np.ndarray
    x: np.ndarray
    p: np.ndarray
    scalars: dict


def load_snap(path, scalars=("h", "rho", "u")) -> Snap:
    d = np.load(path)
    return Snap(
        float(d["t"]), d["ids"].astype(np.int64), d["x"].astype(float), d["p"].astype(float),
        {k: d[k].astype(float) for k in scalars},
    )  # fmt: skip


def state_at(
    a: Snap, b: Snap, t: float, eta: float = ETA_ORB, r_drop: float = 4.0, with_p: bool = False
):
    """Gas at time t between snapshots a and b (a.t <= t <= b.t): (ids, x, scalars), float64;
    with ``with_p`` (Phase 9: the gas velocity for Doppler colours) (ids, x, p, scalars), the
    momenta advanced and cross-faded exactly like the positions.

    t = a.t returns snapshot a exactly; t = b.t returns b's positions for every particle still
    present in b."""
    if not a.t <= t <= b.t:
        raise ValueError(f"t = {t} not between {a.t} and {b.t}")
    frac = (t - a.t) / (b.t - a.t)
    w = fade_weight(frac)
    xf, pf, alive = advance(a.x, a.p, t - a.t, eta, r_drop)
    pos = {k: j for j, k in enumerate(b.ids)}
    jb = np.array([pos.get(k, -1) for k in a.ids])
    both = jb >= 0
    x = xf
    p = pf
    scal = {k: v.copy() for k, v in a.scalars.items()}
    for k in scal:
        scal[k][both] = (1.0 - frac) * a.scalars[k][both] + frac * b.scalars[k][jb[both]]
    if w > 0.0:
        xb, pb, alive_b = advance(b.x[jb[both]], b.p[jb[both]], t - b.t, eta, r_drop)
        idx = np.nonzero(both)[0]
        use = alive_b
        # a forward path that fell in (frozen at r_drop) is not a place the particle was: the
        # straight blend to the backward state on the far side would cut through the hole
        # (r = 1.28 at day 288.6 of R6.6, NaN velocities inside r = 2; AI_LOG 2026-11-18), so
        # such particles take the backward state alone
        wu = np.where(alive[idx[use]], w, 1.0)
        x = x.copy()
        x[idx[use]] = blend_vectors(xf[idx[use]], xb[use], wu)
        p = p.copy()
        p[idx[use]] = blend_vectors(pf[idx[use]], pb[use], wu)
        alive[idx[use]] = True  # present in b: shown even if its forward path fell in
    keep = alive
    if with_p:
        return a.ids[keep], x[keep], p[keep], {k: v[keep] for k, v in scal.items()}
    return a.ids[keep], x[keep], {k: v[keep] for k, v in scal.items()}
