"""Easing the joins between source runs (render.junctions; DECISIONS 2026-11-17).

Two synthetic runs of the same particles on orbits at r ~ 1,000: the new run is the old one
carried to t1 and then shifted by a fixed offset (5 units, momenta by 1e-4, density and the
other scalars x 2). The eased state must equal the old run before the window, reach the new
run at t1, apply half the offset at mid-window, and move smoothly (no frame jump)."""

import numpy as np

from nebula.render.interpolate import Snap, advance, state_at
from nebula.render.junctions import eased_state, make_junction
from tests.helpers import check

N = 200
T = 100.0  # code time between the old run's snapshots
DELTA = np.array([5.0, -3.0, 1.0])  # position offset of the new run (|DELTA| ~ 6)


def _runs():
    rng = np.random.default_rng(7)
    phi = rng.uniform(0, 2 * np.pi, N)
    r = rng.uniform(900, 1100, N)
    x = np.stack([r * np.cos(phi), r * np.sin(phi), rng.normal(0, 5, N)], axis=1)
    v = 1.0 / np.sqrt(r)  # roughly circular (G = M = 1)
    p = np.stack([-v * np.sin(phi), v * np.cos(phi), np.zeros(N)], axis=1)
    ids = np.arange(N, dtype=np.int64)[::-1].copy()  # not sorted: the code must not assume it
    scal = {"h": np.full(N, 3.0), "rho": rng.uniform(1, 2, N), "u": np.full(N, 0.1)}
    snaps = [Snap(0.0, ids, x, p, scal)]
    for k in (1, 2):
        xk, pk, _ = advance(x, p, k * T)
        snaps.append(Snap(k * T, ids, xk, pk, scal))
    t1 = 2 * T + 7.0  # the new run's first snapshot, a little after the old run's last
    x1, p1, _ = advance(x, p, t1)
    keep = np.arange(N) != 0  # one particle missing in the new run (swallowed there)
    new = Snap(t1, ids[keep], x1[keep] + DELTA, p1[keep] + 1e-4,
               {k: 2.0 * v[keep] for k, v in scal.items()})  # fmt: skip
    return snaps, new


def test_junction_easing():
    snaps, new = _runs()
    old = snaps[-1]
    window = 1.5 * T  # starts inside the old run's interval [T, 2T]
    j = make_junction(old, new, window)
    t_start = new.t - window

    def pair(t):
        return (
            (snaps[0], snaps[1]) if t <= T else (snaps[1], snaps[2]) if t <= old.t else (old, new)
        )

    def state(t):
        return eased_state(*pair(t), t, j, with_p=True)

    # before the window: the old run untouched
    t = t_start - 10.0
    ids, x, p, s = state(t)
    ids0, x0, p0, s0 = state_at(*pair(t), t, with_p=True)
    check("junction: before the window = old run (positions)", x0, x, atol=0.0)
    check("junction: before the window = old run (density)", s0["rho"], s["rho"], atol=0.0)

    # at the join: the new run (just before t1, so the old run's eased state is shown)
    ids, x, p, s = state(new.t - 1e-6)
    pos = {k: i for i, k in enumerate(new.ids)}
    jn = np.array([pos.get(k, -1) for k in ids])
    m = jn >= 0
    check("junction: reaches the new run at t1 (positions)", new.x[jn[m]], x[m], atol=1e-4)
    check("junction: reaches the new run at t1 (momenta)", new.p[jn[m]], p[m], atol=1e-8)
    check("junction: reaches the new run at t1 (density x 2)", new.scalars["rho"][jn[m]],
          s["rho"][m], rtol=1e-6)  # fmt: skip
    gone = ids[~m]
    check("junction: a particle missing in the new run is not moved", 1, len(gone))

    # mid-window: half the offset
    tm = new.t - 0.5 * window
    ids, x, p, s = state(tm)
    ids0, x0, _, s0 = state_at(*pair(tm), tm, with_p=True)
    moved = (x - x0)[np.isin(ids, new.ids)]
    check("junction: half the offset at mid-window", np.tile(0.5 * DELTA, (len(moved), 1)),
          moved, atol=0.05)  # fmt: skip

    # smooth: frame-to-frame extra motion never above the smooth step's steepest share
    ts = np.linspace(t_start - 20, new.t - 1e-6, 121)
    extra = []
    prev = None
    for t in ts:
        a, b = pair(t)
        _, x, _ = eased_state(a, b, t, j)
        if t <= old.t:  # the old run's own motion, without the easing
            _, xn, _ = state_at(a, b, t)
        else:
            xn = advance(old.x, old.p, t - old.t)[0]
        off = x - xn
        if prev is not None:
            extra.append(np.abs(off - prev).max())
        prev = off
    steep = 1.5 * np.abs(DELTA).max() * (ts[1] - ts[0]) / window  # smoothstep slope <= 1.5
    print(f"    largest extra step {max(extra):.4f}, limit {steep * 1.05:.4f}")
    check(
        "junction: no jump (largest extra step <= smooth-step limit)",
        True,
        max(extra) <= steep * 1.05,
    )
