"""Easing the joins between source runs of a shot (2026-11-17, DECISIONS).

A shot's gas can come from different runs one after another (render.director.Timeline). When
the new run grew out of the old one (restarted from its snapshot) the join is seamless. When it
is a different run (dense_P188a -> dense_C400, dense_C400 -> R6.1_100k_ph_b_to235) the same
particles sit up to tens of units apart at the join, which the in-betweens would show as a jump
in one frame (the 37.5 s blip of the first low-res render).

Here the jump is spread over ``window`` code time before the new run starts: each particle of
the old run is moved by w(t) x (its offset to the new run), w rising smoothly from 0 to 1 over
the window, so at the new run's first snapshot the old state has become the new one exactly.
The offset is measured at the new run's first snapshot time t1: new snapshot minus the old
run's last snapshot carried forward along its orbit (render.interpolate.advance). Momenta
(Doppler colours) and the scalars (h, rho, u: temperature, brightness) are eased the same way.

A display choice, not physics: the simulation data are untouched; only frames inside the window
show gas moved by part of the offset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nebula.render.director import CODE_DAY, Timeline, smoothstep
from nebula.render.interpolate import ETA_ORB, Snap, advance, state_at


@dataclass(frozen=True)
class Junction:
    """Offsets from the old run's last snapshot (time t0) to the new run's first (t1)."""

    t0: float
    t1: float
    window: float  # code time before t1 over which the offset is eased in
    ids: np.ndarray  # sorted ids of the old run's last snapshot
    dx: np.ndarray  # (n, 3) per id, 0 where the particle is not in both runs
    dp: np.ndarray
    dscal: dict  # ln(new / old) per id for each scalar

    def weight(self, t: float) -> float:
        """Share of the offset applied at time t: 0 before the window, 1 at t1."""
        if t >= self.t1 or self.window <= 0:
            return 0.0  # from t1 the new run is shown as it is
        return float(smoothstep((t - (self.t1 - self.window)) / self.window))


def make_junction(a: Snap, b: Snap, window: float, eta: float = ETA_ORB,
                  r_drop: float = 4.0) -> Junction:  # fmt: skip
    """Junction from the old run's last snapshot a to the new run's first snapshot b."""
    xf, pf, alive = advance(a.x, a.p, b.t - a.t, eta, r_drop)
    order = np.argsort(a.ids)
    pos = {k: j for j, k in enumerate(b.ids)}
    jb = np.array([pos.get(k, -1) for k in a.ids[order]])
    both = (jb >= 0) & alive[order]
    dx = np.zeros((len(order), 3))
    dp = np.zeros((len(order), 3))
    dx[both] = b.x[jb[both]] - xf[order][both]
    dp[both] = b.p[jb[both]] - pf[order][both]
    dscal = {}  # positive scalars (h, rho, u): eased as ratios, ln(new / old), never < 0
    for k, v in a.scalars.items():
        d = np.zeros(len(order))
        old, nw = v[order][both], b.scalars[k][jb[both]]
        ok = (old > 0) & (nw > 0)
        d[np.nonzero(both)[0][ok]] = np.log(nw[ok] / old[ok])
        dscal[k] = d
    return Junction(a.t, b.t, window, a.ids[order], dx, dp, dscal)


def eased_state(a: Snap, b: Snap, t: float, junction: Junction | None, r_drop: float = 4.0,
                with_p: bool = False):  # fmt: skip
    """state_at(a, b, t) with the junction's offset eased in (same return as state_at). For the
    pair that spans the join itself (a = old run's last, b = new run's first) the old run is
    carried along its orbits instead of being cross-faded to b (the offset does that)."""
    if junction is not None and a.t == junction.t0 and b.t == junction.t1 and t < b.t:
        xf, pf, alive = advance(a.x, a.p, t - a.t, ETA_ORB, r_drop)
        ids, x, p = a.ids[alive], xf[alive], pf[alive]
        scal = {k: v[alive].copy() for k, v in a.scalars.items()}
    else:
        ids, x, p, scal = state_at(a, b, t, r_drop=r_drop, with_p=True)
    w = junction.weight(t) if junction is not None else 0.0
    if w > 0.0:
        j = np.searchsorted(junction.ids, ids)
        j = np.minimum(j, len(junction.ids) - 1)
        hit = junction.ids[j] == ids
        x = x.copy()
        p = p.copy()
        x[hit] += w * junction.dx[j[hit]]
        p[hit] += w * junction.dp[j[hit]]
        for k in scal:
            scal[k] = scal[k].copy()
            scal[k][hit] *= np.exp(w * junction.dscal[k][j[hit]])
    if with_p:
        return ids, x, p, scal
    return ids, x, scal


class JunctionEaser:
    """The eased joins of a shot: one window before each source that sets ``ease_days`` and
    follows a different run. ``at(t)`` gives the Junction whose window holds t (or None); each
    is built once, from the two snapshots either side of the join (``load``: path -> Snap)."""

    def __init__(self, sources, timeline: Timeline, load):
        ease = {src["run"]: float(src.get("ease_days", 0.0)) * CODE_DAY for src in sources}
        runs = [Path(p).parent.parent.name for p in timeline.paths]
        self.timeline, self.load = timeline, load
        self.windows = [
            (k, float(timeline.times[k + 1]), ease.get(runs[k + 1], 0.0))
            for k in range(len(runs) - 1)
            if runs[k] != runs[k + 1] and ease.get(runs[k + 1], 0.0) > 0.0
        ]
        self._made = {}

    def at(self, t: float) -> Junction | None:
        for k, t1, w in self.windows:
            if t1 - w <= t < t1:
                if k not in self._made:
                    paths = self.timeline.paths
                    self._made[k] = make_junction(self.load(paths[k]), self.load(paths[k + 1]), w)
                return self._made[k]
        return None
