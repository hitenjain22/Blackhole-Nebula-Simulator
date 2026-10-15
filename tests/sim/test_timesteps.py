"""Step 4.1: integer timeline, levels, sync rules, limiter (phases/phase-04-timestepping.md)."""

import numpy as np
import pytest

from nebula.sim.neighbours import Neighbours
from nebula.sim.timesteps import (
    K_MAX,
    allowed_level,
    is_active,
    level_for_dt,
    limit_levels,
    ticks,
)
from tests.helpers import check


def test_level_for_dt():
    dt_max = 4096.0
    wanted = np.array([4096.0, 4095.0, 2048.0, 1.0, 0.6, 1e-6])
    k = level_for_dt(wanted, dt_max)
    dt = dt_max / 2.0**k
    check("level step <= wanted step", True, bool(np.all(dt <= wanted)))
    check("level step > wanted / 2 (largest allowed)", True, bool(np.all(dt > wanted / 2)))
    check("exact powers map exactly", (0, 1, 12), (int(k[0]), int(k[2]), int(k[3])))
    with pytest.raises(ValueError, match="K_MAX"):
        level_for_dt(np.array([dt_max / 2.0 ** (K_MAX + 2)]), dt_max)


def test_sync_points_and_timeline_exactness():
    """Integer timeline: after many mixed steps every particle's time is exactly its tick count
    times the tick, and all levels meet at multiples of the longest step."""
    levels = np.array([0, 3, 7, 12])
    n = np.int64(0)
    t_part = np.zeros(4, np.int64)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        nxt = np.min(t_part + ticks(levels))
        active = t_part + ticks(levels) == nxt
        t_part[active] = nxt
        n = nxt
        check_ok = np.all(is_active(n, levels[active]))
        if not check_ok:
            break
        if rng.uniform() < 0.1:  # occasional level changes, respecting the rules
            want = rng.integers(0, 14, 4)
            levels[active] = allowed_level(n, levels[active], want[active])
    check("active particles are exactly at their sync points", True, bool(check_ok))
    check("particle times are whole tick counts (int64)", np.int64, t_part.dtype.type)


def test_level_change_rules():
    n = np.int64(3) * ticks(5)  # a sync point of level 5 but not of levels 4 and below
    check("shorter step always allowed", 9, int(allowed_level(n, np.array([5]), np.array([9]))[0]))
    check(
        "longer step refused when out of sync",
        5,
        int(allowed_level(n, np.array([5]), np.array([2]))[0]),
    )
    n2 = np.int64(4) * ticks(5)  # multiple of level 3's step (= 4 level-5 steps)
    check(
        "longer step allowed at a common sync point",
        3,
        int(allowed_level(n2, np.array([5]), np.array([2]))[0]),
    )


def test_limiter_propagates():
    """A chain 0-1-2-...-9 with particle 0 on level 20 and the rest on 0: after limiting,
    level(i) = 20 - 2 i, never lower than requested."""
    n = 10
    rows = [[i - 1, i, i + 1] for i in range(n)]
    rows[0], rows[-1] = [0, 1], [n - 2, n - 1]
    off = np.cumsum([0] + [len(r) for r in rows])
    nb = Neighbours(np.asarray(off, np.int64), np.concatenate(rows).astype(np.int64), None)
    levels = np.zeros(n, np.int64)
    levels[0] = 20
    out = limit_levels(levels, nb)
    check("limiter: levels fall off by 2 per neighbour", np.maximum(20 - 2 * np.arange(n), 0), out)
    check("limiter never lengthens a step", True, bool(np.all(out >= levels)))
