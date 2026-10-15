"""Individual power-of-two time steps on an integer timeline (phases/phase-04-timestepping.md 4.1).

Time is t = t0 + n * tick with n an int64 counter and tick = dt_max / 2^K_MAX; never t += dt
(float accumulation drifts and breaks synchronization). A particle on level k takes steps
dt_k = dt_max / 2^k = 2^(K_MAX - k) ticks and is active when n is a multiple of that.

Rules (Springel 2005 §5, Quinn et al. 1997; plan 4.1):
- a particle may move to a shorter step (higher k) at any of its sync points;
- to a longer step (lower k) only when n is a multiple of the new step, otherwise it would
  fall out of sync;
- neighbouring particles' levels differ by at most 2 (Saitoh & Makino 2009, ApJ 697, L99): the
  limiter only ever shortens steps.
"""

import numba
import numpy as np

K_MAX = 40  # deepest level (sim/integrator.py uses the same); 2^40 ticks per dt_max


def ticks(level) -> np.ndarray:
    """Step length in ticks of level k: 2^(K_MAX - k)."""
    return np.left_shift(np.int64(1), (K_MAX - np.asarray(level, np.int64)))


def level_for_dt(dt_wanted, dt_max: float) -> np.ndarray:
    """Smallest level k with dt_max / 2^k <= dt_wanted (the largest allowed power-of-two step).
    Raises if a particle would need more than K_MAX halvings."""
    dt_wanted = np.asarray(dt_wanted, float)
    if np.any(~(dt_wanted > 0)):
        raise ValueError("time steps must be positive")
    k = np.ceil(np.log2(dt_max / dt_wanted) - 1e-12).astype(np.int64)
    k = np.maximum(k, 0)
    # guard against log2 rounding: ensure dt_max / 2^k <= dt_wanted
    too_long = dt_max / np.exp2(k) > dt_wanted
    k = k + too_long
    if np.any(k > K_MAX):
        raise ValueError(f"time step needs more than K_MAX = {K_MAX} halvings of dt_max")
    return k


def is_active(n: int, level) -> np.ndarray:
    """True where n is a sync point of the level."""
    return np.mod(np.int64(n), ticks(level)) == 0


def allowed_level(n: int, current, wanted) -> np.ndarray:
    """New level at sync point n: shorter (k_wanted >= k_current) always; longer only to the
    longest level k in [k_wanted, k_current) that n is a multiple of."""
    current = np.asarray(current, np.int64)
    wanted = np.asarray(wanted, np.int64)
    new = np.where(wanted >= current, wanted, current)
    longer = wanted < current
    for i in np.flatnonzero(longer):
        for k in range(int(wanted[i]), int(current[i])):
            if np.int64(n) % ticks(k) == 0:
                new[i] = k
                break
    return new


@numba.njit(cache=True)
def _limit(levels, offset, index, max_gap):
    """Raise levels until every particle is within max_gap of its neighbours' max level.
    Repeats until nothing changes (a sharp level jump propagates outward)."""
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for i in range(levels.shape[0]):
            need = levels[i]
            for k in range(offset[i], offset[i + 1]):
                j = index[k]
                if levels[j] - max_gap > need:
                    need = levels[j] - max_gap
            if need > levels[i]:
                levels[i] = need
                changed = True
    return rounds


def limit_levels(levels: np.ndarray, neighbours, max_gap: int = 2) -> np.ndarray:
    """Neighbour time-step limiter (Saitoh & Makino 2009): levels of neighbouring particles
    differ by at most ``max_gap``; steps are only shortened. Returns a new array."""
    out = np.array(levels, np.int64)
    _limit(out, neighbours.offset, neighbours.index, int(max_gap))
    return out
