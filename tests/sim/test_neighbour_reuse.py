"""Reuse of cached neighbour lists (sim/sph.py NeighbourCache, _reuse_limits): whenever the
lists are reused they must hold every pair that can interact, checked here by brute force
over all pairs, including after a few particles have moved far (the near-hole case)."""

import numpy as np

from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import NeighbourCache, _reuse_limits
from tests.helpers import check


def _missing_pairs(nb, pos, h_lim):
    """Pairs with r <= 2 max(h_lim_i, h_lim_j) at pos that are not in the lists."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    need = d <= 2.0 * np.maximum(h_lim[:, None], h_lim[None, :])
    have = np.zeros_like(need)
    for i in range(len(pos)):
        have[i, nb.of(i)] = True
    return int(np.sum(need & ~have))


def test_reused_lists_hold_every_interacting_pair(rng):
    n = 800
    reused = {"none fast": 0, "with fast set": 0}
    missing = 0
    for trial in range(120):
        pos0 = rng.normal(0.0, 4.0, (n, 3))
        h = rng.uniform(0.6, 1.4, n) * (1.0 + 0.3 * np.linalg.norm(pos0, axis=1) / 4.0)
        n_fast = trial % 6
        fast = rng.choice(n, n_fast, replace=False)
        # the near-hole case: sparse outliers with large h (a halo) move far; half the trials
        # instead move ordinary particles far inside the cluster (which must force rebuilds)
        halo = trial % 2 == 0
        if halo:
            u = rng.normal(0.0, 1.0, (n_fast, 3))
            pos0[fast] = 18.0 * u / np.linalg.norm(u, axis=1)[:, None]
            h[fast] = rng.uniform(3.0, 6.0, n_fast)
        h_search = 1.4 * h
        cache = NeighbourCache(find_neighbours_numba(pos0, h_search), pos0.copy(), h_search.copy())
        scale = [0.005, 0.02, 0.05, 0.1][trial % 4]
        pos = pos0 + rng.normal(0.0, scale, (n, 3))
        pos[fast] += rng.normal(0.0, 0.6 if halo else 1.5, (n_fast, 3))  # a few move far
        rows = np.sort(rng.choice(n, n // 5, replace=False))
        others = np.ones(n, bool)
        others[rows] = False
        h_guess = h * rng.uniform(0.97, 1.03, n)
        h_eff = _reuse_limits(cache, pos, h_guess, rows, others, None)
        if h_eff is None:
            continue
        reused["none fast" if n_fast == 0 else "with fast set"] += 1
        h_lim = np.where(others, h_guess, h_eff)  # rows may grow up to h_eff in the solve
        missing += _missing_pairs(cache.neighbours, pos, h_lim)
    print(f"   reuse decisions checked: {reused}")
    check("reused lists: interacting pairs missing (brute force)", 0, missing)
    check("reuse exercised without and with far-moving particles", True,
          reused["none fast"] > 0 and reused["with fast set"] > 0)  # fmt: skip


def test_one_far_mover_no_longer_forces_a_rebuild(rng):
    """The R6.1 case: one particle moves far, everything else barely: the old rule
    (hs - max displacement) rebuilds, the new one reuses with that particle checked directly."""
    n = 1000
    pos0 = rng.normal(0.0, 5.0, (n, 3))
    h = np.full(n, 1.0)
    cache = NeighbourCache(find_neighbours_numba(pos0, 1.4 * h), pos0.copy(), 1.4 * h)
    pos = pos0 + rng.normal(0.0, 0.01, (n, 3))
    far = int(np.argmax(np.linalg.norm(pos0, axis=1)))  # an outlier, few neighbours
    pos[far] += np.array([1.0, 0.0, 0.0])
    rows = np.arange(n)
    others = np.zeros(n, bool)
    old_rule_reuses = bool(np.all(1.4 * h - np.max(np.linalg.norm(pos - pos0, axis=1)) > 0.5 * h)
                           and np.all(1.4 * h - 1.0 > h))  # fmt: skip
    h_eff = _reuse_limits(cache, pos, h, rows, others, None)
    check("old rule would rebuild", False, old_rule_reuses)
    check("new rule reuses", True, h_eff is not None)
    if h_eff is not None:
        check("and the lists are complete", 0, _missing_pairs(cache.neighbours, pos, h_eff))


def test_particle_at_the_h_cap_keeps_its_list(rng):
    """An isolated particle whose h sits at h_max: a small move must not force a rebuild (the
    search keeps its skin margin above the cap), and the density equals a fresh build's."""
    from nebula.sim.sph import compute_density

    pos = rng.normal(0.0, 1.0, (600, 3))
    pos[0] = [30.0, 0.0, 0.0]  # far from the rest: wants h >> h_max
    mass = np.full(len(pos), 1.0 / len(pos))
    h0 = np.full(len(pos), 0.3)
    first = compute_density(pos, mass, h0, h_max=2.0)
    check("isolated particle sits at the cap", 2.0, float(first.h[0]), rtol=0.0)
    moved = pos.copy()
    moved[0] += [0.05, 0.0, 0.0]
    again = compute_density(moved, mass, first.h, h_max=2.0, cache=first.cache, rows=np.array([0]))
    fresh = compute_density(moved, mass, first.h, h_max=2.0, rows=np.array([0]))
    check("small move of the capped particle: lists reused, no rebuild", first.cache.builds,
          again.cache.builds)  # fmt: skip
    check("its density equals a fresh build's", float(fresh.rho[0]), float(again.rho[0]), rtol=0.0)
