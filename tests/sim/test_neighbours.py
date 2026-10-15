"""Step 3.2: neighbour lists and lattices (phases/phase-03-sph-core.md 3.1, 3.3)."""

import numpy as np
import pytest

from nebula.sim.ic import cubic_lattice, fcc_lattice, smoothing_length_uniform
from nebula.sim.neighbours import (
    brute_force_neighbours,
    find_neighbours,
    minimum_image,
)
from tests.helpers import check


def _uneven(rng, n, box):
    """Random positions; smoothing lengths varying 30x so gather-only lists would miss pairs."""
    pos = rng.uniform(0, box, size=(n, 3))
    h = np.exp(rng.uniform(np.log(0.01), np.log(0.3), n)) * box / 8
    return pos, h


@pytest.mark.parametrize("periodic", [False, True], ids=["open", "periodic"])
def test_lists_match_brute_force_and_are_symmetric(rng, periodic):
    box = 2.0
    pos, h = _uneven(rng, 1500, box)
    nb = find_neighbours(pos, h, box=box if periodic else None)
    ref = brute_force_neighbours(pos, h, box=box if periodic else None)
    same = all(np.array_equal(nb.of(i), ref[i]) for i in range(len(pos)))
    check(
        f"{'periodic' if periodic else 'open'}: lists = brute force (1,500 particles)", True, same
    )
    pairs = {(i, int(j)) for i in range(len(pos)) for j in nb.of(i)}
    symmetric = all((j, i) in pairs for i, j in pairs)
    check(f"{'periodic' if periodic else 'open'}: j in N(i) <=> i in N(j)", True, symmetric)
    check("each particle lists itself", True, all(i in set(nb.of(i)) for i in range(len(pos))))
    gather_only = sum(
        int(
            np.sum(
                np.linalg.norm(minimum_image(pos[nb.of(i)] - pos[i], nb.box), axis=1) >= 2 * h[i]
            )
        )
        for i in range(len(pos))
    )
    check("scatter partners present (pairs only the larger h reaches)", True, gather_only > 0)


def test_minimum_image():
    box = np.array([1.0, 2.0, 4.0])
    folded = minimum_image(np.array([0.9, -1.9, 2.1]), box)
    check("minimum image: nearest periodic copy", np.array([-0.1, 0.1, -1.9]), folded, atol=1e-15)
    rng = np.random.default_rng(0)
    dx = rng.uniform(-10, 10, size=(1000, 3))
    f = minimum_image(dx, box)
    check(
        "minimum image: always within half a box", True, bool(np.all(np.abs(f) <= box / 2 + 1e-12))
    )
    check(
        "minimum image: differs by whole boxes",
        np.zeros((1000, 3)),
        np.abs(np.round((dx - f) / box) - (dx - f) / box),
        atol=1e-12,
    )


def test_too_large_h_refused():
    pos = np.zeros((2, 3)) + 0.5
    with pytest.raises(ValueError, match="half the periodic box"):
        find_neighbours(pos, np.array([0.3, 0.3]), box=1.0)
    check("kernel larger than half the box is refused", True, True)


def test_lattices():
    cub = cubic_lattice(10, box=2.0)
    check("cubic lattice: count", 1000, len(cub))
    spacing = np.min(np.linalg.norm(cub[1:] - cub[0], axis=1))
    check("cubic lattice: nearest spacing", 0.2, spacing, rtol=1e-14)
    fcc = fcc_lattice(6, box=1.0)
    check("fcc lattice: count", 864, len(fcc))
    d = np.linalg.norm(minimum_image(fcc[1:] - fcc[0], np.ones(3)), axis=1)
    check("fcc: nearest spacing a/sqrt(2)", 1 / 6 / np.sqrt(2), d.min(), rtol=1e-12)
    check("fcc: 12 nearest neighbours", 12, int(np.sum(np.abs(d - d.min()) < 1e-9)))
    check("fcc: inside the box", True, bool(fcc.min() > 0 and fcc.max() < 1))


def test_neighbour_count_on_lattice():
    """Uniform lattice, h = 1.2 (V/N)^(1/3): close to the §4 estimate of 57.9 neighbours (the
    estimate counts a uniform continuum; a lattice gives the nearby shell counts)."""
    pos = fcc_lattice(8)
    h = np.full(len(pos), smoothing_length_uniform(len(pos), 1.0, 1.2))
    counts = find_neighbours(pos, h, box=1.0).counts - 1  # excluding self
    check("fcc lattice, eta=1.2: neighbours (excl. self) vs 57.9", 57.9, counts.mean(), rtol=0.1)
    check("every particle has the same count (perfect lattice)", True, bool(np.ptp(counts) == 0))


# --------------------------------------------------------------------------- Numba cell grid


def _same_lists(a, b) -> bool:
    return bool(np.array_equal(a.offset, b.offset) and np.array_equal(a.index, b.index))


@pytest.mark.parametrize(
    "case",
    ["open_uneven", "periodic_cube", "periodic_long_box", "isolated_large_h", "clump_and_halo"],
)
def test_numba_grid_matches_kdtree(rng, case):
    """find_neighbours_numba gives exactly the cKDTree reference lists (rule 2)."""
    from nebula.sim.neighbours import find_neighbours_numba

    if case == "open_uneven":
        pos, h = _uneven(rng, 3000, 2.0)
        box = None
    elif case == "periodic_cube":
        pos, h = _uneven(rng, 3000, 2.0)
        box = 2.0
    elif case == "periodic_long_box":  # like the Sod tube: 16x longer than wide
        box = np.array([2.0, 0.125, 0.125])
        pos = rng.uniform(0, 1, (4000, 3)) * box
        h = rng.uniform(0.005, 0.03, 4000)
    elif case == "isolated_large_h":
        pos = np.concatenate([rng.uniform(0, 1, (2000, 3)), [[5.0, 5.0, 5.0], [-3.0, 0.5, 0.5]]])
        h = np.concatenate([np.full(2000, 0.04), [4.0, 2.5]])
        box = None
    else:  # dense clump inside a sparse halo: h varies 100x
        pos = np.concatenate([rng.normal(0, 0.02, (2000, 3)), rng.uniform(-1, 1, (500, 3))])
        h = np.concatenate([np.full(2000, 0.005), np.full(500, 0.5)])
        box = None
    ref = find_neighbours(pos, h, box=box)
    fast = find_neighbours_numba(pos, h, box=box)
    check(f"Numba grid = cKDTree lists ({case})", True, _same_lists(ref, fast))


def test_numba_grid_single_particle():
    from nebula.sim.neighbours import find_neighbours_numba

    nb = find_neighbours_numba(np.array([[0.3, 0.2, 0.1]]), np.array([0.1]))
    check("single particle: lists itself only", [0], nb.of(0).tolist())


def test_tighten_keeps_exactly_the_interacting_pairs(rng):
    """Lists built with a 1.25x search margin, tightened at the true h, equal the lists built
    at h directly; and the force pass gives bit-identical results on either."""
    from nebula.sim.neighbours import find_neighbours_numba, tighten
    from nebula.sim.sph import compute_density, forces_numba

    pos, h = _uneven(rng, 3000, 2.0)
    loose = find_neighbours_numba(pos, 1.25 * h, box=2.0)
    tight = tighten(loose, pos, h)
    direct = find_neighbours_numba(pos, h, box=2.0)
    check("tightened lists = lists built at h", True, _same_lists(tight, direct))
    mass = np.full(len(pos), 1e-3)
    h_guess = np.full(len(pos), 1.6 * (8.0 / len(pos)) ** (1 / 3))  # from the mean density
    d = compute_density(pos, mass, h_guess, box=2.0)
    vel = rng.normal(0, 0.1, pos.shape)
    u = rng.uniform(0.5, 1.5, len(pos))
    p = (2 / 3) * d.rho * u
    cs = np.sqrt(10 / 9 * u)
    alpha = rng.uniform(0, 1, len(pos))
    nb_tight = tighten(d.neighbours, pos, d.h)
    a = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, d.neighbours, u=u, c_s=cs, alpha=alpha)
    b = forces_numba(pos, vel, mass, d.h, d.rho, d.omega, p, nb_tight, u=u, c_s=cs, alpha=alpha)
    same = np.array_equal(a.acc, b.acc) and np.array_equal(a.dudt, b.dudt)
    check("forces on tightened lists bit-identical", True, bool(same))
