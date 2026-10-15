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
