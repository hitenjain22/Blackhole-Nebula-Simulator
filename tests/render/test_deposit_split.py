"""Wide particles on a coarser grid (gasgrid.deposit_log_split; optimization review
2026-11-23): no wide particle -> identical to deposit_log; mass ledger exact; a field of wide
particles matches the exact deposit closely."""

import numpy as np
import pytest

from nebula.render.gasgrid import (
    LogSphericalGrid,
    coarse_log_grid,
    deposit_log,
    deposit_log_split,
    wide_particles,
)
from tests.helpers import check

G = LogSphericalGrid(r_min=2.0, r_max=2000.0, nr=64, nth=32, nph=64)


def _gas(n, h_frac, seed=0):
    rng = np.random.default_rng(seed)
    r = np.exp(rng.uniform(np.log(30.0), np.log(600.0), n))
    d = rng.normal(size=(n, 3))
    x = r[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)
    return (x, h_frac * r, np.full(n, 1e-4), rng.uniform(1e4, 1e5, n),
            rng.normal(0.0, 0.01, (n, 3)))  # fmt: skip


def test_without_wide_particles_it_is_deposit_log():
    x, h, m, t, v = _gas(2000, 0.02)
    check("no particle is wide here", 0, int(wide_particles(x, h, G).sum()))
    a, b = deposit_log(x, h, m, t, v, G), deposit_log_split(x, h, m, t, v, G)
    check("split = deposit_log exactly (rho)", a.rho, b.rho)
    check("split = deposit_log exactly (T_em, v)", True,
          bool(np.array_equal(a.t_em, b.t_em) and np.array_equal(a.vel, b.vel)))  # fmt: skip


def test_wide_particles_ledger_and_closeness():
    """Half the particles wide (h = 1.5 r on this coarse test grid): grid mass + outside = sum m to 1e-12, and the
    density matches deposit_log where it matters (> 1e-2 of the max): median within 1%,
    99th percentile within 5%: the coarse grid has >= 4 cells across every wide kernel's
    support (WIDE_CELLS 24 fine cells / coarsening 4: 6), so interpolation errors stay small."""
    x1, h1, m1, t1, v1 = _gas(1500, 0.02, seed=1)
    x2, h2, m2, t2, v2 = _gas(1500, 1.5, seed=2)
    x, h = np.concatenate([x1, x2]), np.concatenate([h1, h2])
    m, t, v = np.concatenate([m1, m2]), np.concatenate([t1, t2]), np.concatenate([v1, v2])
    check("wide particles present", True, int(wide_particles(x, h, G).sum()) > 1000)
    ex = deposit_log(x, h, m, t, v, G)
    sp = deposit_log_split(x, h, m, t, v, G)
    check("ledger: grid + outside = sum m", m.sum(), sp.mass.sum() + sp.outside, rtol=1e-12)
    sig = ex.rho > 1e-2 * ex.rho.max()
    rel = np.abs(sp.rho[sig] - ex.rho[sig]) / ex.rho[sig]
    check("density median relative difference", 0.0, float(np.median(rel)), atol=0.01)
    check("density 99th percentile relative difference", 0.0, float(np.percentile(rel, 99)),
          atol=0.05)  # fmt: skip


def test_coarse_grid_must_divide():
    with pytest.raises(ValueError):
        coarse_log_grid(LogSphericalGrid(nr=63, nth=32, nph=64), (2, 4, 4))
    gc = coarse_log_grid(G, (2, 4, 4))
    check("coarse grid covers the same radii", (G.r_min, G.r_max), (gc.r_min, gc.r_max))
