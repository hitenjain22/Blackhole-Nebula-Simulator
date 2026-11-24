"""The render-time inner disc (render/inner_disc.py; DECISIONS 2026-11-20): the edge
measurement recovers a known torus, the disc's particles carry exactly the steady-disc surface
density, the planned thickness and prograde circular velocities, and the time average moves
continuously."""

import numpy as np
import pytest

from nebula.physics import steady_disc as sd
from nebula.render.director import Shot, ShotError
from nebula.render.gasgrid import CartesianBox, deposit_box
from nebula.render.inner_disc import (
    EdgeFeed,
    blend_feeds,
    disc_particles,
    edge_feed,
    taper,
    triangle_weights,
)
from tests.helpers import check

TAPER = (35.0, 50.0)


def _torus(n=200_000, normal=(0.0, 0.0, 1.0), seed=1):
    """Uniform-Sigma annulus 30 < R < 80, Gaussian height H/R = 0.04, T = 3e5 K, circular
    velocities about ``normal``."""
    rng = np.random.default_rng(seed)
    big_r = np.sqrt(rng.uniform(30.0**2, 80.0**2, n))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.normal(0.0, 0.04, n) * big_r
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(nrm, a) / np.linalg.norm(np.cross(nrm, a))
    e2 = np.cross(nrm, e1)
    x = (big_r * np.cos(ph))[:, None] * e1 + (big_r * np.sin(ph))[:, None] * e2 + z[:, None] * nrm
    p = (big_r**-0.5)[:, None] * (-np.sin(ph)[:, None] * e1 + np.cos(ph)[:, None] * e2)
    return x, p, np.full(n, 3.0e5), nrm


def test_edge_feed_recovers_a_known_torus():
    x, p, t, nrm = _torus(normal=(0.3, -0.2, 1.0))
    m = 1e-5
    f = edge_feed(x, p, m, t, 40.0, 60.0)
    sigma_true = len(x) * m / (np.pi * (80.0**2 - 30.0**2))
    check("edge feed: Sigma (Poisson ~1%)", sigma_true, f.sigma, rtol=0.03)
    check("edge feed: T_em", 3.0e5, f.t_em, rtol=1e-12)
    check("edge feed: H/R", 0.04, f.h_over_r, rtol=0.03)
    check("edge feed: normal", nrm, f.lvec / np.linalg.norm(f.lvec), atol=1e-3)
    empty = edge_feed(x[:10] * 10.0, p[:10], m, t[:10], 40.0, 60.0)
    check("edge feed: no gas there -> Sigma 0", 0.0, empty.sigma)


@pytest.fixture(scope="module")
def disc():
    feed = EdgeFeed(2e-5, 3e5, 0.04, np.array([0.0, 0.0, 1.0]))
    return feed, disc_particles(feed, 6.0, 50.0, TAPER, spacing=0.6, stripes=0.0)


def test_disc_mass_is_the_steady_disc_sigma(disc):
    """Total mass = int Sigma(r) taper(r) 2 pi r dr (midpoint rings in ln r: 1e-3)."""
    feed, dp = disc
    r = np.geomspace(6.0, 50.0, 200_001)
    integrand = sd.surface_density(r, 6.0, 50.0, feed.sigma) * taper(r, *TAPER) * 2 * np.pi * r
    exact = float(np.trapezoid(integrand, r))
    check("disc mass = int Sigma taper 2 pi r dr", exact, float(dp.m.sum()), rtol=1e-3)
    print(f"    disc: {len(dp.m)} particles, mass {dp.m.sum():.3e}")


def test_disc_profile_and_velocities(disc):
    feed, dp = disc
    r = np.linalg.norm(dp.x, axis=1)
    check("disc: inside r_in .. taper end", True, bool(r.min() > 6.0 and r.max() < 50.0))
    check("disc: in the plane", 0.0, float(np.abs(dp.x[:, 2]).max()), atol=1e-12)
    check("disc: T = steady-disc T", sd.temperature(r, 6.0, 50.0, 3e5), dp.temp, rtol=1e-12)
    check("disc: |gamma beta| = 1/sqrt(r - 3)", sd.circular_gamma_beta(r),
          np.linalg.norm(dp.gamma_beta, axis=1), rtol=1e-12)  # fmt: skip
    lz = np.cross(dp.x, dp.gamma_beta)
    check("disc: prograde about the feed's normal (L parallel)", 1.0,
          float((lz[:, 2] / np.linalg.norm(lz, axis=1)).min()), rtol=1e-12)  # fmt: skip


def test_disc_thickness_is_the_measured_h_over_r(disc):
    """Deposited on a fine box at R = 20 tall enough for the whole layer (support 2h = 3.5):
    the density-weighted rms height = (H/R) R, with the cell width^2 / 12 added (cell
    averaging), within 3%."""
    feed, dp = disc
    box = CartesianBox(centre=(20.0, 0.0, 0.0), size=8.0, n=128)
    near = np.linalg.norm(dp.x - np.array([20.0, 0.0, 0.0]), axis=1) < 8.0
    gg = deposit_box(dp.x[near], dp.h[near], dp.m[near], dp.temp[near],
                     np.zeros((near.sum(), 3)), box)  # fmt: skip
    zc = (np.arange(128) + 0.5) * box.dx - 4.0
    prof = gg.rho[56:72, 56:72, :].sum(axis=(0, 1))
    rms = float(np.sqrt(np.sum(prof * zc**2) / prof.sum()))
    expect = float(np.sqrt((0.04 * 20.0) ** 2 + box.dx**2 / 12.0))
    check("disc: thickness at R = 20", expect, rms, rtol=0.03)


def test_feed_average_is_continuous_and_exact_for_equal_feeds():
    times = np.arange(0.0, 20.0, 1.0)
    ts = np.linspace(5.0, 15.0, 2001)
    sig = np.array([1.0 + 0.3 * (k % 2) for k in range(len(times))])
    feeds = [EdgeFeed(s, 3e5 * s, 0.04, np.array([0, 0, 1.0])) for s in sig]
    avg = np.array([blend_feeds(feeds, triangle_weights(times, t, 4.0)).sigma for t in ts])
    check("triangle average: largest step between t 0.005 apart (continuous)", True,
          bool(np.abs(np.diff(avg)).max() < 2e-3))  # fmt: skip
    same = blend_feeds([feeds[0]] * 3, [0.2, 0.5, 0.3])
    check("average of equal feeds = the feed (Sigma, T)", [1.0, 3e5], [same.sigma, same.t_em],
          rtol=1e-14)  # fmt: skip
    check("no torus: no disc", None, disc_particles(EdgeFeed(0.0, 0.0, 0.0, np.zeros(3)), 6.0,
          50.0, TAPER, 0.6, 0.3))  # fmt: skip


def test_disc_settings_are_checked(tmp_path):
    from pathlib import Path

    base = Path(__file__).resolve().parents[2] / "configs" / "shots" / "full_story.toml"
    import re

    # the story's own [disc] section out, so each case below adds exactly one
    text = re.sub(r"\n\[disc\]\n(?:[^\[\n][^\n]*\n|\n)*", "\n", base.read_text())
    good = tmp_path / "good.toml"
    good.write_text(text + "\n[disc]\nstripes = 0.2\n")
    check("[disc] accepted", 0.2, Shot.load(good).disc["stripes"])
    for bad in ("colour_x = 1", "taper = [60.0, 50.0]", "stripes = 1.5"):
        f = tmp_path / "bad.toml"
        f.write_text(text + f"\n[disc]\n{bad}\n")
        with pytest.raises(ShotError):
            Shot.load(f)


def test_look_settings_scale_the_matched_disc(disc):
    """temperature and density multiply T and the mass exactly; t_stripes changes T only."""
    feed, dp = disc
    dp2 = disc_particles(feed, 6.0, 50.0, TAPER, spacing=0.6, stripes=0.0, t_scale=0.6,
                         density=0.03)  # fmt: skip
    check("look: temperature x 0.6", 0.6 * dp.temp, dp2.temp, rtol=1e-14)
    check("look: mass x 0.03", 0.03 * dp.m, dp2.m, rtol=1e-14)
    dp3 = disc_particles(feed, 6.0, 50.0, TAPER, spacing=0.6, stripes=0.0, t_stripes=0.2)
    ratio = dp3.temp / dp.temp
    check("look: t_stripes leaves the mass", dp.m, dp3.m, rtol=1e-14)
    check("look: t_stripes within 1 +- 0.2 x noise range", True,
          bool(ratio.min() > 0.6 and ratio.max() < 1.4 and ratio.std() > 0.01))  # fmt: skip


def test_feed_snapshots_keeps_sparse_and_thins_dense():
    """Snapshots 0.5 apart (wider than the 0.25 step): every one in the window is kept (the
    average is unchanged). 0.01 apart: ~width/step of them, all inside the window."""
    from nebula.render.inner_disc import feed_snapshots, triangle_weights

    sparse = np.arange(0.0, 40.0, 0.5)
    ks = feed_snapshots(sparse, 20.0, 10.0, 0.25)
    check("sparse snapshots: all in the window kept", np.nonzero(triangle_weights(sparse, 20.0,
          10.0))[0], ks)  # fmt: skip
    dense = np.arange(0.0, 40.0, 0.01)
    kd = feed_snapshots(dense, 20.0, 10.0, 0.25)
    check("dense snapshots: ~ width/step kept (41 grid points, ends excluded)", True,
          bool(38 <= len(kd) <= 41 and np.all(np.abs(dense[kd] - 20.0) < 5.0)))  # fmt: skip


def test_edge_feed_ignores_a_plunging_stream():
    """Gas crossing the annulus on a plunging orbit (radial speed comparable to the orbital one,
    as the first pass's stream) does not feed the disc: Sigma 0. The same gas on circular
    orbits does."""
    x, p, t, _ = _torus()
    r_hat = x / np.linalg.norm(x, axis=1, keepdims=True)
    plunging = p - np.linalg.norm(p, axis=1, keepdims=True) * r_hat  # inward = orbital speed
    check("plunging gas: no feed", 0.0, edge_feed(x, plunging, 1e-5, t, 40.0, 60.0).sigma)
    check("circular gas: feeds", True, edge_feed(x, p, 1e-5, t, 40.0, 60.0).sigma > 0)
