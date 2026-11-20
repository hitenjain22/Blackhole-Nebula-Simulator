"""The inner accretion disc drawn at render time, from the torus's inner edge down to the
innermost stable orbit (DECISIONS 2026-11-20; Hiten 2026-11-19: the dark gap around the hole
made the photon ring look fake).

Why: the simulation removes gas inside r_acc = 4 and its torus sits at r ~ 45-70, so the
region r = 6-45, where a real disc is hottest, is empty: the photon ring (the lensed image of
whatever is near the hole) then shows only the distant torus. A steady thin disc fills it
(physics/steady_disc.py): temperature rising inward to 2.66x the torus edge's at r = 8.2 and
falling to 0 at r_in, surface density from the same steady-disc law.

Fed by the simulation, not set by hand: ``edge_feed`` measures the gas in an annulus at the
torus's inner edge (surface density, emission temperature, thickness, orbital plane) in each
snapshot; ``blend_feeds`` averages them over ``smooth_days`` (a real disc answers its supply on
the viscous time, days to weeks; the raw edge temperature jumps +-30% between dumps); and
``disc_particles`` lays the disc out as extra particles that the normal pipeline deposits and
draws (so the GPU march, colours, Doppler shifts and their tests apply unchanged). Before the
torus forms the annulus is empty and there is no disc.

Layout: rings evenly spaced in ln r and particles evenly spaced in phi, all in the orbital
plane, each with h set so the layer's thickness (axis rms) equals the measured H/R x r. The
disc is azimuthally smooth and does not move: late in the video a frame spans ~350 time units,
more than an orbit at r < 15 (period 2 pi r^1.5), so any lump would strobe. Its rotation is in
the velocities (prograde circular orbits about the plane's normal, Kerr speeds with spin),
which set the Doppler colours. With spin the disc reaches the spinning hole's innermost stable
orbit (r_in = physics.orbits.isco_radius(a); 2.32 at a = 0.9). Faint ring stripes
(Sigma x (1 + A n(ln r)), fixed smooth noise) give it texture.
"""

from dataclasses import dataclass

import numpy as np

from nebula.physics import steady_disc
from nebula.physics.kernels import get_kernel

KERNEL = "wendland_c4"  # the deposit's kernel (gasgrid.deposit_log, bricks.deposit_bricks)
IN_PLANE = 0.3  # |height| / R limit for gas counted in the edge annulus
MIN_H_OVER_R = 0.01


@dataclass(frozen=True)
class EdgeFeed:
    """Gas at the torus's inner edge (one snapshot or a time average)."""

    sigma: float  # surface density in the annulus [code mass / area]
    t_em: float  # emission temperature (mean T^4)^(1/4) [K]
    h_over_r: float  # rms height / R
    lvec: np.ndarray  # total angular momentum of the annulus gas (its direction: the normal)


def _plane_basis(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


def edge_feed(x, p, m, temp, r_lo: float, r_hi: float) -> EdgeFeed:
    """Measure the annulus r_lo < R < r_hi (cylindrical, in the orbital plane of the gas within
    r < 2 r_hi; |height| < IN_PLANE R). m: mass per particle (scalar or array); temp in K."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    m = np.broadcast_to(np.asarray(m, float), (len(x),))
    temp = np.asarray(temp, float)
    near = np.linalg.norm(x, axis=1) < 2.0 * r_hi
    lnear = (m[near, None] * np.cross(x[near], p[near])).sum(axis=0)
    if not near.any() or np.linalg.norm(lnear) == 0.0:
        return EdgeFeed(0.0, 0.0, 0.0, np.zeros(3))
    e1, e2, n = _plane_basis(lnear)
    z = x @ n
    big_r = np.hypot(x @ e1, x @ e2)
    sel = (big_r > r_lo) & (big_r < r_hi) & (np.abs(z) < IN_PLANE * big_r)
    if not sel.any():
        return EdgeFeed(0.0, 0.0, 0.0, np.zeros(3))
    ms = m[sel]
    area = np.pi * (r_hi**2 - r_lo**2)
    return EdgeFeed(
        sigma=float(ms.sum() / area),
        t_em=float((np.sum(ms * temp[sel] ** 4) / ms.sum()) ** 0.25),
        h_over_r=float(np.sqrt(np.sum(ms * (z[sel] / big_r[sel]) ** 2) / ms.sum())),
        lvec=(ms[:, None] * np.cross(x[sel], p[sel])).sum(axis=0),
    )


def blend_feeds(feeds, weights) -> EdgeFeed:
    """Weighted time average: sigma linearly, T through sigma T^4 (emission), H/R weighted by
    sigma, angular momenta summed with the weights."""
    w = np.asarray(weights, float)
    sig = np.array([f.sigma for f in feeds])
    if w.sum() <= 0.0 or np.sum(w * sig) <= 0.0:
        return EdgeFeed(0.0, 0.0, 0.0, np.zeros(3))
    ws = w * sig
    return EdgeFeed(
        sigma=float(np.sum(ws) / w.sum()),
        t_em=float((np.sum(ws * np.array([f.t_em for f in feeds]) ** 4) / ws.sum()) ** 0.25),
        h_over_r=float(np.sum(ws * np.array([f.h_over_r for f in feeds])) / ws.sum()),
        lvec=np.sum(w[:, None] * np.array([f.lvec for f in feeds]), axis=0),
    )


def triangle_weights(times, t: float, width: float) -> np.ndarray:
    """Weights max(0, 1 - |t_k - t| / (width / 2)) (normalised): a triangle of full width
    ``width``, so the average changes continuously as t moves (a box would jump whenever a
    snapshot enters or leaves it)."""
    w = np.clip(1.0 - np.abs(np.asarray(times, float) - t) / (0.5 * width), 0.0, None)
    return w / w.sum() if w.sum() > 0 else w


def stripe_noise(lnr, seed: int, n_waves: int = 12, lam=(0.08, 0.4)) -> np.ndarray:
    """Smooth fixed noise in ln r, roughly in [-1, 1]: a sum of sines with random wavelengths
    (in ln r, between lam[0] and lam[1]) and phases, normalised by its rms (x 1/2)."""
    rng = np.random.default_rng(seed)
    k = 2.0 * np.pi / rng.uniform(lam[0], lam[1], n_waves)
    ph = rng.uniform(0.0, 2.0 * np.pi, n_waves)
    amp = rng.uniform(0.5, 1.0, n_waves)
    s = (amp[:, None] * np.sin(k[:, None] * np.asarray(lnr, float)[None, :] + ph[:, None])).sum(0)
    return s / (2.0 * np.sqrt(0.5 * np.sum(amp**2)))


@dataclass(frozen=True)
class DiscParticles:
    x: np.ndarray  # (N, 3)
    h: np.ndarray
    m: np.ndarray
    temp: np.ndarray  # K
    gamma_beta: np.ndarray  # (N, 3), static frame (physics.steady_disc.circular_gamma_beta)


def taper(r, r0: float, r1: float):
    """1 inside r0, 0 outside r1, a smoothstep between (the disc's outer edge, where it hands
    over to the torus)."""
    z = np.clip((np.asarray(r, float) - r0) / (r1 - r0), 0.0, 1.0)
    return 1.0 - z * z * (3.0 - 2.0 * z)


def disc_particles(
    feed: EdgeFeed,
    r_in: float,
    r_ref: float,
    r_taper,
    spacing: float,
    stripes: float,
    seed: int = 7,
    spin: float = 0.0,
    t_scale: float = 1.0,
    density: float = 1.0,
    t_stripes: float = 0.0,
) -> DiscParticles | None:
    """Particles of the inner disc matched to ``feed`` at r_ref (None if the feed is empty).

    Rings from r_in (exclusive) to r_taper[1], spaced spacing x (H/R) in ln r; particles spaced
    spacing x (H/R) in phi; h = (H/R) r / axis_rms(kernel) (layer thickness H). Mass per
    particle Sigma(r) taper(r) (1 + stripes n(ln r)) x its area r^2 dlnr dphi, so the disc's
    surface density is the steady-disc Sigma through feed.sigma at r_ref.

    Look settings (Hiten 2026-11-20: the matched disc was a solid, washed-out plate): t_scale
    multiplies the temperature (brightness ~ T^4), density the surface density (optical
    depth: < 1 makes it see-through), t_stripes adds ring stripes in temperature
    (T x (1 + t_stripes n'(ln r)), another fixed noise), which show even where the disc is
    opaque. All 1, 1, 0: the disc matched to the torus."""
    if feed.sigma <= 0.0 or not np.any(feed.lvec):
        return None
    hr = max(feed.h_over_r, MIN_H_OVER_R)
    d = spacing * hr
    lo, hi = np.log(r_in), np.log(r_taper[1])
    n_r = int(np.ceil((hi - lo) / d))
    lnr = lo + (np.arange(n_r) + 0.5) * (hi - lo) / n_r
    dlnr = (hi - lo) / n_r
    n_ph = int(np.ceil(2.0 * np.pi / d))
    dph = 2.0 * np.pi / n_ph
    r = np.exp(lnr)
    sig = steady_disc.surface_density(r, r_in, r_ref, feed.sigma) * taper(r, *r_taper)
    sig = density * sig * np.clip(1.0 + stripes * stripe_noise(lnr, seed), 0.0, None)
    temp = steady_disc.temperature(r, r_in, r_ref, feed.t_em) * t_scale
    temp = temp * np.clip(1.0 + t_stripes * stripe_noise(lnr, seed + 1), 0.0, None)
    # each ring starts at a different angle (golden-ratio offsets): no radial spokes
    ph = (np.arange(n_ph)[None, :] + 0.618034 * np.arange(n_r)[:, None]) * dph
    e1, e2, n = _plane_basis(feed.lvec)
    rr = np.repeat(r, n_ph)
    phf = ph.ravel()
    radial = np.cos(phf)[:, None] * e1 + np.sin(phf)[:, None] * e2
    along = -np.sin(phf)[:, None] * e1 + np.cos(phf)[:, None] * e2  # prograde about n
    keep = np.repeat(sig > 0.0, n_ph)
    x = rr[:, None] * radial
    gb = steady_disc.circular_gamma_beta_kerr(rr, spin)[:, None] * along  # prograde, a >= 0
    mass = np.repeat(sig * r * r * dlnr * dph, n_ph)
    h = hr * rr / get_kernel(KERNEL).axis_rms()
    return DiscParticles(x[keep], h[keep], mass[keep], np.repeat(temp, n_ph)[keep], gb[keep])
