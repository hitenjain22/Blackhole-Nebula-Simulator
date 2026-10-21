"""Disk profiles from particles (analysis/disk_profile.py): hand-made rings with known answers,
flat and tilted (phase-07 7.3)."""

import numpy as np
from scipy.spatial.transform import Rotation

from nebula.analysis.disk_profile import disk_frame, disk_profile, viscosity_profile
from tests.helpers import check


def _ring(rng, n=200_000, r_in=40.0, r_out=60.0, h_over_r=0.1):
    """Uniform surface density between r_in and r_out (Sigma = 1 / (pi (r_out^2 - r_in^2)) for
    total mass 1), Gaussian heights sigma_z = h_over_r R, circular velocities 1/sqrt(R)."""
    rr = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0.0, h_over_r * rr)
    x = np.c_[rr * np.cos(ph), rr * np.sin(ph), z]
    p = np.c_[-np.sin(ph), np.cos(ph), np.zeros(n)] / np.sqrt(rr)[:, None]
    return x, p, np.full(n, 1.0 / n)


def test_flat_ring_profile(rng):
    x, p, m = _ring(rng)
    cs = np.full(len(m), 0.01)
    edges = np.linspace(42.0, 58.0, 9)
    prof = disk_profile(x, p, m, cs, edges)
    sigma0 = 1.0 / (np.pi * (60.0**2 - 40.0**2))
    # random heights tilt the sampled total J by ~ sigma_z / R / sqrt(N) ~ 2e-4 (sampling noise)
    check("flat ring: plane normal = +z (within sampling noise)", np.array([0.0, 0.0, 1.0]),
          prof.frame[2], atol=1e-3)  # fmt: skip
    check("flat ring: Sigma per annulus (uniform; sampling noise ~1%)", np.full(8, sigma0),
          prof.sigma, rtol=0.03)  # fmt: skip
    # sigma_z = 0.1 R averaged over the annulus: rms z = 0.1 sqrt(<R^2>)
    want_h = 0.1 * np.sqrt(0.5 * (edges[1:] ** 2 + edges[:-1] ** 2))
    check("flat ring: H = rms z per annulus", want_h, prof.height, rtol=0.03)
    check("flat ring: c_s per annulus", np.full(8, 0.01), prof.c_s, rtol=1e-12)
    check("nu = alpha c_s H", 0.1 * 0.01 * prof.height, viscosity_profile(prof, 0.1), rtol=1e-12)


def test_tilted_ring_profile(rng):
    """The same ring tilted by 30 degrees about a random axis: the plane is found from the total
    angular momentum and the profile equals the flat one's."""
    x, p, m = _ring(rng)
    cs = np.full(len(m), 0.01)
    edges = np.linspace(42.0, 58.0, 9)
    flat = disk_profile(x, p, m, cs, edges)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rot = Rotation.from_rotvec(np.radians(30.0) * axis)
    xt, pt = rot.apply(x), rot.apply(p)
    tilted = disk_profile(xt, pt, m, cs, edges)
    check("tilted ring: plane normal = the flat ring's, rotated (exact)", rot.apply(flat.frame[2]),
          tilted.frame[2], atol=1e-10)  # fmt: skip
    check("tilted ring: Sigma equals the flat ring's", flat.sigma, tilted.sigma, rtol=1e-9)
    check("tilted ring: H equals the flat ring's", flat.height, tilted.height, rtol=1e-9)


def test_frame_is_a_rotation(rng):
    x, p, m = _ring(rng, n=2000)
    f = disk_frame(x, p, m)
    check("frame is orthonormal", np.eye(3), f @ f.T, atol=1e-12)
    check("frame is a proper rotation (det +1)", 1.0, float(np.linalg.det(f)), atol=1e-12)
