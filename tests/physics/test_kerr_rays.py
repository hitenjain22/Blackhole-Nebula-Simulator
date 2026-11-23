"""Kerr light rays, float64 reference (physics/kerr.py): phase-11 11.2 tests 1-5."""

import numpy as np
import pytest

from nebula.physics import kerr as K
from nebula.physics.geodesics import schw_ray_escape_angle, schw_shadow_angle
from tests.helpers import check

D_FAR = 1.0e4


def _status(a, alpha, beta=0.0, D=D_FAR):
    """Ray from a camera on the +x axis at distance D looking at the hole (screen right = +y,
    up = +z = spin axis) at impact parameters (alpha, beta) = D x (screen offsets)."""
    xc = np.array([D, 0.0, 0.0])
    d = np.array([-1.0, alpha / D, beta / D])
    return K.trace_ray(xc, K.flat_to_local(xc, d / np.linalg.norm(d)), a)[0]


def _psi_dir(D, psi):
    """Flat unit direction at angle psi from the inward radial, in the x-y plane."""
    return np.array([-np.cos(psi), np.sin(psi), 0.0])


def test_1_spin_zero_is_phase_2():
    """a = 0: shadow edge at the Synge angle (D = 20, 50; bisection to 1e-7 rad) and the
    escape directions of Phase 2 (physics.geodesics) within 2e-6 rad."""
    for D in (20.0, 50.0):
        xc = np.array([D, 0.0, 0.0])
        lo, hi = 0.0, 0.5 * np.pi  # captured at lo, escaped at hi
        while hi - lo > 1e-7:
            mid = 0.5 * (lo + hi)
            s = K.trace_ray(xc, K.flat_to_local(xc, _psi_dir(D, mid)), 0.0, rtol=1e-12)[0]
            lo, hi = (mid, hi) if s == "captured" else (lo, mid)
        check(f"test 1: shadow angle, a = 0, D = {D:g}", schw_shadow_angle(D), 0.5 * (lo + hi),
              atol=2e-7)  # fmt: skip
        worst = 0.0
        for psi in (0.6, 1.0, 1.6, 2.5):
            psi = max(psi, schw_shadow_angle(D) + 0.05)
            st, _, p, _ = K.trace_ray(xc, K.flat_to_local(xc, _psi_dir(D, psi)), 0.0,
                                      r_escape=1e8, rtol=1e-12)  # fmt: skip
            phi_inf = schw_ray_escape_angle(D, psi)
            # Phase 2: the ray sees the sky toward cos(phi) e1 + sin(phi) e2 (e1 = x, e2 = y)
            want = np.array([np.cos(phi_inf), np.sin(phi_inf), 0.0])
            got = -p / np.linalg.norm(p)  # the light came from where the backward ray goes
            worst = max(worst, float(np.arccos(np.clip(want @ got, -1, 1))))
        check(f"test 1: escape directions, a = 0, D = {D:g} (rad)", 0.0, worst, atol=2e-6)


def test_2_hamiltonian_conserved():
    xc = np.array([60.0, 5.0, 20.0])
    worst = 0.0
    for d in ([-1.0, 0.07, 0.03], [-1.0, 0.11, -0.08], [-1.0, -0.2, 0.15]):
        d = np.asarray(d) / np.linalg.norm(d)
        _, _, _, sol = K.trace_ray(xc, K.flat_to_local(xc, d), 0.9)
        h = [K.hamiltonian(sol.y[:3, j], sol.y[3:, j], 0.9) for j in range(sol.y.shape[1])]
        worst = max(worst, float(np.max(np.abs(h))))
    check("test 2: |H| along rays, a = 0.9", 0.0, worst, atol=1e-10)


def test_3_grad_x_against_finite_differences():
    rng = np.random.default_rng(3)
    worst, n = 0.0, 0
    while n < 100:
        a = rng.uniform(0.0, 0.99)
        x = rng.normal(size=3) * rng.uniform(1.0, 40.0)
        if K.ks_radius(x, a) < K.horizon_radius(a) * 0.8:
            continue
        p = rng.normal(size=3)
        g = K.hamiltonian_grad_x(x, p, a)
        eps = 1e-6 * max(1.0, np.linalg.norm(x))
        fd = np.array([(K.hamiltonian(x + eps * e, p, a) - K.hamiltonian(x - eps * e, p, a))
                       / (2 * eps) for e in np.eye(3)])  # fmt: skip
        worst = max(worst, float(np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd)))))
        n += 1
    check("test 3: dH/dx analytic vs finite difference (100 points)", 0.0, worst, atol=1e-7)


@pytest.mark.parametrize("a", [0.5, 0.9, 0.99])
def test_4_shadow_shape(a):
    """Camera at D = 1e4 in the equatorial plane: rays 0.2% inside the exact critical curve
    (Bardeen 1973) are captured, 0.2% outside escape, at both beta = 0 edges; for a = 0.9 also
    at the top (alpha = 2a, beta = 3 sqrt 3)."""
    want = {0.5: (-4.096267, 6.138156), 0.9: (-2.844421, 6.832319), 0.99: (-2.251724, 6.983323)}
    edges = K.shadow_edge_alpha(a)
    check(f"test 4: exact edges, a = {a}", want[a], edges, atol=2e-6)
    got = [(_status(a, e * 0.998), _status(a, e * 1.002)) for e in edges]
    check(f"test 4: a = {a}, edges 0.2% in / out", [("captured", "escaped")] * 2, got)
    if a == 0.9:
        top = (_status(a, 1.8, 3 * np.sqrt(3) * 0.998), _status(a, 1.8, 3 * np.sqrt(3) * 1.002))
        check("test 4: a = 0.9 top (alpha 1.8, beta 5.196) in / out", ("captured", "escaped"), top)


def test_5_flat_side_faces_the_approaching_side():
    """Spin +z, camera on +x, up +z: screen right is +y, where the rotation (z x y = -x) moves
    gas AWAY from the camera; so the flat edge (closer to the centre) must be on screen left."""
    a = 0.9
    left, right = -2.844421, 6.832319
    check("test 5: flat edge on screen left (approaching side): captured just inside",
          "captured", _status(a, left * 0.998))  # fmt: skip
    check("test 5: the same offset on screen right is deep in the shadow", "captured",
          _status(a, -left * 1.5))  # fmt: skip
    check("test 5: screen left at 1.5 x the flat edge escapes", "escaped", _status(a, left * 1.5))
    check("test 5: far edge on screen right", ("captured", "escaped"),
          (_status(a, right * 0.998), _status(a, right * 1.002)))  # fmt: skip


def test_fixed_steps_match_the_reference_at_spin_0_9():
    """The renderer's fixed-step rays (render/kerr_rays.py, the GPU's step rule) against DOP853
    for the story's camera distance, a = 0.9: 30 random rays and 30 within +-0.05 of the
    prograde shadow edge (where rays skim the hole): same fate, sky directions within 2e-4 rad.
    (a = 0.99 fails this with fixed steps: AI_LOG 2026-11-23; not used.)"""
    from nebula.render import kerr_rays as KR

    a = 0.9
    xc = np.array([270.0, 50.0, 70.0])
    rc = np.linalg.norm(xc)
    fwd = -xc / rc
    e1 = np.cross(fwd, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, fwd)
    c0, amat = KR.camera_map(xc, a)
    rng = np.random.default_rng(2)
    bs = list(rng.uniform(0.0, 9.0, 30)) + list(2.925 + rng.uniform(-0.05, 0.05, 30))
    bad, worst = 0, 0.0
    for k, b in enumerate(bs):
        ph = rng.uniform(0.0, 2.0 * np.pi) if k < 30 else np.pi
        d = np.cos(b / rc) * fwd + np.sin(b / rc) * (np.cos(ph) * e1 + np.sin(ph) * e2)
        _, st, sky = KR.ray_polyline(xc, d, a, 8000.0, c0, amat)
        s2, _, pe, _ = K.trace_ray(xc, K.flat_to_local(xc, d), a, r_escape=1e8)
        if {"captured": KR.RAY_CAPTURED, "escaped": KR.RAY_ESCAPED}.get(s2, -1) != st:
            bad += 1
        elif st == KR.RAY_ESCAPED:
            ex = -pe / np.linalg.norm(pe)
            worst = max(worst, float(np.arcsin(min(1.0, np.linalg.norm(np.cross(ex, sky))))))
    check("fixed steps vs DOP853, a = 0.9: rays with a different fate (60)", 0, bad)
    check("fixed steps vs DOP853, a = 0.9: sky directions (rad)", 0.0, worst, atol=2e-4)
