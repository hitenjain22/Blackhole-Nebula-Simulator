"""Camera rays around a spinning hole as the GPU traces them: the readable float64 twin of the
KERR ray path in shaders/gas_march.wgsl (CLAUDE.md rule 2; phase-11 11.2 test 6).

Same equations as physics/kerr.py (Hamiltonian rays in Cartesian Kerr-Schild coordinates,
the arriving photon followed backward), integrated with fixed RK4 steps in the affine
parameter instead of an adaptive DOP853: step h = H_FAR r min(1, (r - r_h) / R_NEAR), so a
step is a fixed fraction of the radius far out (chords ~ H_FAR rad of arc, like the Phase 2 rays'
0.02 rad) and shrinks toward the horizon, where a backward ray creeps up to r_h. The step
rule is set by the accuracy tests against physics.kerr.trace_ray (test_kerr_rays_fixed.py).

A ray ends captured at r < r_h + HORIZON_MARGIN, escaped when going outward past r_max (then
continued with growing steps to r > R_SKY for its sky direction), or "orbiting" after
MAX_STEPS. The polyline of RK4 points is what the gas march walks along (chords).

The camera: physics.kerr.static_observer_frame and initial_momentum give the photon's
covariant momentum as a linear map of the flat pixel direction d,
p_mu proportional to C0_mu - A_mu_i d_i (C0 = g u, A = g [e_r e_th e_ph] [axes]), normalised
to p_t = -1; ``camera_map`` returns (C0, A), uploaded once per frame (``kerr_camera_bytes``).
"""

import numpy as np

from nebula.physics import kerr as K

H_FAR = 0.02  # step / r far from the hole
R_NEAR = 0.5  # steps shrink inside r_h + R_NEAR (was 2.0: 1.7x the steps, no gain at a = 0.9)
H_MIN = 1.0e-4  # smallest step (affine units)
R_SKY = 1.0e6  # sky direction taken here (remaining bending ~ 4 / R_SKY rad at most)
MAX_STEPS = 20000
RAY_ESCAPED, RAY_CAPTURED, RAY_ORBITING = 0, 1, 2  # as bent_rays / schw_ray.wgsl


def step_size(r: float, r_h: float) -> float:
    """RK4 step in the affine parameter (module doc). Mirrors kerr_step_size (WGSL)."""
    return max(H_FAR * r * min(1.0, (r - r_h) / R_NEAR), H_MIN)


def camera_map(x_cam, a: float):
    """(C0 (4,), A (4, 3)) with p_mu proportional to C0 - A d for a flat unit direction d
    (module doc; the same p as physics.kerr.initial_momentum(x_cam, flat_to_local(x_cam, d)))."""
    x_cam = np.asarray(x_cam, float)
    u, er, eth, eph = K.static_observer_frame(x_cam, a)
    g = K.metric(x_cam, a)
    rh = x_cam / np.linalg.norm(x_cam)
    ph = np.cross(np.array([0.0, 0.0, 1.0]), rh)
    if np.linalg.norm(ph) < 1e-12:
        ph = np.cross(np.array([1.0, 0.0, 0.0]), rh)
    ph /= np.linalg.norm(ph)
    th = np.cross(ph, rh)
    axes = np.stack([rh, th, ph])  # rows: n_local = axes @ d
    frame = np.stack([er, eth, eph], axis=1)  # (4, 3)
    return g @ u, g @ frame @ axes


def initial_p(c0, amat, d) -> np.ndarray:
    pc = np.asarray(c0, float) - np.asarray(amat, float) @ np.asarray(d, float)
    return pc[1:] / (-pc[0])


def _rk4(x, p, h, a):
    def f(xx, pp):
        return K.backward_rhs(xx, pp, a)

    k1x, k1p = f(x, p)
    k2x, k2p = f(x + 0.5 * h * k1x, p + 0.5 * h * k1p)
    k3x, k3p = f(x + 0.5 * h * k2x, p + 0.5 * h * k2p)
    k4x, k4p = f(x + h * k3x, p + h * k3p)
    return (x + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
            p + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p))  # fmt: skip


def ray_polyline(x_cam, d, a: float, r_max: float, c0=None, amat=None):
    """(points (n, 3), status, sky direction or None) of the camera ray along flat unit
    direction d (module doc). Captured: the last point is the first RK4 point inside
    r_h + HORIZON_MARGIN. Escaped: the last chord ends where r crosses r_max (linear in r
    along the step, as the GPU does)."""
    x = np.asarray(x_cam, float).copy()
    if c0 is None:
        c0, amat = camera_map(x, a)
    p = initial_p(c0, amat, d)
    r_h = K.horizon_radius(a)
    pts = [x.copy()]
    for _ in range(MAX_STEPS):
        r = float(K.ks_radius(x, a))
        x1, p1 = _rk4(x, p, step_size(r, r_h), a)
        r1 = float(K.ks_radius(x1, a))
        if r1 < r_h + K.HORIZON_MARGIN:
            pts.append(x1.copy())
            return np.array(pts), RAY_CAPTURED, None
        if r1 > r_max and r1 > r:
            t = min(max((r_max - r) / (r1 - r), 0.0), 1.0)
            pts.append(x + t * (x1 - x))
            return np.array(pts), RAY_ESCAPED, sky_direction(x1, p1, a)
        pts.append(x1.copy())
        x, p = x1, p1
    return np.array(pts), RAY_ORBITING, None


def sky_direction(x, p, a: float) -> np.ndarray:
    """Where an escaping ray's light came from: continue outward with steps 0.1 r to r > R_SKY,
    then the direction of dx/dlambda' (the backward ray's motion)."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    while K.ks_radius(x, a) < R_SKY:
        x, p = _rk4(x, p, 0.1 * float(K.ks_radius(x, a)), a)
    dx, _ = K.backward_rhs(x, p, a)
    return dx / np.linalg.norm(dx)


def kerr_camera_bytes(x_cam, a: float) -> bytes:
    """Binding 17 of the KERR gas shader: (a, r_h, f at the camera, 0), C0, then A's three
    columns as vec4 (float32)."""
    c0, amat = camera_map(x_cam, a)
    _, f_cam, _ = K.metric_parts(np.asarray(x_cam, float), a)
    rows = [[a, K.horizon_radius(a), float(f_cam), 0.0], list(c0)] + [
        list(amat[:, i]) for i in range(3)
    ]
    return np.asarray(rows, np.float32).tobytes()
