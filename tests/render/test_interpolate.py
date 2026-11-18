"""Phase 8 step 5: gas between snapshots (phases/phase-08-gas-rendering.md 8.2; 8.6 test 8)."""

from pathlib import Path

import numpy as np
import pytest

from nebula.physics.orbits import apocentre_state
from nebula.render.interpolate import (
    ETA_ORB,
    FADE,
    Snap,
    advance,
    blend_vectors,
    fade_weight,
    load_snap,
    state_at,
)
from nebula.sim.integrator import integrate_orbit
from tests.helpers import check

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "runs" / "R6.1_100k" / "snapshots" / "snap_00160.npz"
INTERVAL = 6801.46875  # R6.1_100k snapshot spacing (code time; 1.67 days)


def _orbits():
    """Eccentric, close and inclined test orbits (x, p) in float64."""
    xs, ps = [], []
    for r_peri, r_apo, tilt in ((8.0, 300.0, 0.0), (10.0, 60.0, 0.7), (40.0, 2000.0, 1.3)):
        x, p, _ = apocentre_state(r_peri, r_apo)
        c, s = np.cos(tilt), np.sin(tilt)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        xs.append(rot @ x)
        ps.append(rot @ p)
    return np.array(xs), np.array(ps)


def test_advance_matches_phase1_integrator():
    """8.6 test 8a: the interpolation code fed float64 states equals the Phase 1 gravity-only
    generalized leapfrog (sim.integrator.integrate_orbit, same eta) to 1e-12, forward and,
    by time reversal, backward."""
    xs, ps = _orbits()
    for sign, name in ((1.0, "forward"), (-1.0, "backward")):
        x, p, alive = advance(xs, ps, sign * 20000.0, eta=ETA_ORB)
        for i in range(len(xs)):
            o = integrate_orbit(xs[i], sign * ps[i], 20000.0, eta=ETA_ORB)
            check(f"test 8a, orbit {i} {name}: x", o.x[-1], x[i], rtol=1e-12, per_vector=True)
            check(f"test 8a, orbit {i} {name}: p", sign * o.p[-1], p[i], rtol=1e-12,
                  per_vector=True)  # fmt: skip
        check(f"test 8a, {name}: all alive", True, bool(alive.all()))


def _two_snaps(x0, p0, dt, eta=ETA_ORB / 16):
    """Snapshots a (t = 0) and b (t = dt) of a gravity-only 'run' (fine steps), stored in
    float32 like real snapshots; returns (a, b, truth function t -> float64 positions)."""
    xb, pb, alive = advance(x0, p0, dt, eta)
    ids = np.arange(len(x0))

    def f32(v):
        return v.astype(np.float32).astype(float)

    a = Snap(0.0, ids, f32(x0), f32(p0), {"h": np.full(len(x0), 1.0)})
    b = Snap(dt, ids[alive], f32(xb[alive]), f32(pb[alive]), {"h": np.full(alive.sum(), 3.0)})
    return a, b, lambda t: advance(x0, p0, t, eta)[0]


def test_endpoints_scalars_and_no_jump():
    """At the snapshot times the frame is the snapshot itself; scalars blend linearly; and the
    cross-fade leaves no jump: the change of position between frames stays smooth up to and
    across the next snapshot (largest step-to-step change within 1.5x of its neighbours)."""
    xs, ps = _orbits()
    a, b, _ = _two_snaps(xs, ps, INTERVAL)
    ids, x, s = state_at(a, b, 0.0)
    check("frame at snapshot a: positions", a.x, x, rtol=0.0, atol=0.0)
    ids, x, s = state_at(a, b, INTERVAL)
    check("frame at snapshot b: positions", b.x, x, rtol=1e-15, per_vector=True)
    _, _, s = state_at(a, b, 0.25 * INTERVAL)
    check("scalars at a quarter: 1 + 0.25 x (3 - 1)", np.full(3, 1.5), s["h"], rtol=1e-14)
    check("fade weight: 0 before the last 20%, 1 at the end", (0.0, 0.0, 1.0),
          (fade_weight(0.5), fade_weight(1 - FADE), fade_weight(1.0)))  # fmt: skip
    ts = np.linspace(0.7, 1.0, 61) * INTERVAL
    xs_t = np.array([state_at(a, b, t)[1] for t in ts])
    steps = np.linalg.norm(np.diff(xs_t, axis=0), axis=2)  # (60, particles)
    rough = steps[1:-1] / (0.5 * (steps[:-2] + steps[2:]))
    check("no jump: largest step / mean of its neighbours", 1.0, float(np.max(np.abs(rough))),
          atol=0.5)  # fmt: skip


def test_swallowed_particle_is_dropped():
    """A particle on a plunging orbit, absent from the next snapshot, follows its orbit until
    it crosses r_drop and is then dropped; before that it is shown."""
    x0 = np.array([[30.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    p0 = np.array([[0.0, 0.02, 0.0], [0.0, 0.1, 0.0]])  # first one plunges
    a, b, _ = _two_snaps(x0, p0, 2000.0)
    check("plunging particle missing from b", 1, len(b.ids))
    shown_early = 0 in state_at(a, b, 50.0)[0]
    gone_late = 0 not in state_at(a, b, 1999.0)[0]
    check("plunging particle shown early, dropped after crossing r_drop", (True, True),
          (shown_early, gone_late))  # fmt: skip


def test_fallen_forward_path_never_crosses_the_hole():
    """A particle present in both snapshots whose forward path falls in while its backward path
    does not (gas pushed out by pressure in the run; the orbit-only forward path plunges). As
    in R6.6 frame 2016 (AI_LOG 2026-11-18): the old straight blend of the frozen forward state
    and the backward state on the far side put it at r = 1.28, inside the horizon. It must stay
    outside r_drop at every frame and take the backward state alone in the fade window."""
    dt = 2000.0
    xa, pa = np.array([[30.0, 0.0, 0.0]]), np.array([[0.0, 0.02, 0.0]])  # plunges forward
    xb, pb = np.array([[-40.0, 5.0, 0.0]]), np.array([[0.0, -0.158, 0.0]])  # ~circular at 40
    ids = np.arange(1)
    a = Snap(0.0, ids, xa, pa, {"h": np.ones(1)})
    b = Snap(dt, ids, xb, pb, {"h": np.ones(1)})
    check("set-up: forward path falls in", False, bool(advance(xa, pa, dt)[2][0]))
    r_min, worst = np.inf, 0.0
    for t in np.linspace(0.0, dt, 201):
        out_ids, x, p, _ = state_at(a, b, t, with_p=True)
        if len(out_ids):
            r_min = min(r_min, float(np.linalg.norm(x[0])))
        if fade_weight(t / dt) > 0.0:
            back = advance(xb, pb, t - dt)  # backward from b
            worst = max(worst, float(np.abs(x[0] - back[0][0]).max()))
            worst = max(worst, float(np.abs(p[0] - back[1][0]).max()))
    check("fallen forward path: closest approach >= r_drop 4", True, r_min >= 4.0)
    check("fallen forward path: backward state alone in the fade window", 0.0, worst, atol=1e-12)


def test_blend_never_cuts_through_the_hole():
    """blend_vectors: exact ends; for two positions on opposite sides of the hole (R6.6 frame
    2016: forward r = 35, backward r = 16 at 160 deg) the blend stays at r >= 16 where the
    straight blend reached r = 1.28 (AI_LOG 2026-11-18); for nearby points (the usual case)
    it is the straight blend to second order."""
    rng = np.random.default_rng(3)
    u = np.array([[35.0, 0.0, 0.0]])
    v = 16.0 * np.array([[np.cos(np.radians(160)), np.sin(np.radians(160)), 0.0]])
    ws = np.linspace(0.0, 1.0, 1001)
    r = [float(np.linalg.norm(blend_vectors(u, v, w))) for w in ws]
    straight = [float(np.linalg.norm((1 - w) * u + w * v)) for w in ws]
    print(f"    closest approach: blend {min(r):.2f}, straight {min(straight):.2f}")
    check("blend: closest approach >= nearer end (16)", True, min(r) >= 16.0 - 1e-9)
    check("blend: w = 0 is the first vector", u, blend_vectors(u, v, 0.0), atol=0.0, rtol=0.0)
    check("blend: w = 1 is the second vector", v, blend_vectors(u, v, 1.0), atol=0.0, rtol=0.0)
    a = rng.normal(0, 100, (500, 3))
    b = a + rng.normal(0, 0.5, (500, 3))  # forward and backward states 0.5 apart at r ~ 170
    w = rng.uniform(0, 1, 500)
    near = blend_vectors(a, b, w) - ((1 - w)[:, None] * a + w[:, None] * b)
    check("blend: nearby points = straight blend (|diff| < 0.01)", True,
          float(np.abs(near).max()) < 0.01)  # fmt: skip
    check("blend: exactly opposite is finite", True,
          bool(np.isfinite(blend_vectors(u, -u, 0.5)).all()))  # fmt: skip


@pytest.mark.skipif(not REAL.exists(), reason="run R6.1_100k not on this machine")
def test_interpolated_positions_vs_fine_run():
    """8.6 test 8b: ~2,100 real particles (snapshot 160 of R6.1_100k, including all 136 within
    r = 60) in a gravity-only 'run' with 16x finer steps, stored in float32 like real
    snapshots. Interpolated positions at mid-interval match the run to 1e-5 of r (largest
    particle), and the error falls ~4x when eta_orb is halved (second order); at 90% of the
    interval (inside the cross-fade) the float32 snapshots set the floor.

    The plan's reference is `nebula resim` writing snapshots at the frame times; between
    snapshots the interpolation is gravity-only, so a gravity-only run with much finer steps is
    the same reference without a resimulation tool (DECISIONS 2026-11-15)."""
    snap = load_snap(REAL)
    r = np.linalg.norm(snap.x, axis=1)
    rng = np.random.default_rng(0)
    sel = np.unique(np.r_[rng.choice(np.nonzero(r > 4.5)[0], 2000, replace=False),
                          np.nonzero((r > 4.5) & (r < 60))[0]])  # fmt: skip
    a, b, truth = _two_snaps(snap.x[sel], snap.p[sel], INTERVAL)
    errs = {}
    for frac in (0.5, 0.9):
        xt = truth(frac * INTERVAL)
        for k, eta in enumerate((ETA_ORB, ETA_ORB / 2)):
            ids, x, _ = state_at(a, b, frac * INTERVAL, eta=eta)
            e = np.linalg.norm(x - xt[ids], axis=1) / np.linalg.norm(xt[ids], axis=1)
            errs[frac, k] = e.max()
            check(f"test 8b, {frac:.0%} of interval, eta_orb / {2**k}: worst error / r", 0.0,
                  e.max(), atol=1e-5)  # fmt: skip
    check("test 8b: error ratio when eta_orb is halved (mid-interval)", 4.0,
          errs[0.5, 0] / errs[0.5, 1], rtol=0.25)  # fmt: skip


def test_momenta_at_endpoints_and_smooth():
    """Phase 9 (Doppler colours): state_at(with_p=True) gives snapshot a's momenta at a.t,
    snapshot b's at b.t, the same positions as without p, and through the cross-fade momenta
    that follow the true (16x finer, float64) orbit to 1e-5 of |p| with no jump (the frame-to-
    frame change of the error stays below 1e-5 of |p|; the limit of test 8b for positions).
    A step-ratio test as for positions does not apply: at r = 8 the true momentum's own steps
    vary by 1.63x between frames (pericentre)."""
    xs, ps = _orbits()
    a, b, _ = _two_snaps(xs, ps, INTERVAL)
    _, x, p, _ = state_at(a, b, 0.0, with_p=True)
    check("momenta at snapshot a", a.p, p, rtol=0.0, atol=0.0)
    check("positions unchanged by with_p", state_at(a, b, 0.0)[1], x, rtol=0.0, atol=0.0)
    _, _, p, _ = state_at(a, b, INTERVAL, with_p=True)
    check("momenta at snapshot b", b.p, p, rtol=1e-13, per_vector=True)
    ts = np.linspace(0.7, 1.0, 61) * INTERVAL
    ps_t = np.array([state_at(a, b, t, with_p=True)[2] for t in ts])
    truth = np.array([advance(xs, ps, t, ETA_ORB / 16)[1] for t in ts])
    norm = np.linalg.norm(truth, axis=2)
    err = np.linalg.norm(ps_t - truth, axis=2) / norm
    check("momenta vs true orbit through the fade (max rel)", 0.0, float(err.max()), atol=1e-5)
    jump = np.linalg.norm(np.diff(ps_t - truth, axis=0), axis=2) / norm[1:]
    check("momenta: no jump (largest frame-to-frame error change / |p|)", 0.0,
          float(jump.max()), atol=1e-5)  # fmt: skip
