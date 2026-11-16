"""Phase 8 step 6a: the director (phases/phase-08-gas-rendering.md 8.5)."""

from pathlib import Path

import numpy as np
import pytest

from nebula.render.director import (
    CODE_DAY,
    FramePlan,
    Shot,
    ShotError,
    Timeline,
    plan_frames,
    smoothstep,
    spherical,
)
from tests.helpers import check

V = np.array([3.0, -1.0, 0.5])  # synthetic gas velocity (code units)


@pytest.fixture
def runs(tmp_path):
    """A synthetic run: 11 snapshots, days 10-20, 50 particles moving in a straight line
    x = x0 + V t (so every centre is known exactly); particle 0 is swallowed after day 15."""
    rng = np.random.default_rng(1)
    x0 = rng.normal(0.0, 5.0, (50, 3)) + [200.0, 0.0, 0.0]
    folder = tmp_path / "synth" / "snapshots"
    folder.mkdir(parents=True)
    for k, day in enumerate(np.linspace(10.0, 20.0, 11)):
        t = day * CODE_DAY
        keep = np.arange(50) if day <= 15 else np.arange(1, 50)
        x = x0 + V * (t - 10.0 * CODE_DAY) / CODE_DAY
        np.savez(folder / f"snap_{k:05d}.npz", t=t, ids=keep, x=x[keep].astype(np.float32))
    return tmp_path, x0


def write_shot(tmp_path, body, name="s"):
    p = tmp_path / f"{name}.toml"
    p.write_text('name = "test"\nfps = 24\nwidth = 64\nheight = 36\n' + body +
                 '\n[[sources]]\nrun = "synth"\n')  # fmt: skip
    return p


ORBIT = """
duration = 4.0
[[camera]]
t = 0.0
r = 50.0
elev = 20.0
az = 0.0
vfov = 50.0
[[camera]]
t = 2.0
r = 8.0
elev = 5.0
az = 180.0
vfov = 70.0
[[camera]]
t = 4.0
r = 30.0
elev = 30.0
az = 360.0
vfov = 60.0
[[time]]
t = 0.0
day = 11.0
[[time]]
t = 1.0
day = 11.2
[[time]]
t = 4.0
day = 19.0
"""


def test_keyframes_hit_and_rules_hold(runs):
    """The path passes through every keyframe exactly; r never dips below the smallest
    keyframe r (no overshoot), so the r >= 6 rule holds; the time warp increases; the camera
    looks at the hole (forward = -position direction)."""
    tmp, _ = runs
    shot = Shot.load(write_shot(tmp, ORBIT))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    check("director: 96 frames for 4 s at 24 fps", 96, len(plan.video_t))
    for t, r, el, az, fov in ((0, 50, 20, 0, 50), (2, 8, 5, 180, 70)):
        i = int(t * 24)
        check(f"director: keyframe at t = {t} s, position", spherical(r, el, az),
              plan.position[i], atol=1e-9)  # fmt: skip
        check(f"director: keyframe at t = {t} s, vfov", float(fov), plan.vfov[i], rtol=1e-12)
    check("director: time warp hits day 11.2 at t = 1 s", 11.2 * CODE_DAY, plan.sim_t[24],
          rtol=1e-12)  # fmt: skip
    check("director: time warp strictly increasing", True, bool(np.all(np.diff(plan.sim_t) > 0)))
    r = np.linalg.norm(plan.position, axis=1)
    check("director: r never below the smallest keyframe r (8)", 8.0, float(r.min()), rtol=1e-12)
    f = plan.look_at - plan.position
    f /= np.linalg.norm(f, axis=1)[:, None]
    check("director: looks at the hole", -plan.position / r[:, None], f, atol=1e-12)


def test_path_has_no_corners(runs):
    """Smooth path (8.5: no sudden turns): the camera velocity is continuous. Sampled 100x
    finer than the frames, the largest change of velocity between neighbouring samples is
    tiny next to the speed (a corner would show a jump of order the speed itself)."""
    tmp, _ = runs
    shot = Shot.load(write_shot(tmp, ORBIT))
    shot.fps = 2400
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    vel = np.diff(plan.position, axis=0) * shot.fps
    jump = np.linalg.norm(np.diff(vel, axis=0), axis=1).max()
    speed = np.linalg.norm(vel, axis=1).max()
    check("director: largest velocity change between samples / top speed", 0.0, jump / speed,
          atol=0.01)  # fmt: skip
    slope = np.diff(plan.sim_t) * shot.fps
    check("director: time warp slope continuous (largest change between samples / top slope)",
          0.0, float(np.abs(np.diff(slope)).max() / slope.max()), atol=0.01)  # fmt: skip


@pytest.mark.parametrize(
    "change, message",
    [
        (("r = 8.0", "r = 5.0"), "r = 5.0 < 6"),
        (("day = 11.2", "day = 10.9"), "days must increase"),
        (("day = 19.0", "day = 21.0"), "outside the sources"),
        (("elev = 5.0", "elev = 85.0"), "|elev| must be"),
        (("t = 4.0\nr = 30.0", "t = 5.0\nr = 30.0"), "must lie in"),
        (("az = 0.0", 'az = 0.0\nlook_at = "nowhere"'), "unknown target"),
        (("duration = 4.0", "duration = 4.0\n[[camera]]\nt = 4.0\nr = 1"), "increase strictly"),
    ],
)
def test_bad_shots_rejected(runs, change, message):
    """Shots that break a rule are refused before rendering, with a message naming it."""
    tmp, _ = runs
    with pytest.raises(ShotError, match=message.replace("|", r"\|")):
        shot = Shot.load(write_shot(tmp, ORBIT.replace(*change, 1)))
        plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    check(f"director: refused ({message})", True, True)


def test_camera_too_close_between_keyframes_and_vertical_view_refused(runs):
    """A camera around a moving target can pass close to the hole between keyframes, and a
    camera above its target looks straight down: both are found on the fine check grid."""
    tmp, _ = runs
    near = """
duration = 2.0
[targets.gas]
type = "com"
[[camera]]
t = 0.0
r = 200.0
elev = 0.0
az = 180.0
around = "gas"
look_at = "gas"
[[time]]
t = 0.0
day = 10.0
[[time]]
t = 2.0
day = 12.0
"""
    with pytest.raises(ShotError, match="camera comes to r"):
        shot = Shot.load(write_shot(tmp, near))
        plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    top = near.replace("r = 200.0\nelev = 0.0", "r = 50.0\nelev = 80.0")
    shot = Shot.load(write_shot(tmp, top))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))  # 80 deg is allowed
    straight = top.replace('look_at = "gas"', "look_at = [200.0, 0.0, -1000.0]")
    with pytest.raises(ShotError, match="straight up or down"):
        plan_frames(Shot.load(write_shot(tmp, straight)), Timeline.from_sources(shot.sources, tmp))
    check("director: close pass and vertical view refused; 80 deg elevation allowed", 48,
          len(plan.video_t))  # fmt: skip


def test_follows_moving_targets(runs):
    """Camera fixed relative to the gas centre ("com"), looking at it: offset camera - centre
    stays as the keyframe says, and the centre follows the gas (x0 mean + V t, with the
    swallowed particle's exit between days 15 and 16 showing only as a small step in the
    snapshot means). A "group" target (the 10 particles nearest the hole at day 12) follows
    those 10 particles."""
    tmp, x0 = runs
    body = """
duration = 2.0
[targets.gas]
type = "com"
[targets.ten]
type = "group"
n = 10
pick_day = 12.0
[[camera]]
t = 0.0
r = 40.0
elev = 10.0
az = 90.0
around = "gas"
look_at = "gas"
[[camera]]
t = 1.0
r = 40.0
elev = 10.0
az = 90.0
around = "gas"
look_at = "ten"
[[camera]]
t = 2.0
r = 40.0
elev = 10.0
az = 90.0
around = "gas"
look_at = "ten"
[[time]]
t = 0.0
day = 10.0
[[time]]
t = 2.0
day = 14.0
"""
    shot = Shot.load(write_shot(tmp, body))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    days = plan.sim_t / CODE_DAY - 10.0
    centre = x0.astype(np.float32).astype(float).mean(axis=0) + V * days[:, None]
    check("director: camera - gas centre = keyframe offset",
          np.broadcast_to(spherical(40, 10, 90), centre.shape), plan.position - centre,
          atol=1e-4)  # fmt: skip
    check("director: looks at the gas centre at t = 0", centre[0], plan.look_at[0], atol=1e-4)
    x12 = x0 + V * 2.0
    ten = np.argsort(np.linalg.norm(x12.astype(np.float32).astype(float), axis=1))[:10]
    ten_c = x0[ten].astype(np.float32).astype(float).mean(axis=0) + V * days[:, None]
    check("director: looks at the group from t = 1 s", ten_c[24:], plan.look_at[24:], atol=1e-4)
    w = smoothstep(plan.video_t[:24] / 1.0)[:, None]  # keyframes at t = 0 and 1 s
    check("director: look-at turns from gas centre to group by the smoothstep blend",
          (1 - w) * centre[:24] + w * ten_c[:24], plan.look_at[:24], atol=1e-4)  # fmt: skip


def test_plan_saved_and_reloaded(runs, tmp_path):
    tmp, _ = runs
    shot = Shot.load(write_shot(tmp, ORBIT))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    plan.save(tmp_path / "plan.npz")
    again = FramePlan.load(tmp_path / "plan.npz")
    for k in ("video_t", "sim_t", "position", "look_at", "vfov", "stops"):
        check(f"director: saved plan {k} identical", getattr(plan, k), getattr(again, k),
              rtol=0.0, atol=0.0)  # fmt: skip
    cam = again.camera(48, 64, 36)
    check("director: camera of frame 48 at the planned position", plan.position[48],
          np.array(cam.position), rtol=0.0, atol=0.0)  # fmt: skip


def test_sources_join_by_day_range(runs):
    """Two sources (8.2: normal snapshots, then a dense re-run of a stretch): each keeps its
    day range, the shared junction snapshot appears once (from the re-run), and sources that
    overlap are refused instead of one silently hiding the other."""
    tmp, x0 = runs
    folder = tmp / "dense" / "snapshots"
    folder.mkdir(parents=True)
    for k, day in enumerate(np.linspace(15.0, 20.0, 21)):  # every 0.25 day from day 15
        t = day * CODE_DAY
        x = x0[1:] + V * (t - 10.0 * CODE_DAY) / CODE_DAY
        np.savez(folder / f"snap_{k:05d}.npz", t=t, ids=np.arange(1, 50),
                 x=x.astype(np.float32))  # fmt: skip
    tl = Timeline.from_sources([{"run": "synth", "to_day": 15.0},
                                {"run": "dense", "from_day": 15.0}], tmp)  # fmt: skip
    days = tl.times / CODE_DAY
    check("sources: 6 normal (days 10-14 + junction from the re-run) + 20 dense snapshots", 26,
          len(tl.times))  # fmt: skip
    check("sources: times strictly increasing", True, bool(np.all(np.diff(days) > 0)))
    check("sources: junction snapshot taken from the re-run", "dense",
          tl.paths[5].parent.parent.name)  # fmt: skip
    check("sources: days 10-15 then 15.25-20", (10.0, 15.0, 15.25, 20.0),
          tuple(np.round(days[[0, 5, 6, -1]], 9)))  # fmt: skip
    with pytest.raises(ShotError, match="before the previous source ends"):
        Timeline.from_sources([{"run": "synth"}, {"run": "dense"}], tmp)
    with pytest.raises(ShotError, match="pick_day outside"):
        from nebula.render.director import target_track

        target_track({"type": "group", "n": 3, "pick_day": 30.0}, tl)


def test_camera_in_gas_check(tmp_path):
    """camera_in_gas flags a camera within 1.5 h of a particle and only then."""
    from nebula.render.shot_render import camera_in_gas

    folder = tmp_path / "g" / "snapshots"
    folder.mkdir(parents=True)
    np.savez(folder / "snap_00000.npz", t=0.0, x=np.array([[100.0, 0, 0]], np.float32),
             h=np.array([2.0], np.float32), ids=np.array([0]))  # fmt: skip
    tl = Timeline(np.array([0.0]), [folder / "snap_00000.npz"])
    pos = np.array([[100.0, 2.9, 0], [100.0, 3.1, 0], [50.0, 0, 0]])
    plan = FramePlan(np.zeros(3), np.zeros(3), pos, np.zeros((3, 3)), np.full(3, 60.0),
                     np.zeros(3))  # fmt: skip
    check("camera in gas: flagged at 1.45 h, not at 1.55 h or far away", (True, False, False),
          tuple(camera_in_gas(plan, tl)))  # fmt: skip


DENSE = Path(__file__).resolve().parents[2] / "runs" / "dense_B192"


@pytest.mark.skipif(not DENSE.exists(), reason="run dense_B192 not on this machine")
def test_shot_frame_at_a_snapshot_equals_direct_render():
    """Glue check: a shot frame placed exactly at a snapshot time renders the same GPU picture
    as the snapshot rendered directly (snapshot_scene.snapshot_textures), bit for bit."""
    from nebula.render.camera import Camera
    from nebula.render.gpu_march import OUT_PICTURE, render_gas, upload_scene
    from nebula.render.march import GreyBlackbody
    from nebula.render.shot_render import ShotRenderer
    from nebula.render.snapshot_scene import snapshot_textures
    from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

    tl = Timeline.from_sources([{"run": "dense_B192"}], DENSE.parent)
    k = 100
    pos = np.array([[30.0, 25.0, 12.0]])
    plan = FramePlan(np.zeros(1), tl.times[[k]], pos, np.zeros((1, 3)), np.array([60.0]),
                     np.zeros(1))  # fmt: skip
    shot = Shot("t", 24, 1.0, 96, 54, 1, 1e7, [], {}, [], [])
    sky = upload_cube_map(build_cube_map(StarFieldConfig(face_size=256)))
    pic = ShotRenderer(shot, plan, tl, sky=sky).picture(0)
    texs, t = snapshot_textures(DENSE, k)
    cam = Camera(position=tuple(pos[0]), vfov_deg=60.0, width=96, height=54)
    ref, _ = render_gas(cam, upload_scene(texs, GreyBlackbody(kappa=1e7)), sky=sky,
                        out_mode=OUT_PICTURE)  # fmt: skip
    check("shot frame at snapshot 100 = direct render of snapshot 100", ref, pic, rtol=0.0,
          atol=0.0)  # fmt: skip
    check("shot frame time = snapshot time", t, float(plan.sim_t[0]), rtol=0.0, atol=0.0)
