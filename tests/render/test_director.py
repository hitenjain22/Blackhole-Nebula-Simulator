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
    fade_factor,
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


BOXED = """
duration = 2.0
[targets.gas]
type = "com"
[[camera]]
t = 0.0
r = 300.0
elev = 20.0
az = 0.0
around = "gas"
look_at = "gas"
[[time]]
t = 0.0
day = 10.0
[[time]]
t = 2.0
day = 14.0
[grid]
r_max = 3000.0
nr = 200
[[boxes]]
around = "gas"
size = 150.0
n = 64
"""


def test_detail_box_follows_target(runs, tmp_path):
    """8.3 item 2: a detail box whose centre follows the gas centre, saved with the plan; the
    grid settings come from the shot."""
    tmp, x0 = runs
    shot = Shot.load(write_shot(tmp, BOXED))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    days = plan.sim_t / CODE_DAY - 10.0
    centre = x0.astype(np.float32).astype(float).mean(axis=0) + V * days[:, None]
    check("box: centre follows the gas centre", centre, plan.box_centres[:, 0], atol=1e-4)
    b = plan.boxes(30, shot)[0]
    check("box: frame 30 as CartesianBox fields (size, n)", (150.0, 64), (b["size"], b["n"]))
    check("grid: shot settings", (3000.0, 200), (shot.log_grid().r_max, shot.log_grid().nr))
    plan.save(tmp_path / "p.npz")
    check("box: centres saved and reloaded", plan.box_centres,
          FramePlan.load(tmp_path / "p.npz").box_centres, rtol=0.0, atol=0.0)  # fmt: skip


@pytest.mark.parametrize(
    "change, message",
    [
        (("size = 150.0", "size = 6000.0"), "beyond the grid.s"),
        (("r = 300.0", "r = 2900.0"), "camera reaches r"),
        (("nr = 200", "nr = 200\nwrong = 1"), r"\[grid\]"),
        (("n = 64", "n = 64\n[[boxes]]\nsize = 9.0\n[[boxes]]\nsize = 9.0"), "at most 2"),
        (('around = "gas"\nsize = 150.0', 'around = "nowhere"\nsize = 150.0'), "unknown target"),
        (("size = 150.0", "size = 150.0\nsize_by_day = [[10, 100]]"), "either size"),
        (("size = 150.0", "size_by_day = [[12, 100], [11, 200]]"), "increasing days"),
        (("size = 150.0", "size_by_day = [[10, 100], [12, -5]]"), "increasing days"),
        (("size = 150.0", "size_by_day = [10, 100]"), "pairs"),
        (("size = 150.0", "size_by_day = [[10, 100], [14, 7000]]"), "beyond the grid.s"),
    ],
)
def test_bad_grid_and_boxes_rejected(runs, change, message):
    """A box or camera outside the grid (rays never reach / start outside), unknown grid
    settings, more than the shader's two boxes, and unknown box targets are refused."""
    tmp, _ = runs
    with pytest.raises(ShotError, match=message):
        shot = Shot.load(write_shot(tmp, BOXED.replace(*change, 1)))
        plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    check(f"grid/boxes: refused ({message})", True, True)


def test_box_grows_with_the_gas(runs, tmp_path):
    """A box with size_by_day: exact sizes at the key days, never outside the neighbouring
    keys' range between them (monotone: no overshoot, so the gas never pokes out), held
    beyond the last key, and saved with the plan."""
    tmp, _ = runs
    body = BOXED.replace("size = 150.0", "size_by_day = [[10, 100], [12, 150], [13, 400]]")
    shot = Shot.load(write_shot(tmp, body))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    days, size = plan.sim_t / CODE_DAY, plan.box_sizes[:, 0]
    check("growing box: size at day 10 (first key)", 100.0, size[0], rtol=1e-12)
    keyed = [int(np.argmin(np.abs(days - d))) for d in (12.0, 13.0)]
    want = np.interp(days[keyed], [10, 12, 13], np.log([100, 150, 400]))
    check("growing box: frames nearest days 12 and 13 (log-linear estimate, 2%)",
          np.exp(want), size[keyed], rtol=0.02)  # fmt: skip
    check("growing box: never shrinks (keys increase)", True, bool(np.all(np.diff(size) >= 0)))
    check("growing box: held at 400 after day 13", 400.0, size[days > 13.0].min(), rtol=1e-12)
    check("growing box: frame 0 as CartesianBox size", 100.0, plan.boxes(0, shot)[0]["size"],
          rtol=1e-12)  # fmt: skip
    plan.save(tmp_path / "p.npz")
    check("growing box: sizes saved and reloaded", size,
          FramePlan.load(tmp_path / "p.npz").box_sizes[:, 0], rtol=0.0, atol=0.0)  # fmt: skip


def test_gas_outside_boxes(tmp_path):
    """gas_outside_boxes counts the particles not deep inside a box (kernel reach 2h plus
    3 cells from the face): here 1 of 4."""
    from nebula.render.shot_render import gas_outside_boxes

    folder = tmp_path / "g" / "snapshots"
    folder.mkdir(parents=True)
    x = np.array([[0.0, 0, 0], [30, 0, 0], [0, -30, 0], [45, 0, 0]], np.float32)
    np.savez(folder / "snap_00000.npz", t=0.0, x=x, h=np.full(4, 2.0, np.float32),
             ids=np.arange(4))  # fmt: skip
    tl = Timeline(np.array([0.0]), [folder / "snap_00000.npz"])
    plan = FramePlan(np.zeros(1), np.zeros(1), np.array([[500.0, 0, 0]]), np.zeros((1, 3)),
                     np.full(1, 60.0), np.zeros(1), np.zeros((1, 1, 3)))  # fmt: skip
    shot = Shot.__new__(Shot)
    shot.boxes = [{"size": 100.0, "n": 100}]  # dx 1: deep inside means |x| <= 50 - 4 - 3 = 43
    check("gas outside boxes: 1 of 4 (the one at 45)", 0.25, gas_outside_boxes(plan, shot, tl)[0],
          rtol=1e-12)  # fmt: skip


def test_fixed_look_points_join_smoothly(runs):
    """Fixed look-at points are joined by one monotone cubic per axis: the keys are hit
    exactly and the aim does not stop at the middle key (smoothstep blending would)."""
    tmp, _ = runs
    body = BOXED.split("[grid]")[0].replace('look_at = "gas"', "look_at = [100.0, 0.0, 0.0]")
    body = body.replace(
        "[[time]]",
        """[[camera]]
t = 1.0
r = 300.0
elev = 20.0
az = 0.0
around = "gas"
look_at = [150.0, 30.0, 0.0]
[[camera]]
t = 2.0
r = 300.0
elev = 20.0
az = 0.0
around = "gas"
look_at = [250.0, 40.0, 0.0]
[[time]]""",
        1,
    )
    shot = Shot.load(write_shot(tmp, body))
    plan = plan_frames(shot, Timeline.from_sources(shot.sources, tmp))
    i1 = int(np.argmin(np.abs(plan.video_t - 1.0)))
    check("fixed look points: key at t = 1 hit", [150.0, 30.0, 0.0], plan.look_at[i1], atol=1e-9)
    speed = np.linalg.norm(np.diff(plan.look_at, axis=0), axis=1)
    check("fixed look points: aim still moving at the middle key (step > 0.3 x mean)", True,
          bool(speed[i1 - 1:i1 + 1].min() > 0.3 * speed.mean()))  # fmt: skip


def test_bricks_setting(runs):
    """[bricks] in a shot gives the lattice; refused together with boxes or with bad values."""
    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    shot = Shot.load(write_shot(tmp, plain + "[bricks]\ndx = 0.5\nr_fine = 800.0\n"))
    check("bricks setting: lattice (dx, r_fine)", (0.5, 800.0),
          (shot.brick_lattice().dx, shot.brick_lattice().r_fine))  # fmt: skip
    for body, msg in ((BOXED + "[bricks]\ndx = 1.0\n", "one or the other"),
                      (plain + "[bricks]\ndx = -1.0\n", "positive"),
                      (plain + "[bricks]\nsize = 3.0\n", r"\[bricks\]")):  # fmt: skip
        with pytest.raises(ShotError, match=msg):
            Shot.load(write_shot(tmp, body))
    check("bricks setting: bad settings refused", True, True)


def test_adaptation_stops():
    """Automatic exposure curve: flat probes give no change; a sudden 64x brightness jump
    becomes a smooth change (no frame-to-frame step over 0.1 stop at 24 fps, no overshoot)
    compensating `strength` of the 6 stops; bad [auto_exposure] settings are refused."""
    from nebula.render.shot_render import adaptation_stops

    pt = np.arange(0.0, 50.0, 1.0)
    ft = np.arange(0.0, 50.0, 1.0 / 24.0)
    flat = adaptation_stops(pt, np.full(len(pt), 3.0), ft, 4.0, 0.75)
    check("auto exposure: flat probes, no change", np.zeros(len(ft)), flat, atol=1e-12)
    e = np.where(pt < 25.0, 64.0, 1.0)  # gas 64x brighter after t = 25 s: exposure 64x lower
    s = adaptation_stops(pt, e, ft, 4.0, 0.75)
    check("auto exposure: total change = 0.75 x 6 stops (ends of a 50 s shot)", -4.5,
          s[ft > 45].mean() - s[ft < 5].mean(), rtol=0.01)  # fmt: skip
    check("auto exposure: largest step between frames < 0.1 stop", True,
          bool(np.abs(np.diff(s)).max() < 0.1))  # fmt: skip
    check("auto exposure: no overshoot beyond the probes' range", True,
          bool(s.max() - s.min() <= 4.5 + 1e-9))  # fmt: skip
    ramp = 2.0 ** (-0.3 * pt)  # gas brightening steadily: 0.3 stop per second
    sr = adaptation_stops(pt, ramp, ft, 2.0, 1.0)
    want = -0.3 * ft - np.median(-0.3 * pt)
    # 1e-6 stop: the reflection spans 6 smoothing widths (Gaussian tail beyond: ~1e-8)
    check("auto exposure: a steady trend is followed, up to both ends (1e-6 stop)", want, sr,
          atol=1e-6)  # fmt: skip


def test_auto_exposure_setting(runs):
    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    shot = Shot.load(write_shot(tmp, plain + "[auto_exposure]\nsmooth = 3.0\n"))
    check("auto exposure setting read", {"smooth": 3.0}, shot.auto_exposure)
    for body, msg in ((plain + "[auto_exposure]\nspeed = 1.0\n", "unknown"),
                      (plain + "[auto_exposure]\nstrength = 0.0\n", "positive")):  # fmt: skip
        with pytest.raises(ShotError, match=msg):
            Shot.load(write_shot(tmp, body))
    check("auto exposure setting: bad settings refused", True, True)


def test_colour_setting(runs):
    """Phase 9: colour = "true" selects true colour (default grey); other values are refused."""
    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    check("colour default", "grey", Shot.load(write_shot(tmp, plain)).colour)
    head, rest = plain.split("\n", 1)
    shot = Shot.load(write_shot(tmp, head + '\ncolour = "true"\n' + rest))
    check("colour = true read", "true", shot.colour)
    with pytest.raises(ShotError, match="colour"):
        Shot.load(write_shot(tmp, head + '\ncolour = "orange"\n' + rest))


def test_fade_out(runs):
    """The end fade (Hiten 2026-11-17): brightness 1 until fade_out seconds before the last
    frame, a smooth step down to exactly 0 on the last frame, never rising; fade_out = 0 (the
    default) changes nothing; a negative or too long fade is refused."""
    t = np.arange(0, 108 * 24) / 24.0
    f = fade_factor(t, t[-1], 6.0)
    start = t[-1] - 6.0
    check("fade: 1 before it starts", np.ones(np.sum(t <= start)), f[t <= start])
    check("fade: 0 on the last frame", 0.0, float(f[-1]))
    check("fade: half way at the middle", 0.5, float(fade_factor(start + 3.0, t[-1], 6.0)),
          atol=1e-12)  # fmt: skip
    check("fade: never rises", True, bool(np.all(np.diff(f) <= 0)))
    check("fade: smooth (largest frame-to-frame drop < 1/(fps x fade) x 1.5)", True,
          bool(np.max(-np.diff(f)) < 1.5 / (24 * 6.0)))  # fmt: skip
    check("fade: none when fade_out = 0", np.ones_like(t), fade_factor(t, t[-1], 0.0))
    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    check("fade_out default", 0.0, Shot.load(write_shot(tmp, plain)).fade_out)
    head, rest = plain.split("\n", 1)
    check(
        "fade_out read",
        1.5,
        Shot.load(write_shot(tmp, head + "\nfade_out = 1.5\n" + rest)).fade_out,
    )
    for bad in ("-1.0", "2.5"):  # negative, or longer than the 2 s test shot
        with pytest.raises(ShotError, match="fade_out"):
            Shot.load(write_shot(tmp, head + f"\nfade_out = {bad}\n" + rest))


def test_opacity_setting(runs):
    """opacity = "thermal" (phase-09 9.4 option 1) is read for true colour; default constant;
    unknown values and thermal without true colour are refused."""
    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    head, rest = plain.split("\n", 1)
    check("opacity default", "constant", Shot.load(write_shot(tmp, plain)).opacity)
    shot = Shot.load(write_shot(tmp, head + '\ncolour = "true"\nopacity = "thermal"\n' + rest))
    check("opacity thermal read", "thermal", shot.opacity)
    for bad in ('\ncolour = "true"\nopacity = "fog"\n', '\nopacity = "thermal"\n'):
        with pytest.raises(ShotError, match="opacity"):
            Shot.load(write_shot(tmp, head + bad + rest))


def test_bricks_max_support_by_day(runs):
    """[bricks] max_support_by_day: the widest particle support kept on the bricks steps to each
    value from its day on (full_story: 4 through the pass, 1 after day 105, where spread-out gas
    cost ~90,000 bricks per frame); default 4; bad pairs refused."""
    from nebula.render.bricks import MAX_SUPPORT_BRICKS

    tmp, _ = runs
    plain = BOXED.split("[grid]")[0]
    shot = Shot.load(write_shot(tmp, plain + "[bricks]\ndx = 1.0\n"))
    check("max_support default", float(MAX_SUPPORT_BRICKS), shot.brick_lattice(200.0).max_support)
    body = plain + "[bricks]\ndx = 1.0\nmax_support_by_day = [[0.0, 4.0], [105.0, 1.0]]\n"
    shot = Shot.load(write_shot(tmp, body))
    got = [shot.brick_lattice(d).max_support for d in (None, 10.0, 104.99, 105.0, 300.0)]
    check("max_support steps by day", [4.0, 4.0, 4.0, 1.0, 1.0], got)
    body2 = plain + "[bricks]\ndx_by_day = [[0.0, 1.0], [105.0, 2.0]]\n"
    shot2 = Shot.load(write_shot(tmp, body2))
    got2 = [shot2.brick_lattice(d).dx for d in (None, 104.99, 105.0)]
    check("dx steps by day (full_story: 2-unit cells after day 105)", [1.0, 1.0, 2.0], got2)
    with pytest.raises(ShotError, match="dx_by_day"):
        Shot.load(write_shot(tmp, plain + "[bricks]\ndx_by_day = [[0.0, 1.0], [105.0, -2.0]]\n"))
    with pytest.raises(ShotError, match="max_support_by_day"):
        Shot.load(write_shot(tmp, plain + "[bricks]\nmax_support_by_day = [[0.0, -1.0]]\n"))
