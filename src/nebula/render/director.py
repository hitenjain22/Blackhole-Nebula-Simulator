"""Camera and shots (phases/phase-08-gas-rendering.md 8.5): a shot file (TOML) -> the list of
frames (video time, simulation time, camera, exposure), computed once, checked, and saved.

Shot file layout (all times in video seconds unless named ``day``):

    name = "orbit_pass"
    fps = 24
    duration = 12.0
    width = 640                 # render settings
    height = 360
    n_sub = 1
    kappa = 1e7

    [[sources]]                 # snapshot runs in time order (8.2)
    run = "R6.1_100k_ph"
    to_day = 96.385             # optional day range of this source (inclusive)
    [[sources]]
    run = "dense_B192"          # e.g. a dense re-run of the hero stretch
    from_day = 96.385
    ease_days = 0.3             # optional: a different run from the one before; spread the
                                # jump between them over this many days (render.junctions)

    [targets.stream]            # named moving targets (besides "hole" and [x, y, z] points)
    type = "group"              # centre of the n particles nearest the hole at pick_day
    n = 2000
    pick_day = 97.0
                                # or type = "com": centre of all gas present

    [[camera]]                  # keyframes; at least one
    t = 0.0
    r = 60.0                    # distance from `around` (GM/c^2)
    elev = 20.0                 # degrees above the x-y plane (|elev| <= 80)
    az = 0.0                    # degrees round z; written unwrapped (0, 180, 360 = a full turn)
    around = "hole"             # target the position is measured from (default "hole")
    look_at = "hole"            # target the camera looks at (default "hole")
    vfov = 60.0                 # vertical field of view, degrees

    [[time]]                    # time warp keyframes: video t -> simulation day, increasing
    t = 0.0
    day = 96.4

    [[exposure]]                # optional: brightness change of the gas in photographic stops
    t = 0.0
    stops = 0.0

    colour = "true"             # optional (top level): "true" = Phase 9 true colour (blackbody
                                # at g T, Doppler and gravitational shifts); default "grey"
                                # (Phase 8 placeholder fire ramp)

    [grid]                      # optional: the global log-spherical grid (gasgrid.py defaults)
    r_max = 8000.0              # rays march out to r_max: must exceed the camera and the gas
    nr = 230                    # shown (8.3: "5,000 for early shots")

    [bricks]                    # optional (not with boxes): sparse fine grid for thin gas
    dx = 1.0                    # (render/bricks.py); cell size and outer radius of the fine
    r_fine = 1000.0             # region

    [auto_exposure]             # optional: smooth automatic exposure over the shot
    every = 1.0                 # probe frames every this many seconds
    smooth = 4.0                # Gaussian smoothing width (s): no sudden changes
    strength = 0.75             # fraction of the change compensated (1 = fully)

    [[boxes]]                   # optional, at most 2: detail boxes of fixed size whose centre
    around = "cloud"            # follows a target (8.3 item 2); n^3 cells
    size = 400.0                # or size_by_day = [[day, size], ...]: a box that grows with
    n = 160                     # the gas (log size, monotone cubic in day; n fixed, so the
                                # cells are finest while the gas is small)

How the path is made smooth: r (as log r), elev, az, vfov, the time warp and the exposure are
each interpolated through their keyframes with monotone cubic (PCHIP) curves. They are smooth
(continuous slope, so no sudden turns) and never overshoot: between two keyframes a value
stays between their values. So r never dips below the smaller keyframe r, and the time warp is
increasing when its keyframes are. When `around` or `look_at` changes between two keyframes the
two targets are blended with a smoothstep weight (zero slope at both ends); when all of a
shot's look-at (or around) targets are fixed points, they are joined by one monotone cubic
instead (no pause at each keyframe). Moving targets are
PCHIP tracks through their position at each snapshot.

Rules checked before any rendering (8.5): camera at r >= 6 from the hole (checked on a grid 8x
finer than the frames), never looking (nearly) straight along the z axis (the camera's up
vector would be undefined and the picture would spin), all simulation times inside the
sources' snapshot range, keyframe times inside the shot and strictly increasing, the camera
and every detail box inside the grid (rays start at the camera and end at r_max), at most two
boxes (the GPU marcher's limit).
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from nebula import units
from nebula.render.bricks import BrickLattice
from nebula.render.camera import Camera
from nebula.render.gasgrid import LogSphericalGrid

CODE_DAY = units.DAY / units.BlackHoleUnits().time_s  # one day in code time (GM/c^3)
R_CAMERA_MIN = 6.0  # 8.5: camera stays at r >= 6
ELEV_MAX = 80.0  # keyframe elevation limit (degrees)
MAX_BOXES = 2  # shaders/gas_march.wgsl samples box0 and box1
CAMERA_GRID_FRACTION = 0.9  # camera within 0.9 r_max: rays start inside the grid
VERTICAL_LIMIT = 0.985  # |forward . z| above this: the up vector is too close to the view
CHECK_OVERSAMPLE = 8


class ShotError(ValueError):
    """A shot file that cannot be rendered as written; the message says what to change."""


# ---------------------------------------------------------------- curves


def _curve(ts, values):
    """Monotone cubic through (ts, values); constant for one keyframe. Vectorised in t."""
    ts, values = np.asarray(ts, float), np.asarray(values, float)
    if len(ts) == 1:
        return lambda t: np.full(np.shape(t), values[0])
    f = PchipInterpolator(ts, values, extrapolate=False)
    return lambda t: f(np.clip(t, ts[0], ts[-1]))


def smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def spherical(r, elev_deg, az_deg):
    """(r, elevation, azimuth) -> Cartesian offset, last axis xyz."""
    e, a = np.radians(elev_deg), np.radians(az_deg)
    return np.stack([r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)], -1)


# ---------------------------------------------------------------- snapshots and targets


@dataclass
class Timeline:
    """Snapshot files of the shot's sources in time order: (t, path) with t in code time."""

    times: np.ndarray
    paths: list

    @classmethod
    def from_sources(cls, sources, runs_dir: Path) -> "Timeline":
        times, paths = [], []
        for src in sources:
            folder = Path(runs_dir) / src["run"] / "snapshots"
            files = sorted(folder.glob("snap_*.npz"))
            lo = float(src.get("from_day", -np.inf)) * CODE_DAY
            hi = float(src.get("to_day", np.inf)) * CODE_DAY
            tol = 1e-6 * CODE_DAY  # day values written to 3-6 decimals in shot files
            kept = [(t, f) for f in files
                    if lo - tol <= (t := float(np.load(f)["t"])) <= hi + tol]  # fmt: skip
            if not kept:
                raise ShotError(f"source run {src['run']!r}: no snapshots in {folder} within "
                                "its day range")  # fmt: skip
            if times and kept[0][0] < times[-1]:
                raise ShotError(
                    f"source run {src['run']!r} starts at day {kept[0][0] / CODE_DAY:.4f}, "
                    f"before the previous source ends (day {times[-1] / CODE_DAY:.4f}): set "
                    "to_day / from_day so the sources follow each other"
                )
            if times and kept[0][0] == times[-1]:
                times.pop()  # shared junction snapshot (a re-run starts from it): use the new run's
                paths.pop()
            times += [t for t, _ in kept]
            paths += [f for _, f in kept]
        return cls(np.array(times), paths)


def target_track(spec, timeline: Timeline):
    """Position of a moving target as a function of code time: PCHIP through its position at
    every snapshot. spec: {"type": "com"} or {"type": "group", "n": N, "pick_day": D}."""
    kind = spec.get("type")
    if kind not in ("com", "group"):
        raise ShotError(f"target type {kind!r}: use 'com' or 'group'")
    ids = None
    if kind == "group":
        t_pick = float(spec["pick_day"]) * CODE_DAY
        if not timeline.times[0] <= t_pick <= timeline.times[-1]:
            raise ShotError(f"target {spec}: pick_day outside the sources' snapshots")
        k = int(np.argmin(np.abs(timeline.times - t_pick)))
        d = np.load(timeline.paths[k])
        r = np.linalg.norm(d["x"].astype(float), axis=1)
        ids = d["ids"][np.argsort(r)[: int(spec["n"])]]
    pts = []
    for f in timeline.paths:
        d = np.load(f)
        x = d["x"].astype(float)
        if ids is not None:
            x = x[np.isin(d["ids"], ids)]
        if len(x) == 0:
            raise ShotError(f"target {spec}: no particles left in {Path(f).name}")
        pts.append(x.mean(axis=0))
    pts = np.array(pts)
    f = PchipInterpolator(timeline.times, pts, axis=0, extrapolate=False)
    lo, hi = timeline.times[0], timeline.times[-1]
    return lambda t: f(np.clip(t, lo, hi))


# ---------------------------------------------------------------- shot


@dataclass
class Shot:
    name: str
    fps: int
    duration: float
    width: int
    height: int
    n_sub: int
    kappa: float
    sources: list
    targets: dict
    camera: list
    time: list
    exposure: list = field(default_factory=list)
    grid: dict = field(default_factory=dict)
    boxes: list = field(default_factory=list)
    bricks: dict | None = None
    auto_exposure: dict | None = None
    colour: str = "grey"
    fade_out: float = 0.0  # seconds: the whole picture fades to black at the end (fade_factor)
    opacity: str = "constant"  # true colour: "constant" (kappa) or "thermal" (march.py)

    def log_grid(self) -> LogSphericalGrid:
        return LogSphericalGrid(**self.grid)

    def brick_lattice(self) -> BrickLattice | None:
        return BrickLattice(**self.bricks) if self.bricks is not None else None

    @classmethod
    def load(cls, path) -> "Shot":
        try:
            d = tomllib.loads(Path(path).read_text())
        except tomllib.TOMLDecodeError as e:
            raise ShotError(f"{path}: not valid TOML ({e})") from e
        missing = [k for k in ("name", "duration", "sources", "camera", "time") if k not in d]
        if missing:
            raise ShotError(
                f"{path}: missing {', '.join(missing)} (top-level settings must come "
                "before the first [section]: TOML puts later keys inside it)"
            )
        shot = cls(
            name=d["name"], fps=int(d.get("fps", 24)), duration=float(d["duration"]),
            width=int(d.get("width", 640)), height=int(d.get("height", 360)),
            n_sub=int(d.get("n_sub", 1)), kappa=float(d.get("kappa", 1e7)),
            sources=d["sources"], targets=d.get("targets", {}), camera=d["camera"],
            time=d["time"], exposure=d.get("exposure", []), grid=d.get("grid", {}),
            boxes=d.get("boxes", []),
            bricks=d.get("bricks"),
            auto_exposure=d.get("auto_exposure"),
            colour=str(d.get("colour", "grey")),
            fade_out=float(d.get("fade_out", 0.0)),
            opacity=str(d.get("opacity", "constant")),
        )  # fmt: skip
        shot.validate()
        return shot

    def validate(self):
        if self.duration <= 0 or self.fps <= 0:
            raise ShotError("duration and fps must be positive")
        if self.colour not in ("grey", "true"):
            raise ShotError(f'colour = {self.colour!r}: use "grey" or "true"')
        for src in self.sources:
            if float(src.get("ease_days", 0.0)) < 0:
                raise ShotError(f"[[sources]] {src.get('run')!r}: ease_days must be >= 0")
        if self.opacity not in ("constant", "thermal"):
            raise ShotError(f'opacity = {self.opacity!r}: use "constant" or "thermal"')
        if self.opacity == "thermal" and self.colour != "true":
            raise ShotError('opacity = "thermal" needs colour = "true"')
        if not 0.0 <= self.fade_out <= self.duration:
            raise ShotError(f"fade_out = {self.fade_out}: must be between 0 and the duration")
        try:
            self.log_grid()
        except TypeError as e:
            raise ShotError(f"[grid]: {e}") from e
        if self.bricks is not None:
            if self.boxes:
                raise ShotError("[bricks] and [[boxes]]: use one or the other")
            try:
                lat = self.brick_lattice()
            except TypeError as e:
                raise ShotError(f"[bricks]: {e}") from e
            if lat.dx <= 0 or lat.r_fine <= 0:
                raise ShotError("[bricks]: dx and r_fine must be positive")
        if self.auto_exposure is not None:
            bad = set(self.auto_exposure) - {"every", "smooth", "strength"}
            if bad:
                raise ShotError(f"[auto_exposure]: unknown settings {sorted(bad)}")
            if any(float(v) <= 0 for v in self.auto_exposure.values()):
                raise ShotError("[auto_exposure]: settings must be positive")
        if len(self.boxes) > MAX_BOXES:
            raise ShotError(f"[[boxes]]: at most {MAX_BOXES} detail boxes")
        for b in self.boxes:
            tgt = b.get("around", "hole")
            if isinstance(tgt, str) and tgt != "hole" and tgt not in self.targets:
                raise ShotError(f"[[boxes]]: unknown target {tgt!r}")
            if ("size" in b) == ("size_by_day" in b):
                raise ShotError("[[boxes]]: give either size or size_by_day")
            if "size" in b and float(b["size"]) <= 0:
                raise ShotError("[[boxes]]: size must be positive")
            if "size_by_day" in b:
                keys = np.asarray(b["size_by_day"], float)
                if keys.ndim != 2 or keys.shape[1] != 2 or len(keys) < 1:
                    raise ShotError("[[boxes]]: size_by_day must be a list of [day, size] pairs")
                if np.any(np.diff(keys[:, 0]) <= 0) or np.any(keys[:, 1] <= 0):
                    raise ShotError("[[boxes]]: size_by_day needs increasing days and positive "
                                    "sizes")  # fmt: skip
        for name, keys in (("camera", self.camera), ("time", self.time),
                           ("exposure", self.exposure)):  # fmt: skip
            if name != "exposure" and not keys:
                raise ShotError(f"[[{name}]]: at least one keyframe needed")
            ts = [float(k["t"]) for k in keys]
            if any(b <= a for a, b in zip(ts[:-1], ts[1:], strict=False)):
                raise ShotError(f"[[{name}]]: keyframe times must increase strictly ({ts})")
            if ts and (ts[0] < 0 or ts[-1] > self.duration):
                raise ShotError(f"[[{name}]]: keyframe times must lie in [0, {self.duration}]")
        days = [float(k["day"]) for k in self.time]
        if any(b <= a for a, b in zip(days[:-1], days[1:], strict=False)):
            raise ShotError(f"[[time]]: days must increase strictly ({days})")
        for k in self.camera:
            for key in ("r", "elev", "az"):
                if key not in k:
                    raise ShotError(f"[[camera]] at t = {k.get('t')}: missing {key}")
            if abs(float(k["elev"])) > ELEV_MAX:
                raise ShotError(f"[[camera]] at t = {k['t']}: |elev| must be <= {ELEV_MAX}")
            if k.get("around", "hole") == "hole" and float(k["r"]) < R_CAMERA_MIN:
                raise ShotError(f"[[camera]] at t = {k['t']}: r = {k['r']} < {R_CAMERA_MIN}")
            for role in ("around", "look_at"):
                tgt = k.get(role, "hole")
                if isinstance(tgt, str) and tgt != "hole" and tgt not in self.targets:
                    raise ShotError(f"[[camera]] at t = {k['t']}: unknown target {tgt!r}")


def fade_factor(video_t, t_last: float, fade_out: float) -> np.ndarray:
    """Brightness multiplier of the end fade (Hiten 2026-11-17: the video ends by fading to
    black): 1 until t_last - fade_out, then a smooth step (3u^2 - 2u^3) down to 0 at t_last,
    the last frame's time. fade_out = 0: no fade."""
    t = np.asarray(video_t, float)
    if fade_out <= 0:
        return np.ones_like(t)
    return smoothstep((t_last - t) / fade_out)


@dataclass
class FramePlan:
    """Everything needed to render each frame: arrays of length n_frames."""

    video_t: np.ndarray
    sim_t: np.ndarray
    position: np.ndarray  # (n, 3)
    look_at: np.ndarray  # (n, 3)
    vfov: np.ndarray
    stops: np.ndarray
    box_centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3)))  # (n, boxes, 3)
    box_sizes: np.ndarray | None = None  # (n, boxes); None in plans saved before size_by_day

    def boxes(self, i: int, shot: "Shot") -> list[dict]:
        """Detail boxes of frame i as gasgrid.CartesianBox keyword dicts."""
        out = []
        for j, b in enumerate(shot.boxes):
            size = self.box_sizes[i, j] if self.box_sizes is not None else b["size"]
            out.append({"centre": tuple(self.box_centres[i, j]), "size": float(size),
                        "n": int(b.get("n", 160))})  # fmt: skip
        return out

    def camera(self, i: int, width: int, height: int) -> Camera:
        return Camera(position=tuple(self.position[i]), look_at=tuple(self.look_at[i]),
                      vfov_deg=float(self.vfov[i]), width=width, height=height)  # fmt: skip

    def save(self, path):
        np.savez(path, **{k: v for k, v in self.__dict__.items() if v is not None})

    @classmethod
    def load(cls, path) -> "FramePlan":
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _target_fn(name_or_point, tracks):
    if isinstance(name_or_point, str):
        if name_or_point == "hole":
            return lambda t: np.zeros((np.size(t), 3))
        track = tracks[name_or_point]
        return lambda t: np.atleast_2d(track(np.atleast_1d(t)))
    p = np.asarray(name_or_point, float)
    if p.shape != (3,):
        raise ShotError(f"target point {name_or_point!r}: give [x, y, z]")
    return lambda t: np.tile(p, (np.size(t), 1))


def _blended(keys, role, tracks, vt, sim_t):
    """Position of the `role` target at video times vt: each key's target, smoothstep-blended
    across each keyframe interval where the target changes. When every key's target is fixed
    (a point or the hole), the points are joined by one monotone cubic per axis instead, so in
    a long shot the aim keeps moving through the keyframes rather than pausing at each one."""
    kt = np.array([float(k["t"]) for k in keys])
    names = [k.get(role, "hole") for k in keys]
    if len(keys) > 1 and all(not isinstance(n, str) or n == "hole" for n in names):
        pts = np.array([np.zeros(3) if isinstance(n, str) else _target_fn(n, tracks)(0.0)[0]
                        for n in names])  # fmt: skip
        return np.stack([_curve(kt, pts[:, a])(vt) for a in range(3)], axis=-1)
    fns = [_target_fn(k.get(role, "hole"), tracks) for k in keys]
    out = fns[0](sim_t)
    for i in range(1, len(keys)):
        if keys[i].get(role, "hole") == keys[i - 1].get(role, "hole"):
            sel = vt >= kt[i]
            out[sel] = fns[i](sim_t[sel])
            continue
        w = smoothstep((vt - kt[i - 1]) / (kt[i] - kt[i - 1]))[:, None]
        sel = vt > kt[i - 1]
        out[sel] = (1 - w[sel]) * fns[i - 1](sim_t[sel]) + w[sel] * fns[i](sim_t[sel])
    return out


def _sample(shot: Shot, vt, tracks):
    kt = [float(k["t"]) for k in shot.camera]
    log_r = _curve(kt, [np.log(float(k["r"])) for k in shot.camera])
    elev = _curve(kt, [float(k["elev"]) for k in shot.camera])
    az = _curve(kt, [float(k["az"]) for k in shot.camera])
    vfov = _curve(kt, [float(k.get("vfov", 60.0)) for k in shot.camera])
    warp = _curve([float(k["t"]) for k in shot.time], [float(k["day"]) for k in shot.time])
    if shot.exposure:
        stops = _curve([float(k["t"]) for k in shot.exposure],
                       [float(k["stops"]) for k in shot.exposure])(vt)  # fmt: skip
    else:
        stops = np.zeros_like(vt)
    sim_t = warp(vt) * CODE_DAY
    anchor = _blended(shot.camera, "around", tracks, vt, sim_t)
    pos = anchor + spherical(np.exp(log_r(vt)), elev(vt), az(vt))
    look = _blended(shot.camera, "look_at", tracks, vt, sim_t)
    return sim_t, pos, look, vfov(vt), stops


def plan_frames(shot: Shot, timeline: Timeline) -> FramePlan:
    """Frame list of the shot, after checking the rules (raises ShotError with the reason)."""
    t_lo, t_hi = timeline.times[0], timeline.times[-1]
    tracks = {name: target_track(spec, timeline) for name, spec in shot.targets.items()}
    n = int(round(shot.duration * shot.fps))
    fine = np.linspace(0.0, (n - 1) / shot.fps, (n - 1) * CHECK_OVERSAMPLE + 1)
    sim_t, pos, look, _, _ = _sample(shot, fine, tracks)
    tol = 1e-9 * max(abs(t_lo), abs(t_hi))
    if sim_t.min() < t_lo - tol or sim_t.max() > t_hi + tol:
        raise ShotError(
            f"simulation days {sim_t.min() / CODE_DAY:.3f}-{sim_t.max() / CODE_DAY:.3f} are "
            f"outside the sources' snapshots ({t_lo / CODE_DAY:.3f}-{t_hi / CODE_DAY:.3f})"
        )
    r = np.linalg.norm(pos, axis=1)
    if r.min() < R_CAMERA_MIN:
        i = int(np.argmin(r))
        raise ShotError(f"camera comes to r = {r[i]:.2f} < {R_CAMERA_MIN} at video "
                        f"t = {fine[i]:.2f} s")  # fmt: skip
    fwd = look - pos
    dist = np.linalg.norm(fwd, axis=1)
    if dist.min() < 1e-6:
        raise ShotError(f"camera sits on its look-at point at t = {fine[np.argmin(dist)]:.2f} s")
    vert = np.abs(fwd[:, 2]) / dist
    if vert.max() > VERTICAL_LIMIT:
        raise ShotError(f"camera looks almost straight up or down at t = "
                        f"{fine[np.argmax(vert)]:.2f} s: lower the elevation")  # fmt: skip
    r_max = shot.log_grid().r_max
    if r.max() > CAMERA_GRID_FRACTION * r_max:
        raise ShotError(f"camera reaches r = {r.max():.0f}, beyond {CAMERA_GRID_FRACTION} x the "
                        f"grid's r_max = {r_max:g}: raise [grid] r_max")  # fmt: skip
    boxes, sizes = _box_centres(shot, tracks, sim_t), box_sizes(shot, sim_t)
    for j in range(len(shot.boxes)):
        far = np.linalg.norm(boxes[:, j], axis=1) + 0.5 * np.sqrt(3) * sizes[:, j]
        if far.max() > r_max:
            raise ShotError(f"detail box {j} reaches r = {far.max():.0f}, beyond the grid's "
                            f"r_max = {r_max:g}: rays never get there")  # fmt: skip
    vt = np.arange(n) / shot.fps
    sim_t, pos, look, vfov, stops = _sample(shot, vt, tracks)
    sim_t = np.clip(sim_t, t_lo, t_hi)
    return FramePlan(vt, sim_t, pos, look, vfov, stops, _box_centres(shot, tracks, sim_t),
                     box_sizes(shot, sim_t))  # fmt: skip


def box_sizes(shot: Shot, sim_t) -> np.ndarray:
    """(n, boxes): each box's edge length at the simulation times (fixed `size`, or log size
    through the `size_by_day` keys with the same monotone cubic as the camera, so it never
    overshoots between keys and is held at the end values outside them)."""
    out = np.zeros((len(sim_t), len(shot.boxes)))
    for j, b in enumerate(shot.boxes):
        if "size_by_day" in b:
            keys = np.asarray(b["size_by_day"], float)
            out[:, j] = np.exp(_curve(keys[:, 0], np.log(keys[:, 1]))(sim_t / CODE_DAY))
        else:
            out[:, j] = float(b["size"])
    return out


def _box_centres(shot: Shot, tracks, sim_t) -> np.ndarray:
    """(n, boxes, 3): each box's centre (its target's position) at the simulation times."""
    out = np.zeros((len(sim_t), len(shot.boxes), 3))
    for j, b in enumerate(shot.boxes):
        out[:, j] = _target_fn(b.get("around", "hole"), tracks)(sim_t)
    return out
