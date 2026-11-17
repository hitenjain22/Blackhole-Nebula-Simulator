"""Fast camera-path preview of a shot: the gas as glowing dots seen from the planned camera,
the hole as a black disc of the shadow's size, no light bending and no grid rendering (a
fraction of a second per frame, so a path can be tried and changed in minutes). For designing
camera paths only; the real look comes from scripts/render_trailer.py.

    uv run python scripts/path_preview.py configs/shots/continuous.toml [scale=1] [every=1]
        [tag=]

Writes outputs/checks/path_<shot>[_tag].mp4, a frame sheet path_<shot>[_tag]_sheet.png, and
prints the director's camera checks (nearest gas, camera inside gas)."""

import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.director import CODE_DAY, Shot, Timeline, plan_frames
from nebula.render.interpolate import load_snap, state_at
from nebula.render.shot_render import camera_gas_clearance
from nebula.render.snapshot_scene import run_eos_and_mass
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
B_SHADOW = 3.0 * np.sqrt(3.0)  # GM/c^2: critical impact parameter (the shadow's edge)
DECADES = 3.5  # brightness range shown
N_STARS = 6000  # preview star background (fixed directions: stars are infinitely far)


def star_field(seed=7):
    """(directions (N, 3), brightness (N,)): a fixed random sky for the preview only."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(N_STARS, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d, np.clip(rng.pareto(2.5, N_STARS) * 0.15 + 0.05, 0, 1)


STARS = star_field()


def draw_sky(cam, w, h):
    """Stars (no lensing) and, at the hole, a thin ring of the background stars' Einstein
    radius sqrt(4 GM / (c^2 D)): a stand-in for where the real render bends the starlight."""
    img = np.zeros((h, w, 3))
    d, b = STARS
    front = d @ cam.forward > 0.05
    px, py = cam.pixel_of_direction(d[front])
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    np.add.at(img, (py[ok].astype(int), px[ok].astype(int)), b[front][ok][:, None] * 0.8)
    img = gaussian_filter(img, (0.6, 0.6, 0))
    hd = -np.asarray(cam.position)
    hdist = np.linalg.norm(hd)
    if hd @ cam.forward > 0:
        hx, hy = cam.pixel_of_direction(hd / hdist)
        ring = np.sqrt(4.0 / hdist) / np.radians(cam.vfov_deg) * h
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.hypot(xx - hx, yy - hy)
        img += (np.exp(-(((rr - ring) / 1.2) ** 2)) * 0.25)[..., None] * [0.6, 0.7, 1.0]
    return img


def colour(temp):
    """Rough glow colour by temperature: deep red -> orange -> yellow-white -> blue-white."""
    lt = np.clip((np.log10(np.maximum(temp, 1.0)) - 3.8) / 2.2, 0.0, 1.0)[..., None]
    stops = np.array([[0.8, 0.15, 0.02], [1.0, 0.55, 0.1], [1.0, 0.9, 0.7], [0.75, 0.85, 1.0]])
    pos = lt * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)[..., 0]
    f = pos - i[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def frame(cam, x, temp, w, h, level):
    """uint8 frame: mass per pixel (log, fixed `level`) coloured by mass-weighted temperature."""
    d = x - np.asarray(cam.position)
    z = d @ cam.forward
    front = z > 1e-3
    px, py = cam.pixel_of_direction(d[front] / np.linalg.norm(d[front], axis=1)[:, None])
    t, dist = temp[front], np.linalg.norm(d[front], axis=1)
    # the hole hides gas behind its shadow (straight-line view: no lensing in this preview)
    hd = -np.asarray(cam.position)
    hdist = np.linalg.norm(hd)
    hx, hy = cam.pixel_of_direction(hd / hdist)
    rad = np.arcsin(min(B_SHADOW / hdist, 1.0)) / np.radians(cam.vfov_deg) * h
    behind = (dist > hdist) & ((px - hx) ** 2 + (py - hy) ** 2 < rad**2)
    keep = (px >= 0) & (px < w) & (py >= 0) & (py < h) & ~behind
    ix, iy = px[keep].astype(int), py[keep].astype(int)
    mass = np.zeros((h, w))
    np.add.at(mass, (iy, ix), 1.0)
    tsum = np.zeros((h, w))
    np.add.at(tsum, (iy, ix), t[keep])
    mass_b = gaussian_filter(mass, 0.8)
    tm = gaussian_filter(tsum, 0.8) / np.maximum(mass_b, 1e-12)
    bright = np.clip((np.log10(np.maximum(mass_b, 1e-12)) - (level - DECADES)) / DECADES, 0, 1)
    img = colour(tm) * bright[..., None] ** 1.3
    img = img + draw_sky(cam, w, h) * (1.0 - bright[..., None])
    yy, xx = np.mgrid[0:h, 0:w]
    img[(xx - hx) ** 2 + (yy - hy) ** 2 < rad**2] = 0.0  # the shadow itself
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), mass_b.max()


def _blend(a, b, t):
    """Positions blended linearly between snapshots a and b for particles in both (others from
    a); scalars from a. Preview only: gas swinging round the hole moves on curves."""
    f = min(max((t - a.t) / (b.t - a.t), 0.0), 1.0)
    pos = {k: j for j, k in enumerate(b.ids)}
    jb = np.array([pos.get(k, -1) for k in a.ids])
    x = a.x.copy()
    both = jb >= 0
    x[both] = (1 - f) * a.x[both] + f * b.x[jb[both]]
    return x, a.scalars


def main():
    shot_path = Path(sys.argv[1])
    o = dict(a.split("=", 1) for a in sys.argv[2:])
    scale, every = float(o.get("scale", 1.0)), int(o.get("every", 1))
    shot = Shot.load(shot_path)
    timeline = Timeline.from_sources(shot.sources, ROOT / "runs")
    plan = plan_frames(shot, timeline)
    if o.get("checks", "0") == "1":  # scans every snapshot: slow for long shots
        dist, inside = camera_gas_clearance(plan, timeline)
        i = int(np.argmin(dist))
        print(f"nearest gas to the camera: {dist[i]:.1f} at video t {plan.video_t[i]:.1f} s "
              f"(day {plan.sim_t[i] / CODE_DAY:.2f}); camera inside gas in {inside.sum()} "
              "frames")  # fmt: skip
    w, h = 2 * round(shot.width * scale / 2), 2 * round(shot.height * scale / 2)
    snaps, eos_mass = {}, {}
    level = None
    frames, sheet, low_frames = [], [], []
    idx = range(0, len(plan.video_t), every)
    t0 = time.perf_counter()
    for n, i in enumerate(idx):
        t = float(plan.sim_t[i])
        k = min(max(int(np.searchsorted(timeline.times, t, side="right")) - 1, 0),
                len(timeline.times) - 2)  # fmt: skip
        for kk in (k, k + 1):
            if kk not in snaps:
                snaps[kk] = load_snap(timeline.paths[kk])
        for old in [kk for kk in snaps if kk < k]:
            del snaps[old]
        a, b = snaps[k], snaps[k + 1]
        if o.get("exact", "0") == "1":  # orbit-advanced in-betweens (slow near the hole)
            _, x, s = state_at(a, b, min(max(t, a.t), b.t), r_drop=4.0)
        else:  # straight blend between the two snapshots: enough to judge a camera path
            x, s = _blend(a, b, t)
        run_dir = Path(timeline.paths[k]).parent.parent
        if run_dir not in eos_mass:
            eos_mass[run_dir] = run_eos_and_mass(run_dir)
        temp = snapshot_temperature(eos_mass[run_dir][0], s["rho"], s["u"])
        cam = plan.camera(i, w, h)
        rel = x - np.asarray(cam.position)
        fr = rel @ cam.forward > 0
        gx, gy = cam.pixel_of_direction(rel[fr] / np.linalg.norm(rel[fr], axis=1)[:, None])
        if ((gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)).sum() < 0.1 * len(x) and n % 6 == 0:
            low_frames.append(f"{plan.video_t[i]:.1f}")
        if level is None:  # one brightness scale for the whole video (no pumping)
            _, peak = frame(cam, x, temp, w, h, 0.0)
            level = np.log10(peak) + 0.3
        img, _ = frame(cam, x, temp, w, h, level)
        frames.append(img)
        if n % max(1, len(idx) // 12) == 0 and len(sheet) < 12:
            sheet.append(img)
        if n % 20 == 0:
            print(f"frame {i}/{len(plan.video_t)} (day {t / CODE_DAY:.1f}), "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    if low_frames:
        print(f"NOTE: under 10% of the gas in frame at video t {', '.join(low_frames[:20])} s")
    tag = f"_{o['tag']}" if o.get("tag") else ""
    out = OUT / f"path_{shot.name}{tag}.mp4"
    write_video(frames, out, fps=int(round(shot.fps / every)))
    rows = [np.concatenate(sheet[r : r + 4], axis=1) for r in range(0, len(sheet) - 3, 4)]
    iio.imwrite(OUT / f"path_{shot.name}{tag}_sheet.png", np.concatenate(rows, axis=0))
    print(f"wrote {out.relative_to(ROOT)} ({len(frames)} frames) and the frame sheet")


if __name__ == "__main__":
    main()
