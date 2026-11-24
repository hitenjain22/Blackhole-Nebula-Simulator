"""A short low-cost clip of part of a shot (for judging a camera move with the real look:
lensing, spin, gas), e.g. the orbit around the spinning hole (Hiten 2026-11-23).

    uv run python scripts/shot_clip.py configs/shots/full_story_orbit.toml T0 T1 \
        [width=640] [every=2] [n_sub=1] [tag=] [h_scale=]

Every ``every``-th frame from T0 to T1 seconds, held so the clip plays at real speed. Exposure:
the approved final render's (outputs/renders/final_story_v1_final1080p_*/exposures.json),
corrected for the new camera once per second the way the automatic exposure (strength from the
shot) would: low-res probes of this shot and of configs/shots/full_story.toml, smoothed in
time (saved in the frames folder; a rerun resumes, keeping frames already made).
h_scale: draw the gas with this smoothing factor instead of the shot's (a preview may
use 1.0: clumpier but ~3x less deposition work than the story's 1.5). Writes
outputs/checks/clip_<shot>_<tag>.mp4 and its frames under outputs/checks/clips/.
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

import nebula.render.shot_render as SR
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.gas_frames import auto_gas_exposure
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/renders/final_story_v1_final1080p_20261120-031806"
args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
shot_path, t0, t1 = Path(args[0]), float(args[1]), float(args[2])
width, every = int(opts.get("width", 640)), int(opts.get("every", 2))
n_sub, tag = int(opts.get("n_sub", 1)), opts.get("tag", "try")
height = width * 9 // 16

shot = Shot.load(shot_path)
if "h_scale" in opts:  # preview only: the shot file is not changed
    shot = dataclasses.replace(shot, gas={**(shot.gas or {}), "h_scale": float(opts["h_scale"])})
base = Shot.load(ROOT / "configs/shots/full_story.toml")
tl = Timeline.from_sources(shot.sources, ROOT / "runs")
plan, plan0 = plan_frames(shot, tl), plan_frames(base, tl)
e = json.loads((FINAL / "exposures.json").read_text())["full_story"]
strength = float(shot.auto_exposure["strength"])


def renderer(s, p, w, h, ns, probing=False):
    r = SR.ShotRenderer(s, p, tl, width=w, height=h, n_sub=ns)
    r.base, r.adapt = SR.Exposure(*e[:2]), np.asarray(e[2]).copy()
    r._probing = probing
    return r


def level(r, i):
    return auto_gas_exposure(r._gas_level(r.picture(i)), SR.GAS_PERCENTILE, SR.GAS_LEVEL)


fps = shot.fps
frames = list(range(int(round(t0 * fps)), int(round(t1 * fps)), every))
# exposure correction: probes once per second, interpolated (and so smooth) between them
pa, pb = renderer(shot, plan, 320, 180, 1, True), renderer(base, plan0, 320, 180, 1, True)
pb._feeds = pa._feeds  # same edge measurements
pt = np.arange(t0, t1 + 0.5, 1.0)
out = ROOT / "outputs/checks/clips" / f"{shot.name}_{tag}"
out.mkdir(parents=True, exist_ok=True)
stops_file = out / "exposure_stops.json"
if stops_file.exists():  # resumed: the same exposure as the frames already made
    stops = json.loads(stops_file.read_text())
else:
    stops = []
    for t in pt:
        i = min(int(round(t * fps)), len(plan.video_t) - 1)
        stops.append(strength * float(np.log2(level(pa, i) / level(pb, i))))
        print(f"probe {t:5.1f} s: {stops[-1]:+.2f} stops", flush=True)
    stops_file.write_text(json.dumps(stops))
r = renderer(shot, plan, width, height, n_sub)
r._feeds = pa._feeds
imgs = []
for i in frames:
    if (out / f"{i:05d}.png").exists():  # resumed: keep the frame
        imgs.append(iio.imread(out / f"{i:05d}.png"))
        continue
    r.adapt[i] = e[2][i] + float(np.interp(i / fps, pt, stops))
    t_start = time.perf_counter()
    img = r.frame(i)
    iio.imwrite(out / f"{i:05d}.png", img)
    imgs.append(img)
    print(f"frame {i} ({i / fps:.2f} s): {time.perf_counter() - t_start:.0f} s", flush=True)
video = ROOT / "outputs/checks" / f"clip_{shot.name}_{tag}.mp4"
write_video((im for im in imgs for _ in range(every)), video, fps=fps)
print(f"wrote {video.relative_to(ROOT)}: {len(imgs)} frames x {every}")
