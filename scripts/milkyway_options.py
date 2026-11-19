"""Milky Way band options on real full_story frames, with the render's own exposures.
argv: options (comma list of asis,neutral,dim,none) then frame numbers."""
import dataclasses, json, sys, time
from pathlib import Path
import numpy as np, imageio.v2 as iio
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.shot_render import ShotRenderer, Exposure
from nebula.render import starfield as SF
ROOT = Path(".").resolve()
R = ROOT / "outputs/renders/final_story_v1_lowres_20261118-145413"
opts = sys.argv[1].split(","); frames = [int(a) for a in sys.argv[2:]]
shot = Shot.load(R / "plans/full_story.toml")
tl = Timeline.from_sources(shot.sources, ROOT / "runs"); plan = plan_frames(shot, tl)
WARM = np.array([1.0, 0.93, 0.85]); LUM = np.array([0.2126, 0.7152, 0.0722])
orig_band = SF.band_radiance
def neutral_band(d, cfg, m):  # same luminance, no tint
    return orig_band(d, cfg, m) / WARM * float(WARM @ LUM)
def sky(opt):
    SF.band_radiance = neutral_band if opt == "neutral" else orig_band
    cfg = SF.StarFieldConfig()
    if opt == "dim": cfg = dataclasses.replace(cfg, band_brightness=5.0)
    if opt == "none": cfg = dataclasses.replace(cfg, band=False)
    s = SF.upload_cube_map(SF.build_cube_map(cfg)); SF.band_radiance = orig_band; return s
e = json.loads((R / "exposures.json").read_text())["full_story"]
r = None
out = ROOT / "outputs/checks/milkyway"
for opt in opts:
    s = sky(opt)
    if r is None:
        r = ShotRenderer(shot, plan, tl, width=640, height=360, n_sub=1, sky=s)
        r.base = Exposure(*e[:2]); r.adapt = np.asarray(e[2])
    else:
        r.sky = s
    for i in frames:
        t0 = time.perf_counter(); iio.imwrite(out / f"{opt}_{i:05d}.png", r.frame(i))
        print(f"{opt} frame {i}: {time.perf_counter()-t0:.0f} s", flush=True)
    del s
