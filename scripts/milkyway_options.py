"""Milky Way band options on real full_story frames, with the render's own exposures
(outputs/checks/milkyway_options.png, 2026-11-19). argv: options (comma list of warm, neutral,
dim, none; neutral is the default since Hiten chose it) then frame numbers."""

import dataclasses
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from nebula.render import starfield as SF
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.shot_render import Exposure, ShotRenderer

ROOT = Path(".").resolve()
R = ROOT / "outputs/renders/final_story_v1_lowres_20261118-145413"
opts = sys.argv[1].split(",")
frames = [int(a) for a in sys.argv[2:]]
shot = Shot.load(R / "plans/full_story.toml")
tl = Timeline.from_sources(shot.sources, ROOT / "runs")
plan = plan_frames(shot, tl)


def sky(opt):
    cfg = SF.StarFieldConfig()
    if opt == "warm":
        cfg = dataclasses.replace(cfg, band_colour=SF.WARM_BAND)
    if opt == "dim":
        cfg = dataclasses.replace(cfg, band_brightness=5.0)
    if opt == "none":
        cfg = dataclasses.replace(cfg, band=False)
    return SF.upload_cube_map(SF.build_cube_map(cfg))


e = json.loads((R / "exposures.json").read_text())["full_story"]
r = None
out = ROOT / "outputs/checks/milkyway"
for opt in opts:
    s = sky(opt)
    if r is None:
        r = ShotRenderer(shot, plan, tl, width=640, height=360, n_sub=1, sky=s)
        r.base = Exposure(*e[:2])
        r.adapt = np.asarray(e[2])
    else:
        r.sky = s
    for i in frames:
        t0 = time.perf_counter()
        iio.imwrite(out / f"{opt}_{i:05d}.png", r.frame(i))
        print(f"{opt} frame {i}: {time.perf_counter() - t0:.0f} s", flush=True)
    del s
