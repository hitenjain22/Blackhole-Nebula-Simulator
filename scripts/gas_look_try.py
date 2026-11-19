"""Gas look options on real full_story frames (outputs/checks/gaslook/): argv tag, JSON [gas]
settings, frames. Low-res render's exposures; old cube stars (the gas is what is compared)."""

import dataclasses
import json
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.shot_render import Exposure, ShotRenderer

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/renders/final_story_v1_lowres_20261118-145413"
tag, gas = sys.argv[1], json.loads(sys.argv[2])
frames = [int(a) for a in sys.argv[3:]]
shot = dataclasses.replace(Shot.load(R / "plans/full_story.toml"), gas=gas or None)
tl = Timeline.from_sources(shot.sources, ROOT / "runs")
plan = plan_frames(shot, tl)
r = ShotRenderer(shot, plan, tl, width=1920, height=1080, n_sub=1)
e = json.loads((R / "exposures.json").read_text())["full_story"]
r.base, r.adapt = Exposure(*e[:2]), np.asarray(e[2])
out = ROOT / "outputs/checks/gaslook"
for i in frames:
    t0 = time.perf_counter()
    iio.imwrite(out / f"{tag}_{i:05d}.png", r.frame(i))
    print(f"{tag} frame {i}: {time.perf_counter() - t0:.0f} s", flush=True)
