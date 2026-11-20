"""The inner disc (render/inner_disc.py) side by side with the frames as rendered now
(outputs/checks/inner_disc/). Frames of the full story at the final render's exposures; with
the disc the automatic exposure would react, so its frame also gets the stops the 90%
adaptation would apply (from a low-res probe of that frame, with and without the disc).

    uv run python scripts/inner_disc_try.py [frames, comma-separated] [width] [tag] [disc TOML]
"""

import dataclasses
import json
import sys
import tomllib
from pathlib import Path

import imageio.v2 as iio
import numpy as np

import nebula.render.shot_render as SR
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.gas_frames import auto_gas_exposure

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/renders/final_story_v1_final1080p_20261120-031806"
frames = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "2112,2352").split(",")]
width = int(sys.argv[2]) if len(sys.argv) > 2 else 960
tag = sys.argv[3] if len(sys.argv) > 3 else "try"
disc = tomllib.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
out = ROOT / "outputs/checks/inner_disc"
out.mkdir(parents=True, exist_ok=True)

base = Shot.load(ROOT / "configs/shots/full_story.toml")
tl = Timeline.from_sources(base.sources, ROOT / "runs")
plan = plan_frames(base, tl)
e = json.loads((FINAL / "exposures.json").read_text())["full_story"]
strength = float(base.auto_exposure["strength"])
height = width * 9 // 16


def renderer(shot, w, h, n_sub):
    r = SR.ShotRenderer(shot, plan, tl, width=w, height=h, n_sub=n_sub)
    r.base, r.adapt = SR.Exposure(*e[:2]), np.asarray(e[2]).copy()
    return r


without = renderer(base, width, height, 2)
with_disc = renderer(dataclasses.replace(base, disc=disc), width, height, 2)
probe_a = renderer(base, 320, 180, 1)
probe_b = renderer(dataclasses.replace(base, disc=disc), 320, 180, 1)
for p in (probe_a, probe_b):
    p._probing = True
for i in frames:
    lvl = [auto_gas_exposure(p._gas_level(p.picture(i)), SR.GAS_PERCENTILE, SR.GAS_LEVEL)
           for p in (probe_a, probe_b)]  # fmt: skip
    stops = strength * float(np.log2(lvl[1] / lvl[0]))
    with_disc.adapt[i] = without.adapt[i] + stops
    print(f"frame {i} (t = {i / base.fps:.1f} s): exposure change with the disc {stops:+.2f} "
          "stops", flush=True)  # fmt: skip
    a = without.frame(i)
    b = with_disc.frame(i)
    iio.imwrite(out / f"{tag}_{i:05d}_now.png", a)
    iio.imwrite(out / f"{tag}_{i:05d}_disc.png", b)
    c = np.concatenate([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b], axis=1)
    iio.imwrite(out / f"{tag}_{i:05d}_side_by_side.png", c)
    dp = with_disc.disc(float(plan.sim_t[i]))
    if dp is not None:
        print(f"  disc: {len(dp.m)} particles, T max {dp.temp.max():.3g} K", flush=True)
