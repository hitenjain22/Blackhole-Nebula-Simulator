"""Several looks of the inner disc on one frame, side by side with the frame as rendered now
(outputs/checks/inner_disc/). Each version's exposure gets the stops the 90% automatic
adaptation would apply (as scripts/inner_disc_try.py).

    uv run python scripts/inner_disc_variants.py frame width tag "A:toml" "B:toml" ...
e.g. "A:temperature = 0.7\ndensity = 0.1"; a key spin = 0.9 sets the shot's spin instead
(Kerr light rays; the disc then reaches the spinning hole's innermost stable orbit).
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
frame, width, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
variants = [(v.split(":", 1)[0], tomllib.loads(v.split(":", 1)[1].replace("\\n", "\n")))
            for v in sys.argv[4:]]  # fmt: skip
out = ROOT / "outputs/checks/inner_disc"
out.mkdir(parents=True, exist_ok=True)
base = Shot.load(ROOT / "configs/shots/full_story.toml")
tl = Timeline.from_sources(base.sources, ROOT / "runs")
plan = plan_frames(base, tl)
e = json.loads((FINAL / "exposures.json").read_text())["full_story"]
strength = float(base.auto_exposure["strength"])
height = width * 9 // 16
feeds = {}  # the edge measurements are the same for every version: measure once


def renderer(shot, w, h, n_sub, probing=False):
    r = SR.ShotRenderer(shot, plan, tl, width=w, height=h, n_sub=n_sub)
    r.base, r.adapt = SR.Exposure(*e[:2]), np.asarray(e[2]).copy()
    r._feeds, r._probing = feeds, probing
    return r


probe0 = renderer(base, 320, 180, 1, probing=True)
lvl0 = auto_gas_exposure(probe0._gas_level(probe0.picture(frame)), SR.GAS_PERCENTILE,
                         SR.GAS_LEVEL)  # fmt: skip
now = renderer(base, width, height, 2)
imgs, labels = [now.frame(frame)], ["now"]
for name, disc in variants:
    spin = float(disc.pop("spin", 0.0))
    shot = dataclasses.replace(base, disc=disc, spin=spin)
    probe = renderer(shot, 320, 180, 1, probing=True)
    lvl = auto_gas_exposure(probe._gas_level(probe.picture(frame)), SR.GAS_PERCENTILE,
                            SR.GAS_LEVEL)  # fmt: skip
    r = renderer(shot, width, height, 2)
    stops = strength * float(np.log2(lvl / lvl0))
    r.adapt[frame] += stops
    img = r.frame(frame)
    iio.imwrite(out / f"{tag}_{frame:05d}_{name}.png", img)
    imgs.append(img)
    labels.append(name)
    print(f"{name}: {disc} spin {spin} exposure {stops:+.2f} stops", flush=True)
gap = np.full((height, 8, 3), 255, np.uint8)
rows = [np.concatenate([imgs[k], gap, imgs[k + 1]], axis=1) if k + 1 < len(imgs)
        else np.concatenate([imgs[k], gap, np.zeros_like(imgs[k])], axis=1)
        for k in range(0, len(imgs), 2)]  # fmt: skip
sheet = np.concatenate([np.concatenate([r, np.full((8, r.shape[1], 3), 255, np.uint8)])
                        for r in rows])  # fmt: skip
iio.imwrite(out / f"{tag}_{frame:05d}_sheet.png", sheet)
print("order (left to right, top to bottom):", ", ".join(labels))
