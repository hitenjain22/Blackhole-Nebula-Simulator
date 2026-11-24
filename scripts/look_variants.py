"""Full-frame look versions of story frames side by side (Hiten: full frames, not crops):
each version is a TOML snippet with any of [nebula], [disc], [gas] and spin, applied on top of
configs/shots/full_story.toml. Exposure: the final render's, plus the stops the automatic
exposure (90%) would apply to each version (low-res probes, as scripts/inner_disc_variants.py).

    uv run python scripts/look_variants.py FRAMES WIDTH TAG "A:toml" "B:toml" ...
e.g. "lines:[nebula]\\ngain = 3.0"  (FRAMES comma-separated; the first picture is "now")

Writes outputs/checks/looks/<tag>_<frame>_<name>.png and <tag>_<frame>_sheet.png (labelled).
"""

import dataclasses
import json
import sys
import tomllib
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

import nebula.render.shot_render as SR
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.gas_frames import auto_gas_exposure

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/renders/final_story_v1_final1080p_20261120-031806"
frames = [int(v) for v in sys.argv[1].split(",")]
width, tag = int(sys.argv[2]), sys.argv[3]
variants = [(v.split(":", 1)[0], tomllib.loads(v.split(":", 1)[1].replace("\\n", "\n")))
            for v in sys.argv[4:]]  # fmt: skip
out = ROOT / "outputs/checks/looks"
out.mkdir(parents=True, exist_ok=True)
base = Shot.load(ROOT / "configs/shots/full_story.toml")
tl = Timeline.from_sources(base.sources, ROOT / "runs")
plan = plan_frames(base, tl)
e = json.loads((FINAL / "exposures.json").read_text())["full_story"]
strength = float(base.auto_exposure["strength"])
height = width * 9 // 16
feeds = {}


def shot_for(v):
    changes = {k: v[k] for k in ("nebula", "disc", "gas") if k in v}
    if "spin" in v:
        changes["spin"] = float(v["spin"])
    s = dataclasses.replace(base, **changes)
    s.validate()
    return s


def renderer(shot, w, h, n_sub, probing=False):
    r = SR.ShotRenderer(shot, plan, tl, width=w, height=h, n_sub=n_sub)
    r.base, r.adapt = SR.Exposure(*e[:2]), np.asarray(e[2]).copy()
    r._feeds, r._probing = feeds, probing
    return r


def level(r, i):
    return auto_gas_exposure(r._gas_level(r.picture(i)), SR.GAS_PERCENTILE, SR.GAS_LEVEL)


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


probe0 = renderer(base, 320, 180, 1, True)
shots = [(name, shot_for(v)) for name, v in variants]
for i in frames:
    lvl0 = level(probe0, i)
    imgs = [label(renderer(base, width, height, 2).frame(i), "now")]
    for name, shot in shots:
        stops = strength * float(np.log2(level(renderer(shot, 320, 180, 1, True), i) / lvl0))
        r = renderer(shot, width, height, 2)
        r.adapt[i] += stops
        img = r.frame(i)
        iio.imwrite(out / f"{tag}_{i:05d}_{name}.png", img)
        imgs.append(label(img, f"{name} ({stops:+.2f} stops)"))
        print(f"frame {i} {name}: exposure {stops:+.2f} stops", flush=True)
    gap = np.full((height, 8, 3), 255, np.uint8)
    rows = []
    for k in range(0, len(imgs), 2):
        right = imgs[k + 1] if k + 1 < len(imgs) else np.zeros_like(imgs[k])
        rows.append(np.concatenate([imgs[k], gap, right], axis=1))
    sep = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    sheet = np.concatenate([x for r in rows for x in (r, sep)][:-1])
    iio.imwrite(out / f"{tag}_{i:05d}_sheet.png", sheet)
    print("wrote", (out / f"{tag}_{i:05d}_sheet.png").relative_to(ROOT), flush=True)
