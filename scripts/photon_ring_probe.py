"""What lights the thin ring at the shadow edge? Frame 2352 (98 s) rendered normally, with a
black sky (no stars, no Milky Way), and with no gas (outputs/checks/photon_ring/)."""

import dataclasses
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

import nebula.render.shot_render as SR
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/renders/final_story_v1_lowres_20261118-145413"
frame = int(sys.argv[1]) if len(sys.argv) > 1 else 2352
base = Shot.load(ROOT / "configs/shots/full_story.toml")
tl = Timeline.from_sources(base.sources, ROOT / "runs")
e = json.loads((R / "exposures.json").read_text())["full_story"]
out = ROOT / "outputs/checks/photon_ring"
orig_tex = SR.textures_from_particles


def render(tag, shot, sky=None, no_gas=False):
    SR.textures_from_particles = (
        (lambda x, h, m, *a, **k: orig_tex(x, h, 0.0 * np.asarray(m, float), *a, **k))
        if no_gas else orig_tex
    )  # fmt: skip
    r = SR.ShotRenderer(shot, plan_frames(shot, tl), tl, width=1280, height=720, n_sub=2, sky=sky)
    r.base, r.adapt = SR.Exposure(*e[:2]), np.asarray(e[2])
    img = r.frame(frame)
    iio.imwrite(out / f"{tag}_{frame:05d}.png", img)
    print(tag, "done", flush=True)


dark = upload_cube_map(build_cube_map(StarFieldConfig(band=False, cube_stars=False, n_stars=10)))
no_points = dataclasses.replace(base, stars={**base.stars, "mode": "cube"})
render("normal", base)
render("black_sky", no_points, sky=dark)
render("no_gas", base, no_gas=True)
