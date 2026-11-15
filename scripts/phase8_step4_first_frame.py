"""Phase 8 step 4: the first real lensed frame (phases/phase-08-gas-rendering.md 8.9 item 2):
gas of a real snapshot marched along bent rays, stars behind it, placeholder fire colours.

    uv run python scripts/phase8_step4_first_frame.py [run=R6.1_100k] [snap=170] [kappa=1e7]
        [r=40] [elev=5] [vfov=60] [width=1280] [height=720] [n_sub=2] [variants=1]

Writes outputs/checks/phase8_step4_first_frame_<run>_<snap>.png and, with variants=1, a
labelled sheet of other snapshots / opacities / camera heights
(phase8_step4_variants_<run>.png) for choosing the look.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.render.camera import Camera
from nebula.render.frames import write_png
from nebula.render.gas_frames import Exposure, auto_gas_exposure, auto_star_exposure, compose
from nebula.render.gpu_march import OUT_PICTURE, render_gas, upload_scene
from nebula.render.march import GreyBlackbody
from nebula.render.snapshot_scene import snapshot_textures
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
DAY = 4080.0
GAS_PERCENTILE, GAS_LEVEL = 99.9, 1.0  # this gas brightness percentile -> aces(1) = 0.6


def camera(r, elev_deg, vfov, width, height):
    e = np.radians(elev_deg)
    return Camera(
        position=(r * np.cos(e), 0.0, r * np.sin(e)), vfov_deg=vfov, width=width, height=height
    )


def frame(textures, sky, kappa, cam, n_sub):
    scene = upload_scene(textures, GreyBlackbody(kappa=kappa))
    t0 = time.perf_counter()
    pic, tm = render_gas(cam, scene, n_sub=n_sub, sky=sky, out_mode=OUT_PICTURE)
    seconds = time.perf_counter() - t0
    exp = Exposure(
        gas=auto_gas_exposure(pic[..., 0], GAS_PERCENTILE, GAS_LEVEL),
        stars=auto_star_exposure(pic[..., 1:4]),
    )
    return compose(pic, exp), seconds, tm


def main():
    o = dict(a.split("=") for a in sys.argv[1:])
    run, snap = o.get("run", "R6.1_100k"), int(o.get("snap", 170))
    kappa, r, elev = float(o.get("kappa", 1e7)), float(o.get("r", 40)), float(o.get("elev", 5))
    vfov, n_sub = float(o.get("vfov", 60)), int(o.get("n_sub", 2))
    w, h = int(o.get("width", 1280)), int(o.get("height", 720))
    sky = upload_cube_map(build_cube_map(StarFieldConfig()))  # full 2048 faces, as Phase 2
    cache = {}

    def tex(s):
        if s not in cache:
            cache[s] = snapshot_textures(ROOT / "runs" / run, s)
        return cache[s]

    textures, t = tex(snap)
    img, sec, tm = frame(textures, sky, kappa, camera(r, elev, vfov, w, h), n_sub)
    path = OUT / f"phase8_step4_first_frame_{run}_{snap}.png"
    write_png(path, img)
    print(
        f"{run} snap {snap} (day {t / DAY:.0f}), {w}x{h}, {n_sub}x{n_sub} rays/pixel: "
        f"{sec:.1f} s on the GPU (slowest tile {tm.max:.2f} s); wrote {path.relative_to(ROOT)}"
    )
    if o.get("variants", "1") != "1":
        return
    variants = [
        (snap - 20, kappa, elev, "earlier"),
        (snap, kappa, elev, "chosen"),
        (snap, kappa, 15.0, "camera higher"),
        (snap, 3 * kappa, elev, "3x more opaque"),
        (snap, kappa / 3, elev, "3x less opaque"),
        (snap, kappa, 2.0, "camera lower"),
    ]
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for a, (s, k, el, lab) in zip(ax.flat, variants, strict=True):
        tx, ts = tex(s)
        im, _, _ = frame(tx, sky, k, camera(r, el, vfov, 640, 360), n_sub)
        a.imshow(im)
        a.set_title(f"{lab}: day {ts / DAY:.0f}, opacity {k:.0e}, camera {el:g} deg up")
        a.axis("off")
    fig.suptitle(f"{run}: variants of the first lensed frame (camera r = {r:g})")
    plt.tight_layout()
    vpath = OUT / f"phase8_step4_variants_{run}.png"
    plt.savefig(vpath, dpi=80)
    print(f"wrote {vpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
