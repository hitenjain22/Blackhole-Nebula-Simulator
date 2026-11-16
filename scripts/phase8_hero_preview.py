"""Preview clip of a densely saved stretch (DECISIONS 2026-11-15: dense re-run of the hero
stretch): frames at even times, gas between saves from render/interpolate.py, a slow camera
swing round the hole. A preview only; Phase 8 step 6 (the director) makes the real shots.

    uv run python scripts/phase8_hero_preview.py [run=dense_B192] [frames=240] [r=80] [elev=12]
        [swing=40] [vfov=60] [width=640] [height=360] [n_sub=1] [kappa=1e7] [stride=1]
        [first=0] [last=-1] [tag=]  (snapshot indices of the stretch; tag names the outputs)

Writes outputs/checks/phase8_hero_preview_<run>.mp4 (24 fps) and a 6-frame sheet
phase8_hero_preview_<run>.png. Exposure is fixed for the whole clip: the median of the auto
exposures of 5 frames spread over it (no pumping).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.camera import Camera
from nebula.render.gas_frames import Exposure, auto_gas_exposure, auto_star_exposure, compose
from nebula.render.gpu_march import OUT_PICTURE, render_gas, upload_scene
from nebula.render.interpolate import load_snap, state_at
from nebula.render.march import GreyBlackbody
from nebula.render.snapshot_scene import run_eos_and_mass, textures_from_particles
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
DAY = 4080.0
FPS = 24
R_ACC = 4.0  # the Phantom run's accretion radius (accradius1)


def main():
    o = dict(a.split("=") for a in sys.argv[1:])
    run = o.get("run", "dense_B192")
    n_frames, stride = int(o.get("frames", 240)), int(o.get("stride", 1))
    r, elev, swing = float(o.get("r", 80)), float(o.get("elev", 12)), float(o.get("swing", 40))
    vfov, n_sub, kappa = float(o.get("vfov", 60)), int(o.get("n_sub", 1)), float(o.get("kappa", 1e7))
    w, h = int(o.get("width", 640)), int(o.get("height", 360))
    folder = ROOT / "runs" / run
    eos, mass = run_eos_and_mass(folder)
    first, last = int(o.get("first", 0)), int(o.get("last", -1))
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    files = files[first : (last + 1 if last >= 0 else None) : stride]
    tag = o.get("tag", "")
    times = np.array([float(np.load(f)["t"]) for f in files])
    cache = {}

    def snap(i):
        if i not in cache:
            cache.clear() if len(cache) > 3 else None
            cache[i] = load_snap(files[i])
        return cache[i]

    sky = upload_cube_map(build_cube_map(StarFieldConfig()))
    model = GreyBlackbody(kappa=kappa)
    e = np.radians(elev)

    def picture(k):
        t = times[0] + (times[-1] - times[0]) * k / (n_frames - 1)
        i = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        ids, x, s = state_at(snap(i), snap(i + 1), t, r_drop=R_ACC)
        temp = snapshot_temperature(eos, s["rho"], s["u"])
        texs = textures_from_particles(x, s["h"], mass, temp, np.zeros_like(x))
        az = np.radians(swing * k / max(n_frames - 1, 1))
        pos = (r * np.cos(e) * np.cos(az), r * np.cos(e) * np.sin(az), r * np.sin(e))
        cam = Camera(position=pos, vfov_deg=vfov, width=w, height=h)
        pic, _ = render_gas(cam, upload_scene(texs, model), n_sub=n_sub, sky=sky,
                            out_mode=OUT_PICTURE)  # fmt: skip
        return t, pic

    t0 = time.perf_counter()
    probes = [picture(k)[1] for k in np.linspace(0, n_frames - 1, 5).astype(int)]
    exposure = Exposure(
        float(np.median([auto_gas_exposure(p[..., 0], 99.9, 1.0) for p in probes])),
        float(np.median([auto_star_exposure(p[..., 1:4]) for p in probes])),
    )
    frames, labels = [], []
    for k in range(n_frames):
        t, pic = picture(k)
        if not np.isfinite(pic).all():
            raise FloatingPointError(f"frame {k}: non-finite pixels")
        frames.append(compose(pic, exposure))
        labels.append(f"day {t / DAY:.2f}")
        if k % 10 == 0:
            print(f"  {labels[-1]}: {k + 1}/{n_frames} frames, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    out = OUT / f"phase8_hero_preview_{run}{tag}.mp4"
    write_video(frames, out, fps=FPS)
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for a, k in zip(ax.flat, np.linspace(0, n_frames - 1, 6).astype(int), strict=True):
        a.imshow(frames[k])
        a.set_title(f"frame {k}: {labels[k]}")
        a.axis("off")
    fig.suptitle(f"{run}: preview frames (camera r = {r:g}, {elev:g} deg up, swinging {swing:g} deg)")
    plt.tight_layout()
    sheet = OUT / f"phase8_hero_preview_{run}{tag}.png"
    plt.savefig(sheet, dpi=80)
    print(f"wrote {out.relative_to(ROOT)} ({n_frames} frames, "
          f"{(time.perf_counter() - t0) / (n_frames + 5):.1f} s per frame) and {sheet.name}")  # fmt: skip


if __name__ == "__main__":
    main()
