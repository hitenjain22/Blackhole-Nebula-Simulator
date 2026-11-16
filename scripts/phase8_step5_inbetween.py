"""Phase 8 step 5 check: gas between snapshots (phases/phase-08-gas-rendering.md 8.2).

    uv run python scripts/phase8_step5_inbetween.py [run=R6.1_100k] [first=165] [n=3]
        [per=24] [width=640] [height=360] [kappa=1e7]

Writes outputs/checks/phase8_step5_tracks_<run>.png (paths of gas near the hole: following the
orbit vs straight-line blending, with the snapshot positions) and
outputs/checks/phase8_step5_clip_<run>.mp4 (a lensed clip across n snapshot intervals with
`per` frames each, fixed camera and exposure: look for jumps at the snapshot boundaries).
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


def tracks(run, snaps, out):
    pairs = list(zip(snaps[:-1], snaps[1:], strict=True))
    a0 = snaps[0]
    r = np.linalg.norm(a0.x, axis=1)
    common = set(snaps[-1].ids)
    pick = [i for i in np.argsort(r) if a0.ids[i] in common and r[i] > 6][:8]
    pid = a0.ids[pick]
    fig, ax = plt.subplots(figsize=(9, 9))
    for k, p in enumerate(pid):
        orb, line, dots = [], [], []
        for a, b in pairs:
            ia, ib = np.nonzero(a.ids == p)[0][0], np.nonzero(b.ids == p)[0][0]
            dots.append(a.x[ia])
            for f in np.linspace(0, 1, 40, endpoint=False):
                ids, x, _ = state_at(a, b, a.t + f * (b.t - a.t))
                orb.append(x[np.nonzero(ids == p)[0][0]])
                line.append((1 - f) * a.x[ia] + f * b.x[ib])
        dots.append(snaps[-1].x[np.nonzero(snaps[-1].ids == p)[0][0]])
        orb, line, dots = np.array(orb), np.array(line), np.array(dots)
        c = f"C{k}"
        ax.plot(orb[:, 0], orb[:, 1], "-", color=c, lw=1.5)
        ax.plot(line[:, 0], line[:, 1], ":", color=c, lw=1)
        ax.plot(dots[:, 0], dots[:, 1], "o", color=c, ms=5)
    ax.plot(0, 0, "k*", ms=12)
    ax.set_aspect("equal")
    ax.set_title(
        f"{run}: 8 gas particles nearest the hole over {len(pairs)} snapshot intervals\n"
        "dots = saved snapshots; solid = our in-between (follows the orbit); "
        "dotted = straight-line blend (cuts corners)"
    )
    ax.set_xlabel("x (GM/c^2)")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out, dpi=80)


def main():
    o = dict(a.split("=") for a in sys.argv[1:])
    run = o.get("run", "R6.1_100k")
    first, n, per = int(o.get("first", 165)), int(o.get("n", 3)), int(o.get("per", 24))
    w, h, kappa = int(o.get("width", 640)), int(o.get("height", 360)), float(o.get("kappa", 1e7))
    folder = ROOT / "runs" / run
    eos, mass = run_eos_and_mass(folder)
    r_acc = 4.0
    snaps = [
        load_snap(folder / "snapshots" / f"snap_{s:05d}.npz") for s in range(first, first + n + 1)
    ]
    tracks(run, snaps, OUT / f"phase8_step5_tracks_{run}.png")
    print(f"wrote outputs/checks/phase8_step5_tracks_{run}.png")

    sky = upload_cube_map(build_cube_map(StarFieldConfig()))
    e = np.radians(5.0)
    cam = Camera(position=(40 * np.cos(e), 0.0, 40 * np.sin(e)), vfov_deg=60, width=w, height=h)
    model = GreyBlackbody(kappa=kappa)
    exposure = None
    frames = []
    t0 = time.perf_counter()
    for a, b in zip(snaps[:-1], snaps[1:], strict=True):
        for f in np.arange(per) / per:
            ids, x, s = state_at(a, b, a.t + f * (b.t - a.t), r_drop=r_acc)
            temp = snapshot_temperature(eos, s["rho"], s["u"])
            texs = textures_from_particles(x, s["h"], mass, temp, np.zeros_like(x))
            pic, _ = render_gas(cam, upload_scene(texs, model), n_sub=2, sky=sky,
                                out_mode=OUT_PICTURE)  # fmt: skip
            if exposure is None:  # fixed for the whole clip (no pumping)
                exposure = Exposure(auto_gas_exposure(pic[..., 0], 99.9, 1.0),
                                    auto_star_exposure(pic[..., 1:4]))  # fmt: skip
            frames.append(compose(pic, exposure))
            print(f"  day {(a.t + f * (b.t - a.t)) / DAY:.2f}: {len(frames)} frames, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    out = OUT / f"phase8_step5_clip_{run}.mp4"
    write_video(frames, out, fps=FPS)
    print(f"wrote {out.relative_to(ROOT)}: {len(frames)} frames, "
          f"{(time.perf_counter() - t0) / len(frames):.1f} s per frame")  # fmt: skip


if __name__ == "__main__":
    main()
