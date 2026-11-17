"""True-colour still of one shot frame with the CPU reference renderer (Phase 9 steps 9d-9e
check; no GPU, so it is safe to run unattended, but slow: plain Python per sample).

    uv run python scripts/phase9_cpu_colour_still.py configs/shots/continuous.toml t=27
        [w=96] [h=54] [workers=3] [pixels=0]

Builds the frame's gas (positions and momenta between snapshots, gamma beta from the GN
momenta) on the shot's log grid (no bricks: the readable brick sampler is too slow here),
renders X, Y, Z, bolometric with bent_rays.render_reference_colour in worker processes (rows
split between them), and writes outputs/checks/phase9_cpu_colour_<shot>_t<t>.png (true colour,
exposure from the 99.5th percentile of luminance, no stars) plus the raw .npy. pixels=N:
time N pixels only and stop."""

import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
_W = {}


def _init(path):
    with open(path, "rb") as f:
        _W.update(pickle.load(f))


def _rows(rows):
    from nebula.render.bent_rays import render_reference_colour

    cam, texs, model = _W["cam"], _W["texs"], _W["model"]
    pix = [(i, j) for j in rows for i in range(cam.width)]
    out = render_reference_colour(cam, texs, model, pixels=pix)
    return rows, out[rows[0] : rows[-1] + 1]


def main():
    from nebula.analysis.cloud_metrics import snapshot_temperature
    from nebula.physics.colour import chroma_lut
    from nebula.physics.redshift import gn_static_frame_velocity
    from nebula.render.colour_data import load_cmf
    from nebula.render.director import Shot, Timeline, plan_frames
    from nebula.render.gas_frames import Exposure, auto_gas_exposure, compose_colour
    from nebula.render.interpolate import load_snap, state_at
    from nebula.render.march import RelativisticBlackbody
    from nebula.render.snapshot_scene import run_eos_and_mass, textures_from_particles

    shot_path = Path(sys.argv[1])
    o = dict(a.split("=", 1) for a in sys.argv[2:])
    t_video = float(o.get("t", 27))
    w, h = int(o.get("w", 96)), int(o.get("h", 54))
    workers = int(o.get("workers", 3))
    shot = Shot.load(shot_path)
    timeline = Timeline.from_sources(shot.sources, ROOT / "runs")
    plan = plan_frames(shot, timeline)
    i = int(round(t_video * shot.fps))
    t = float(plan.sim_t[i])
    k = min(max(int(np.searchsorted(timeline.times, t, side="right")) - 1, 0),
            len(timeline.times) - 2)  # fmt: skip
    a, b = load_snap(timeline.paths[k]), load_snap(timeline.paths[k + 1])
    _, x, p, s = state_at(a, b, min(max(t, a.t), b.t), with_p=True)
    eos, mass = run_eos_and_mass(Path(timeline.paths[k]).parent.parent)
    temp = snapshot_temperature(eos, s["rho"], s["u"])
    gb = gn_static_frame_velocity(x, p)[1]
    texs = textures_from_particles(x, s["h"], mass, temp, gb, shot.log_grid())
    cam = plan.camera(i, w, h)
    lam, cmf = load_cmf()
    lt, lc = chroma_lut(1e3, 1e8, 4096, lam, cmf)
    model = RelativisticBlackbody(kappa=shot.kappa, r_cam=float(np.linalg.norm(cam.position)),
                                  log_t=lt, log_chroma=lc)  # fmt: skip
    print(f"frame {i} (video t {t_video} s, day {t / 4079.27:.2f}), {len(x)} particles, "
          f"camera r {np.linalg.norm(cam.position):.0f}, |gamma beta| max {np.abs(gb).max():.3f}",
          flush=True)  # fmt: skip
    if int(o.get("pixels", 0)):
        from nebula.render.bent_rays import render_reference_colour

        n = int(o["pixels"])
        pix = [(int(c), h // 2) for c in np.linspace(0, w - 1, n)]
        t0 = time.perf_counter()
        render_reference_colour(cam, texs, model, pixels=pix)
        dt = (time.perf_counter() - t0) / n
        print(f"{dt:.2f} s per pixel -> {w}x{h} with {workers} workers ~ "
              f"{dt * w * h / workers / 3600:.1f} h", flush=True)  # fmt: skip
        return
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"cam": cam, "texs": texs, "model": model}, f)
        state = f.name
    del texs
    img = np.full((h, w, 7), np.nan)
    chunks = [list(range(j, min(j + 2, h))) for j in range(0, h, 2)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(state,)) as ex:
        for n, (rows, part) in enumerate(ex.map(_rows, chunks), 1):
            img[rows[0] : rows[-1] + 1] = part
            print(f"rows {n}/{len(chunks)}, {time.perf_counter() - t0:.0f} s", flush=True)
    Path(state).unlink()
    tag = f"phase9_cpu_colour_{shot.name}_t{t_video:g}"
    np.save(OUT / f"{tag}.npy", img)
    exp = auto_gas_exposure(img[..., 1], 99.5, 2.0)  # on luminance Y
    pic = np.zeros((h, w, 8))
    pic[..., :4] = img[..., :4]
    iio.imwrite(OUT / f"{tag}.png", compose_colour(pic, Exposure(exp, 0.0)))
    print(f"wrote {OUT / tag}.png", flush=True)


if __name__ == "__main__":
    main()
