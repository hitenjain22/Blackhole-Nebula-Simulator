"""Phase 8 step 6d: the timing table and render budget (phases/phase-08-gas-rendering.md 8.7,
8.9 item 5). Measured on this machine, as is (CLAUDE.md: the CPU throttles when hot; report
what is measured).

    uv run python scripts/phase8_timing.py [shot=configs/shots/orbit_pass.toml] [frame=144]
        [repeats=2]

For one frame of the shot: the CPU part (gas state between snapshots + deposition onto the
grids) and the GPU part (bent-ray march) at 720p / 1080p / 4K with 1 / 4 / 16 rays per pixel
(n_sub = 1 / 2 / 4), best of `repeats`, with the slowest GPU tile (must stay < 1 s). Writes
outputs/checks/phase8_timing.md with the table and the budget for a 30 s trailer.
"""

import sys
import time
from pathlib import Path

import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.gpu_march import OUT_PICTURE, render_gas, upload_scene
from nebula.render.interpolate import state_at
from nebula.render.shot_render import ShotRenderer
from nebula.render.snapshot_scene import textures_from_particles

ROOT = Path(__file__).resolve().parents[1]
SIZES = [("720p", 1280, 720), ("1080p", 1920, 1080), ("4K", 3840, 2160)]
SUBS = [1, 2, 4]


def main():
    o = dict(a.split("=", 1) for a in sys.argv[1:])
    shot = Shot.load(ROOT / o.get("shot", "configs/shots/orbit_pass.toml"))
    i, repeats = int(o.get("frame", 144)), int(o.get("repeats", 2))
    timeline = Timeline.from_sources(shot.sources, ROOT / "runs")
    plan = plan_frames(shot, timeline)
    rend = ShotRenderer(shot, plan, timeline)
    t = float(plan.sim_t[i])
    k = min(max(int(np.searchsorted(timeline.times, t, side="right")) - 1, 0),
            len(timeline.times) - 2)  # fmt: skip
    a, b = rend._snap(k), rend._snap(k + 1)
    cpu = {"state between snapshots": [], "deposit onto grids": []}
    for _ in range(repeats):
        t0 = time.perf_counter()
        ids, x, s = state_at(a, b, t, r_drop=4.0)
        t1 = time.perf_counter()
        eos, mass = rend._eos_mass(k)
        temp = snapshot_temperature(eos, s["rho"], s["u"])
        texs = textures_from_particles(x, s["h"], mass, temp, np.zeros_like(x), rend.grid,
                                       plan.boxes(i, shot))  # fmt: skip
        t2 = time.perf_counter()
        cpu["state between snapshots"].append(t1 - t0)
        cpu["deposit onto grids"].append(t2 - t1)
    scene = upload_scene(texs, rend.model)
    rows = []
    for label, w, h in SIZES:
        for n_sub in SUBS:
            cam = plan.camera(i, w, h)
            best, tile = np.inf, 0.0
            for _ in range(repeats):
                t0 = time.perf_counter()
                pic, tm = render_gas(cam, scene, n_sub=n_sub, sky=rend.sky, out_mode=OUT_PICTURE)
                dt = time.perf_counter() - t0
                if dt < best:
                    best, tile = dt, tm.max
            rows.append((label, n_sub * n_sub, best, tile))
            print(f"{label} x {n_sub * n_sub:2d} rays: GPU {best:7.1f} s (slowest tile "
                  f"{tile:.2f} s)", flush=True)  # fmt: skip
    cpu_s = sum(min(v) for v in cpu.values())
    lines = [
        f"# Phase 8 timing ({time.strftime('%Y-%m-%d')}, this Mac: Intel i3 2 cores, Iris Plus)",
        "",
        f"Frame {i} of shot `{shot.name}` (day {t / 4079.27:.2f}, {len(x):,} particles).",
        "",
        "CPU part per frame (same for every resolution):",
        "",
    ]
    lines += [f"- {name}: {min(v):.1f} s" for name, v in cpu.items()]
    lines += [
        "",
        "| size | rays/pixel | GPU s/frame | slowest tile s | total s/frame | 30 s (720 frames) |",
        "|---|---|---|---|---|---|",
    ]
    for label, rays, g, tile in rows:
        tot = g + cpu_s
        lines.append(f"| {label} | {rays} | {g:.1f} | {tile:.2f} | {tot:.1f} | "
                     f"{tot * 720 / 3600:.1f} h |")  # fmt: skip
    out = ROOT / "outputs" / "checks" / "phase8_timing.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
