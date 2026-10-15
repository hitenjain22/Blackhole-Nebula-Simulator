"""Renderer timing (phases/phase-02-blackhole-render.md Step 2.6).

Times the star-field black-hole view at 720p and 1080p with 1, 4 and 16 rays per pixel:
GPU render (median of 3 after a warm-up), slowest tile (macOS watchdog: keep < ~1 s), and
the CPU time to tone-map and encode to 8-bit. Writes outputs/checks/render_benchmark.txt.

Run: uv run python scripts/render_benchmark.py
"""

import time
from pathlib import Path

import numpy as np

from nebula.render.camera import MODE_HOLE_STARS, Camera
from nebula.render.frames import render, to_png_array
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "render_benchmark.txt"
RESOLUTIONS = [(1280, 720), (1920, 1080)]
N_SUB = [1, 2, 4]  # rays per pixel = n_sub^2
REPEATS = 3
VIDEO_FRAMES = 3600  # 2 minutes at 30 fps


def main() -> None:
    sky = upload_cube_map(build_cube_map(StarFieldConfig()))
    rows = []
    for w, h in RESOLUTIONS:
        cam = Camera(position=(40.0, -15.0, 8.0), up=(0, 0, 1), vfov_deg=60, width=w, height=h)
        for n in N_SUB:
            args = (cam, MODE_HOLE_STARS, n, (sky.scale, 0.0), sky.entries(), "render_stars.wgsl")
            render(*args)  # warm-up
            times, tile_max = [], 0.0
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                img, tiles = render(*args)
                times.append(time.perf_counter() - t0)
                tile_max = max(tile_max, tiles.max)
            t0 = time.perf_counter()
            to_png_array(img[..., :3], 1.0)
            t_cpu = time.perf_counter() - t0
            rows.append((w, h, n * n, float(np.median(times)), tile_max, t_cpu))

    lines = [
        "Renderer timing: star field + black hole, camera at r = 43, 60 deg field of view.",
        "GPU = render on the Iris Plus (median of 3). CPU = tone map + sRGB + dither (NumPy).",
        "",
        f"{'resolution':>11} {'rays/px':>8} {'GPU s':>8} {'slowest tile s':>15} {'CPU s':>7} "
        f"{'s/frame':>8} {'2-min video (3,600 frames)':>28}",
    ]
    for w, h, rays, t_gpu, tile, t_cpu in rows:
        per = t_gpu + t_cpu
        lines.append(
            f"{w:>5}x{h:<5} {rays:>8} {t_gpu:>8.2f} {tile:>15.3f} {t_cpu:>7.2f} {per:>8.2f} "
            f"{per * VIDEO_FRAMES / 60:>21.0f} min"
        )
    worst_tile = max(r[4] for r in rows)
    lines += [
        "",
        f"Slowest single tile: {worst_tile:.3f} s ({'OK' if worst_tile < 1.0 else 'TOO SLOW'}: "
        "limit ~1 s, macOS GPU watchdog).",
    ]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
