"""Star-field previews (Phase 2, Step 2.4).

outputs/checks/starfield_allsky.png   the whole sky, unrolled (longitude across, latitude up)
outputs/checks/starfield_view.png     what a camera sees with no hole (60 deg field of view)
outputs/checks/starfield_lensed.png   the same camera with the hole in front (preview of 2.5)

Run: uv run python scripts/starfield_preview.py
"""

import time
from pathlib import Path

import numpy as np

from nebula.render.camera import MODE_FLAT_STARS, MODE_HOLE_STARS, Camera
from nebula.render.frames import render, to_png_array, write_png
from nebula.render.starfield import (
    LUMA_WEIGHTS,
    StarFieldConfig,
    build_cube_map,
    cube_gpu_bytes,
    sample_cube,
    upload_cube_map,
)

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"
EXPOSURE_PERCENTILE = 99.98  # this luminance percentile maps to display value 0.5 (chosen by eye)


def main() -> None:
    cfg = StarFieldConfig()
    t0 = time.perf_counter()
    cube = build_cube_map(cfg)
    t_build = time.perf_counter() - t0
    sky = upload_cube_map(cube)
    print(
        f"Built {cfg.n_stars:,} stars + band, {cfg.face_size}^2 x 6 faces in {t_build:.1f} s; "
        f"GPU memory {cube_gpu_bytes(cfg.face_size) / 1e6:.0f} MB"
    )

    w, h = 2048, 1024
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    lo, la = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    allsky = sample_cube(sky, dirs.reshape(-1, 3), lod=0.0)[:, :3].reshape(h, w, 3) * cube.scale
    exposure = 0.5 / np.percentile(allsky @ LUMA_WEIGHTS, EXPOSURE_PERCENTILE)
    write_png(CHECKS / "starfield_allsky.png", to_png_array(allsky, exposure))

    cam = Camera(position=(40.0, -12.0, 6.0), look_at=(0.0, 0.0, 0.0), up=(0, 0, 1), vfov_deg=60)
    sky_params = (cube.scale, 0.0)
    flat, _ = render(cam, MODE_FLAT_STARS, 4, sky_params, sky.entries(), "render_stars.wgsl")
    write_png(CHECKS / "starfield_view.png", to_png_array(flat[..., :3], exposure))
    t0 = time.perf_counter()
    lensed, timing = render(cam, MODE_HOLE_STARS, 4, sky_params, sky.entries(), "render_stars.wgsl")
    t_frame = time.perf_counter() - t0
    write_png(CHECKS / "starfield_lensed.png", to_png_array(lensed[..., :3], exposure))
    print(f"Lensed 1280x720, 16 rays/pixel: {t_frame:.2f} s (slowest tile {timing.max:.2f} s)")
    print(f"Written starfield_allsky.png, starfield_view.png, starfield_lensed.png to {CHECKS}")


if __name__ == "__main__":
    main()
