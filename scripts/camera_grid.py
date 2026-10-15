"""Camera check images (Phase 2, Step 2.3).

outputs/checks/camera_grid_flat.png    no hole, straight rays: the sky grid must look straight
outputs/checks/camera_grid_lensed.png  same camera idea with the hole: a latitude-longitude
                                       sky grid (orange = equator, green = longitude 0) as
                                       bent by gravity, next to the same view without the hole

Run: uv run python scripts/camera_grid.py
"""

import time
from pathlib import Path

import numpy as np

from nebula.render.camera import (
    MODE_FLAT_GNOMONIC_GRID,
    MODE_FLAT_LATLONG_GRID,
    MODE_HOLE_LATLONG_GRID,
    Camera,
)
from nebula.render.frames import render, to_png_array, write_png

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"


def main() -> None:
    flat_cam = Camera(position=(0.0, 0.0, -100.0), look_at=(0, 0, 0), up=(0, 1, 0), vfov_deg=60)
    img, _ = render(flat_cam, MODE_FLAT_GNOMONIC_GRID, n_sub=4, sky_params=(0.1, 0.002))
    write_png(CHECKS / "camera_grid_flat.png", to_png_array(img[..., :3], exposure=1.6))

    cam = Camera(
        position=(30.0, 0.0, 3.0),
        look_at=(0, 0, 0),
        up=(0, 0, 1),
        vfov_deg=75,
        width=960,
        height=720,
    )
    t0 = time.perf_counter()
    lensed, _ = render(cam, MODE_HOLE_LATLONG_GRID, n_sub=4, sky_params=(15.0, 0.2))
    t_lensed = time.perf_counter() - t0
    flat, _ = render(cam, MODE_FLAT_LATLONG_GRID, n_sub=4, sky_params=(15.0, 0.2))
    gap = np.ones((cam.height, 16, 3), dtype=np.float32)
    both = np.concatenate([flat[..., :3], gap, lensed[..., :3]], axis=1)
    write_png(CHECKS / "camera_grid_lensed.png", to_png_array(both, exposure=1.6))
    print(f"Lensed 960x720, 16 rays/pixel: {t_lensed:.2f} s")
    print(f"Written camera_grid_flat.png and camera_grid_lensed.png to {CHECKS}")


if __name__ == "__main__":
    main()
