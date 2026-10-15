"""GPU check image and timing (Phase 0, Step 0.4).

Renders the check pattern at 1280 x 720 in 256 x 256 tiles, compares it with the CPU
float32 reference, saves outputs/checks/gpu_test.png and outputs/checks/gpu_timing.txt.

Run: uv run python scripts/gpu_check.py
"""

import time
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nebula.render.device import get_device, load_shader, run_tiled
from nebula.render.gpu_pattern import gpu_pattern_reference

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"
WIDTH, HEIGHT, TILE = 1280, 720, 256
REPEATS = 3


def to_png(img: np.ndarray) -> np.ndarray:
    """Stretch each of the three channels to 0..1 so the pattern is visible; 8-bit RGB."""
    rgb = img[..., :3].astype(np.float64)
    lo = rgb.min(axis=(0, 1))
    hi = rgb.max(axis=(0, 1))
    return np.round(255 * (rgb - lo) / (hi - lo)).astype(np.uint8)


def main() -> None:
    shader = load_shader("gpu_pattern.wgsl")
    run_tiled(shader, WIDTH, HEIGHT, TILE)  # warm-up: shader compilation, driver start-up
    runs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        img, tiles = run_tiled(shader, WIDTH, HEIGHT, TILE)
        runs.append((time.perf_counter() - t0, tiles))
    cpu = gpu_pattern_reference(WIDTH, HEIGHT)
    rel = float(np.max(np.abs(img - cpu) / np.abs(cpu)))

    frame_times = [r[0] for r in runs]
    tile_times = np.concatenate([r[1].per_tile for r in runs])
    info = get_device().adapter.info
    lines = [
        f"GPU check, {datetime.now():%Y-%m-%d %H:%M}",
        f"GPU: {info['device']} ({info['backend_type']})",
        f"Image {WIDTH} x {HEIGHT}, tiles {TILE} x {TILE} ({len(runs[0][1].per_tile)} per image), "
        f"2,000 loop iterations per pixel; {REPEATS} timed runs after a warm-up.",
        "",
        f"Per image:  median {np.median(frame_times) * 1e3:.0f} ms, "
        f"max {max(frame_times) * 1e3:.0f} ms",
        f"Per tile:   median {np.median(tile_times) * 1e3:.1f} ms, "
        f"max {tile_times.max() * 1e3:.1f} ms  (limit ~1,000 ms: "
        f"{'OK' if tile_times.max() < 1.0 else 'TOO SLOW'})",
        f"GPU vs CPU reference: max rel diff {rel:.2e} "
        f"(tol 1e-5: {'PASS' if rel < 1e-5 else 'FAIL'})",
    ]
    text = "\n".join(lines) + "\n"
    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / "gpu_timing.txt").write_text(text)
    iio.imwrite(CHECKS / "gpu_test.png", to_png(img))
    print(text)
    print(f"Written {CHECKS / 'gpu_test.png'} and {CHECKS / 'gpu_timing.txt'}")


if __name__ == "__main__":
    main()
