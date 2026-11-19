"""Step 0.4: the GPU test shader must match its float32 CPU reference, tile by tile."""

from nebula.render.device import load_shader, run_tiled
from nebula.render.gpu_pattern import gpu_pattern_reference
from tests.helpers import check


def test_gpu_matches_cpu_reference():
    # 600 x 400 with 256-px tiles: includes partial tiles on the right and bottom edges.
    width, height = 600, 400
    gpu, timings = run_tiled(load_shader("gpu_pattern.wgsl"), width, height, tile=256)
    cpu = gpu_pattern_reference(width, height)
    for c, name in enumerate(["x final", "x after 20", "mean x", "alpha"]):
        check(f"GPU vs CPU, {name}", cpu[..., c], gpu[..., c], rtol=1e-5)
    check("GPU slowest tile under 1 s", True, timings.max < 1.0)


def test_adaptive_tiles_do_not_change_pixels():
    """Tiles shrink when slow (target 1e-6 s: every tile at the 8-px minimum, uneven rows) and
    grow when fast (target 10 s); the image is the same pixel for pixel (AI_LOG 2026-11-19)."""
    width, height = 300, 200
    shader = load_shader("gpu_pattern.wgsl")
    ref, _ = run_tiled(shader, width, height, tile=64)
    small, t_small = run_tiled(shader, width, height, tile=64, target_s=1e-6)
    big, t_big = run_tiled(shader, width, height, tile=64, target_s=10.0)
    check("adaptive tiles, slow target: same pixels", ref, small, atol=0.0)
    check("adaptive tiles, fast target: same pixels", ref, big, atol=0.0)
    check("slow target: tiles shrink to the minimum (many tiles)", True,
          len(t_small.per_tile) >= (width // 8) * (height // 8) * 0.9)  # fmt: skip
    check("fast target: fewer tiles than the slow one", True,
          len(t_big.per_tile) < len(t_small.per_tile) / 10)  # fmt: skip
