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
