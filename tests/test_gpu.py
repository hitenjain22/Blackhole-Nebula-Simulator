"""Step 0.4: the GPU test shader must match its float32 CPU reference, tile by tile."""

import numpy as np

from nebula.render.device import load_shader, run_tiled
from nebula.render.gpu_pattern import gpu_pattern_reference


def test_gpu_matches_cpu_reference():
    # 600 x 400 with 256-px tiles: includes partial tiles on the right and bottom edges.
    width, height = 600, 400
    gpu, timings = run_tiled(load_shader("gpu_pattern.wgsl"), width, height, tile=256)
    cpu = gpu_pattern_reference(width, height)
    rel = np.abs(gpu - cpu) / np.abs(cpu)
    names = ["x final", "x after 20", "mean x", "alpha"]
    print()
    for c, name in enumerate(names):
        worst = rel[..., c].max()
        verdict = "PASS" if worst < 1e-5 else "FAIL"
        print(f"GPU vs CPU, {name:11s} | max rel diff {worst:.2e} | tol 1e-5 | {verdict}")
    print(f"tiles: {len(timings.per_tile)}, slowest {timings.max * 1e3:.1f} ms")
    assert rel.max() < 1e-5
