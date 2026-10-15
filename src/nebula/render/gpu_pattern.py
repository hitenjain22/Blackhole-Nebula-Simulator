"""CPU float32 reference for shaders/gpu_pattern.wgsl (Phase 0, Step 0.4)."""

import numpy as np

N_BLOCKS = 50
BLOCK = 40  # N_BLOCKS * BLOCK = 2,000 iterations
N_EARLY = 20


def gpu_pattern_reference(width: int, height: int) -> np.ndarray:
    """Same arithmetic as the shader, in float32. Returns (H, W, 4):
    (x after 2,000 steps, x after N_EARLY steps, mean x over all steps (block-summed), 1)."""
    f = np.float32
    u = (np.arange(width, dtype=f) + f(0.5)) / f(width)
    v = (np.arange(height, dtype=f) + f(0.5)) / f(height)
    r = np.broadcast_to(f(2.5) + f(0.45) * u, (height, width))
    x = np.broadcast_to((f(0.05) + f(0.9) * v)[:, None], (height, width)).copy()
    total = np.zeros((height, width), dtype=f)
    early = None
    for b in range(N_BLOCKS):
        block_sum = np.zeros((height, width), dtype=f)
        for k in range(BLOCK):
            x = r * x * (f(1.0) - x)
            block_sum += x
            if b * BLOCK + k == N_EARLY - 1:
                early = x.copy()
        total += block_sum
    out = np.ones((height, width, 4), dtype=f)
    out[..., 0] = x
    out[..., 1] = early
    out[..., 2] = total / f(N_BLOCKS * BLOCK)
    return out
