"""Python wrappers around the GPU light-ray integrator (shaders/schw_ray.wgsl)."""

from dataclasses import dataclass

import numpy as np

from nebula.render.device import load_shader, run_items

RAY_ESCAPED, RAY_CAPTURED, RAY_ORBITING = 0, 1, 2


@dataclass
class GPURays:
    """Per ray: escape angle phi_inf (rad; NaN unless escaped), status code, half-turns
    around the hole, and RK4 steps taken."""

    phi_inf: np.ndarray
    status: np.ndarray
    half_turns: np.ndarray
    steps: np.ndarray


def gpu_escape_angles(d: float, psi: np.ndarray) -> GPURays:
    """GPU (float32) escape angles for rays at angles psi (rad) from the direction to the hole,
    seen by a static observer at r = d. Twin of physics.geodesics.schw_ray_escape_angle."""
    psi = np.asarray(psi, dtype=np.float64)
    rays = np.zeros((len(psi), 4), dtype=np.float32)
    rays[:, 0] = 1.0 / d
    rays[:, 1] = np.cos(psi)
    rays[:, 2] = np.sin(psi)
    uniform = np.array([len(psi), 0, 0, 0], dtype=np.uint32).tobytes()
    raw = run_items(
        load_shader("schw_ray.wgsl", "escape_angles.wgsl"), uniform, [rays], len(psi), 16
    )
    out = np.frombuffer(raw, dtype=np.float32).reshape(len(psi), 4).astype(np.float64)
    status = out[:, 1].astype(int)
    phi = np.where(status == RAY_ESCAPED, out[:, 0], np.nan)
    return GPURays(phi, status, out[:, 2].astype(int), out[:, 3].astype(int))
