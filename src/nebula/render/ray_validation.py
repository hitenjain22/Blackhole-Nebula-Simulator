"""GPU-vs-reference comparison of escape angles (Phase 2, Step 2.2): shared by the test and by
scripts/ray_gpu_errors.py (the error-curve plot)."""

from dataclasses import dataclass

import numpy as np

from nebula.physics.geodesics import (
    B_CRIT,
    schw_impact_parameter,
    schw_ray_escape_angle,
    schw_shadow_angle,
)
from nebula.render.rays import RAY_ESCAPED, gpu_escape_angles


def test_angles(d: float, n: int = 2000) -> np.ndarray:
    """n viewing angles psi (rad) in [0, 180 deg]: half uniform, half packed toward the shadow
    edge from both sides (b / b_crit - 1 = +-10^s, s from -6 to 0, log-spaced)."""
    n_edge = n // 2
    uniform = np.linspace(0.0, np.pi, n - n_edge)
    s = np.linspace(-6.0, 0.0, n_edge // 2)
    b = np.concatenate([B_CRIT * (1 + 10**s), B_CRIT * (1 - 10**s * 0.999)])
    sin_psi = np.clip(b * np.sqrt(1 - 2 / d) / d, 0.0, 1.0)
    edge = np.arcsin(sin_psi)  # looking inward (psi < 90 deg), where the shadow is
    psi = np.sort(np.concatenate([uniform, edge]))
    assert np.all(np.abs(psi - schw_shadow_angle(d)) >= 0) and len(psi) == n
    return psi


@dataclass
class Comparison:
    d: float
    psi: np.ndarray
    b_over_bcrit: np.ndarray
    ref: np.ndarray  # reference phi_inf (NaN = captured)
    gpu: np.ndarray  # GPU phi_inf (NaN = captured or orbiting)
    gpu_escaped: np.ndarray


def compare(d: float, n: int = 2000) -> Comparison:
    psi = test_angles(d, n)
    ref = np.array([schw_ray_escape_angle(d, p) for p in psi])
    g = gpu_escape_angles(d, psi)
    b = np.array([schw_impact_parameter(d, p) for p in psi])
    return Comparison(d, psi, b / B_CRIT, ref, g.phi_inf, g.status == RAY_ESCAPED)
