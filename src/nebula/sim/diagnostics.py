"""Run diagnostics (phases/phase-05-cloud-infall.md 5.2; DECISIONS D5).

``hole_resolution``: does SPH resolve the gas near the hole? For particles at r < r_near:
the median and largest 2h/r (kernel diameter over distance to the hole), and the share of
neighbour pairs (within 2 h of the particle) more than 90 degrees apart as seen from the hole,
i.e. the kernel reaching across the hole, where SPH would make gas moving in opposite
directions interact although it never meets (review docs/review-2026-10-15-plunge.md §2b).
A warning is raised above 1%. Also the internal energy of the gas near the hole (r < r_gas),
typical (median) and hottest 1%, to compare with the gravity-only ceiling (review §2d).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

ACROSS_WARNING = 0.01  # share of pairs across the hole that triggers a warning (D5)


@dataclass
class HoleResolution:
    n_near: int
    two_h_over_r_median: float
    two_h_over_r_max: float
    across_fraction: float  # share of neighbour pairs more than 90 degrees apart
    pairs: int
    n_gas: int
    u_median: float
    u_p99: float
    u_max: float

    @property
    def warning(self) -> bool:
        return self.across_fraction > ACROSS_WARNING

    def line(self) -> str:
        flag = "  WARNING: kernels span the hole" if self.warning else ""
        return (
            f"r<60: {self.n_near} particles, 2h/r median {self.two_h_over_r_median:.2f} max "
            f"{self.two_h_over_r_max:.2f}, across the hole {100 * self.across_fraction:.1f}% of "
            f"{self.pairs} pairs; gas r<300: {self.n_gas}, u median {self.u_median:.2e}, "
            f"hottest 1% {self.u_p99:.2e}, max {self.u_max:.2e}{flag}"
        )


def hole_resolution(
    x: np.ndarray, h: np.ndarray, u: np.ndarray, r_near: float = 60.0, r_gas: float = 300.0
) -> HoleResolution:
    """See the module docstring. x (N, 3), h and u (N,), code units."""
    r = np.linalg.norm(x, axis=1)
    near = np.flatnonzero(r < r_near)
    across = pairs = 0
    if len(near):
        tree = cKDTree(x)
        for a in near:
            nb = np.array([b for b in tree.query_ball_point(x[a], 2.0 * h[a]) if b != a], int)
            if len(nb) == 0:
                continue
            cos = (x[nb] @ x[a]) / (r[nb] * r[a])
            pairs += len(nb)
            across += int(np.sum(cos < 0.0))  # more than 90 degrees apart
        ratio = 2.0 * h[near] / r[near]
        med, mx = float(np.median(ratio)), float(np.max(ratio))
    else:
        med = mx = float("nan")
    gas = r < r_gas
    ug = u[gas]
    stats = np.percentile(ug, [50, 99]) if len(ug) else [np.nan, np.nan]
    return HoleResolution(
        n_near=len(near), two_h_over_r_median=med, two_h_over_r_max=mx,
        across_fraction=across / pairs if pairs else 0.0, pairs=pairs, n_gas=int(gas.sum()),
        u_median=float(stats[0]), u_p99=float(stats[1]),
        u_max=float(ug.max()) if len(ug) else float("nan"),
    )  # fmt: skip
