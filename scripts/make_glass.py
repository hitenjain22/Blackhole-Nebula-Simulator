"""Make and store the relaxed glass (Phase 3, Step 3.4) and a picture of it.

data/glass_4096.npy, glass_512.npy   relaxed glasses in the periodic unit box (noise < 1%)
outputs/checks/glass.png        thin slice: random points vs the glass, coloured by density

Run: uv run python scripts/make_glass.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.sim.glass import density_noise, glass_path, make_glass  # noqa: E402
from nebula.sim.ic import smoothing_length_uniform  # noqa: E402
from nebula.sim.sph import compute_density  # noqa: E402

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"
N = 4096


def rho_of(pos):
    n = len(pos)
    h0 = np.full(n, smoothing_length_uniform(n, 1.0))
    return compute_density(pos, np.full(n, 1.0 / n), h0, box=1.0).rho * 1.0


def main() -> None:
    for n in (512, N):
        g, log = make_glass(n, seed=1)
        path = glass_path(n)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, g)
        print(f"Glass of {n}: {log.steps} steps, noise {log.noise[0]:.3f} -> {log.noise[-1]:.4f}")
        if n == N:
            glass = g

    random = np.random.default_rng(1).uniform(0, 1, (N, 3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, pos, title in ((axes[0], random, "random points"), (axes[1], glass, "relaxed glass")):
        rho = rho_of(pos)
        sl = np.abs(pos[:, 2] - 0.5) < 0.06
        sc = ax.scatter(
            pos[sl, 0], pos[sl, 1], c=rho[sl], s=14, cmap="coolwarm", vmin=0.6, vmax=1.4
        )
        ax.set_title(f"{title}: density noise {density_noise(pos):.1%}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.colorbar(sc, ax=axes, shrink=0.8, label="SPH density (mean = 1)")
    fig.suptitle(f"Thin slice of {N} particles in a periodic box", fontsize=13)
    CHECKS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHECKS / "glass.png", dpi=130)
    print(f"Written {CHECKS / 'glass.png'}")


if __name__ == "__main__":
    main()
