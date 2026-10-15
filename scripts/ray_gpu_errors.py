"""Error curve of the GPU ray integrator vs the float64 reference (Phase 2, Step 2.2).

Writes outputs/checks/ray_gpu_errors.png: |d phi_inf| against how close each ray passes to
the critical one (b / b_crit - 1), with the requirement lines.

Run: uv run python scripts/ray_gpu_errors.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.render.ray_validation import compare  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "ray_gpu_errors.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for d, col in ((50.0, "#1f5fbf"), (20.0, "#e8590c")):
        c = compare(d)
        both = np.isfinite(c.ref) & c.gpu_escaped & (c.b_over_bcrit > 1)
        x = c.b_over_bcrit[both] - 1.0
        err = np.maximum(np.abs(c.gpu - c.ref)[both], 1e-9)
        ax.loglog(x, err, ".", ms=3.5, color=col, label=f"camera at D = {d:g}")
    ax.hlines(2e-4, 0.05, 20, colors="#2b8a3e", lw=2, label="requirement for b > 1.05 b_crit")
    ax.hlines(1e-2, 1e-3, 20, colors="#c92a2a", lw=2, label="requirement for b > 1.001 b_crit")
    ax.axvline(1e-4, color="#495057", ls=":", lw=1)
    ax.text(1.1e-4, 3e-8, "classification\nmust match\nto the right", fontsize=8, color="#495057")
    ax.set_xlabel("how close the ray passes to the critical one:  b / b_crit - 1")
    ax.set_ylabel("|GPU escape angle - exact|  (rad)")
    ax.set_title(
        "GPU ray tracer (float32) vs exact reference: 4,000 rays\n"
        "errors grow only for rays that circle the hole many times (left)",
        fontsize=11,
    )
    ax.set_ylim(1e-9, 1)
    ax.set_xlim(5e-7, 20)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
