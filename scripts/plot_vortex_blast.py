"""Gresho vortex and Sedov blast pictures (Phase 3, Step 3.7).

outputs/checks/gresho_vortex.png  rotation speed vs radius at t = 1: exact, constant viscosity,
                                  with the switch (32 particles per unit length)
outputs/checks/sedov_blast.png    density vs radius at three times, SPH vs the exact solution
outputs/checks/vortex_blast.txt   measured numbers

The Sedov run (~1-2 hours) is cached under the code fingerprint and shared with the slow test.
Run: uv run python scripts/plot_vortex_blast.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.physics.sedov import sedov_solution  # noqa: E402
from nebula.sim.testproblems import (  # noqa: E402
    cached,
    gresho_l1,
    gresho_setup,
    gresho_vphi,
    run_problem,
    sedov_run,
    sedov_setup,
)

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"


def gresho(lines):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    r = np.linspace(0, 0.5, 500)
    ax.plot(r, gresho_vphi(r), color="#c92a2a", lw=2, label="exact (steady: should not change)")
    for switch, col, label in (
        (False, "#868e96", "constant viscosity (alpha = 1)"),
        (True, "#1f5fbf", "with the switch"),
    ):
        st = cached(f"gresho_high_{switch}", lambda s=switch: run_problem(gresho_setup("high", s)))
        l1, centres, prof = gresho_l1(st)
        ax.plot(centres, prof, "o-", ms=4, color=col, label=f"SPH, {label}: L1 {l1:.3f}")
        lines.append(f"Gresho (16,384 particles), t = 1, {label}: velocity L1 {l1:.4f}")
    ax.set_xlabel("distance from the vortex centre")
    ax.set_ylabel("rotation speed")
    ax.set_title("Gresho vortex after one time unit (32 particles per unit length)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECKS / "gresho_vortex.png", dpi=140)
    plt.close(fig)


def sedov(lines):
    results = cached("sedov_3tiles", lambda: sedov_run(sedov_setup(3)))
    exact = sedov_solution(5 / 3)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cols = ("#2b8a3e", "#1f5fbf", "#862e9c")
    for res, col in zip(results, cols, strict=True):
        ax.plot(
            res["centres"], res["profile"], ".", ms=4, color=col, label=f"SPH, t = {res['t']:.4f}"
        )
        r = np.linspace(0, 0.5, 600)
        rho, _, _ = exact.profile(r / res["r_exact"])
        ax.plot(r, rho, color=col, lw=1.2, alpha=0.8)
        ax.axvline(res["r_sph"], color=col, ls=":", lw=1)
        err = res["r_sph"] / res["r_exact"] - 1
        lines.append(
            f"Sedov t = {res['t']:.4f}: shock radius SPH {res['r_sph']:.4f}, "
            f"exact {res['r_exact']:.4f} ({err:+.2%}); total energy {res['energy']:.6f}"
        )
    ax.set_xlabel("distance from the blast centre")
    ax.set_ylabel("density (undisturbed = 1)")
    ax.set_title(
        "Sedov blast (110,592 particles): SPH dots vs exact lines; dotted = measured shock",
        fontsize=11,
    )
    ax.set_xlim(0, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECKS / "sedov_blast.png", dpi=140)
    plt.close(fig)


def main() -> None:
    CHECKS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    gresho(lines)
    sedov(lines)
    (CHECKS / "vortex_blast.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
