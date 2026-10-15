"""Sod shock tube and sound wave against their exact solutions (Phase 3, Step 3.6).

outputs/checks/sod_shock_tube.png   density, velocity, pressure, internal energy vs x at t = 0.2
outputs/checks/sound_wave.png       the wave after one period, two resolutions, vs exact
outputs/checks/standard_problems.txt measured numbers

Runs the plan's full-resolution set-ups (~20 minutes on this Mac).
Run: uv run python scripts/plot_standard_problems.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.physics.riemann import solve_riemann  # noqa: E402
from nebula.sim.testproblems import (  # noqa: E402
    run_problem,
    run_sound,
    sod_exact,
    sod_measure,
    sod_setup,
    sound_measure,
    sound_setup,
)

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"


def sod(lines):
    t0 = time.perf_counter()
    prob = sod_setup("full")
    st = run_problem(prob)
    m = sod_measure(prob, st)
    ex = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)

    def row(name, got, exact):
        return f"  {name:6s}{got:.4f} (exact {exact:.4f}, {got / exact - 1:+.2%})"

    lines += [
        f"Sod shock tube (36,864 particles), t = 0.2, {time.perf_counter() - t0:.0f} s",
        row("rho*L", m.rho_star_l, ex.rho_star_l),
        row("rho*R", m.rho_star_r, ex.rho_star_r),
        row("p*", m.p_star, ex.p_star),
        row("u*", m.u_star, ex.u_star),
        f"  shock at {m.shock_x - prob.x0:.4f} (exact {m.shock_x_exact - prob.x0:.4f}): "
        f"{abs(m.shock_x - m.shock_x_exact) / m.h_shock:.2f} smoothing lengths",
        f"  post-shock oscillation {m.oscillation:.2%}",
    ]
    x = st.x[:, 0] - prob.x0
    win = (x > -0.4) & (x < 0.45)
    xs = np.linspace(-0.4, 0.45, 2000)
    rho_e, v_e, p_e = sod_exact(prob, xs + prob.x0, st.t)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        (st.rho, rho_e, "density"),
        (st.v[:, 0], v_e, "velocity"),
        (st.pressure, p_e, "pressure"),
        (st.u, p_e / (0.4 * rho_e), "internal energy u"),
    ]
    for ax, (sph, exact, name) in zip(axes.ravel(), panels, strict=True):
        ax.plot(x[win], sph[win], ".", ms=1.5, color="#1f5fbf", alpha=0.35, label="SPH particles")
        ax.plot(xs, exact, color="#c92a2a", lw=1.4, label="exact solution")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=9, markerscale=6)
    for ax in axes[1]:
        ax.set_xlabel("x (diaphragm at 0)")
    fig.suptitle(
        "Sod shock tube at t = 0.2: SPH (Wendland C4, 36,864 particles) vs exact", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(CHECKS / "sod_shock_tube.png", dpi=140)
    plt.close(fig)


def sound(lines):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xx = np.linspace(0, 1, 400)
    ax.plot(
        xx, np.cos(2 * np.pi * xx), color="#c92a2a", lw=1.5, label="exact (back where it started)"
    )
    for res, col, label in (
        ("low", "#e8590c", "32 per wavelength"),
        ("high", "#1f5fbf", "64 per wavelength"),
    ):
        t0 = time.perf_counter()
        prob = sound_setup(res)
        m = sound_measure(prob, *run_sound(prob))
        ax.plot(m.profile_x, m.profile, "o-", ms=4, color=col, label=f"SPH, {label}")
        lines.append(
            f"Sound wave {label} ({len(prob.state.x):,} particles), one period, "
            f"{time.perf_counter() - t0:.0f} s: amplitude {m.amplitude:.4f}, "
            f"phase error {m.phase_cycles:+.2%} of a wavelength, L1 {m.l1:.4f}"
        )
    ax.set_xlabel("x (one wavelength)")
    ax.set_ylabel("density perturbation / amplitude (1e-4)")
    ax.set_title(
        "Sound wave after one full period (Wendland C4): slice-averaged density", fontsize=12
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECKS / "sound_wave.png", dpi=140)
    plt.close(fig)


def main() -> None:
    CHECKS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    sod(lines)
    sound(lines)
    text = "\n".join(lines) + "\n"
    (CHECKS / "standard_problems.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
