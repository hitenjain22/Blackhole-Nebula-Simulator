"""Orbit pictures for Hiten and the report (phases/phase-01-orbits.md Step 1.5).

Writes to outputs/checks/: orbit_rosette.png, orbit_zoom_pericentre.png, isco_demo.png,
timeline.txt; copies the rosette and ISCO plots to docs/figures/.

Run: uv run python scripts/plot_orbits.py
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula import units  # noqa: E402
from nebula.analysis.orbit_metrics import pericentres  # noqa: E402
from nebula.physics.gravity import gn_p_from_v  # noqa: E402
from nebula.physics.orbits import (  # noqa: E402
    apocentre_state,
    orbit_angle_from_pericentre,
    precession_exact,
)
from nebula.sim.integrator import GLFConvergenceError, integrate_orbit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "outputs" / "checks"
FIGURES = ROOT / "docs" / "figures"
RP, RA = 25.0, 5000.0
ETA = 0.00125  # docs/DECISIONS.md, Step 1.4
N_ORBITS = 5
RADIAL_PERIOD = 792_447.254  # expected-values §2
BH = units.BlackHoleUnits()

COL_GN, COL_EXACT, COL_NEWTON = "#1f5fbf", "#e8590c", "#6c757d"


def exact_schwarzschild_path(n_orbits: int, n_r: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """The exact orbit shape r(phi), from the Step 1.3 integral (independent of the
    integrator). Starts at apocentre on +x, like the simulation."""
    th = np.linspace(0.0, np.pi, n_r)  # dense near both turning points
    u_p, u_a = 1 / RP, 1 / RA
    r = 1.0 / (0.5 * (u_p + u_a) + 0.5 * (u_p - u_a) * np.cos(th))  # pericentre -> apocentre
    phi_branch = orbit_angle_from_pericentre(RP, RA, r)
    phi_half = phi_branch[-1]  # pericentre -> apocentre angle (192.02 deg)
    step = 2 * phi_half  # angle between successive apocentres = 2 pi + precession
    xs, ys = [], []
    for k in range(n_orbits):
        phi_peri = phi_half + k * step
        for rr, ph in ((r[::-1], phi_peri - phi_branch[::-1]), (r, phi_peri + phi_branch)):
            xs.append(rr * np.cos(ph))
            ys.append(rr * np.sin(ph))
    return np.concatenate(xs), np.concatenate(ys)


def newton_ellipse() -> tuple[np.ndarray, np.ndarray]:
    """Kepler ellipse with the same r_p and r_a, apocentre on +x (it never rotates)."""
    a, e = 0.5 * (RP + RA), (RA - RP) / (RA + RP)
    phi = np.linspace(0, 2 * np.pi, 4000)
    r = a * (1 - e * e) / (1 - e * np.cos(phi))  # r(0) = r_a
    return r * np.cos(phi), r * np.sin(phi)


def draw_hole(ax, labels: bool = True) -> None:
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(
        2 * np.cos(th),
        2 * np.sin(th),
        color="black",
        zorder=5,
        label="event horizon (r = 2)" if labels else None,
    )
    ax.plot(
        6 * np.cos(th),
        6 * np.sin(th),
        ls="--",
        lw=0.8,
        color="#c92a2a",
        zorder=5,
        label="innermost stable orbit (r = 6)" if labels else None,
    )


def plot_rosette(glf, ex, ey, nx, ny) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(nx, ny, ls=":", lw=1.4, color=COL_NEWTON, label="Newton (closed ellipse)")
    ax.plot(glf.x[:, 0], glf.x[:, 1], lw=2.2, color=COL_GN, label="simulation (GN gravity)")
    ax.plot(ex, ey, ls="--", lw=1.0, color=COL_EXACT, label="exact relativistic shape")
    draw_hole(ax, labels=False)
    ax.plot([RA], [0], "o", color=COL_GN, ms=7, zorder=6)
    ax.annotate(
        "start (r = 5,000)",
        (RA, 0),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x  [GM/c² = 6.35 million km]")
    ax.set_ylabel("y  [GM/c²]")
    ax.set_title(
        f"Cloud-centre orbit, {N_ORBITS} orbits: it turns 24.04° each time round\n"
        "(Newton's orbit would close on itself)",
        fontsize=12,
    )
    au = units.AU / BH.length_m
    sec = ax.secondary_xaxis("top", functions=(lambda v: v / au, lambda v: v * au))
    sec.set_xlabel("AU")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHECKS / "orbit_rosette.png", dpi=150)
    plt.close(fig)


def plot_zoom(glf, ex, ey, t_peri, phi_peri) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(glf.x[:, 0], glf.x[:, 1], lw=2.2, color=COL_GN, label="simulation (GN gravity)")
    ax.plot(ex, ey, ls="--", lw=1.0, color=COL_EXACT, label="exact relativistic shape")
    draw_hole(ax)
    lim = 120
    for k, ph in enumerate(phi_peri):
        ax.plot(
            [0, 1.6 * lim * np.cos(ph)],
            [0, 1.6 * lim * np.sin(ph)],
            lw=0.8,
            color="#495057",
            alpha=0.7,
        )
        ax.annotate(
            f"pass {k + 1}\n(day {t_peri[k] * BH.time_s / units.DAY:.0f})",
            (108 * np.cos(ph), 108 * np.sin(ph)),
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.8},
        )
    # Angle arc between passes 1 and 2.
    arc = np.linspace(phi_peri[0], phi_peri[1] - 2 * np.pi, 100)
    ax.plot(70 * np.cos(arc), 70 * np.sin(arc), color="#2b8a3e", lw=2)
    mid = arc[50]
    prec = np.degrees(phi_peri[1] - phi_peri[0] - 2 * np.pi)
    ax.annotate(
        f"{prec:.2f}°",
        (70 * np.cos(mid), 70 * np.sin(mid)),
        xytext=(-30, -12),
        textcoords="offset points",
        fontsize=13,
        color="#2b8a3e",
        weight="bold",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [GM/c²]")
    ax.set_ylabel("y  [GM/c²]")
    ax.set_title(
        "Closest approaches (r = 25 = 1.06 AU): each pass is rotated by "
        f"{prec:.2f}°\n(exact relativistic value 24.04°)",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHECKS / "orbit_zoom_pericentre.png", dpi=150)
    plt.close(fig)


def nudged_circular_run(r0: float, t_end: float):
    """Circular orbit at r0 given the same small inward push (radial velocity 0.1% of the
    orbital speed, angular momentum unchanged); integrated until t_end or r < 3.

    Not a 0.1% slow-down: just outside r = 6 that removes enough angular momentum to drop
    below the minimum any circular orbit has (L^2 = 12), and both particles plunge."""
    vt = (r0 - 2) / (r0 * np.sqrt(r0 - 3))
    x = np.array([r0, 0.0, 0.0])
    p = gn_p_from_v(x, np.array([-0.001 * vt, vt, 0.0]))
    ts, rs, t0 = [0.0], [r0], 0.0
    while t0 < t_end:
        try:
            seg = integrate_orbit(x, p, 20.0, eta=0.002, dt_max=1.0)
        except GLFConvergenceError:
            break
        ts.extend(t0 + seg.t[1:])
        rs.extend(np.linalg.norm(seg.x[1:], axis=1))
        x, p, t0 = seg.x[-1], seg.p[-1], t0 + 20.0
        if rs[-1] < 3.0:
            break
    return np.array(ts), np.array(rs)


def plot_isco() -> tuple[float, float]:
    t_end = 2500.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ends = []
    for r0, col, label in (
        (6.2, COL_GN, "starts at r = 6.2 (outside): stays in orbit"),
        (5.8, "#c92a2a", "starts at r = 5.8 (inside): plunges"),
    ):
        t, r = nudged_circular_run(r0, t_end)
        ax.plot(t, r, color=col, lw=1.6, label=label)
        ends.append(t[-1])
    ax.axhline(6, ls="--", color="#495057", lw=0.9)
    ax.text(t_end * 0.99, 5.92, "innermost stable orbit, r = 6", ha="right", va="top", fontsize=9)
    ax.axhline(2, color="black", lw=1.2)
    ax.text(t_end * 0.99, 2.08, "event horizon, r = 2", ha="right", va="bottom", fontsize=9)
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, 7)
    ax.set_xlabel(
        f"time  [GM/c³ = {BH.time_s:.2f} s]   (one orbit at r = 6 ≈ 98, about 35 minutes)"
    )
    ax.set_ylabel("distance from the hole  r  [GM/c²]")
    ax.set_title(
        "No stable orbits inside r = 6: two particles on circular orbits, each given the same\n"
        "tiny inward push (radial velocity 0.1% of the orbital speed)",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHECKS / "isco_demo.png", dpi=150)
    plt.close(fig)
    return ends[0], ends[1]


def write_timeline(t_peri: np.ndarray) -> None:
    rows = [("Start (apocentre, r = 5,000)", 0.0)]
    rows += [(f"Pericentre passage {k + 1} (simulation)", t) for k, t in enumerate(t_peri)]
    rows += [
        ("Radial period (§2)", RADIAL_PERIOD),
        ("Ring orbit at r_circ = 50.89 (§2)", 2303.257),
        ("Circular orbit at r = 6, the innermost stable orbit (§2)", 97.9452),
    ]
    lines = [f"{'Event':58s} {'code time':>14s}   {'physical':>12s}"]
    for name, t in rows:
        lines.append(f"{name:58s} {t:>14,.1f}   {units.format_time(BH.time_to_s(t)):>12s}")
    lines.append("")
    lines.append(f"Precession per orbit (exact): {np.degrees(precession_exact(RP, RA)):.5f} deg")
    (CHECKS / "timeline.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    CHECKS.mkdir(parents=True, exist_ok=True)
    x0, _, p0 = apocentre_state(RP, RA)
    glf = integrate_orbit(x0, p0, N_ORBITS * RADIAL_PERIOD, eta=ETA)
    t_peri, _, phi_peri = pericentres(glf.t, glf.x)
    ex, ey = exact_schwarzschild_path(N_ORBITS)
    nx, ny = newton_ellipse()
    plot_rosette(glf, ex, ey, nx, ny)
    plot_zoom(glf, ex, ey, t_peri, phi_peri)
    t_stay, t_plunge = plot_isco()
    write_timeline(t_peri)
    print(
        f"ISCO demo: r=6.2 run ended at t={t_stay:.0f}; "
        f"r=5.8 run ended (r < 3 or t_end) at t={t_plunge:.0f}"
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ("orbit_rosette.png", "isco_demo.png"):
        shutil.copy(CHECKS / name, FIGURES / name)
    print(f"Plots in {CHECKS}; copies in {FIGURES}")


if __name__ == "__main__":
    main()
