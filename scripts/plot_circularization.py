"""Circularization pictures (phases/phase-06-shocks-cooling-ring.md 6.6, pictures 2 and 3):
face-on views at chosen times (e.g. passage 1, the crash, torus forming, settled torus), and an
edge-on view of the gas near the hole at the last of them (shows the torus thickness).

    uv run python scripts/plot_circularization.py RUN T1 T2 T3 T4 [half=400]

Particles coloured by orbital eccentricity (blue: circular, red: very eccentric); unbound gas
grey. Titles give the ring share (bound mass with e < 0.3 and r < 200, the plan's torus measure).

outputs/checks/<RUN>_circularization.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.analysis.torus_metrics import orbit_elements, ring_measures  # noqa: E402
from nebula.units import DAY, BlackHoleUnits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def snapshot_near(run: str, t: float):
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    ts = np.array([float(np.load(f)["t"]) for f in files])
    return np.load(files[int(np.argmin(np.abs(ts - t)))])


def main():
    run = sys.argv[1]
    times = [float(a) for a in sys.argv[2:6]]
    opts = dict(a.split("=") for a in sys.argv[6:])
    half = float(opts.get("half", 400))
    day = DAY / BlackHoleUnits().time_s
    fig = plt.figure(figsize=(20, 9.5))
    axes = [fig.add_subplot(2, 4, k + 1) for k in range(4)]
    for ax, t_want in zip(axes, times, strict=True):
        d = snapshot_near(run, t_want)
        x, p = d["x"].astype(float), d["p"].astype(float)
        el = orbit_elements(x, p)
        rm = ring_measures(x, np.ones(len(x)), el)
        bound = el.kind == 0
        ax.scatter(x[~bound, 0], x[~bound, 1], s=0.3, c="#999999", linewidths=0)
        sc = ax.scatter(x[bound, 0], x[bound, 1], s=0.5, c=el.ecc[bound], cmap="coolwarm",
                        vmin=0, vmax=1, linewidths=0)  # fmt: skip
        ax.plot(0, 0, "o", color="k", ms=4)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_facecolor("#eef0f2")
        t = float(d["t"])
        ax.set_title(f"t = {t:,.0f} ({t / day:.0f} days)\nring share {rm.mass_fraction:.1%}",
                     fontsize=10)  # fmt: skip
    fig.colorbar(sc, ax=axes, label="orbital eccentricity (0 = circular)", shrink=0.8)
    d = snapshot_near(run, times[-1])
    x = d["x"].astype(float)
    near = np.linalg.norm(x, axis=1) < half
    ax = fig.add_subplot(2, 1, 2)
    el = orbit_elements(x, d["p"].astype(float))
    ax.scatter(x[near, 0], x[near, 2], s=0.6, c=el.ecc[near], cmap="coolwarm", vmin=0, vmax=1,
               linewidths=0)  # fmt: skip
    ax.plot(0, 0, "o", color="k", ms=4)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half / 3, half / 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#eef0f2")
    ax.set_xlabel("x")
    ax.set_ylabel("z (height above the orbital plane)")
    ax.set_title(f"edge-on at t = {float(d['t']):,.0f}: gas within r = {half:.0f} of the hole")
    fig.suptitle(f"{run}: circularization (face-on, coloured by eccentricity; grey: unbound)",
                 fontsize=13)  # fmt: skip
    out = ROOT / "outputs" / "checks" / f"{run}_circularization.png"
    fig.savefig(out, dpi=95, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
