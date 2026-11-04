"""Our R6.1 v2 (50k) vs Phantom with our patch (ph_r2) over the whole run: face-on views at
several days (wide and zoomed on the hole, coloured by gas energy u) and the swallowed count
against time.

    uv run --with sarracen --with scikit-learn python scripts/plot_phantom_vs_ours.py PH_DIR

PH_DIR holds Phantom full dumps cloud_NNNNN (NNNNN = day) and run.out.
outputs/checks/phantom_vs_ours_full_run.png
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sarracen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 4079.27  # code time per day
DAYS = [200, 400, 600, 800, 928]
RUN = ROOT / "runs" / "R6.1_50k_v2"


def phantom(ph, day):
    s = sarracen.read_phantom(str(ph / f"cloud_{day:05d}"))
    s = s[0] if isinstance(s, tuple) else s
    return float(s.params["time"]), s[["x", "y", "z"]].values, s["u"].values


def ours(day, snaps, ts):
    o = np.load(snaps[int(np.argmin(abs(ts - day * DAY)))])
    return float(o["t"]), o["x"], o["u"]


def swallowed_phantom(ph):
    t, n, cur = [], [], None
    for line in open(ph / "run.out"):
        m = re.search(r"TIME =\s+([0-9.E+-]+)", line)
        if m:
            cur = float(m.group(1))
        m = re.search(r"n_dead_or_accreted=\s*(\d+)", line)
        if m and cur is not None:
            t.append(cur / DAY)
            n.append(int(m.group(1)))
    return np.array(t), np.array(n)


def main():
    ph = Path(sys.argv[1])
    snaps = sorted((RUN / "snapshots").glob("snap_*.npz"))
    ts = np.array([float(np.load(f)["t"]) for f in snaps])
    fig = plt.figure(figsize=(4.0 * len(DAYS), 17))
    gs = fig.add_gridspec(5, len(DAYS), height_ratios=[1, 1, 1, 1, 0.9])
    norm = matplotlib.colors.Normalize(-9, -2)
    for k, day in enumerate(DAYS):
        for row, (lab, (t, x, u)) in enumerate(
            (("our code", ours(day, snaps, ts)), ("Phantom + patch", phantom(ph, day)))
        ):
            for z, span in enumerate((6000.0, 400.0)):
                a = fig.add_subplot(gs[2 * z + row, k])
                sel = np.all(np.abs(x[:, :2]) < span, axis=1)
                sc = a.scatter(x[sel, 0], x[sel, 1], s=0.3 if z == 0 else 0.6,
                               c=np.log10(u[sel]), cmap="inferno", norm=norm, lw=0)  # fmt: skip
                a.plot(0, 0, "c*", ms=8)
                a.set_xlim(-span, span); a.set_ylim(-span, span); a.set_aspect("equal")  # fmt: skip
                a.set_facecolor("#ddd")
                a.set_title(f"{lab}, day {t / DAY:.0f}" + (" (zoom)" if z else "")
                            + f": {len(x):,} left", fontsize=8)  # fmt: skip
                a.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[4, :])
    rows = list(csv.DictReader(open(RUN / "diagnostics.csv")))
    ax.plot([float(r["t"]) / DAY for r in rows], [int(r["removed"]) for r in rows],
            label="our code (R6.1 v2)")  # fmt: skip
    tp, npd = swallowed_phantom(ph)
    ax.plot(tp, npd, label="Phantom + our patch (ph_r2)")
    ax.set_xlabel("day"); ax.set_ylabel("particles swallowed (of 50,015)")  # fmt: skip
    ax.legend(); ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=fig.axes[:-1], shrink=0.4, label="log10 gas energy u (c$^2$): brighter = hotter")
    fig.suptitle("Whole run, 50k particles: our code (rows 1, 3) vs Phantom with our patch "
                 "(rows 2, 4); rows 3-4 zoom on the hole (cyan star)")  # fmt: skip
    out = ROOT / "outputs" / "checks" / "phantom_vs_ours_full_run.png"
    fig.savefig(out, dpi=70)
    print(out)


if __name__ == "__main__":
    main()
