"""The stream crash (phases/phase-06-shocks-cooling-ring.md 6.6, picture 1): zoom on the region
where the stream crosses itself, three moments, particles coloured by temperature, shock-heated
gas outlined.

    uv run python scripts/plot_crash.py RUN T1 T2 T3 [half=1500]

The view is centred on the median position of the shock-heated gas between r = 800 and 2,500
(the crossing region; r_SI = 1,664) at the middle time, and kept fixed for all three panels.
"Shock-heated": viscous heating rate above Omega_K u (DECISIONS: the Phase 6 shock marker).

outputs/checks/<RUN>_crash.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.analysis.cloud_metrics import snapshot_temperature  # noqa: E402
from nebula.physics.cooling import omega_kepler  # noqa: E402
from nebula.units import DAY, BlackHoleUnits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R_SI = 1663.76  # expected-values §2b


def snapshot_near(run: str, t: float):
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    ts = np.array([float(np.load(f)["t"]) for f in files])
    return np.load(files[int(np.argmin(np.abs(ts - t)))])


def shocked(d) -> np.ndarray:
    x, u = d["x"].astype(float), d["u"].astype(float)
    r = np.linalg.norm(x, axis=1)
    return d["visc_heating"].astype(float) > omega_kepler(np.maximum(r, 4.0)) * u


def main():
    run = sys.argv[1]
    times = [float(a) for a in sys.argv[2:5]]
    opts = dict(a.split("=") for a in sys.argv[5:])
    half = float(opts.get("half", 1500))
    eos = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"].get(
        "eos", "ideal"
    )
    snaps = [snapshot_near(run, t) for t in times]
    mid = snaps[1]
    x = mid["x"].astype(float)
    r = np.linalg.norm(x, axis=1)
    sel = shocked(mid) & (r > 800) & (r < 2500)
    centre = np.median(x[sel, :2], axis=0) if sel.sum() > 20 else np.array([R_SI, 0.0])
    day = DAY / BlackHoleUnits().time_s
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    t_top = 6.5 if eos == "gasrad" else 8.0
    for ax, d, label in zip(axes, snaps, ("approach", "impact", "splash"), strict=True):
        x = d["x"].astype(float)
        logt = np.log10(snapshot_temperature(eos, d["rho"], d["u"]))
        box = np.all(np.abs(x[:, :2] - centre) < half * 1.05, axis=1)
        sc = ax.scatter(x[box, 0], x[box, 1], c=logt[box], s=0.8, cmap="inferno", vmin=4.0,
                        vmax=t_top, linewidths=0)  # fmt: skip
        s = shocked(d) & box
        ax.scatter(x[s, 0], x[s, 1], s=3, facecolors="none", edgecolors="#00d0ff", linewidths=0.4)
        ax.add_patch(plt.Circle((0, 0), R_SI, fill=False, ls="--", color="g", lw=0.8))
        ax.plot(0, 0, "o", color="#4dabf7", ms=6)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.set_facecolor("#d0d4d8")
        t = float(d["t"])
        ax.set_title(
            f"{label}: t = {t:,.0f} ({t / day:.0f} days)\nshock-heated particles: {s.sum():,}",
            fontsize=11,
        )
    fig.colorbar(sc, ax=axes, label="log10 temperature (K)", shrink=0.8)
    fig.suptitle(f"{run}: the stream crashing into itself\ncyan rings: gas being shock-heated; "
                 "dashed green: r = 1,664, where the stream crosses itself; blue dot: the hole",
                 fontsize=12)  # fmt: skip
    out = ROOT / "outputs" / "checks" / f"{run}_crash.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
