"""Pictures of a cloud run for Hiten (phases/phase-05-cloud-infall.md 5.2).

    uv run python scripts/plot_cloud_run.py RUN_NAME

outputs/checks/<RUN>_panels.png   six x-y views coloured by temperature: start, halfway in,
                                  approaching (centre of mass at r ~ 500), at pericentre, just
                                  after, the last snapshot
outputs/checks/<RUN>_edge_on.png  the x-z view at pericentre (the vertical squeeze)
Each panel is framed on the gas (1st-99th percentile of positions); the black dot is the hole.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.analysis.cloud_metrics import snapshot_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_eos(run: str) -> str:
    """The run's equation of state (manifest config; Phase 5 runs are ideal gas)."""
    cfg = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"]
    return cfg.get("eos", "ideal")


def load(run):
    snaps = []
    for f in sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz")):
        d = np.load(f)
        snaps.append({k: d[k] for k in d.files})
    return snaps


def pick(snaps):
    t = np.array([float(s["t"]) for s in snaps])
    com_r = np.array([np.linalg.norm(np.mean(s["x"], axis=0)) for s in snaps])
    i_peri = int(np.argmin(com_r))
    i_after = min(len(snaps) - 2, i_peri + max(1, (len(snaps) - 1 - i_peri) // 2))
    before = np.arange(i_peri)
    i_500 = int(before[np.argmin(np.abs(com_r[before] - 500))]) if len(before) else 0
    return [
        (0, "start"),
        (int(np.argmin(np.abs(t - t[i_peri] / 2))), "halfway in"),
        (i_500, "approaching (centre at r ≈ 500)"),
        (i_peri, "at pericentre"),
        (i_after, "just after"),
        (len(snaps) - 1, "last snapshot"),
    ]


def frame(ax, x2, y2, c, title, show_hole=True):
    lo = np.percentile(np.column_stack([x2, y2]), 1, axis=0)
    hi = np.percentile(np.column_stack([x2, y2]), 99, axis=0)
    centre, half = (lo + hi) / 2, max(np.max(hi - lo) / 2 * 1.3, 5.0)
    sc = ax.scatter(x2, y2, c=c, s=1.5, cmap="inferno", vmin=4.0, vmax=8.0, linewidths=0)
    if show_hole and np.all(np.abs(centre) < half + 50):
        ax.plot(0, 0, "o", color="#4dabf7", ms=5)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_facecolor("#d0d4d8")
    return sc


def main():
    run = sys.argv[1]
    eos = run_eos(run)
    snaps = load(run)
    out = ROOT / "outputs" / "checks"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, (i, label) in zip(axes.ravel(), pick(snaps), strict=True):
        s = snaps[i]
        logt = np.log10(snapshot_temperature(eos, s["rho"], s["u"]))
        r_com = np.linalg.norm(np.mean(s["x"], axis=0))
        sc = frame(ax, s["x"][:, 0], s["x"][:, 1], logt,
                   f"{label}\nt = {float(s['t']):,.0f}, centre at r = {r_com:,.0f}")  # fmt: skip
    fig.colorbar(sc, ax=axes, label="log10 temperature (K)", shrink=0.8)
    fig.suptitle(f"{run}: the cloud seen from above (x-y), coloured by temperature; "
                 "blue dot = the hole", fontsize=12)  # fmt: skip
    fig.savefig(out / f"{run}_panels.png", dpi=120)

    i_peri = pick(snaps)[3][0]
    s = snaps[i_peri]
    logt = np.log10(snapshot_temperature(eos, s["rho"], s["u"]))
    fig, ax = plt.subplots(figsize=(12, 5))
    near = np.linalg.norm(s["x"], axis=1) < 600
    sc = ax.scatter(s["x"][near, 0], s["x"][near, 2], c=logt[near], s=1.5, cmap="inferno",
                    vmin=4.0, vmax=8.0, linewidths=0)  # fmt: skip
    ax.plot(0, 0, "o", color="#4dabf7", ms=5)
    ax.set_aspect("equal")
    ax.set_facecolor("#d0d4d8")
    ax.set_xlabel("x")
    ax.set_ylabel("z (height above the orbital plane)")
    ax.set_title(f"{run} at pericentre (t = {float(s['t']):,.0f}), seen edge-on (x-z), gas "
                 "within r < 600: the vertical squeeze", fontsize=11)  # fmt: skip
    fig.colorbar(sc, ax=ax, label="log10 temperature (K)")
    fig.tight_layout()
    fig.savefig(out / f"{run}_edge_on.png", dpi=120)
    print("saved", out / f"{run}_panels.png", out / f"{run}_edge_on.png")


if __name__ == "__main__":
    main()
