"""Preview animation of a cloud run (phases/phase-05-cloud-infall.md 5.2: "a quick matplotlib
animation (MP4) of the whole phase: no lensing, just particles").

    uv run python scripts/cloud_video.py RUN_NAME [hold=6]

outputs/checks/<RUN>_preview.mp4  1280 x 720, 30 fps: every snapshot held for ``hold`` frames
(no interpolation: snapshots near the hole are a few hundred time units apart, and straight-line
interpolation would cut the corners of the curved paths, showing motion that never happened).
The view follows the gas (smoothed), particles coloured by temperature, the hole as a dot.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula import units  # noqa: E402
from nebula.analysis.cloud_metrics import snapshot_temperature  # noqa: E402
from nebula.render.video import write_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_eos(run: str) -> str:
    """The run's equation of state (manifest config; Phase 5 runs are ideal gas)."""
    cfg = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"]
    return cfg.get("eos", "ideal")


def frames(run: str, hold: int):
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    eos = run_eos(run)
    day = units.DAY / units.BlackHoleUnits().time_s  # code time units per day
    centre = half = None
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    for f in files:
        d = np.load(f)
        x, t = d["x"].astype(float), float(d["t"])
        logt = np.log10(snapshot_temperature(eos, d["rho"], d["u"]))
        lo, hi = np.percentile(x[:, :2], 1, axis=0), np.percentile(x[:, :2], 99, axis=0)
        c_new = 0.5 * (lo + hi)
        h_new = max(0.65 * float(np.max(hi - lo)), 15.0)
        if centre is None:
            centre, half = c_new, h_new
        else:  # smooth the camera so the view does not jump between snapshots
            centre = 0.5 * centre + 0.5 * c_new
            half = float(np.exp(0.5 * np.log(half) + 0.5 * np.log(h_new)))
        fig.clf()
        ax = fig.add_axes([0.06, 0.08, 0.72, 0.84])
        sc = ax.scatter(x[:, 0], x[:, 1], c=logt, s=0.6, cmap="inferno", vmin=4.0, vmax=8.0,
                        linewidths=0)  # fmt: skip
        ax.set_facecolor("#d0d4d8")
        if np.all(np.abs(centre) < 1.6 * half):
            ax.plot(0, 0, "o", color="#4dabf7", ms=6)
        ax.set_xlim(centre[0] - 16 / 9 * half, centre[0] + 16 / 9 * half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.set_title(f"{run}: t = {t:,.0f} GM/c³ ({t / day:.1f} days), {len(x):,} particles; "
                     "blue dot = the hole", fontsize=11)  # fmt: skip
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        fig.colorbar(sc, cax=cax, label="log10 temperature (K)")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        for _ in range(hold):
            yield img
    plt.close(fig)


def main():
    run = sys.argv[1]
    opts = dict(a.split("=") for a in sys.argv[2:])
    out = ROOT / "outputs" / "checks" / f"{run}_preview.mp4"
    n = write_video(frames(run, int(opts.get("hold", 6))), out)
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
