"""Resolution check of the numerical gas loss near the hole (DECISIONS, D4 update 2026-10-16).

    uv run python scripts/loss_check.py LOW_RUN HIGH_RUN [t=404000]

"Lost" = swallowed (removal log) + still present but below the capture line L < 4 (it will be
swallowed on a later pass), as a share of the starting particles. The check: at time t (default:
the last time both runs reached), the higher-resolution run loses a smaller share.

outputs/checks/loss_<LOW>_vs_<HIGH>.png   lost share against time for both runs
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.physics.orbits import L_CAPTURE_PARABOLIC  # noqa: E402
from nebula.sim import coupled  # noqa: E402
from nebula.sim.coupled import CoupledConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def removal_log(folder: Path) -> list:
    """removed.json of a finished run, else the log inside the latest checkpoint."""
    if (folder / "removed.json").exists():
        log = json.loads((folder / "removed.json").read_text())
        ckpt = folder / "checkpoint.pkl"
        if not ckpt.exists() or ckpt.stat().st_mtime <= (folder / "removed.json").stat().st_mtime:
            return log
    return pickle.loads((folder / "checkpoint.pkl").read_bytes())["st"].removed.log


def lost_history(run: str):
    folder = ROOT / "runs" / run
    cfg = json.loads((folder / "manifest.json").read_text())["config"]
    cc = CoupledConfig(gravity=cfg.get("gravity", "gn"))
    acc_t = np.sort([e["t"] for e in removal_log(folder) if e["reason"] == "accreted"])
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    n0 = len(np.load(files[0])["x"])
    ts, lost, below = [], [], []
    for f in files:
        d = np.load(f)
        t = float(d["t"])
        ell = np.linalg.norm(coupled.angmom(cc, d["x"].astype(float), d["p"].astype(float)), axis=1)
        b = int(np.sum(ell < L_CAPTURE_PARABOLIC))
        swallowed = int(np.searchsorted(acc_t, t, side="right"))
        ts.append(t)
        below.append(b)
        lost.append((b + swallowed) / n0)
    return np.array(ts), np.array(lost), np.array(below), n0


def at(ts, vals, t):
    """Value at the last snapshot at or before t."""
    return vals[np.searchsorted(ts, t, side="right") - 1]


def main():
    low, high = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=") for a in sys.argv[3:])
    hist = {r: lost_history(r) for r in (low, high)}
    t_end = float(opts.get("t", min(h[0][-1] for h in hist.values())))
    print(f"lost share (swallowed + below L < {L_CAPTURE_PARABOLIC:g}) at t = {t_end:,.0f}")
    end, peak = {}, {}
    for r, (ts, lost, below, n0) in hist.items():
        end[r] = at(ts, lost, t_end)
        peak[r] = lost[ts <= t_end].max()
        print(
            f"  {r:28s} {n0:6d} particles: lost at t {100 * end[r]:.3f}% (below the line "
            f"{at(ts, below, t_end)}); peak up to t {100 * peak[r]:.3f}%"
        )
    # below the line is not final (pressure can lift L back above 4), so with a handful of
    # particles one moment is noisy: the higher resolution must win on the peak and at t
    for name, v in (("peak share up to t", peak), ("share at t", end)):
        ok = v[high] < v[low] or v[high] == v[low] == 0
        print(f"check ({name}): {high} {100 * v[high]:.3f}% < {low} {100 * v[low]:.3f}%: "
              f"{'PASS' if ok else 'FAIL'}")  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 5))
    for r, (ts, lost, _, n0) in hist.items():
        ax.plot(ts, 100 * lost, ".-", label=f"{r} ({n0:,} particles)")
    ax.axvline(t_end, color="grey", ls=":", label="compared here")
    ax.set_xlabel("time (GM/c³)")
    ax.set_ylabel("gas lost (% of the cloud)")
    ax.set_title("Gas swallowed or bound to be swallowed (L < 4): should shrink with resolution")
    first = [h[0][np.argmax(h[1] > 0)] for h in hist.values() if h[1].max() > 0]
    if first:  # zoom on the pass where the loss happens
        ax.set_xlim(left=0.99 * min(first))
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / "outputs" / "checks" / f"loss_{low}_vs_{high}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
