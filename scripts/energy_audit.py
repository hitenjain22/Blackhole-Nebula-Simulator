"""Energy audit of a run's snapshots (analysis/energy_audit.py), to locate the ledger drift.

    uv run python scripts/energy_audit.py RUN [RUN ...]

For each pair of consecutive snapshots: the heat, motion and removal channels (the last for gas
swallowed or escaped in between, from the run's removed.json). Their running sums, divided by
sum m |H| at the start (the ledger's own scale), are plotted against the run's ledger error
(diagnostics.csv), and the biggest intervals are listed. Writes
outputs/checks/energy_audit_<RUN>.png and .csv.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nebula import units  # noqa: E402
from nebula.analysis.energy_audit import energy_channels  # noqa: E402
from nebula.physics.gravity import gn_hamiltonian, newton_hamiltonian  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("t", "ids", "x", "p", "u", "h", "heat_visc", "heat_cond", "heat_pdv", "heat_disk",
          "heat_cool")  # fmt: skip


def load(f):
    d = np.load(f)
    return {k: d[k] for k in FIELDS if k in d.files}


def audit(run):
    folder = ROOT / "runs" / run
    cfg = json.loads((folder / "manifest.json").read_text())["config"]
    ham = gn_hamiltonian if cfg.get("gravity", "gn") == "gn" else newton_hamiltonian
    log = json.loads((folder / "removed.json").read_text())
    masses = {e["m"] for e in log}
    removed = {e["id"]: e["H"] + e["u"] for e in log}
    if len(masses) != 1:
        raise ValueError(f"expected equal particle masses, got {len(masses)} values")
    m = masses.pop()
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    a = load(files[0])
    scale = float(np.sum(m * np.abs(ham(a["x"].astype(float), a["p"].astype(float)))))
    rows, s_heat, s_motion, s_rem = [], 0.0, 0.0, 0.0
    for f in files[1:]:
        b = load(f)
        r = energy_channels(a, b, m, ham, removed)
        s_heat += r["r_heat"]
        s_motion += r["r_motion"]
        s_rem += r["r_removed"]
        capped = float(np.mean(b["h"] >= 0.999 * cfg.get("h_max", np.inf)))
        rows.append({"t0": float(a["t"]), "t1": float(b["t"]), **r, "cum_heat": s_heat / scale,
                     "cum_motion": s_motion / scale, "cum_removed": s_rem / scale,
                     "capped": capped})  # fmt: skip
        a = b
    return rows, scale


def main():
    day = units.DAY / units.BlackHoleUnits().time_s
    for run in sys.argv[1:]:
        rows, scale = audit(run)
        out = ROOT / "outputs" / "checks" / f"energy_audit_{run}"
        with open(f"{out}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        diag = list(csv.DictReader(open(ROOT / "runs" / run / "diagnostics.csv")))
        td = np.array([float(r["t"]) for r in diag]) / day
        ed = np.array([float(r["energy_err"]) for r in diag])
        t = np.array([r["t1"] for r in rows]) / day
        ch, cm = np.array([r["cum_heat"] for r in rows]), np.array([r["cum_motion"] for r in rows])
        cr = np.array([r["cum_removed"] for r in rows])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].semilogy(td, ed, "k", lw=2.5, alpha=0.4, label="run's own ledger error")
        ax[0].semilogy(t, np.abs(ch + cm + cr), "C3", lw=1, label="audit: three channels added")
        ax[0].set_title(f"{run}: does the audit reproduce the ledger?")
        ax[1].plot(t, ch, label="heat channel (heat not explained by its causes)")
        ax[1].plot(t, cm, label="motion channel (motion energy lost minus heat made)")
        ax[1].plot(t, cr, label="removal channel (swallowed/escaped gas, last moments)")
        ax[1].plot(t, ch + cm + cr, "k", lw=1.5, label="total = ledger drift")
        ax[1].set_title("Where the extra energy comes from (fraction of sum m|H|)")
        for a in ax:
            a.set_xlabel("days since start")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out}.png", dpi=130)
        print(f"== {run}: saved {out}.png")
        for d in (100, 200, 300, 330, 400, 450, 500, 550, 600, 800, 931):
            i = np.searchsorted(t, d)
            if i < len(t):
                print(f"  day {t[i]:5.0f}: heat {ch[i]:+.2e}  motion {cm[i]:+.2e}  removal "
                      f"{cr[i]:+.2e}  sum {ch[i] + cm[i] + cr[i]:+.2e}  ledger "
                      f"{np.interp(t[i], td, ed):.2e}  "
                      f"capped {100 * rows[i]['capped']:4.1f}%")  # fmt: skip
        tot = [r["r_heat"] + r["r_motion"] + r["r_removed"] for r in rows]
        big = [rows[i] for i in np.argsort(-np.abs(tot))[:6]]
        print("  biggest intervals:")
        for r in big:
            print(f"    days {r['t0'] / day:6.1f}-{r['t1'] / day:6.1f}: heat "
                  f"{r['r_heat'] / scale:+.2e}  motion {r['r_motion'] / scale:+.2e}  removal "
                  f"{r['r_removed'] / scale:+.2e} ({r['n_removed']} removed)")  # fmt: skip


if __name__ == "__main__":
    main()
