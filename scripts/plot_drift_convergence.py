"""Time-step convergence of the R6.1 v2 ledger drift (HANDOFF 2026-11-03): one-day restarts from
day 416 (scripts/energy_steps.py) with the orbit criterion eta_orb at 1, 1/2 and 1/4 of the
run's, plus the dt_max = 512 and cfl = 0.5 restarts. Left: ledger change after one day vs the
orbit-step factor, with a slope-2 line. Right: the energy change per distance bin
(open + flow + close, summed over all particle steps), full steps minus quarter steps.

    uv run python scripts/plot_drift_convergence.py

outputs/checks/drift_convergence_orbit_step.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORB = {"es416": 1.0, "es416_orb05": 0.5, "es416_orb025": 0.25}
OTHER = {"es416_dt512": "dt_max 512", "es416_cfl05": "Courant/force x 0.5"}
LABELS = ["<10", "10-30", "30-100", "100-300", "300-1000", ">1000"]


def last_sync(run):
    rows = list(csv.DictReader(open(ROOT / "runs" / run / "energy_steps.csv")))
    return [r for r in rows if r["synced"] == "True"][-1]


def by_bin(row):
    return np.array([sum(float(row[f"{k}_{i}"]) for k in ("open", "flow", "close"))
                     for i in range(6)])  # fmt: skip


def main():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8))
    f = np.array(list(ORB.values()))
    led = np.array([float(last_sync(r)["ledger"]) for r in ORB])
    ax.loglog(f, led, "o-", label="orbit step factor (eta_orb x f)")
    ax.loglog(f, led[0] * f**2, "k--", lw=0.8, label="slope 2 (error ~ step$^2$)")
    for run, name in OTHER.items():
        ax.plot(1.0, float(last_sync(run)["ledger"]), "s", ms=5, label=name + " (f = 1)")
    for x, y in zip(f, led, strict=True):
        ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points", xytext=(6, -12))
    ax.set_xlabel("orbit step factor f")
    ax.set_ylabel("ledger change after 1 day / sum m|H| (day 416)")
    ax.set_title("Drift shrinks as the orbit step squared")
    ax.legend(fontsize=8)
    d = by_bin(last_sync("es416")) - by_bin(last_sync("es416_orb025"))
    bx.bar(LABELS, d, color=["C0"] * 5 + ["C3"])
    bx.axhline(0, color="k", lw=0.6)
    bx.set_xlabel("distance from the hole at step end (code units; r_acc = 4)")
    bx.set_ylabel("energy change: full steps - quarter steps")
    bx.set_title("All of the difference is in gas beyond r = 1000")
    fig.tight_layout()
    out = ROOT / "outputs" / "checks" / "drift_convergence_orbit_step.png"
    fig.savefig(out, dpi=120)
    print(out, "ledger", led, "bins", d)


if __name__ == "__main__":
    main()
