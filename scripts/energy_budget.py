"""Energy budget of a run (phases/phase-06-shocks-cooling-ring.md 6.6, picture 6): where the
energy has gone by a given time.

    uv run python scripts/energy_budget.py RUN [t=...]

Total energy E = sum m (H + u) (H: specific orbital energy, GN) is conserved together with what
left: E_start = [bound gas: orbital + thermal] + [unbound gas] + [accreted] + [escaped]
+ [radiated] + (ledger error). Each term is printed as a change from the start where that is the
natural reading. The radiated energy is the run's ledger value (checkpoint) when the checkpoint
is at that time; otherwise the per-particle cooling budgets of the gas still present (gas
swallowed or escaped before then took its radiated share out of that sum: noted).

outputs/checks/<RUN>_energy_budget.png and .txt
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.sim import coupled  # noqa: E402
from nebula.sim.coupled import CoupledConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = sys.argv[1]
    opts = dict(a.split("=") for a in sys.argv[2:])
    folder = ROOT / "runs" / run
    cfg = json.loads((folder / "manifest.json").read_text())["config"]
    cc = CoupledConfig(gravity=cfg.get("gravity", "gn"))
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    ts = np.array([float(np.load(f)["t"]) for f in files])
    t_want = float(opts.get("t", ts[-1]))
    d = np.load(files[int(np.argmin(np.abs(ts - t_want)))])
    d0 = np.load(files[0])
    t = float(d["t"])
    m0 = np.full(len(d0["x"]), 1.0 / len(d0["x"]))
    h0 = coupled.hamiltonian(cc, d0["x"].astype(float), d0["p"].astype(float))
    e_start = float(np.sum(m0 * (h0 + d0["u"])))
    m = np.full(len(d["x"]), 1.0 / len(d0["x"]))
    H = coupled.hamiltonian(cc, d["x"].astype(float), d["p"].astype(float))
    u = d["u"].astype(float)
    bound = H + u < 0
    removed_path = folder / "removed.json"
    log = json.loads(removed_path.read_text()) if removed_path.exists() else []
    if not log and (folder / "checkpoint.pkl").exists():
        log = pickle.loads((folder / "checkpoint.pkl").read_bytes())["st"].removed.log
    log = [e for e in log if e["t"] <= t]

    def e_removed(reason):
        return sum(e["m"] * (e["H"] + e["u"]) for e in log if e["reason"] == reason)

    radiated, note = None, ""
    ck = folder / "checkpoint.pkl"
    if ck.exists():
        state = pickle.loads(ck.read_bytes())
        if abs(state["st"].t - t) < 1.0:
            radiated = float(state["st"].e_radiated)
    if radiated is None:
        radiated = (
            float(np.sum(m * d["heat_cool"].astype(float))) if "heat_cool" in d.files else 0.0
        )
        note = " (present gas only)"
    terms = {
        "bound gas: orbital energy": float(np.sum(m[bound] * H[bound])),
        "bound gas: heat": float(np.sum(m[bound] * u[bound])),
        "unbound gas (orbital + heat)": float(np.sum(m[~bound] * (H[~bound] + u[~bound]))),
        "swallowed by the hole": e_removed("accreted"),
        "escaped (r > r_escape)": e_removed("escaped"),
        f"radiated{note}": radiated,
    }
    total = sum(terms.values())
    heat_made = float(np.sum(m * d["heat_visc"].astype(float))) if "heat_visc" in d.files else 0.0
    lines = [f"{run}: energy budget at t = {t:,.0f} (code units, total mass 1)",
             f"start: E = {e_start:+.4e}", ""]  # fmt: skip
    for k, v in terms.items():
        lines.append(f"  {k:36s} {v:+.4e}")
    lines += ["", f"  sum                                  {total:+.4e}",
              f"  ledger error (sum - start)           {total - e_start:+.2e}",
              f"  heat made by shocks so far (present gas): {heat_made:.4e}"]  # fmt: skip
    text = "\n".join(lines)
    out = ROOT / "outputs" / "checks"
    (out / f"{run}_energy_budget.txt").write_text(text + "\n")
    print(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(terms) + ["start (for comparison)"]
    vals = list(terms.values()) + [e_start]
    ax.barh(names, vals, color=["#4c72b0"] * len(terms) + ["#999999"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("energy (code units, G = c = M = 1; total cloud mass 1)")
    ax.set_title(f"{run}: where the energy is at t = {t:,.0f} (they add up to the start value)")
    ax.invert_yaxis()
    fig.savefig(out / f"{run}_energy_budget.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
