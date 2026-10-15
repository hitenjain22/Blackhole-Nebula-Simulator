"""The energy-ledger figure for the report (Phase 4, test 4): the hot gas blob on the
r_a = 500 -> r_p = 25 orbit, hydro kicks applied to the canonical momentum p (right) or to the
velocity v (wrong), at two step settings.

outputs/checks/energy_ledger.png   left: ledger error over the orbit; right: max error vs step
outputs/checks/energy_ledger.txt   the numbers

Runs are cached with the slow tests (sim/blob.py); ~30 minutes if none are cached yet.
Run: uv run python scripts/plot_ledger.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nebula.sim.blob import cached_hot_blob  # noqa: E402

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"
SETTINGS = ((0.0025, 0.3), (0.00125, 0.15))  # (eta_orb, Courant): coarse, fine


def main():
    CHECKS.mkdir(parents=True, exist_ok=True)
    runs = {
        (wrong, s): cached_hot_blob(*s, wrong=wrong) for wrong in (False, True) for s in SETTINGS
    }
    blocks = {s: cached_hot_blob(*s, block_steps=True) for s in SETTINGS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    style = {
        False: ("#1f5fbf", "kicks on momentum p (right)"),
        True: ("#c92a2a", "kicks on v (wrong)"),
    }
    for (wrong, s), r in runs.items():
        col, label = style[wrong]
        fine = s == SETTINGS[1]
        ax1.semilogy(
            r["t"], r["energy_error"] + 1e-14, color=col, lw=1.4 if fine else 0.8,
            ls="-" if fine else "--", label=f"{label}, {'fine' if fine else 'coarse'} steps",
        )  # fmt: skip
    ax1.axhline(1e-5, color="k", lw=0.8, ls=":", label="plan limit 1e-5")
    ax1.set_xlabel("time (GM/c^3); pericentre at ~13,500")
    ax1.set_ylabel("|change of total energy| / sum m|H|")
    ax1.set_title("Energy ledger of a gas blob from apocentre through pericentre")
    ax1.legend(fontsize=8, loc="lower right")
    x = [1, 2]
    for wrong in (False, True):
        col, label = style[wrong]
        ax2.semilogy(
            x,
            [runs[(wrong, s)]["max_energy_error"] for s in SETTINGS],
            "o-",
            color=col,
            label=label,
        )
    ax2.semilogy(
        x, [blocks[s]["max_energy_error"] for s in SETTINGS], "s--", color="#2b8a3e",
        label="kicks on p, block time steps",
    )  # fmt: skip
    ax2.set_xticks(x, ["coarse\n(eta 0.0025, C 0.3)", "fine\n(eta 0.00125, C 0.15)"])
    ax2.set_xlim(0.6, 2.4)
    ax2.set_ylabel("largest ledger error over the run")
    ax2.set_title("Halving the steps: right falls ~4x, wrong stays")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHECKS / "energy_ledger.png", dpi=130)

    lines = []
    for (wrong, s), r in runs.items():
        lines.append(
            f"{'wrong' if wrong else 'right'} coupling, eta {s[0]}, Courant {s[1]}: max ledger "
            f"error {r['max_energy_error']:.3e}, max |dL|/|L| {r['max_angmom_error']:.1e}, "
            f"{r['steps']:,} steps, largest u {r['u_max']:.2e}, lowest particle angular momentum "
            f"{r['min_particle_angmom']:.2f} (start 7.20)"
        )
    for s, r in blocks.items():
        lines.append(
            f"block steps, eta {s[0]}, Courant {s[1]}: max ledger error "
            f"{r['max_energy_error']:.3e}, max |dL|/|L| {r['max_angmom_error']:.1e}, "
            f"{r['steps']:,} particle-steps ({r['n']} particles)"
        )
    text = "\n".join(lines)
    print(text)
    (CHECKS / "energy_ledger.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
