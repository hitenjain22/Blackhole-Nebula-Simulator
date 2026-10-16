"""Phase 5 diagnostics of a finished (or running) cloud run (phases/phase-05-cloud-infall.md 5.2).

    uv run python scripts/cloud_metrics.py RUN_NAME

outputs/checks/<RUN>_metrics.txt  per snapshot: energy spread, centre-of-mass offset from the
                                  point-mass orbit, stream length/width/thickness, maximum
                                  density and temperature, mass above 10^5 and 10^6 K
outputs/checks/<RUN>_metrics.png  dM/d(eps) at the start, just before and after pericentre; the
                                  same quantities against time
Reads the float32 snapshots; particles all have mass 1/N (N at the start).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.analysis.cloud_metrics import (  # noqa: E402
    centre_of_mass_offset,
    energy_distribution,
    principal_axes,
    thermal_summary,
    thickness,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = sys.argv[1]
    eos = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"].get(
        "eos", "ideal"
    )
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    snaps = [{k: v for k, v in np.load(f).items()} for f in files]
    n0 = len(snaps[0]["x"])
    rows = []
    for s in snaps:
        x, p = s["x"].astype(float), s["p"].astype(float)
        m = np.full(len(x), 1.0 / n0)
        e = energy_distribution(x, p, m)
        ax = principal_axes(x, m)
        th = thermal_summary(s["u"].astype(float), s["rho"].astype(float), m, eos)
        com = np.sum(m[:, None] * x, axis=0) / np.sum(m)
        rows.append({"t": float(s["t"]), "n": len(x), "eps_mean": e.mean, "unbound": e.unbound,
                     "eps_p05": e.p05, "eps_p25": e.p25, "eps_p75": e.p75, "eps_p95": e.p95,
                     "length": ax[0], "thickness": thickness(x, m),
                     "rho_max": th.rho_max, "t_max": th.t_max, "f_1e5": th.mass_above_1e5,
                     "f_1e6": th.mass_above_1e6, "com": com, "r_com": float(np.linalg.norm(com)),
                     "eps": e.eps})  # fmt: skip
    first = snaps[0]
    offsets = centre_of_mass_offset(
        first["x"].astype(float), first["p"].astype(float), np.full(n0, 1.0 / n0),
        [r["t"] for r in rows], [r["com"] for r in rows], t0=rows[0]["t"],
    )  # fmt: skip
    i_peri = int(np.argmin([r["r_com"] for r in rows]))
    lines = [f"{run}: {n0} particles; centre of mass closest to the hole at t = "
             f"{rows[i_peri]['t']:,.0f} (r = {rows[i_peri]['r_com']:.0f})", "",
             "eps spreads: half-width of the middle 50% and 90% of the gas, in % of |eps_mean|",
             "t  N  eps_mean  spread50%  spread90%  unbound  com_offset  length  thickness(rms z)"
             "  rho_max  T_max(K)  f(>1e5K)  f(>1e6K)"]  # fmt: skip
    for r, off in zip(rows, offsets, strict=True):
        em = abs(r["eps_mean"])
        lines.append(
            f"{r['t']:9.0f} {r['n']:6d} {r['eps_mean']:.6e} "
            f"{100 * (r['eps_p75'] - r['eps_p25']) / 2 / em:8.3f} "
            f"{100 * (r['eps_p95'] - r['eps_p05']) / 2 / em:8.3f} {r['unbound']:6.3f} {off:9.3e} "
            f"{r['length']:8.2f} {r['thickness']:7.3f} {r['rho_max']:.3e} {r['t_max']:.3e} "
            f"{r['f_1e5']:.3f} {r['f_1e6']:.3f}"
        )
    text = "\n".join(lines)
    out = ROOT / "outputs" / "checks"
    (out / f"{run}_metrics.txt").write_text(text + "\n")
    print(text)

    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    before = np.arange(i_peri)
    i_app = int(before[np.argmin(np.abs(np.array([rows[i]["r_com"] for i in before]) - 500))])
    picks = [(0, "start", "#2b8a3e"), (i_app, "approaching (centre at r ≈ 500)", "#1f5fbf"),
             (len(rows) - 1, "last snapshot (after pericentre)", "#c92a2a")]  # fmt: skip
    allp = np.concatenate([rows[i]["eps"] for i, _, _ in picks])
    lo, hi = np.percentile(allp, [0.5, 99.5])
    edges = np.linspace(lo, hi, 120)
    from nebula.analysis.cloud_metrics import EnergyDistribution

    for i, label, col in picks:
        d = EnergyDistribution(rows[i]["eps"], 0, 0, 0, 0, 0)
        c, dm = d.histogram(np.full(len(rows[i]["eps"]), 1.0 / n0), edges=edges)
        axes[0, 0].semilogy(c * 1e4, dm / 1e4 + 1e-6, color=col,
                            label=f"{label}, t = {rows[i]['t']:,.0f}")  # fmt: skip
    axes[0, 0].set_xlabel("specific orbital energy eps (10⁻⁴ c²)")
    axes[0, 0].set_ylabel("dM/d(eps) (cloud mass per 10⁻⁴ c²)")
    axes[0, 0].set_title("Energy distribution of the gas (middle 99%)")
    axes[0, 0].set_ylim(1e-3, None)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].semilogy(t, offsets + 1e-12, "o-", ms=3)
    axes[0, 1].set_title("Centre of mass vs the point-mass orbit (distance)")
    for key, col, lab in (
        ("length", "#c92a2a", "length (rms, longest axis)"),
        ("thickness", "#2b8a3e", "thickness (rms height above the plane)"),
    ):
        axes[1, 0].semilogy(t, [r[key] for r in rows], "o-", ms=3, color=col, label=lab)
    axes[1, 0].set_title("Stream shape")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].semilogy(t, [r["t_max"] for r in rows], "o-", ms=3, color="#e8590c",
                        label="hottest particle (K)")  # fmt: skip
    ax2 = axes[1, 1].twinx()
    ax2.plot(t, [r["f_1e5"] for r in rows], "--", color="#495057", label="mass > 10⁵ K")
    ax2.plot(t, [r["f_1e6"] for r in rows], ":", color="#495057", label="mass > 10⁶ K")
    ax2.set_ylabel("mass fraction")
    axes[1, 1].set_title("Temperature")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="center left", fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("time (GM/c³)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{run}: Phase 5 diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{run}_metrics.png", dpi=120)


if __name__ == "__main__":
    main()
