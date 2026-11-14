"""Phase 8 step 1 check: deposit a real snapshot onto the log-spherical render grid and draw
slices through it next to the raw particles (phases/phase-08-gas-rendering.md 8.3).

    uv run python scripts/gasgrid_slices.py RUN SNAP [r_max=2000] [view=600]

Writes outputs/checks/phase8_step1_gasgrid_RUN_SNAP.png and prints the mass ledger, timing and
memory. Particle mass: the cloud is 1 code mass unit split equally (sim/cloud.py), so
m = 1 / N(snapshot 0).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.gasgrid import LogSphericalGrid, deposit_log

ROOT = Path(__file__).resolve().parents[1]
DAY = 4080.0  # code time units per day (M = 4.3e6 Msun)


def main():
    run, snap = sys.argv[1], int(sys.argv[2])
    opts = dict(a.split("=") for a in sys.argv[3:])
    r_max, view = float(opts.get("r_max", 2000)), float(opts.get("view", 600))
    folder = ROOT / "runs" / run
    eos = json.loads((folder / "manifest.json").read_text())["config"].get("eos", "gasrad")
    n0 = len(np.load(folder / "snapshots" / "snap_00000.npz")["h"])
    d = np.load(folder / "snapshots" / f"snap_{snap:05d}.npz")
    x, h = d["x"].astype(float), d["h"].astype(float)
    rho, u = d["rho"].astype(float), d["u"].astype(float)
    t = snapshot_temperature(eos, rho, u)
    m = np.full(len(h), 1.0 / n0)
    v = d["p"].astype(float)  # momentum per mass; velocity proper comes in Phase 9
    g = LogSphericalGrid(r_max=r_max)

    deposit_log(x[:50], h[:50], m[:50], t[:50], v[:50], g)  # compile once
    t0 = time.perf_counter()
    gg = deposit_log(x, h, m, t, v, g)
    dt = time.perf_counter() - t0
    print(
        f"{run} snap {snap} (day {float(d['t']) / DAY:.1f}): {len(h):,} particles, "
        f"grid {g.nr}x{g.nth}x{g.nph}, deposition {dt:.2f} s"
    )
    print(
        f"  mass: grid {gg.mass.sum():.12g} + beyond grid {gg.outside:.6g} = "
        f"{gg.mass.sum() + gg.outside:.12g}; particles {m.sum():.12g}; "
        f"relative difference {(gg.mass.sum() + gg.outside) / m.sum() - 1:.1e}"
    )
    print(
        f"  trilinear fallback: {gg.n_trilinear:,} particles; float32 fields "
        f"{(gg.rho.nbytes + gg.t_mass.nbytes + gg.t_em.nbytes + gg.vel.nbytes) / 1e6:.0f} MB"
    )

    re = g.r_min * np.exp(np.arange(g.nr + 1) * g.dlnr)
    pe = np.arange(g.nph + 1) * g.dph
    mid = g.nth // 2  # cells nth/2-1 and nth/2 touch the equator: average them
    eq_rho = 0.5 * (gg.rho[:, mid - 1, :] + gg.rho[:, mid, :])
    eq_t = 0.5 * (gg.t_em[:, mid - 1, :] + gg.t_em[:, mid, :])
    R, P = np.meshgrid(re, pe, indexing="ij")
    X, Y = R * np.cos(P), R * np.sin(P)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.3), facecolor="k")
    sel = np.abs(x[:, 2]) < 0.5 * view * np.pi / g.nth * 4  # thin slab around the plane
    ax[0].scatter(
        x[sel, 0], x[sel, 1], s=0.4, c=np.log10(t[sel]), cmap="inferno", vmin=3.8, vmax=6.5, lw=0
    )
    ax[0].set_title("particles near the orbital plane (colour = log T)", color="w")
    with np.errstate(divide="ignore"):
        lr = np.log10(np.where(eq_rho > 0, eq_rho, np.nan))
        lt = np.log10(np.where(eq_t > 0, eq_t, np.nan))
    pos = lr[np.isfinite(lr)]
    ax[1].pcolormesh(X, Y, lr, cmap="magma", vmin=np.percentile(pos, 5), vmax=pos.max())
    ax[1].set_title("grid: density in the plane (log)", color="w")
    ax[2].pcolormesh(X, Y, lt, cmap="inferno", vmin=3.8, vmax=6.5)
    ax[2].set_title("grid: emission-weighted temperature in the plane (log K)", color="w")
    for a in ax:
        a.set_facecolor("k")
        a.set_xlim(-view, view)
        a.set_ylim(-view, view)
        a.set_aspect("equal")
        a.tick_params(colors="w")
        a.add_patch(plt.Circle((0, 0), 2.0, color="c", fill=False, lw=0.6))
    fig.suptitle(
        f"{run}, day {float(d['t']) / DAY:.1f}: particles vs render grid "
        f"({g.nr}x{g.nth}x{g.nph}, r {g.r_min:g}-{g.r_max:g}); mass ledger exact to "
        f"{abs((gg.mass.sum() + gg.outside) / m.sum() - 1):.0e}",
        color="w",
    )
    out = ROOT / "outputs" / "checks" / f"phase8_step1_gasgrid_{run}_{snap:05d}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=80, facecolor="k")
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
