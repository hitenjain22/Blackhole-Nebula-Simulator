"""Fast vs slow cooling at matching days: top and edge-on density maps of the gas near the hole
plus numbers (thickness, round orbits, temperature). Run on the VM from ~/nebula:

    uv run --no-sync --with sarracen python ~/cool_compare.py OUT.png DUMP [DUMP ...]

Reads Phantom dumps main100k_DUMP from R6.1_100k_ph_b (beta 230) and R6.6_100k_cool_b1 (beta 1).
Analysis only (picture for Hiten); the numbers use our units and EOS."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sarracen

from nebula import units
from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.physics.gravity import gn_hamiltonian, gn_p_from_v
from nebula.sim.coupled import CoupledConfig

RUNS = {
    "slow cooling (beta 230)": Path.home() / "runs_ph/R6.1_100k_ph_b",
    "fast cooling (beta 1)": Path.home() / "runs_ph/R6.6_100k_cool_b1",
}
CODE_DAY = 4079.27
R_VIEW = 400.0


def load(path):
    s = sarracen.read_phantom(str(path))
    s = s[0] if isinstance(s, tuple) else s
    keep = s["h"].values > 0
    m_ph = float(s.params["massoftype"])
    gu = CoupledConfig().hydro.gas_units
    m_ours = m_ph * units.BlackHoleUnits().mass_kg / gu.mass_kg_per_code
    x = s[["x", "y", "z"]].values[keep].astype(float)
    v = s[["vx", "vy", "vz"]].values[keep].astype(float)
    rho = s["rho"].values[keep].astype(float) * m_ours / m_ph
    t = snapshot_temperature("gasrad", rho, s["u"].values[keep].astype(float))
    return float(s.params["time"]), x, gn_p_from_v(x, v), t


def metrics(x, p, t):
    r = np.linalg.norm(x, axis=1)
    near = r < R_VIEW
    h = gn_hamiltonian(x, p)
    bound = h < 0
    sel = near & bound
    L = np.cross(x[sel], p[sel]).sum(axis=0)
    nz = L / np.linalg.norm(L)
    z = x[sel] @ nz
    R = np.sqrt(np.maximum(r[sel] ** 2 - z**2, 1e-30))
    # Newtonian-like eccentricity from E and L (GN energy ~ (E^2-1)/2); enough to rank
    l2 = np.sum(np.cross(x[sel], p[sel]) ** 2, axis=1)
    e = np.sqrt(np.clip(1 + 2 * h[sel] * l2, 0, None))
    return {
        "near": int(near.sum()),
        "bound_near": int(sel.sum()),
        "thick |z|/R (median)": float(np.median(np.abs(z) / R)) if sel.any() else np.nan,
        "round orbits e<0.3": int((e < 0.3).sum()),
        "T median near (K)": float(np.median(t[near])) if near.any() else np.nan,
    }, nz


def main():
    out = sys.argv[1]
    dumps = [int(d) for d in sys.argv[2:]]
    fig, axes = plt.subplots(
        2 * len(dumps), 2, figsize=(9, 4.6 * 2 * len(dumps)), facecolor="black"
    )
    for row, d in enumerate(dumps):
        for col, (name, run) in enumerate(RUNS.items()):
            f = run / f"main100k_{d:05d}"
            ax_top, ax_side = axes[2 * row, col], axes[2 * row + 1, col]
            for ax in (ax_top, ax_side):
                ax.set_facecolor("black")
                ax.set_xticks([])
                ax.set_yticks([])
            if not f.exists():
                ax_top.set_title(f"{name}: dump {d} not yet written", color="white", fontsize=9)
                continue
            tt, x, p, t = load(f)
            m, nz = metrics(x, p, t)
            ex = np.cross(nz, [0, 0, 1.0] if abs(nz[2]) < 0.9 else [1.0, 0, 0])
            ex /= np.linalg.norm(ex)
            ey = np.cross(nz, ex)
            a, b, c = x @ ex, x @ ey, x @ nz
            rng = [[-R_VIEW, R_VIEW], [-R_VIEW, R_VIEW]]
            for ax, (u, w) in ((ax_top, (a, b)), (ax_side, (a, c))):
                hst, _, _ = np.histogram2d(u, w, bins=160, range=rng)
                img = np.log10(hst.T + 0.5)
                ax.imshow(
                    img,
                    origin="lower",
                    cmap="inferno",
                    vmin=np.log10(0.5),
                    vmax=max(img.max(), 0.5),
                    extent=(-R_VIEW, R_VIEW, -R_VIEW, R_VIEW),
                )
                ax.plot(0, 0, "c+", ms=6)
            ax_top.set_title(
                f"{name}\nday {tt / CODE_DAY:.1f} (top view, {2 * R_VIEW:.0f} wide)",
                color="white",
                fontsize=9,
            )
            ax_side.set_title(
                f"edge-on | gas within r {R_VIEW:.0f}: {m['near']}, thickness "
                f"{m['thick |z|/R (median)']:.3f}\nround orbits {m['round orbits e<0.3']}"
                f", median T {m['T median near (K)']:,.0f} K",
                color="white",
                fontsize=8,
            )
            print(f"dump {d} {name}: day {tt / CODE_DAY:.2f} {m}", flush=True)
    fig.tight_layout()
    fig.savefig(out, dpi=90, facecolor="black")
    print("wrote", out)


if __name__ == "__main__":
    main()
