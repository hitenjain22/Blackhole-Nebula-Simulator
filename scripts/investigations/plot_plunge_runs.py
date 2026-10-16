"""Figure for Hiten: the 1k plunge, three ways (review runs A, B, C, rerun on this Mac).
Left: lowest particle angular momentum (capture below 4). Right: hottest gas (u, c^2 units).
Bottom: the cloud's leading edge near the hole just before the plunge (run C), coloured by u.
Output: outputs/checks/plunge_runs.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
runs = {"A: gravity only": "runA", "B: viscosity off": "runB", "C: as before the fix": "runC"}
colors = {"runA": "#2b8a3e", "runB": "#1f5fbf", "runC": "#c92a2a"}
fig = plt.figure(figsize=(13, 9))
ax1, ax2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
for label, key in runs.items():
    d = np.load(ROOT / "outputs" / "review" / f"{key}.npz")
    t, x, p, u = d["t"], d["x"], d["p"], d["u"]
    sel = t >= 360_000
    lmin = np.linalg.norm(np.cross(x[sel], p[sel]), axis=2).min(axis=1)
    ax1.plot(t[sel], lmin, color=colors[key], lw=1.6, label=label)
    ax2.semilogy(t[sel], u[sel].max(axis=1), color=colors[key], lw=1.6, label=label)
ax1.axhline(4.0, color="k", ls=":", lw=1, label="below this the hole captures the gas")
ax1.set_title("Lowest angular momentum of any particle")
ax2.set_title("Hottest particle (internal energy, units of c²)")
for ax in (ax1, ax2):
    ax.set_xlabel("time (GM/c³)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
d = np.load(ROOT / "outputs" / "review" / "runC.npz")
i = np.searchsorted(d["t"], 386_400)
x, u, h = d["x"][i], d["u"][i], d["h"][i]
ax3 = fig.add_subplot(2, 1, 2)
near = np.linalg.norm(x, axis=1) < 400
sc = ax3.scatter(x[near, 0], x[near, 1], c=np.log10(u[near]), s=8, cmap="inferno")
for k in np.flatnonzero(np.linalg.norm(x, axis=1) < 60):
    ax3.add_patch(plt.Circle(x[k, :2], 2 * h[k], fill=False, color="#1f5fbf", lw=0.6, alpha=0.6))
ax3.add_patch(plt.Circle((0, 0), 2.0, color="k"))
ax3.set_aspect("equal")
ax3.set_xlim(-250, 250)
ax3.set_ylim(-150, 150)
fig.colorbar(sc, ax=ax3, label="log10 u (hotter = brighter)")
ax3.set_title(f"Run C, t = {d['t'][i]:.0f}: gas near the hole (black dot). Blue circles: the "
              "neighbour bubbles (2h) of particles at r < 60", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "checks" / "plunge_runs.png", dpi=120)
print("saved")
