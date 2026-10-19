"""Overheating investigation (DECISIONS 2026-10-19): compare first-pass runs that differ only in
their artificial conductivity, at the same times.

    uv run python scripts/investigations/compare_cond_runs.py RUN [RUN ...]

Per run and time: typical heat of the gas near the hole, the spread of orbital energy, unbound
and lost gas, the near-vacuum halo (h > 100), and where the heat came from (median of the
per-particle budgets: viscosity, conductivity, P dV).
"""

import json
import sys
from pathlib import Path

import numpy as np

from nebula.physics.orbits import L_CAPTURE_PARABOLIC
from nebula.sim import coupled
from nebula.sim.coupled import CoupledConfig

ROOT = Path(__file__).resolve().parents[2]
TIMES = (393700.0, 396300.0, 399300.0, 404000.0, 410000.0, 420000.0)


def snapshot_near(run: str, t: float):
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    ts = np.array([float(np.load(f)["t"]) for f in files])
    k = int(np.argmin(np.abs(ts - t)))
    return np.load(files[k]) if abs(ts[k] - t) < 1500 else None


def main():
    runs = sys.argv[1:]
    cc = CoupledConfig()
    for run in runs:
        cfg = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"]
        log = (ROOT / "runs" / run / "log.txt").read_text().splitlines()
        wall = [ln for ln in log if "finished at" in ln]
        status = wall[-1].split("; ")[-1] if wall else "running"
        first = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))[0]
        n_start = len(np.load(first)["x"])
        print(f"\n{run}: cond {cfg.get('cond', 'velocity')}, alpha_u {cfg.get('alpha_u', 1.0)}, "
              f"h_max {cfg.get('h_max', 'inf')}; {status}")  # fmt: skip
        print("     t   u_med(r<300)  spread50  spread90  unbound  lost   halo(h>100)"
              "  heat: visc  cond  pdv")  # fmt: skip
        for t in TIMES:
            d = snapshot_near(run, t)
            if d is None:
                continue
            x, p = d["x"].astype(float), d["p"].astype(float)
            u, h = d["u"].astype(float), d["h"].astype(float)
            H = coupled.hamiltonian(cc, x, p)
            ell = np.linalg.norm(coupled.angmom(cc, x, p), axis=1)
            med = np.median(H)
            q = np.percentile(H, [25, 75, 5, 95])
            near = np.linalg.norm(x, axis=1) < 300
            lost = (n_start - len(x) + np.sum(ell < L_CAPTURE_PARABOLIC)) / n_start
            parts = ""
            if "heat_visc" in d.files:
                parts = "  ".join(f"{np.median(d[k].astype(float)):9.2e}"
                                  for k in ("heat_visc", "heat_cond", "heat_pdv"))  # fmt: skip
            umed = np.median(u[near]) if near.any() else np.nan
            print(f"{float(d['t']):8.0f}  {umed:9.2e}  {(q[1] - q[0]) / 2 / abs(med):8.1%}  "
                  f"{(q[3] - q[2]) / 2 / abs(med):8.1%}  {np.mean(H > 0):7.2%}  {lost:6.3%}"
                  f"  {np.sum(h > 100):6d}      {parts}")  # fmt: skip


if __name__ == "__main__":
    main()
