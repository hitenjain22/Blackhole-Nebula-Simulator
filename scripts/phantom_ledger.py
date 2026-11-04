"""Conservation ledger (CLAUDE.md rule 9) for a Phantom run with our patch, computed with our
own physics (physics/gravity.py), as sim/coupled.ledger does for our code:

    E = sum_all m (H(x, p) + u) + E_rad,    L = sum_all m (x cross p)

"all" includes the swallowed (r < r_acc) and escaped (r > rkill) particles: Phantom keeps them
in its dumps, frozen at removal, with h < 0, so their sum is the energy and angular momentum
they carried away (our E_acc + E_esc). E_rad is the patch's running total of radiated energy
(header nb_erad; nebula_cool). p = GN canonical momentum from Phantom's velocity (or p = v for
the Newtonian control). Masses are in Phantom's unit (M_bh); the drift is reported relative to
sum m |H| at the first dump (phase-06 done-when).

    uv run --with sarracen --with scikit-learn python scripts/phantom_ledger.py RUN_DIR [every=10]

Writes RUN_DIR/ledger.csv and prints a summary.
"""

import csv

import sys
from pathlib import Path

import numpy as np
import sarracen

from nebula.physics.gravity import gn_angmom, gn_hamiltonian, gn_p_from_v, newton_hamiltonian

DAY = 4079.27


def read_all(path):
    s = sarracen.read_phantom(str(path), ignore_inactive=False)
    return s[0] if isinstance(s, tuple) else s


def totals(s, newton: bool):
    m = float(s.params["massoftype"])
    x = s[["x", "y", "z"]].values.astype(float)
    v = s[["vx", "vy", "vz"]].values.astype(float)
    if newton:
        p = v
        h = newton_hamiltonian(x, p)
        lvec = np.cross(x, p)
    else:
        p = gn_p_from_v(x, v)
        h = gn_hamiltonian(x, p)
        lvec = gn_angmom(x, p)
    u = s["u"].values.astype(float)
    dead = s["h"].values <= 0
    erad = float(s.params.get("nb_erad", 0.0))
    return {
        "t_day": float(s.params["time"]) / DAY,
        "n_alive": int((~dead).sum()),
        "n_removed": int(dead.sum()),
        "e_present": m * float(np.sum((h + u)[~dead])),
        "e_removed": m * float(np.sum((h + u)[dead])),
        "e_rad": erad,
        "energy": m * float(np.sum(h + u)) + erad,
        "abs_h": m * float(np.sum(np.abs(h))),
        "lz": m * float(np.sum(lvec[:, 2])),
        "l_abs": m * float(np.linalg.norm(np.sum(lvec, axis=0))),
    }


def main():
    run = Path(sys.argv[1])
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    every = int(opts.get("every", 10))
    dumps = sorted(d for d in run.glob("*_[0-9][0-9][0-9][0-9][0-9]"))  # Phantom dumps NAME_NNNNN
    pick = dumps[::every] + ([dumps[-1]] if (len(dumps) - 1) % every else [])
    newton = "iexternalforce =           1" in (run / "cloud.in").read_text()
    rows = []
    for d in pick:
        r = totals(read_all(d), newton)
        rows.append(r)
    e0, a0, l0 = rows[0]["energy"], rows[0]["abs_h"], rows[0]["l_abs"]
    with open(run / "ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*rows[0], "dE_rel", "dL_rel"])
        w.writeheader()
        for r in rows:
            r["dE_rel"] = (r["energy"] - e0) / a0
            r["dL_rel"] = (r["l_abs"] - l0) / l0
            w.writerow(r)
    print(f"{'day':>7s} {'alive':>7s} {'removed':>7s} {'E_rad/sum|H|0':>14s} {'dE/sum|H|0':>12s} {'dL/L0':>10s}")
    for r in rows:
        print(f"{r['t_day']:7.1f} {r['n_alive']:7d} {r['n_removed']:7d} {r['e_rad'] / a0:14.4e} "
              f"{r['dE_rel']:12.3e} {r['dL_rel']:10.2e}")  # fmt: skip
    print(run / "ledger.csv")


if __name__ == "__main__":
    main()
