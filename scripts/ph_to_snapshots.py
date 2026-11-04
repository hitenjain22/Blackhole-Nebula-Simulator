"""Convert a Phantom run (our patch) to our snapshot format, so our analysis and video scripts
(cinematic_preview.py, cloud_metrics, ...) read it like one of our runs.

    uv run --with sarracen --with scikit-learn python scripts/ph_to_snapshots.py PH_RUN_DIR NAME [every=1]

Writes runs/NAME/snapshots/snap_NNNNN.npz (t, x, p, u, rho, h, ids, as float32 for rendering
like our snapshots; rule 5) and runs/NAME/manifest.json (the Phantom run's manifest plus the
config keys our scripts read). Only particles still present are written (swallowed and
escaped ones are in the Phantom dumps, with h < 0). p is the GN canonical momentum from
Phantom's velocity (p = v for the Newtonian control). rho is converted to our gas mass unit
(Phantom's masses are in M_bh; PLAN §3), so temperatures from our EOS match.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sarracen

from nebula import units
from nebula.physics.gravity import gn_p_from_v
from nebula.sim.coupled import CoupledConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    src, name = Path(sys.argv[1]), sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    every = int(opts.get("every", 1))
    out = ROOT / "runs" / name / "snapshots"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty (rule 8: never overwrite a run)")
    out.mkdir(parents=True, exist_ok=True)
    cin = (src / "cloud.in").read_text()
    newton = "iexternalforce =           1" in cin
    gu = CoupledConfig().hydro.gas_units
    dumps = sorted(src.glob("*_[0-9][0-9][0-9][0-9][0-9]"))[::every]
    for k, f in enumerate(dumps):
        s = sarracen.read_phantom(str(f))
        s = s[0] if isinstance(s, tuple) else s
        m_ph = float(s.params["massoftype"])
        m_ours = m_ph * units.BlackHoleUnits().mass_kg / gu.mass_kg_per_code
        x = s[["x", "y", "z"]].values.astype(float)
        v = s[["vx", "vy", "vz"]].values.astype(float)
        p = v if newton else gn_p_from_v(x, v)
        np.savez(
            out / f"snap_{k:05d}.npz", t=np.float64(s.params["time"]),
            x=x.astype(np.float32), p=p.astype(np.float32),
            u=s["u"].values.astype(np.float32),
            rho=(s["rho"].values.astype(float) * m_ours / m_ph).astype(np.float32),
            h=s["h"].values.astype(np.float32), ids=s["iorig"].values.astype(np.int64),
        )  # fmt: skip
    man = json.loads((src / "manifest.json").read_text()) if (src / "manifest.json").exists() else {}
    man["config"] = {"eos": "gasrad", "gravity": "newton" if newton else "gn", "r_acc": 4.0,
                     "source": "Phantom run " + str(src), "every": every}  # fmt: skip
    (ROOT / "runs" / name / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {len(dumps)} snapshots to {out}")


if __name__ == "__main__":
    main()
