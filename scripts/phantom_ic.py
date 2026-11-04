"""Write our cloud initial conditions (sim/cloud.py, as cloud_run.py builds them) as a Phantom
starting dump, for production runs in Phantom (DECISIONS 2026-11-03, Phantom engine).

    uv run --with sarracen --with scikit-learn python scripts/phantom_ic.py OUT TEMPLATE n=50000 [seed=1]

Phantom units: length GM/c^2, mass M_bh, time GM/c^3 (so G = c = 1 as in our code; Phantom's
generalised Newtonian force reads c from the units). Our gas masses use a separate scale (the
cloud is units.CLOUD_MASS_MSUN); Phantom gets the real particle masses in M_bh, so its equation
of state sees the same physical densities. The gas has no self-gravity, so this changes nothing
else. Phantom computes h itself at start-up; ours is only its first guess. TEMPLATE: any dump
written by the same Phantom build (e.g. ``phantomsetup`` with SETUP=empty); its header is
copied, so the file has every field that build expects, and only our values are changed.
"""

import sys
from dataclasses import replace

import numpy as np
import sarracen

from nebula import units
from nebula.physics.kernels import DEFAULT_ETA
from nebula.sim.cloud import cloud_ic
from nebula.sim.coupled import CoupledConfig

PH_G, PH_C = 6.672041e-8, 2.997924e10  # cgs, Phantom physcon.f90 (gg, c)


def main():
    out, template = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    n, radius = int(opts.get("n", 50_000)), float(opts.get("radius", 60.0))
    hc = replace(CoupledConfig().hydro, eos=opts.get("eos", "gasrad"))
    seed = int(opts["seed"]) if "seed" in opts else None  # random glass carving (R6.5)
    ic = cloud_ic(n, radius, seed, hydro=hc)
    bh = units.BlackHoleUnits()
    m_code = hc.gas_units.mass_kg_per_code * ic.mass[0] / bh.mass_kg  # particle mass in M_bh
    # u: ours is in c^2 per unit mass, the same in Phantom's units (velocity unit c)
    sdf = sarracen.SarracenDataFrame(
        {"x": ic.x[:, 0], "y": ic.x[:, 1], "z": ic.x[:, 2], "vx": ic.v[:, 0], "vy": ic.v[:, 1],
         "vz": ic.v[:, 2], "h": ic.h, "u": ic.u}  # fmt: skip
    )
    # Phantom's own G and c (src/main/physcon.f90), so G = c = 1 hold exactly in its code units
    # (its GN force checks G = 1; ours, CODATA 2018, differ by 3e-4: the length and time units
    # are 0.03% apart from units.BlackHoleUnits, which changes nothing in code units)
    umass = bh.mass_kg * 1000.0
    udist, utime = PH_G * umass / PH_C**2, PH_G * umass / PH_C**3
    tmpl = sarracen.read_phantom(template)
    params = dict((tmpl[0] if isinstance(tmpl, tuple) else tmpl).params)
    params.update(
        nparttot=np.int32(len(sdf)), nparttot_2=np.int64(len(sdf)),
        npartoftype=np.int32(len(sdf)), massoftype=m_code, mass=m_code,
        hfact=DEFAULT_ETA, gamma=hc.gamma, time=0.0, ieos=np.int32(12),
        iexternalforce=np.int32(12), udist=udist, umass=umass, utime=utime,
        umagfd=np.sqrt(4 * np.pi * umass / udist) / utime,
    )  # fmt: skip
    sdf.params = params
    sarracen.write_phantom(out, sdf)
    print(f"wrote {out}: N = {len(sdf)}, particle mass {m_code:.4e} M_bh, u {ic.u.mean():.4e}, "
          f"h {np.median(ic.h):.3f}, x0 {ic.x.mean(0)}, v0 {ic.v[0]}")  # fmt: skip


if __name__ == "__main__":
    main()
