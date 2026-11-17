"""Export the CIE 1931 2-degree standard observer colour-matching functions (1 nm, 360-830 nm)
to data/cie1931_2deg_1nm.csv, from the colour-science package (BSD-3-Clause), which carries
the CIE's published table. Run once (colour-science is not a project dependency):

    uv run --with colour-science python scripts/export_cie_cmf.py

Phase 9 step 9a (phases/phase-09-colour-physics.md 9.1); source and licences in data/README.md."""

from pathlib import Path

import colour
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "cie1931_2deg_1nm.csv"


def main():
    cm = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl = cm.wavelengths
    sel = (wl >= 360) & (wl <= 830)
    vals = cm.values[sel]
    assert len(wl[sel]) == 471 and np.all(np.diff(wl[sel]) == 1.0)
    with open(OUT, "w") as f:
        f.write("# CIE 1931 2-degree standard observer colour-matching functions, 1 nm\n")
        f.write(
            f"# exported from colour-science {colour.__version__} "
            "(MSDS_CMFS['CIE 1931 2 Degree Standard Observer'])\n"
        )
        f.write("wavelength_nm,xbar,ybar,zbar\n")
        for w, (x, y, z) in zip(wl[sel], vals, strict=True):
            f.write(f"{w:.0f},{x!r},{y!r},{z!r}\n")
    print(f"wrote {OUT} ({len(vals)} rows)")


if __name__ == "__main__":
    main()
