"""Reads the committed colour data (data/cie1931_2deg_1nm.csv) for the colour physics in
nebula.physics.colour, which does no I/O itself (rule 1). Phase 9 step 9a."""

from functools import cache
from pathlib import Path

import numpy as np

CMF_PATH = Path(__file__).resolve().parents[3] / "data" / "cie1931_2deg_1nm.csv"


@cache
def load_cmf() -> tuple[np.ndarray, np.ndarray]:
    """(wavelengths in nm (471,), colour-matching functions xbar, ybar, zbar (471, 3)).

    CIE 1931 2-degree standard observer, 1 nm, 360-830 nm (source and licence in
    data/README.md). Arrays are read-only: the cache hands the same arrays to every caller."""
    a = np.loadtxt(CMF_PATH, delimiter=",", comments="#", skiprows=3)
    lam, cmf = a[:, 0].copy(), a[:, 1:].copy()
    lam.flags.writeable = False
    cmf.flags.writeable = False
    return lam, cmf
