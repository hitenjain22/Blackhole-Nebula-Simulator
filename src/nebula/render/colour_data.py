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


EMISSION_PATH = Path(__file__).resolve().parents[3] / "data" / "emission_lines.npz"


@cache
def load_emission_tables() -> dict:
    """The nebula line tables for nebula.physics.emission_lines (Phase 14 N1): log_coll
    (6, n_t, n_n) [log10 erg cm^3 s^-1], log_t, log_n, coll_names, coll_wave_nm, ext_names,
    ext_wave_nm, ext_a_over_av (CCM89, R_V 3.1), rec_check. Made by
    scripts/export_emission_tables.py (PyNeb, dust_extinction; data/README.md). Read-only."""
    with np.load(EMISSION_PATH) as d:
        out = {k: d[k].copy() for k in d.files}
    for v in out.values():
        v.flags.writeable = False
    return out
