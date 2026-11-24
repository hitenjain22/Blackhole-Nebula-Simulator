"""Export the nebula emission-line tables (Phase 14 N1, phases/phase-14-nebula.md) to
data/emission_lines.npz, from PyNeb (atomic data and line emissivities) and dust_extinction
(the Cardelli, Clayton & Mathis 1989 extinction curve). Run once; neither package is a project
dependency:

    uv run --no-project --python 3.12 --with pyneb --with dust_extinction --with numpy \
        python scripts/export_emission_tables.py

Contents (wavelengths in air, nm):
- log_t (65,): log10 T [K] from 3.3 to 6.5; log_n (33,): log10 n_e [cm^-3] from 0 to 8.
- coll_names, coll_wave_nm: the collisional lines [O III] 500.68, 495.89, [N II] 658.35,
  654.80, [S II] 671.65, 673.08.
- log_coll (6, 65, 33): log10 of PyNeb's emissivity 4 pi j / (n_e n_ion) [erg cm^3 s^-1] for
  each line on the (T, n_e) grid (five-level atoms, PyNeb's default atomic data).
- rec_check (3, 3): PyNeb's case-B H alpha, H beta, H gamma 4 pi j / (n_e n_p) at 5,000,
  10,000, 20,000 K and n_e = 100 (a cross-check of the Dong & Draine fit and the case-B ratios
  used in physics/emission_lines.py; not read by the renderer).
- ext_names, ext_wave_nm, ext_a_over_av: A(lambda) / A(V) of CCM89 with R_V = 3.1 at every
  line's wavelength.
Source and licences in data/README.md."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pyneb as pn
from dust_extinction.parameter_averages import CCM89

OUT = Path(__file__).resolve().parents[1] / "data" / "emission_lines.npz"
COLL = [("OIII_5007", "O", 3, 5007, 500.68), ("OIII_4959", "O", 3, 4959, 495.89),
        ("NII_6584", "N", 2, 6584, 658.35), ("NII_6548", "N", 2, 6548, 654.80),
        ("SII_6716", "S", 2, 6716, 671.65), ("SII_6731", "S", 2, 6731, 673.08)]  # fmt: skip
ALL = [("Halpha", 656.28), ("Hbeta", 486.13), ("Hgamma", 434.05)] + [(c[0], c[4]) for c in COLL]


def main():
    log_t = np.round(np.linspace(3.3, 6.5, 65), 6)
    log_n = np.round(np.linspace(0.0, 8.0, 33), 6)
    tem, den = 10.0**log_t, 10.0**log_n
    table = np.empty((len(COLL), len(log_t), len(log_n)))
    for k, (_, el, ion, wave, _) in enumerate(COLL):
        atom = pn.Atom(el, ion)
        e = np.asarray(atom.getEmissivity(tem, den, wave=wave), float)  # (T, n) grid
        if e.shape != (len(log_t), len(log_n)) or not np.all(np.isfinite(e)) or e.min() <= 0:
            raise RuntimeError(f"{el}{ion} {wave}: unexpected PyNeb output {e.shape}")
        table[k] = np.log10(e)
    h1 = pn.RecAtom("H", 1)
    rec = np.array([[float(h1.getEmissivity(t, 100.0, label=lab)) for lab in ("3_2", "4_2", "5_2")]
                    for t in (5e3, 1e4, 2e4)])  # fmt: skip
    ext = CCM89(Rv=3.1)
    a_av = np.asarray(ext(np.array([w for _, w in ALL]) * u.nm), float)
    np.savez(OUT, log_t=log_t, log_n=log_n, coll_names=np.array([c[0] for c in COLL]),
             coll_wave_nm=np.array([c[4] for c in COLL]), log_coll=table, rec_check=rec,
             ext_names=np.array([n for n, _ in ALL]), ext_wave_nm=np.array([w for _, w in ALL]),
             ext_a_over_av=a_av, pyneb_version=np.array(pn.__version__))  # fmt: skip
    print(f"wrote {OUT} (PyNeb {pn.__version__}); log_coll range "
          f"{table.min():.2f} .. {table.max():.2f}; A/A_V {np.round(a_av, 4)}")


if __name__ == "__main__":
    main()
