"""Nebula emission lines and dust (phases/phase-14-nebula.md N1; expected values §12).

Ionized gas near 10^4 K glows in a few lines rather than as a blackbody: hydrogen recombination
lines (H alpha red + H beta / H gamma blue = pink), the forbidden [O III] doublet (teal) and the
red [N II] and [S II] lines; dust absorbs blue more than red. Rendered on top of the existing
blackbody (physics/colour.py), which takes over in hot gas.

Formulas and sources:
- H alpha (case B): j / (n_e n_p) = 2.82e-26 T4^(-0.942 - 0.031 ln T4) erg cm^3 s^-1 sr^-1,
  T4 = T / 10^4 K (Dong & Draine 2011, ApJ 727, 35, Eq. 3; as used in SILCC, Peters et al.
  2017). Within 0.6% of PyNeb's case B for 5,000-20,000 K (§12).
- H beta = H alpha / 2.86, H gamma = 0.468 H beta: case-B ratios at 10^4 K (Osterbrock & Ferland
  2006, Table 4.4), used at all T (PyNeb: within 6% for 5,000-20,000 K; §12).
- Forbidden lines: j = eps(T, n_e) n_e n_ion / (4 pi), eps = PyNeb's 4 pi j / (n_e n_ion)
  [erg cm^3 s^-1] (Luridiana, Morisset & Shaw 2015), tabulated by scripts/
  export_emission_tables.py (data/emission_lines.npz, read by the caller: no I/O here) and
  interpolated linearly in (log T, log n_e), clamped at the table's edges.
- Abundances by number relative to H (solar; Asplund et al. 2009, ARA&A 47, 481): O 4.90e-4,
  N 6.76e-5, S 1.32e-5, He 0.085 (n_e = x_H n_H (1 + 0.085), helium singly ionized).
- Dust: A(lambda) = A_V (A(lambda)/A_V) from the CCM89 curve, R_V = 3.1 (tabulated);
  transmission 10^(-0.4 A) = exp(-tau), tau = A ln 10 / 2.5.
- Line colour: CIE X, Y, Z of a line of radiance L is L (xbar, ybar, zbar)(lambda) (the
  blackbody's XYZ are integrals of B_lambda xbar d lambda, so the units match, W m^-2 sr^-1);
  CMF interpolated linearly in the 1 nm table. Seen with redshift factor g: wavelength lambda / g
  and radiance g^4 L (I_nu / nu^3 invariant, integrated over the line).

Model choices (labelled, DECISIONS 2026-11-23/24):
- Which gas is ionized and how much O++, N+, S+ it holds is set by tags (x_h, x_o3, x_ns in
  [0, 1]), not computed.
- n_cap: the forbidden lines' eps is looked up at min(n_e, n_cap). The simulated cloud is
  ~10^13-10^14 cm^-3, where forbidden lines are physically quenched; looking eps up at a typical
  nebula density keeps their colour balance (the teal accent) at a nebula's values.
- hot_fade: all lines x 1 / (1 + (T / t_fade)^power), default t_fade 4e4 K, power 8: lines
  dominate to ~5e4 K and are < 10% of dense gas's light at 10^5 K. Above ~10^5 K infalling
  clumps stop glowing in recombination lines (Guillochon et al. 2014) and collisions strip O++;
  without a steep fade [O III] (which brightens with T) kept hot dense gas teal (88% of its
  light at 10^5 K with 5e4 / 4). The fade lets the blackbody take over: the gas turns white-hot.

Units: T in K; densities in cm^-3 for the _cgs functions (rule 4: line emission is naturally
written in physical units), SI for _si; ``line_emission_code`` is the code-unit wrapper.
Pure functions; NumPy arrays or scalars.
"""

import numpy as np

from nebula import units

LINE_NAMES = ("Halpha", "Hbeta", "Hgamma", "OIII_5007", "OIII_4959", "NII_6584", "NII_6548",
              "SII_6716", "SII_6731")  # fmt: skip
LINE_WAVE_NM = np.array([656.28, 486.13, 434.05, 500.68, 495.89, 658.35, 654.80, 671.65, 673.08])
HA_HB, HG_HB = 2.86, 0.468  # case B at 10^4 K
ABUND = {"O": 4.90e-4, "N": 6.76e-5, "S": 1.32e-5, "He": 0.085}
# forbidden lines in data/emission_lines.npz order: element of each
COLL_ELEMENT = ("O", "O", "N", "N", "S", "S")
ERG_CM3_TO_W_M3 = 0.1  # 1 erg s^-1 cm^-3 = 1e-7 W / 1e-6 m^3


def halpha_emissivity_cgs(t_k, n_e, n_p):
    """H alpha emissivity j [erg cm^-3 s^-1 sr^-1] = 2.82e-26 T4^(-0.942 - 0.031 ln T4) n_e n_p
    (Dong & Draine 2011 Eq. 3; densities in cm^-3)."""
    t4 = np.asarray(t_k, float) / 1e4
    return 2.82e-26 * t4 ** (-0.942 - 0.031 * np.log(t4)) * n_e * n_p


def recombination_emissivities_cgs(t_k, n_e, n_p):
    """(H alpha, H beta, H gamma) emissivities [erg cm^-3 s^-1 sr^-1] on the last axis, with the
    case-B ratios H alpha / H beta = 2.86 and H gamma / H beta = 0.468 (module doc)."""
    ha = halpha_emissivity_cgs(t_k, n_e, n_p)
    hb = ha / HA_HB
    return np.stack(np.broadcast_arrays(ha, hb, HG_HB * hb), axis=-1)


def table_lookup(log_eps, log_t, log_n, t_k, n_e):
    """10^(log_eps interpolated linearly in (log10 T, log10 n_e)), clamped at the table edges.
    log_eps (..., n_t, n_n) on the grids log_t (n_t,), log_n (n_n,) (uniform); returns
    (..., *shape of t_k) for each leading table."""
    lt = np.clip(np.log10(np.asarray(t_k, float)), log_t[0], log_t[-1])
    ln = np.clip(np.log10(np.maximum(np.asarray(n_e, float), 1e-300)), log_n[0], log_n[-1])
    ft = (lt - log_t[0]) / (log_t[1] - log_t[0])
    fn = (ln - log_n[0]) / (log_n[1] - log_n[0])
    i = np.minimum(np.floor(ft).astype(int), len(log_t) - 2)
    j = np.minimum(np.floor(fn).astype(int), len(log_n) - 2)
    a, b = ft - i, fn - j
    v = (log_eps[..., i, j] * (1 - a) * (1 - b) + log_eps[..., i + 1, j] * a * (1 - b)
         + log_eps[..., i, j + 1] * (1 - a) * b + log_eps[..., i + 1, j + 1] * a * b)  # fmt: skip
    return 10.0**v


def collisional_emissivities_cgs(tables, t_k, n_e, n_ions, n_cap=np.inf):
    """Forbidden-line emissivities [erg cm^-3 s^-1 sr^-1] on the last axis (data order:
    [O III] 5007, 4959, [N II] 6584, 6548, [S II] 6716, 6731): eps(T, min(n_e, n_cap)) n_e n_ion
    / (4 pi). tables: dict with log_coll, log_t, log_n (data/emission_lines.npz); n_ions (..., 6)
    the emitting ion's density for each line [cm^-3]."""
    eps = table_lookup(tables["log_coll"], tables["log_t"], tables["log_n"], t_k,
                       np.minimum(n_e, n_cap))  # fmt: skip
    eps = np.moveaxis(np.atleast_1d(eps), 0, -1)
    return eps * np.asarray(n_e, float)[..., None] * n_ions / (4.0 * np.pi)


def hot_fade(t_k, t_fade: float, power: float = 4.0):
    """Model factor 1 / (1 + (T / t_fade)^power) on every line (module doc: hot gas stops
    glowing in lines; the blackbody takes over)."""
    return 1.0 / (1.0 + (np.asarray(t_k, float) / t_fade) ** power)


def line_emissivities_si(n_h, t_k, x_h, x_o3, x_ns, tables, n_cap=1e9, t_fade=4e4,
                         fade_power=8.0):  # fmt: skip
    """All nine line emissivities [W m^-3 sr^-1] (LINE_NAMES order on the last axis) of gas with
    hydrogen density n_h [m^-3], temperature T [K] and tags: x_h = ionized hydrogen fraction,
    x_o3 = O++ fraction of oxygen, x_ns = N+ and S+ fraction of nitrogen and sulphur. n_cap
    [cm^-3] for the forbidden lines' eps; hot_fade(T, t_fade, fade_power) on all lines."""
    nh = np.asarray(n_h, float) * 1e-6  # cm^-3
    n_p = np.asarray(x_h, float) * nh
    n_e = n_p * (1.0 + ABUND["He"])
    rec = recombination_emissivities_cgs(t_k, n_e, n_p)
    ion = {"O": np.asarray(x_o3, float) * ABUND["O"], "N": np.asarray(x_ns, float) * ABUND["N"],
           "S": np.asarray(x_ns, float) * ABUND["S"]}  # fmt: skip
    n_ions = np.stack(np.broadcast_arrays(*[ion[e] * nh for e in COLL_ELEMENT]), axis=-1)
    coll = collisional_emissivities_cgs(tables, t_k, n_e, n_ions, n_cap)
    lead = np.broadcast_shapes(rec.shape[:-1], coll.shape[:-1])
    j = np.concatenate([np.broadcast_to(rec, (*lead, 3)), np.broadcast_to(coll, (*lead, 6))],
                       axis=-1) * ERG_CM3_TO_W_M3  # fmt: skip
    return j * hot_fade(t_k, t_fade, fade_power)[..., None]


def line_emission_code(rho_code, t_k, x_h, x_o3, x_ns, tables, gas_units=None, **kw):
    """line_emissivities_si for gas of code density rho, as radiance per code length of path
    [W m^-2 sr^-1 per GM/c^2] (the march multiplies by the step's path in code lengths). The
    rule-4 wrapper: n_H = X rho / m_p with units.GasUnits (default: the runs' gas units)."""
    gas_units = gas_units or units.GasUnits()
    n_h = units.X_HYDROGEN * gas_units.density_to_si(np.asarray(rho_code, float)) / units.M_P
    j = line_emissivities_si(n_h, t_k, x_h, x_o3, x_ns, tables, **kw)
    return j * gas_units.bh.length_m


def line_xyz(wave_nm, lam_nm, cmf):
    """CIE (xbar, ybar, zbar) at each wavelength [nm] (..., 3), linear interpolation of the 1 nm
    table (lam_nm (n,), cmf (n, 3)); 0 outside it."""
    w = np.asarray(wave_nm, float)
    return np.stack([np.interp(w, lam_nm, cmf[:, k], left=0.0, right=0.0) for k in range(3)], -1)


def lines_to_xyz(j_lines, g, lam_nm, cmf, wave_nm=LINE_WAVE_NM):
    """Observed CIE X, Y, Z [same unit as j] of line emission j (..., n_lines) seen with
    redshift factor g (...): sum over lines of g^4 j (xbar, ybar, zbar)(lambda / g)."""
    g = np.asarray(g, float)
    xyz = line_xyz(np.asarray(wave_nm)[None, :] / np.reshape(g, (-1, 1)), lam_nm, cmf)
    xyz = xyz.reshape(*np.shape(g), len(wave_nm), 3)
    return (g**4)[..., None] * np.einsum("...l,...lc->...c", np.asarray(j_lines, float), xyz)


def dust_transmission(a_over_av, a_v):
    """Fraction of light passing dust of visual extinction A_V [mag] at wavelengths with
    A(lambda)/A_V = a_over_av: 10^(-0.4 A_V a_over_av) (= exp(-tau), tau = 0.4 ln 10 A)."""
    return 10.0 ** (-0.4 * np.asarray(a_v, float) * np.asarray(a_over_av, float))
