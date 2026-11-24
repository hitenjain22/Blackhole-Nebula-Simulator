"""Nebula emission lines (physics/emission_lines.py; phase-14 N1 tests 1-3 and 5; §12)."""

import numpy as np

from nebula.physics import emission_lines as el
from nebula.render.colour_data import load_cmf, load_emission_tables
from tests.helpers import check

# expected-values §12 (reference/scripts/25_emission_lines.py)
CHROMA = {"Halpha": (0.728754229, 0.271245771), "Hbeta": (0.063404782, 0.219704533),
          "Hgamma": (0.167306182, 0.008194842), "OIII_5007": (0.006886720, 0.555191722),
          "OIII_4959": (0.020123307, 0.434962850), "NII_6584": (0.729467707, 0.270532293),
          "NII_6548": (0.728186661, 0.271813339), "SII_6716": (0.732245327, 0.267754673),
          "SII_6731": (0.732453290, 0.267546710)}  # fmt: skip
HALPHA_FIT = {5e3: 5.3376606988e-26, 1e4: 2.82e-26, 2e4: 1.4461403754e-26}


def test_1_line_chromaticity_is_the_cie_table():
    lam, cmf = load_cmf()
    xyz = el.line_xyz(el.LINE_WAVE_NM, lam, cmf)
    xy = xyz[:, :2] / xyz.sum(axis=1, keepdims=True)
    want = np.array([CHROMA[n] for n in el.LINE_NAMES])
    check("N1 test 1: line chromaticities (9 lines) vs CIE table", want, xy, atol=1e-6)


def test_2_halpha_emissivity_is_the_fit():
    t = np.array(list(HALPHA_FIT))
    got = el.halpha_emissivity_cgs(t, 1.0, 1.0)
    check("N1 test 2: H alpha j/(n_e n_p) at 5,000 / 10,000 / 20,000 K",
          np.array(list(HALPHA_FIT.values())), got, rtol=1e-6)  # fmt: skip
    tab = load_emission_tables()["rec_check"]  # PyNeb case B, 4 pi j / (n_e n_p)
    check("H alpha fit vs PyNeb case B (independent; within 1%)", tab[:, 0] / (4 * np.pi), got,
          rtol=1e-2)  # fmt: skip
    rec = el.recombination_emissivities_cgs(1e4, 1.0, 1.0)
    check("case-B ratios at 10^4 K vs PyNeb (H alpha/H beta, H gamma/H beta), 0.5%",
          [tab[1, 0] / tab[1, 1], tab[1, 2] / tab[1, 1]], [rec[0] / rec[1], rec[2] / rec[1]],
          rtol=5e-3)  # fmt: skip


def test_3_dust_slab_transmits_exp_minus_tau():
    """A uniform slab of A_V = 1.7 marched in 1,000 thin layers (each layer's extinction
    multiplied in) transmits exp(-tau) at every line, tau = 0.4 ln 10 A_V A(lambda)/A_V."""
    a = load_emission_tables()["ext_a_over_av"]
    layers = np.prod([el.dust_transmission(a, 1.7 / 1000)] * 1000, axis=0)
    tau = 0.4 * np.log(10.0) * 1.7 * a
    check("N1 test 3: dust slab transmission = exp(-tau), 9 lines", np.exp(-tau), layers,
          rtol=1e-6)  # fmt: skip
    check("dust reddens: blue H beta dimmed more than red H alpha", True,
          bool(layers[1] < layers[0]))  # fmt: skip


def test_table_lookup_hits_the_nodes_and_clamps():
    tab = load_emission_tables()
    i, j = 20, 10
    t, n = 10.0 ** tab["log_t"][i], 10.0 ** tab["log_n"][j]
    got = el.table_lookup(tab["log_coll"], tab["log_t"], tab["log_n"], t, n)
    check("table lookup at a node = the table (6 lines)", 10.0 ** tab["log_coll"][:, i, j], got,
          rtol=1e-12)  # fmt: skip
    lo = el.table_lookup(tab["log_coll"], tab["log_t"], tab["log_n"], 1.0, 1e-5)
    check("table lookup below the grid = the corner (clamped)", 10.0 ** tab["log_coll"][:, 0, 0],
          lo, rtol=1e-12)  # fmt: skip


def test_5_no_nan_or_inf_at_the_edges():
    tab = load_emission_tables()
    t = np.array([10.0, 1e4, 1e6])
    for n_h in (0.0, 1e6, 1e20):
        j = el.line_emissivities_si(np.full(3, n_h), t, 1.0, 0.2, 0.5, tab, n_cap=1e3)
        check(f"N1 test 5: emissivities finite and >= 0 at n_H {n_h:g} m^-3 (10 K, 1e4 K, 1e6 K)",
              True, bool(np.all(np.isfinite(j)) and np.all(j >= 0)))  # fmt: skip
    j0 = el.line_emissivities_si(0.0, 1e4, 1.0, 0.2, 0.5, tab)
    check("zero density: no emission", 0.0, float(np.max(j0)))


def test_redshift_moves_the_line_and_scales_it_by_g4():
    lam, cmf = load_cmf()
    j = np.zeros(9)
    j[0] = 2.0  # H alpha only
    g = 656.28 / 600.0  # H alpha seen at 600 nm
    xyz = el.lines_to_xyz(j, g, lam, cmf)
    want = 2.0 * g**4 * el.line_xyz(600.0, lam, cmf)
    check("H alpha blueshifted to 600 nm: XYZ = g^4 j CMF(600 nm)", want, xyz, rtol=1e-12)
    check("g = 1: XYZ = sum j CMF(line)", 2.0 * el.line_xyz(656.28, lam, cmf),
          el.lines_to_xyz(j, 1.0, lam, cmf), rtol=1e-12)  # fmt: skip


def test_hot_fade_and_nebula_colours():
    """10^4 K photoionized gas: hydrogen alone is magenta (H alpha + H beta), with [N II] and
    [S II] pink; O++ turns it teal; at 10^6 K lines are faded by (1 + 20^4)^-1."""
    lam, cmf = load_cmf()
    tab = load_emission_tables()
    from nebula.physics.colour import xyz_to_linear_srgb

    j = el.line_emissivities_si(1e9, 1e4, 1.0, 0.0, 0.3, tab, n_cap=1e3)
    rgb = xyz_to_linear_srgb(el.lines_to_xyz(j, 1.0, lam, cmf))
    check("10^4 K with N+/S+, no O++: pink (sRGB red > blue > green)", True,
          bool(rgb[0] > rgb[2] > rgb[1]))  # fmt: skip
    jt = el.line_emissivities_si(1e9, 1e4, 1.0, 1.0, 0.3, tab, n_cap=1e3)
    rgb_t = xyz_to_linear_srgb(el.lines_to_xyz(jt, 1.0, lam, cmf))
    check("all O++: teal (sRGB green > blue > red)", True, bool(rgb_t[1] > rgb_t[2] > rgb_t[0]))
    check("hot fade at 10^6 K, t_fade 5e4, power 4", 1.0 / (1.0 + 20.0**4),
          float(el.hot_fade(1e6, 5e4)), rtol=1e-14)  # fmt: skip
