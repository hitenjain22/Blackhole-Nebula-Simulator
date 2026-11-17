"""Blackbody colour and sRGB (phase-09 9.1 / 9.5; tests 1-6 and 8 of 9.7).

Expected values: reference/expected-values.md §7 (CIE 1931 2-degree, 1 nm, 360-830 nm,
Planck with exact constants, summation at 1 nm; script 17)."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from nebula.physics.colour import (
    SRGB_KNEE,
    blackbody_chroma,
    blackbody_xyz_si,
    chroma_lut,
    chromaticity,
    luminous_efficacy,
    lut_chroma,
    planck_lambda_si,
    srgb_decode,
    srgb_encode,
    xyz_from_xyy,
    xyz_to_linear_srgb,
)
from nebula.render.colour_data import load_cmf
from tests.helpers import check

LAM, CMF = load_cmf()

# expected-values §7: T -> (x, y, linear sRGB normalised to max = 1)
TABLE = {
    1000: (0.65275, 0.34446, (1, 0.0087, -0.0195)),
    1500: (0.58572, 0.39312, (1, 0.1331, -0.0196)),
    2000: (0.52668, 0.41330, (1, 0.2568, 0.0081)),
    3000: (0.43693, 0.40407, (1, 0.4769, 0.1537)),
    4000: (0.38044, 0.37675, (1, 0.6537, 0.3773)),
    5000: (0.34510, 0.35161, (1, 0.7917, 0.6284)),
    5778: (0.32643, 0.33572, (1, 0.8773, 0.8218)),
    6500: (0.31353, 0.32363, (1, 0.9429, 0.9922)),
    10_000: (0.28063, 0.28829, (0.6093, 0.6952, 1)),
    20_000: (0.25646, 0.25763, (0.4055, 0.5364, 1)),
    30_000: (0.25007, 0.24879, (0.3613, 0.4975, 1)),
    100_000: (0.24258, 0.23803, (0.3137, 0.4534, 1)),
    1_000_000: (0.24013, 0.23442, (0.2992, 0.4393, 1)),
}
T_INFINITE = 1e15  # Rayleigh-Jeans limit for the visible band (h c / (lambda k T) ~ 1e-10)


def test_1_cmf_table_integrity():
    check("CMF rows (360-830 nm at 1 nm)", 471, len(LAM))
    check("CMF first wavelength (nm)", 360.0, LAM[0])
    check("CMF last wavelength (nm)", 830.0, LAM[-1])
    check("CMF step uniform 1 nm", True, bool(np.all(np.diff(LAM) == 1.0)))
    x, y = chromaticity(CMF.sum(axis=0))  # equal-energy spectrum: S(lambda) = 1
    check("equal-energy white x = 1/3", 1 / 3, x, atol=1e-4)
    check("equal-energy white y = 1/3", 1 / 3, y, atol=1e-4)


def test_2_planck_chromaticity():
    for t, (xe, ye) in ((2000, (0.52668, 0.41330)), (6500, (0.31353, 0.32363)),
                        (T_INFINITE, (0.23988, 0.23404))):  # fmt: skip
        x, y = chromaticity(blackbody_xyz_si(t, LAM, CMF))
        label = "T -> inf" if t == T_INFINITE else f"{t} K"
        check(f"Planck x at {label} (§7)", xe, x, atol=2e-4)
        check(f"Planck y at {label} (§7)", ye, y, atol=2e-4)
    temps = np.array(list(TABLE))
    x, y = chromaticity(blackbody_xyz_si(temps, LAM, CMF))
    xe = np.array([v[0] for v in TABLE.values()])
    ye = np.array([v[1] for v in TABLE.values()])
    check("Planck x, full §7 table (13 temperatures)", xe, x, atol=5e-4)
    check("Planck y, full §7 table (13 temperatures)", ye, y, atol=5e-4)
    # linear sRGB normalised to the largest channel, printed to 4 decimals in §7
    rgb = xyz_to_linear_srgb(blackbody_xyz_si(temps, LAM, CMF))
    rgb /= rgb.max(axis=1, keepdims=True)
    rgbe = np.array([v[2] for v in TABLE.values()], float)
    check("linear sRGB (max = 1), full §7 table", rgbe, rgb, atol=5e-4)


def test_3_luminous_efficacy():
    check("luminous efficacy at 6,500 K (lm/W, §7)", 95.36,
          float(luminous_efficacy(6500.0, LAM, CMF)), atol=0.1)  # fmt: skip
    temps = np.arange(3000.0, 12001.0, 10.0)
    t_peak = temps[np.argmax(luminous_efficacy(temps, LAM, CMF))]
    print(f"   peak efficacy at {t_peak:.0f} K")
    check("peak efficacy between 6,000 and 7,000 K", True, 6000.0 < t_peak < 7000.0)
    for t, e in ((10_000, 73.47), (2000, 1.61), (30_000, 8.31)):
        check(f"luminous efficacy at {t:,} K (§7)", e,
              float(luminous_efficacy(float(t), LAM, CMF)), atol=0.01)  # fmt: skip


def test_4_wien_peak():
    for t in (1e4, 1e5, 1e6):
        guess = 2.9e-3 / t
        b0 = planck_lambda_si(guess, t)
        res = minimize_scalar(lambda lam, t=t, b0=b0: -planck_lambda_si(lam, t) / b0,
                              bounds=(0.5 * guess, 2.0 * guess), method="bounded",
                              options={"xatol": guess * 1e-10})  # fmt: skip
        check(f"Wien lambda_peak T at {t:.0e} K (m K)", 2.897771955e-3, res.x * t, rtol=1e-6)


def test_5_srgb_matrix_d65_white():
    rgb = xyz_to_linear_srgb(xyz_from_xyy(0.3127, 0.3290, 1.0))
    check("D65 white (Y = 1) -> linear sRGB (1, 1, 1)", np.ones(3), rgb, atol=1e-3)


def test_6_srgb_transfer():
    lin = np.concatenate([np.linspace(0.0, 1.0, 10_001), [SRGB_KNEE], np.geomspace(1e-9, 1, 500)])
    back = srgb_decode(srgb_encode(lin))
    check("sRGB encode -> decode round trip (max abs)", 0.0, float(np.max(np.abs(back - lin))),
          atol=1e-7)  # fmt: skip
    enc = np.linspace(0.0, 1.0, 10_001)
    check("sRGB decode -> encode round trip (max abs)", 0.0,
          float(np.max(np.abs(srgb_encode(srgb_decode(enc)) - enc))), atol=1e-7)  # fmt: skip
    lin_branch = 12.92 * SRGB_KNEE
    pow_branch = 1.055 * SRGB_KNEE ** (1 / 2.4) - 0.055
    print(f"   branches at the knee: {lin_branch:.9f} vs {pow_branch:.9f}")
    check("sRGB curve continuous at the 0.0031308 knee", lin_branch, pow_branch, atol=1e-7)


@pytest.fixture(scope="module")
def lut():
    return chroma_lut(1e3, 1e8, 4096, LAM, CMF)


def test_8_lut_accuracy(lut):
    log_t, log_chroma = lut
    rng = np.random.default_rng(9)
    step = log_t[1] - log_t[0]
    i = rng.integers(0, len(log_t) - 1, 50)
    t = np.exp(log_t[i] + rng.uniform(0.05, 0.95, 50) * step)  # strictly between nodes
    rel = lut_chroma(t, log_t, log_chroma) / blackbody_chroma(t, LAM, CMF) - 1.0
    check("LUT vs direct, 50 random T between nodes (max rel)", 0.0,
          float(np.max(np.abs(rel))), atol=5e-5)  # fmt: skip
    mid = np.exp(log_t[:-1] + 0.5 * step)  # worst place: halfway between every pair of nodes
    rel = lut_chroma(mid, log_t, log_chroma) / blackbody_chroma(mid, LAM, CMF) - 1.0
    print(f"   worst at mid-nodes over 1e3-1e8 K: {np.max(np.abs(rel)):.2e} "
          "(expected-values §7: 2.38e-5)")  # fmt: skip
    check("LUT vs direct at every mid-node (max rel)", 0.0, float(np.max(np.abs(rel))),
          atol=5e-5)  # fmt: skip
    check("LUT exact at its nodes", blackbody_chroma(np.exp(log_t[::97]), LAM, CMF),
          lut_chroma(np.exp(log_t[::97]), log_t, log_chroma), rtol=1e-12)  # fmt: skip
