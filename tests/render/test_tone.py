"""Hue-preserving display mapping (phase-09 9.5; test 7 of 9.7) and its supporting checks."""

import numpy as np

from nebula.physics.colour import (
    blackbody_xyz_si,
    chromaticity,
    srgb_encode,
    xyz_to_linear_srgb,
)
from nebula.render.colour_data import load_cmf
from nebula.render.tone import (
    LUMA,
    display_rgb8,
    filmic,
    luminance,
    to_gamut,
    tone_map,
)
from tests.helpers import check

LAM, CMF = load_cmf()
SRGB_TO_XYZ = np.linalg.inv(
    np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
)


def xy_of_rgb(rgb):
    return np.stack(chromaticity(np.asarray(rgb) @ SRGB_TO_XYZ.T), axis=-1)


def test_luma_is_the_y_row_of_the_srgb_matrix():
    check("luminance weights = Y row of inverse sRGB matrix", SRGB_TO_XYZ[1], LUMA, atol=1e-4)


def test_7_tone_map_keeps_in_gamut_chromaticity():
    """In-gamut colours (random linear sRGB in [0, 1]^3, and blackbodies 1,900 K-1e6 K) at
    exposures from -6 to +12 stops: after the tone curve and gamut step, whatever ends inside
    the gamut has the same chromaticity (10^-3); the tone curve alone changes none (any colour)."""
    rng = np.random.default_rng(7)
    rgb = rng.uniform(0.0, 1.0, (4000, 3))
    temps = np.geomspace(1900.0, 1e6, 200)
    bb = xyz_to_linear_srgb(blackbody_xyz_si(temps, LAM, CMF))
    bb /= luminance(bb)[:, None]  # unit luminance; all are in gamut above ~1,900 K
    colours = np.concatenate([rgb, bb])
    xy0 = xy_of_rgb(colours)
    worst_tm, worst_all, n_desat, n_total = 0.0, 0.0, 0, 0
    for stops in range(-6, 13, 2):
        lin = colours * 2.0**stops
        tm = tone_map(lin)
        worst_tm = max(worst_tm, float(np.max(np.abs(xy_of_rgb(tm) - xy0))))
        out = to_gamut(tm)
        kept = np.all((tm >= 0.0) & (tm <= 1.0), axis=1)  # in gamut after the tone curve
        n_desat += int((~kept).sum())
        n_total += len(kept)
        if kept.any():  # at +12 stops Y -> 1, so every non-grey colour needs desaturating
            worst_all = max(worst_all, float(np.max(np.abs(xy_of_rgb(out[kept]) - xy0[kept]))))
    print(f"   desaturated (outside [0,1] after the tone curve): {n_desat} of {n_total}")
    check("tone curve alone: chromaticity change (max)", 0.0, worst_tm, atol=1e-3)
    check("in-gamut colours: chromaticity change (max)", 0.0, worst_all, atol=1e-3)


def test_gamut_step_keeps_luminance_and_lands_in_gamut():
    rng = np.random.default_rng(3)
    rgb = rng.normal(0.4, 0.6, (5000, 3))
    y = luminance(rgb)
    ok = (y > 0.0) & (y < 1.0)
    out = to_gamut(rgb[ok])
    check("gamut step: all channels in [0, 1]", True, bool(np.all((out >= 0) & (out <= 1))))
    check("gamut step: luminance unchanged (max abs)", 0.0,
          float(np.max(np.abs(luminance(out) - y[ok]))), atol=1e-12)  # fmt: skip
    # cool blackbody (1,000 K: negative blue) -> desaturated, same hue direction from grey
    c = xyz_to_linear_srgb(blackbody_xyz_si(1000.0, LAM, CMF))
    c = 0.3 * c / luminance(c)
    g = to_gamut(c)
    d0, d1 = c - 0.3, g - 0.3
    cos = float(d0 @ d1 / np.linalg.norm(d0) / np.linalg.norm(d1))
    check("1,000 K: desaturated along the line to grey (cos = 1)", 1.0, cos, atol=1e-12)


def test_filmic_curve_shape():
    y = np.geomspace(1e-6, 1e6, 2000)
    f = filmic(y)
    check("filmic(0) = 0", 0.0, float(filmic(0.0)), atol=1e-15)
    check("filmic monotonic", True, bool(np.all(np.diff(f) >= 0.0)))
    check("filmic in [0, 1]", True, bool(np.all((f >= 0) & (f <= 1))))


def test_display_dither_mean_is_unbiased():
    """A flat mid-grey field: dithering spreads values over neighbouring codes but keeps the mean
    of the 8-bit image equal to the undithered value to 1/16 code."""
    lin = np.full((64, 64, 3), 0.18)
    a = display_rgb8(lin, dither=True).astype(float)
    b = display_rgb8(lin, dither=False).astype(float)
    exact = 255.0 * float(srgb_encode(float(filmic(0.18))))
    print(f"   exact code {exact:.3f}, dithered mean {a.mean():.3f}, undithered {b.mean():.0f}")
    check("dithered mean = exact code value", exact, a.mean(), atol=1 / 16)
