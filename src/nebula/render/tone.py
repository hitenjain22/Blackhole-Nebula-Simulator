"""From linear sRGB light to display pixels without changing hue (phase-09 9.1 and 9.5).

Pipeline (readable reference; the GPU copy in Phase 9 step 9e mirrors these functions):
    linear sRGB (any brightness, possibly out of gamut)
    -> exposure (a multiplier, 2**stops)
    -> tone curve on luminance only: Y_out = filmic(Y_in), RGB scaled by Y_out / Y_in
       (keeps chromaticity exactly; a per-channel curve would shift hue toward white/cyan)
    -> gamut handling: colours still outside [0, 1] are desaturated toward the grey of the
       same luminance (never clamped per channel), by the smallest amount that brings them in
    -> sRGB transfer function (physics.colour.srgb_encode) -> 8 bits with ordered dithering.

Display mapping, not physics: no physical constants here.

Sources: luminance weights of the sRGB primaries (IEC 61966-2-1; the Y row of the inverse of
physics.colour.XYZ_TO_SRGB, Rec. ITU-R BT.709); filmic curve: K. Narkowicz 2015, "ACES Filmic
Tone Mapping Curve" (fit to the ACES reference rendering transform),
f(x) = x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14), clipped to [0, 1]; ordered dithering:
Bayer 1973 (4 x 4 threshold matrix).
"""

import numpy as np

from nebula.physics.colour import srgb_encode

LUMA = np.array([0.2126, 0.7152, 0.0722])  # Y of linear sRGB (BT.709 / sRGB primaries, D65)
BAYER4 = (
    np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float) + 0.5
) / 16.0


def luminance(rgb):
    """Relative luminance Y of linear sRGB (..., 3)."""
    return np.asarray(rgb, float) @ LUMA


def filmic(y):
    """Narkowicz 2015 ACES fit on luminance: maps [0, inf) onto [0, 1], monotonic,
    f(0) = 0, slope ~0.21 at 0 and ~1 near mid grey, f -> 1.03 as y -> inf (clipped at 1)."""
    y = np.maximum(np.asarray(y, float), 0.0)
    return np.clip(y * (2.51 * y + 0.03) / (y * (2.43 * y + 0.59) + 0.14), 0.0, 1.0)


def tone_map(rgb):
    """Luminance-only tone curve: RGB * filmic(Y) / Y (chromaticity unchanged; 0 stays 0)."""
    rgb = np.asarray(rgb, float)
    y = luminance(rgb)
    scale = np.divide(filmic(y), y, out=np.zeros_like(y), where=y > 0.0)
    return rgb * scale[..., None]


def to_gamut(rgb):
    """Bring linear sRGB into [0, 1]^3 at constant luminance by desaturating toward grey (Y, Y, Y).

    rgb' = Y + s (rgb - Y) with the largest s in [0, 1] that keeps every channel in [0, 1].
    Requires 0 <= Y <= 1 (true after tone_map); in-gamut colours are returned unchanged (s = 1).
    Luminance is preserved because the weights sum to 1.
    """
    rgb = np.asarray(rgb, float)
    y = np.clip(luminance(rgb), 0.0, 1.0)[..., None]
    d = rgb - y
    with np.errstate(divide="ignore", invalid="ignore"):
        s_hi = np.where(rgb > 1.0, (1.0 - y) / d, 1.0)
        s_lo = np.where(rgb < 0.0, y / -d, 1.0)
    s = np.clip(np.minimum(s_hi, s_lo).min(axis=-1, keepdims=True), 0.0, 1.0)
    return np.clip(y + s * d, 0.0, 1.0)  # the clip only removes round-off


def display_rgb8(rgb_linear, exposure_stops=0.0, dither=True):
    """uint8 image (H, W, 3) from linear sRGB (H, W, 3): exposure, tone curve, gamut, sRGB
    encoding, ordered dithering (removes banding in smooth glows)."""
    lin = to_gamut(tone_map(np.asarray(rgb_linear, float) * 2.0**exposure_stops))
    enc = srgb_encode(lin) * 255.0
    if dither:
        h, w = enc.shape[:2]
        enc = enc + (np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w, None] - 0.5)
    return np.clip(np.floor(enc + 0.5), 0, 255).astype(np.uint8)
