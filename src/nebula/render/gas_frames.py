"""Gas frames for display (phases/phase-08-gas-rendering.md 8.1, last stage): GPU picture ->
displayable sRGB.

Phase 8 colour is a placeholder (phase-08 intro: "a temperature colour map or grey"). The gas
is grey, so its brightness is tone mapped (frames.aces_curve) and shown through a fire colour
ramp (matplotlib "afmhot": black -> red -> orange -> yellow -> white). The stars behind it are
already dimmed by the gas on the GPU (T x star radiance) and are tone mapped as in Phase 2
(frames.tone_map), then added in linear light. Phase 9 replaces this with blackbody colour,
Doppler beaming and redshift, computed per sample on the GPU; this module then reduces to
frames.to_png_array.
"""

from dataclasses import dataclass

import matplotlib
import numpy as np

from nebula.render.frames import LUMA, aces_curve, srgb_decode, srgb_encode, to_uint8, tone_map

_RAMP = matplotlib.colormaps["afmhot"]


@dataclass(frozen=True)
class Exposure:
    """gas: multiplies gas intensity [W m^-2 sr^-1] before the tone curve; stars: multiplies
    star radiance (cube-map flux units) before tone mapping (artistic, like a camera)."""

    gas: float
    stars: float = 1.0


def auto_gas_exposure(intensity: np.ndarray, percentile: float = 99.5, level: float = 2.0):
    """Exposure putting the given percentile of the (non-zero) gas brightness at ``level`` on
    the tone curve (aces(2) = 0.8: bright gas near white, the hottest few pixels saturate)."""
    lit = intensity[intensity > 0]
    return level / float(np.percentile(lit, percentile)) if lit.size else 1.0


def auto_star_exposure(star_rgb: np.ndarray, percentile: float = 99.98) -> float:
    """Phase 2's rule (scripts/validation_images.py, DECISIONS Step 2.4): this percentile of
    the star luminance maps to 0.5 before the tone curve."""
    y = np.asarray(star_rgb, np.float64) @ LUMA
    top = float(np.percentile(y, percentile))
    return 0.5 / top if top > 0 else 1.0


def gas_display_linear(intensity: np.ndarray, exposure: float) -> np.ndarray:
    """(H, W) gas intensity -> (H, W, 3) linear display RGB through the fire ramp."""
    level = aces_curve(np.asarray(intensity, np.float64) * exposure)
    return srgb_decode(_RAMP(level)[..., :3])  # the ramp's colours are sRGB-encoded


def compose(picture: np.ndarray, exposure: Exposure) -> np.ndarray:
    """GPU picture (H, W, 4) = (gas I, T x star RGB) -> uint8 sRGB (H, W, 3)."""
    gas = gas_display_linear(picture[..., 0], exposure.gas)
    stars = tone_map(picture[..., 1:4], exposure.stars)
    return to_uint8(srgb_encode(np.clip(gas + stars, 0.0, 1.0)))


def compose_colour(picture: np.ndarray, exposure: Exposure) -> np.ndarray:
    """Phase 9 GPU picture (H, W, 8) = (X, Y, Z, bolometric, T x star RGB, 0) -> uint8 sRGB.

    Gas: CIE XYZ -> linear sRGB (physics.colour.xyz_to_linear_srgb), times the exposure, then
    the hue-preserving tone curve and gamut step of render.tone (true colour, phase-09 9.5).
    Stars: as in Phase 8 (frames.tone_map, already dimmed by the gas on the GPU), added in
    linear light; their brightness is an artistic multiplier (9.5)."""
    from nebula.physics.colour import xyz_to_linear_srgb
    from nebula.render.tone import to_gamut
    from nebula.render.tone import tone_map as tone_luminance

    gas = to_gamut(tone_luminance(xyz_to_linear_srgb(picture[..., 0:3]) * exposure.gas))
    stars = tone_map(picture[..., 4:7], exposure.stars)
    return to_uint8(srgb_encode(np.clip(gas + stars, 0.0, 1.0)))
