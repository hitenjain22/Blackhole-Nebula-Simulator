"""From camera to PNG: GPU render, tone mapping, sRGB encoding, dithering
(phases/phase-02-blackhole-render.md Step 2.3; Phase 9 extends the colour handling).

Pipeline: linear HDR RGB (float32) -> exposure -> tone curve on luminance only (keeps hue,
phase-09 9.5) -> hue-preserving fit into [0, 1] -> sRGB transfer function (IEC 61966-2-1)
-> 8-bit with ordered dithering (avoids banding in smooth gradients) -> PNG.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nebula.render.camera import Camera
from nebula.render.device import TileTimings, load_shader, run_tiled, uniform_buffer

# Rec. 709 / sRGB luminance weights (linear RGB).
LUMA = np.array([0.2126, 0.7152, 0.0722])

# 4x4 Bayer matrix, thresholds in (0, 1).
_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16


def render(
    camera: Camera,
    mode: int,
    n_sub: int = 4,
    sky_params=(0.0, 0.0),
    extra: list[dict] | None = None,
    shader: str = "render_sky.wgsl",
    tile: int = 256,
) -> tuple[np.ndarray, TileTimings]:
    """Render one HDR frame: (H, W, 4) float32 = linear RGB + fraction of sub-rays captured.
    ``extra`` adds bind-group entries after the camera uniform (e.g. the star cube map)."""
    cam_buf = uniform_buffer(camera.uniform_bytes(n_sub, mode, sky_params))
    entries = [{"binding": 2, "resource": {"buffer": cam_buf}}, *(extra or [])]
    return run_tiled(
        load_shader("schw_ray.wgsl", shader), camera.width, camera.height, tile, entries
    )


def aces_curve(x: np.ndarray) -> np.ndarray:
    """Filmic tone curve (Narkowicz 2015 fit to ACES): maps [0, inf) to [0, ~1)."""
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def tone_map(rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Linear HDR RGB -> linear display RGB in [0, 1], keeping each pixel's hue.

    The curve acts on luminance Y; RGB is scaled by Y_out / Y_in. A pixel whose largest
    channel still exceeds 1 is desaturated toward its own luminance (not clipped per channel,
    which would shift the hue)."""
    rgb = np.asarray(rgb, np.float64) * exposure
    y_in = rgb @ LUMA
    y_out = aces_curve(y_in)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(y_in > 0, y_out / y_in, 0.0)
    out = rgb * scale[..., None]
    peak = out.max(axis=-1)
    over = peak > 1.0
    if np.any(over):
        y = y_out[over][:, None]
        t = (1.0 - y) / (peak[over][:, None] - y)  # largest blend keeping all channels <= 1
        out[over] = y + t * (out[over] - y)
    return np.clip(out, 0.0, 1.0)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 transfer function: linear [0, 1] -> encoded [0, 1]."""
    x = np.asarray(linear, np.float64)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """Inverse of ``srgb_encode``."""
    v = np.asarray(encoded, np.float64)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def to_uint8(encoded: np.ndarray, dither: bool = True) -> np.ndarray:
    """Encoded [0, 1] -> uint8, with 4x4 ordered dithering (fixed pattern: no flicker)."""
    v = np.clip(np.asarray(encoded, np.float64), 0.0, 1.0) * 255.0
    if dither:
        h, w = v.shape[:2]
        thr = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        v = np.floor(v + thr[..., None] if v.ndim == 3 else v + thr)
    else:
        v = np.round(v)
    return np.clip(v, 0, 255).astype(np.uint8)


def to_png_array(hdr_rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Linear HDR RGB (H, W, 3) -> displayable uint8 sRGB (H, W, 3)."""
    return to_uint8(srgb_encode(tone_map(hdr_rgb, exposure)))


def write_png(path: Path, img8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, img8)
