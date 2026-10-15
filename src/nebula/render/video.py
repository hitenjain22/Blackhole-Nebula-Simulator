"""Frames -> MP4 (Phase 0, Step 0.5; extended with overlays in Phase 10).

H.264 (libx264), yuv420p, CRF 18, BT.709 colour tags: plays correctly everywhere,
including QuickTime and phones. Uses the ffmpeg binary bundled with imageio-ffmpeg.
"""

from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as iio
import numpy as np

FPS = 30
CRF = 18
# Tag the frames as BT.709 *before* the RGB -> YUV conversion, so the conversion uses the
# BT.709 matrix and all three tags (matrix, primaries, transfer) are written. With no colour
# settings ffmpeg converts with BT.601 (saturated colours ~25/255 off in players expecting
# BT.709; tests/test_video.py::test_colours_are_bt709); the output flags -colorspace etc.
# converted correctly but wrote only the matrix tag (test_video_format).
BT709 = ["-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"]


def write_video(frames: Iterable[np.ndarray], path: Path, fps: int = FPS, crf: int = CRF) -> int:
    """Write (H, W, 3) uint8 sRGB frames to an MP4. Returns the number of frames written.

    Width and height must be even (yuv420p stores colour at half resolution).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    try:
        for frame in frames:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"frames must be (H, W, 3) uint8, got {frame.shape} {frame.dtype}")
            h, w = frame.shape[:2]
            if h % 2 or w % 2:
                raise ValueError(f"width and height must be even for yuv420p, got {w} x {h}")
            if writer is None:
                writer = iio.get_writer(
                    path,
                    format="FFMPEG",
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    quality=None,  # use CRF below instead of imageio's quality scale
                    macro_block_size=1,  # never silently resize; evenness is checked above
                    ffmpeg_params=["-crf", str(crf), "-movflags", "+faststart", *BT709],
                )
            writer.append_data(frame)
            count += 1
    finally:
        if writer is not None:
            writer.close()
    return count
