"""Step 0.5: videos are written in the agreed format with true BT.709 colour."""

import re
import subprocess

import imageio_ffmpeg
import numpy as np
import pytest

from nebula.render.video import write_video
from tests.helpers import check

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def video_stream_line(path) -> str:
    """ffmpeg's one-line description of the video stream (imageio-ffmpeg ships no ffprobe)."""
    info = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    return next(line for line in info.splitlines() if "Video:" in line)


def test_video_format(tmp_path):
    n, w, h = 12, 64, 48
    frames = (np.full((h, w, 3), 10 * k, dtype=np.uint8) for k in range(n))
    path = tmp_path / "t.mp4"
    assert write_video(frames, path) == n

    stream = video_stream_line(path)
    # e.g. "Video: h264 (High) (...), yuv420p(tv, bt709, progressive), 64x48 ..., 30 fps"
    # ffmpeg prints a single "bt709" only when matrix, primaries and transfer are all bt709.
    pix = re.search(r"(yuv\w+)\(([^)]*)\)", stream)
    rows = {
        "frame count": (n, imageio_ffmpeg.count_frames_and_secs(str(path))[0]),
        "codec": ("h264", re.search(r"Video: (\w+)", stream).group(1)),
        "pixel format": ("yuv420p", pix.group(1)),
        "range, colour tags": ("tv, bt709", ", ".join(pix.group(2).split(", ")[:2])),
        "size": (f"{w}x{h}", re.search(r", (\d+x\d+)", stream).group(1)),
        "fps": ("30", re.search(r"([\d.]+) fps", stream).group(1)),
    }
    for name, (expected, actual) in rows.items():
        check(f"video {name}", expected, actual)


def test_colours_are_bt709(tmp_path):
    """Decode the stored YUV ourselves with the BT.709 formula (as a player trusting the tags
    would): saturated colours must come back within 3/255. With BT.601 data they are ~25 off."""
    colours = np.array(
        [[200, 40, 40], [40, 200, 40], [40, 40, 200], [230, 180, 60], [128, 128, 128]], np.uint8
    )
    frame = np.repeat(np.repeat(colours[None], 64, axis=0), 64, axis=1)  # 64 x 320 blocks
    h, w = frame.shape[:2]
    path = tmp_path / "c.mp4"
    write_video([frame] * 3, path)
    raw = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(path), "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
    ).stdout
    y = np.frombuffer(raw[: h * w], np.uint8).reshape(h, w).astype(float)
    u = np.frombuffer(raw[h * w : h * w * 5 // 4], np.uint8).reshape(h // 2, w // 2).astype(float)
    v = (
        np.frombuffer(raw[h * w * 5 // 4 : h * w * 3 // 2], np.uint8)
        .reshape(h // 2, w // 2)
        .astype(float)
    )
    kr, kb = 0.2126, 0.0722  # BT.709 luma coefficients
    worst = 0
    for i, x in enumerate(range(32, w, 64)):
        yy = (y[32, x] - 16) / 219  # limited ("tv") range
        cb = (u[16, x // 2] - 128) / 224
        cr = (v[16, x // 2] - 128) / 224
        r = yy + 2 * (1 - kr) * cr
        b = yy + 2 * (1 - kb) * cb
        g = (yy - kr * r - kb * b) / (1 - kr - kb)
        rgb = np.round(255 * np.array([r, g, b]))
        worst = max(worst, int(np.abs(rgb - colours[i]).max()))
    check("video BT.709 colour round trip (levels)", 0, worst, atol=3)


def test_odd_size_refused(tmp_path):
    with pytest.raises(ValueError, match="even"):
        write_video([np.zeros((47, 64, 3), dtype=np.uint8)], tmp_path / "odd.mp4")
