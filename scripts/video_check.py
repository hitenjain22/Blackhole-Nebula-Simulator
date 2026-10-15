"""Test video (Phase 0, Step 0.5): 90 frames (3 s) of a moving pattern -> outputs/checks/test.mp4.

Run: uv run python scripts/video_check.py   then open the file in QuickTime.
"""

from pathlib import Path

import numpy as np

from nebula.render.video import write_video

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "test.mp4"
W, H, N = 1280, 720, 90


def frame(k: int) -> np.ndarray:
    """Colour wheel turning once over the clip, a white ball crossing left to right,
    and a grey ramp along the bottom (shows banding or colour-range problems)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    angle = np.arctan2(y - H / 2, x - W / 2) + 2 * np.pi * k / N
    rgb = 0.5 + 0.4 * np.stack([np.cos(angle + s) for s in (0, 2.094, 4.189)], axis=-1)
    bx, by = 100 + (W - 200) * k / (N - 1), H / 2
    rgb[(x - bx) ** 2 + (y - by) ** 2 < 60**2] = 1.0
    rgb[H - 80 :] = (x[H - 80 :, :, None] / (W - 1)).repeat(3, axis=-1)
    return np.round(255 * rgb).astype(np.uint8)


def main() -> None:
    n = write_video((frame(k) for k in range(N)), OUT)
    print(f"Wrote {n} frames ({W} x {H}, 30 fps, H.264 yuv420p CRF 18, BT.709) to {OUT}")
    print(f"Size: {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
