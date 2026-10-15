"""Phase 2 validation images and the orbiting-camera video (phases/phase-02 Step 2.5).

outputs/checks/shadow_D50.png          shadow size, measured vs expected (printed on it)
outputs/checks/einstein_ring_D50.png   one star behind the hole: its two Einstein rings
outputs/checks/blackhole_starfield.png the composed image
(these three are copied to docs/figures/)
outputs/checks/orbit_cam.mp4           240 frames, camera circling at r = 40, 720p, 4 rays/px
outputs/checks/render_timing.txt       times per frame

The camera is a static observer at each frame's position (a moving camera would also see
aberration; not modelled, by the plan's choice).

Run: uv run python scripts/validation_images.py
"""

import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nebula.render.camera import MODE_HOLE_STARS, Camera
from nebula.render.frames import render, to_png_array, write_png
from nebula.render.starfield import LUMA_WEIGHTS, StarFieldConfig, build_cube_map, upload_cube_map
from nebula.render.validation import measure_rings, measure_shadow, ring_camera, shadow_camera
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "outputs" / "checks"
FIGURES = ROOT / "docs" / "figures"
EXPOSURE_PERCENTILE = 99.98  # docs/DECISIONS.md, Step 2.4


def auto_exposure(hdr: np.ndarray) -> float:
    return 0.5 / np.percentile(hdr[..., :3] @ LUMA_WEIGHTS, EXPOSURE_PERCENTILE)


def annotate(
    img8: np.ndarray,
    lines: list[str],
    ticks: list[tuple[float, float, float, str]],
    bottom: bool = False,
):
    """Text box at the top left, plus short tick marks at expected radii (cx, cy, r, colour)
    left and right of the centre (marks only: nothing drawn over the feature itself)."""
    im = Image.fromarray(img8)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=17)
    pad, line_h = 10, 22
    box_h = pad * 2 + line_h * len(lines)
    y0 = im.height - box_h if bottom else 0
    draw.rectangle([0, y0, 660, y0 + box_h], fill=(0, 0, 0))
    for k, text in enumerate(lines):
        draw.text((pad, y0 + pad + k * line_h), text, fill=(235, 235, 235), font=font)
    for cx, cy, r, colour in ticks:
        for sgn in (-1, 1):
            x = cx + sgn * r
            draw.line([(x, cy + 14), (x, cy + 34)], fill=colour, width=2)
            draw.line([(x, cy - 14), (x, cy - 34)], fill=colour, width=2)
    return np.asarray(im)


def main() -> None:
    CHECKS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cube = build_cube_map(StarFieldConfig())
    sky = upload_cube_map(cube)
    print(f"Star field built and uploaded in {time.perf_counter() - t0:.1f} s")

    # 1. Shadow
    img, (rad, off), (xc, yc, _) = measure_shadow(sky)
    cam = shadow_camera()
    img8 = to_png_array(img[..., :3], auto_exposure(img))
    lines = [
        "Shadow seen from D = 50 (2.1 AU)",
        f"expected radius {rad.expected_deg:.6f} deg = {rad.expected_px:.2f} px",
        f"measured radius {rad.measured_deg:.6f} deg = {rad.measured_px:.2f} px",
        f"difference {rad.error_px:+.3f} px (requirement: within 1 px)",
        f"centre offset {off.measured_px:.3f} px. Ticks: expected edge",
    ]
    write_png(
        CHECKS / "shadow_D50.png",
        annotate(img8, lines, [(xc, yc, rad.expected_px, (80, 220, 120))]),
    )

    # 2. Einstein rings
    img, (prim, sec) = measure_rings()
    cam = ring_camera()
    # Rings of a single star: boost exposure so the faint secondary rim is visible.
    img8 = to_png_array(img[..., :3], 0.5 / np.percentile(img[..., :3] @ LUMA_WEIGHTS, 99.9))
    lines = [
        "One star exactly behind the hole, camera at D = 50",
        f"primary ring:   expected {prim.expected_deg:.6f} deg = {prim.expected_px:.2f} px, "
        f"measured {prim.measured_px:.2f} px ({prim.error_px:+.3f})",
        f"secondary ring: expected {sec.expected_deg:.6f} deg = {sec.expected_px:.2f} px, "
        f"measured {sec.measured_px:.2f} px ({sec.error_px:+.3f})",
        "Requirement: within 1 px. The secondary ring is 0.1 px outside the",
        "shadow edge at this distance: it shows as a thin rim on the shadow.",
    ]
    c = (cam.width / 2, cam.height / 2)
    ticks = [(*c, prim.expected_px, (80, 220, 120)), (*c, sec.expected_px, (230, 180, 60))]
    img8 = annotate(img8, lines, ticks, bottom=True)
    # Inset: the shadow's right edge magnified 10x, 300x brighter, so the secondary rim shows.
    lum = img[..., :3] @ LUMA_WEIGHTS
    x_edge = int(c[0] + sec.expected_px)
    crop = img[int(c[1]) - 15 : int(c[1]) + 15, x_edge - 15 : x_edge + 15, :3]
    boost = 300 * 0.5 / np.percentile(lum, 99.9)
    inset = np.kron(to_png_array(crop, boost), np.ones((10, 10, 1), dtype=np.uint8))
    img8 = img8.copy()
    img8[10:310, -310:-10] = inset
    im = Image.fromarray(img8)
    d = ImageDraw.Draw(im)
    d.rectangle([im.width - 311, 9, im.width - 9, 311], outline=(230, 180, 60), width=2)
    d.text(
        (im.width - 305, 315),
        "inset: shadow's right edge, 10x zoom,",
        fill=(230, 180, 60),
        font=ImageFont.load_default(size=15),
    )
    d.text(
        (im.width - 305, 333),
        "300x brighter: the secondary ring",
        fill=(230, 180, 60),
        font=ImageFont.load_default(size=15),
    )
    write_png(CHECKS / "einstein_ring_D50.png", np.asarray(im))
    for m in (rad, prim, sec):
        print(f"{m.name:16s} expected {m.expected_px:8.3f} px, measured {m.measured_px:8.3f} px")

    # 3. Composed image: Milky Way passing behind the hole, hole right of centre.
    pole = np.asarray(StarFieldConfig().band_pole, float)
    pole /= np.linalg.norm(pole)
    e1 = np.cross(pole, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    pos = -45.0 * e1 + np.array([0.0, 0.0, 6.0])  # band's bright core (+e1) behind the hole
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    hero = Camera(position=tuple(pos), look_at=tuple(-7.0 * right), up=(0, 0, 1), vfov_deg=55)
    t0 = time.perf_counter()
    img, _ = render(hero, MODE_HOLE_STARS, 4, (sky.scale, 0.0), sky.entries(), "render_stars.wgsl")
    t_hero = time.perf_counter() - t0
    write_png(CHECKS / "blackhole_starfield.png", to_png_array(img[..., :3], auto_exposure(img)))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ("blackhole_starfield.png", "shadow_D50.png", "einstein_ring_D50.png"):
        shutil.copy(CHECKS / name, FIGURES / name)

    # 4. Orbiting camera: r = 40, orbit plane tilted 20 deg, 240 frames, 2x2 rays per pixel.
    tilt = np.radians(20.0)
    exposure = None
    frame_times = []

    def frames():
        nonlocal exposure
        for k in range(240):
            th = 2 * np.pi * k / 240
            p = 40.0 * np.array([np.cos(th), np.sin(th) * np.cos(tilt), np.sin(th) * np.sin(tilt)])
            cam = Camera(position=tuple(p), up=(0, 0, 1), vfov_deg=60)
            t = time.perf_counter()
            hdr, _ = render(
                cam, MODE_HOLE_STARS, 2, (sky.scale, 0.0), sky.entries(), "render_stars.wgsl"
            )
            exposure = exposure or auto_exposure(hdr)  # fixed after frame 0: no pumping
            out = to_png_array(hdr[..., :3], exposure)
            frame_times.append(time.perf_counter() - t)
            yield out

    write_video(frames(), CHECKS / "orbit_cam.mp4")
    ft = np.array(frame_times)
    timing = [
        f"Composed image 1280x720, 16 rays/px: {t_hero * 1e3:.0f} ms",
        f"Orbit video 1280x720, 4 rays/px, 240 frames: median {np.median(ft) * 1e3:.0f} ms/frame "
        f"(render + tone map), total {ft.sum():.1f} s",
    ]
    (CHECKS / "render_timing.txt").write_text("\n".join(timing) + "\n")
    print("\n".join(timing))


if __name__ == "__main__":
    main()
