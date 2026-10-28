"""A cinematic preview of a run: glowing gas on black (not the Phase 8-9 physical render, which
ray-traces the gas through the hole's light bending; this is an artistic look at the particles).

    uv run python scripts/cinematic_preview.py RUN [hold=2] [still=DAY]

Brightness follows the gas column (mass per pixel, log scaled, with a soft bloom); colour follows
the mass-weighted temperature (dull red at 10^4 K through orange and white to blue-white at
10^6 K). The camera follows the gas near the hole like scripts/cloud_video.py. Writes
outputs/checks/<RUN>_cinematic.mp4, and with still=DAY a 1920x1080 frame at that day,
outputs/checks/<RUN>_cinematic_dayDAY.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from nebula import units
from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
FRAME_R = 12_000.0
# temperature palette (log10 K, stretched toward the 1-10x10^4 K most gas has): ember red ->
# orange -> pale yellow -> white -> blue-white
PAL_T = np.array([4.0, 4.25, 4.55, 4.9, 5.4, 6.2])
PAL_C = np.array([[0.55, 0.07, 0.02], [0.95, 0.30, 0.05], [1.00, 0.62, 0.20],
                  [1.00, 0.88, 0.62], [0.98, 0.97, 0.95], [0.70, 0.82, 1.00]])  # fmt: skip


def _font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _stars(w, h, seed=7):
    rng = np.random.default_rng(seed)
    sky = np.zeros((h, w, 3))
    n = int(w * h / 900)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    b = rng.pareto(2.2, n) * 0.05
    tint = rng.uniform(0.85, 1.0, (n, 3))
    sky[ys, xs] = np.minimum(b, 0.6)[:, None] * tint
    return gaussian_filter(sky, (0.6, 0.6, 0))


def render(x, logt, m, centre, half, w, h, stars):
    """(h, w, 3) float image of the gas within the camera window."""
    xlo, xhi = centre[0] - half * w / h, centre[0] + half * w / h
    ylo, yhi = centre[1] - half, centre[1] + half
    rng = ((xlo, xhi), (ylo, yhi))
    mass, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(w, h), range=rng, weights=m)
    heat, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(w, h), range=rng, weights=m * logt)
    mass, heat = mass.T[::-1], heat.T[::-1]
    s = max(w / 1280, 1.0)
    core = gaussian_filter(mass, 1.2 * s)
    bloom = gaussian_filter(mass, 9.0 * s) + 0.5 * gaussian_filter(mass, 30.0 * s)
    tmap = gaussian_filter(heat, 1.2 * s) / np.maximum(core, 1e-30)
    tmap = np.where(core > 0, tmap, 4.0)
    lum = core + 0.35 * bloom
    ref = np.percentile(lum[lum > 0], 99.7) if np.any(lum > 0) else 1.0
    bright = np.log1p(lum / (0.004 * ref)) / np.log1p(1.0 / 0.004)
    bright = np.clip(bright, 0, 1.15)
    colour = np.stack([np.interp(tmap, PAL_T, PAL_C[:, k]) for k in range(3)], axis=-1)
    img = stars + colour * bright[..., None] ** 1.15
    return np.clip(img, 0, 1)


def _hole(draw, centre, half, w, h):
    px = (w / 2 + (0 - centre[0]) / half * h / 2, h / 2 - (0 - centre[1]) / half * h / 2)
    if not (0 <= px[0] < w and 0 <= px[1] < h):
        return
    r = max(4, int(h / 180))
    draw.ellipse([px[0] - 2.2 * r, px[1] - 2.2 * r, px[0] + 2.2 * r, px[1] + 2.2 * r],
                 outline=(150, 190, 255), width=max(1, r // 3))  # fmt: skip
    draw.ellipse([px[0] - r, px[1] - r, px[0] + r, px[1] + r], fill=(0, 0, 0))


def frames(run, hold, w=1280, h=720, only_day=None):
    files = sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz"))
    eos = json.loads((ROOT / "runs" / run / "manifest.json").read_text())["config"].get("eos", "ideal")
    day = units.DAY / units.BlackHoleUnits().time_s
    stars = _stars(w, h)
    big, small = _font(int(h / 24)), _font(int(h / 42))
    centre = half = None
    for f in files:
        d = np.load(f)
        t = float(d["t"])
        if only_day is not None and t / day < only_day:
            continue
        x = d["x"].astype(float)
        logt = np.log10(snapshot_temperature(eos, d["rho"].astype(float), d["u"].astype(float)))
        m = np.ones(len(x))
        inner = x[np.linalg.norm(x, axis=1) < FRAME_R]
        inner = inner if len(inner) > 100 else x
        lo, hi = np.percentile(inner[:, :2], 2, axis=0), np.percentile(inner[:, :2], 98, axis=0)
        c_new, h_new = 0.5 * (lo + hi), max(0.62 * float(np.max(hi - lo)), 40.0)
        if centre is None or only_day is not None:
            centre, half = c_new, h_new
        else:  # smooth camera
            centre = 0.8 * centre + 0.2 * c_new
            half = float(np.exp(0.8 * np.log(half) + 0.2 * np.log(h_new)))
        img = render(x, logt, m, centre, half, w, h, stars)
        im = Image.fromarray((img ** (1 / 1.1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        _hole(dr, centre, half, w, h)
        pad = int(h / 30)
        dr.text((pad, pad), f"Day {t / day:,.0f}", font=big, fill=(235, 235, 240))
        dr.text((pad, pad + int(h / 18)), "A gas cloud falling onto a black hole of 4.3 million "
                "suns", font=small, fill=(170, 175, 185))  # fmt: skip
        width_au = 2 * half * w / h * 0.0424
        dr.text((pad, h - pad - int(h / 30)), f"View {width_au:,.0f} AU across · {len(x):,} gas "
                "particles · colour = temperature", font=small, fill=(150, 155, 165))  # fmt: skip
        frame = np.asarray(im)
        if only_day is not None:
            yield frame
            return
        for _ in range(hold):
            yield frame


def main():
    run = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if "still" in opts:
        day = float(opts["still"])
        frame = next(frames(run, 1, 1920, 1080, only_day=day))
        out = ROOT / "outputs" / "checks" / f"{run}_cinematic_day{int(day)}.png"
        Image.fromarray(frame).save(out)
        print(f"saved {out}")
        return
    out = ROOT / "outputs" / "checks" / f"{run}_cinematic.mp4"
    n = write_video(frames(run, int(opts.get("hold", 2))), out)
    print(f"saved {out} ({n} frames)")


if __name__ == "__main__":
    main()
