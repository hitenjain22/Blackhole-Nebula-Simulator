"""Phase 9 steps 9a-9b picture: the true colour of glowing gas from 1,000 K to 1,000,000 K.

    uv run python scripts/phase9_colour_strip.py

Top strip: each colour at its brightest (largest channel = 1). Bottom strip: true relative
brightness (sigma T^4 per unit area) through the display pipeline (tone curve on luminance,
gamut step, sRGB), exposure set so 10,000 K is mid-grey.
Writes outputs/checks/phase9_temperature_colours.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.physics.colour import blackbody_xyz_si, srgb_encode, xyz_to_linear_srgb
from nebula.render.colour_data import load_cmf
from nebula.render.tone import display_rgb8, luminance, to_gamut

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "phase9_temperature_colours.png"


def main():
    lam, cmf = load_cmf()
    temps = np.geomspace(1e3, 1e6, 900)
    rgb = xyz_to_linear_srgb(blackbody_xyz_si(temps, lam, cmf))
    equal = to_gamut(rgb / rgb.max(axis=1, keepdims=True))  # each colour at its brightest
    strip_equal = (srgb_encode(equal) * 255).astype(np.uint8)
    y10k = luminance(xyz_to_linear_srgb(blackbody_xyz_si(1e4, lam, cmf)))
    true = rgb / y10k * 0.18  # 10,000 K at mid grey
    strip_true = display_rgb8(true[None, :, :], dither=False)[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.6), facecolor="black")
    for ax, strip, title in (
        (axes[0], strip_equal, "colour (each at its brightest)"),
        (axes[1], strip_true, "true relative brightness (10,000 K at mid-grey; hotter is far "
         "brighter, so it goes white)"),
    ):  # fmt: skip
        ax.imshow(np.repeat(strip[None], 60, axis=0), aspect="auto",
                  extent=(3, 6, 0, 1))  # fmt: skip
        ax.set_yticks([])
        ax.set_xticks([3, np.log10(2000), np.log10(4000), np.log10(6500), 4, np.log10(3e4), 5, 6])
        ax.set_xticklabels(["1,000 K", "2,000", "4,000", "6,500", "10,000\n(gas floor)",
                            "30,000", "100,000", "1,000,000"], color="white")  # fmt: skip
        ax.set_title(title, color="white", fontsize=10)
        for s in ax.spines.values():
            s.set_color("grey")
    fig.suptitle("Phase 9: true colour of glowing gas (CIE 1931 colour + Planck's law)",
                 color="white")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, facecolor="black")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
