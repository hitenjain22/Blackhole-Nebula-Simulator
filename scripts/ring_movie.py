"""Quick-look movie of the fast-cooling ring forming: top and edge-on gas maps, one frame per
Phantom dump of R6.6_100k_cool_b1 (beta 1). Analysis picture for Hiten, not the final render.

    uv run --no-sync --with sarracen --with scikit-learn python scripts/ring_movie.py OUT.mp4 \
        FIRST LAST [half_width=200] [fps=6]

Views are in the plane of the gas near the hole (its total angular momentum, fixed from the last
dump so the camera does not wobble); colour is log column density (particle counts), the same
scale in every frame."""

import sys
from pathlib import Path

import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cool_compare import CODE_DAY, RUNS, load  # noqa: E402

RUN = RUNS["fast cooling (beta 1)"]


def plane_axes(x, p):
    r = np.linalg.norm(x, axis=1)
    sel = r < 400
    nz = np.cross(x[sel], p[sel]).sum(axis=0)
    nz /= np.linalg.norm(nz)
    ex = np.cross([0.0, 0.0, 1.0], nz)
    ex = ex / np.linalg.norm(ex) if np.linalg.norm(ex) > 1e-6 else np.array([1.0, 0, 0])
    return ex, np.cross(nz, ex), nz


def main():
    out, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    o = dict(a.split("=", 1) for a in sys.argv[4:])
    hw, fps = float(o.get("half_width", 200)), int(o.get("fps", 6))
    dumps = [d for d in range(first, last + 1) if (RUN / f"main100k_{d:05d}").exists()]
    _, x, p, _ = load(RUN / f"main100k_{dumps[-1]:05d}")
    ex, ey, nz = plane_axes(x, p)
    w = iio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    for d in dumps:
        t, x, _, temp = load(RUN / f"main100k_{d:05d}")
        a, b, c = x @ ex, x @ ey, x @ nz
        fig, (top, side) = plt.subplots(2, 1, figsize=(6.4, 9.6), facecolor="black",
                                        gridspec_kw={"height_ratios": [2, 1]})  # fmt: skip
        for ax, (u, v), ext in ((top, (a, b), hw), (side, (a, c), hw / 2)):
            hst, _, _ = np.histogram2d(u, v, bins=(320, int(320 * ext / hw)),
                                       range=[[-hw, hw], [-ext, ext]])  # fmt: skip
            ax.imshow(np.log10(hst.T + 0.5), origin="lower", extent=(-hw, hw, -ext, ext),
                      cmap="inferno", vmin=-0.3, vmax=2.5, interpolation="nearest")  # fmt: skip
            ax.plot(0, 0, "+", color="cyan", ms=8)
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
        near = np.linalg.norm(x, axis=1) < 100
        title = (f"fast cooling (beta 1)   day {t / CODE_DAY:.1f}\ntop view, {2 * hw:.0f} wide; "
                 "dashed: predicted ring radius 50.9")  # fmt: skip
        top.set_title(title, color="white", fontsize=10)
        top.add_patch(plt.Circle((0, 0), 50.89, fill=False, ls="--", lw=0.6, color="cyan",
                                 alpha=0.5))  # fmt: skip
        side.set_title(f"edge-on | gas within r 100: {near.sum()}, median T "
                       f"{np.median(temp[near]) if near.any() else 0:,.0f} K", color="white",
                       fontsize=9)  # fmt: skip
        fig.tight_layout()
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
        print(f"dump {d} day {t / CODE_DAY:.1f}", flush=True)
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
