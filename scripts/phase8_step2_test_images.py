"""Phase 8 step 2 check images: the CPU reference marcher on scenes with exact answers
(phases/phase-08-gas-rendering.md 8.6 tests 3-5), straight parallel rays, no hole.

    uv run python scripts/phase8_step2_test_images.py [n=64]

Writes outputs/checks/phase8_step2_test_images.png: images of a thin glowing sphere and an
opaque sphere, their brightness profiles against the exact answers, and the slab test.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.physics.radiative_transfer import blackbody_intensity_si
from nebula.render.gas_scenes import sphere_volume_fraction
from nebula.render.gasgrid import CartesianBox
from nebula.render.march import GreyBlackbody, march_segment, texture_from_fields

ROOT = Path(__file__).resolve().parents[1]
BOX = CartesianBox(centre=(0.0, 0.0, 0.0), size=100.0, n=128)
R, RHO0, T0 = 40.0, 1e-8, 1e4


def image(tex, model, n):
    xs = np.linspace(-48.0, 48.0, n)
    img = np.zeros((n, n))
    for a, y in enumerate(xs):
        for b, x in enumerate(xs):
            img[a, b] = march_segment([None, tex], model, (x, y, -49.0), (x, y, 49.0))[0]
    return xs, img


def main():
    n = int(dict(a.split("=") for a in sys.argv[1:]).get("n", 64))
    s0 = float(blackbody_intensity_si(T0))
    tex = texture_from_fields(BOX, RHO0 * sphere_volume_fraction(BOX, R), np.full(BOX.shape, T0))
    thin, opaque = GreyBlackbody(kappa=1.0), GreyBlackbody(kappa=10.0 / RHO0)
    j = RHO0 * s0

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    xs, im_thin = image(tex, thin, n)
    _, im_opq = image(tex, opaque, n)
    ax[0, 0].imshow(im_thin / (2 * j * R), extent=[xs[0], xs[-1]] * 2, cmap="inferno")
    ax[0, 0].set_title(
        "test 3: see-through glowing ball\n(brighter where the ray crosses more gas)"
    )
    ax[1, 0].imshow(im_opq / s0, extent=[xs[0], xs[-1]] * 2, cmap="inferno", vmin=0, vmax=1)
    ax[1, 0].set_title("test 4: solid (opaque) ball\n(a flat disk: only the surface is seen)")

    bs = np.linspace(0.0, 47.0, 400)
    prof_t = np.array(
        [march_segment([None, tex], thin, (b, 0.3, -49), (b, 0.3, 49))[0] for b in bs]
    )
    prof_o = np.array([march_segment([None, tex], opaque, (b, 0, -49), (b, 0, 49))[0] for b in bs])
    exact_t = np.where(bs < R, 2 * j * np.sqrt(np.clip(R**2 - bs**2 - 0.09, 0, None)), 0.0)
    exact_o = np.where(bs < R, s0, 0.0)
    ax[0, 1].plot(bs, exact_t / (2 * j * R), "w-", lw=4, label="exact: j 2 sqrt(R^2 - b^2)")
    ax[0, 1].plot(bs, prof_t / (2 * j * R), "C1--", lw=1.5, label="marcher")
    ax[0, 1].set_title("test 3 profile (brightness vs distance from centre)")
    ax[1, 1].plot(bs, exact_o / s0, "w-", lw=4, label="exact: S inside, 0 outside")
    ax[1, 1].plot(bs, prof_o / s0, "C1--", lw=1.5, label="marcher")
    ax[1, 1].set_title("test 4 profile")
    m = bs < 0.95 * R
    err_t = np.max(np.abs(prof_t[m] / exact_t[m] - 1))
    inside = bs < R - 2 * BOX.dx
    err_o = np.max(np.abs(prof_o[inside] / s0 - 1))
    ax[0, 1].text(
        1,
        0.15,
        f"largest error for b < 0.95 R: {100 * err_t:.2f}%  (limit 1%)",
        color="w",
        fontsize=11,
    )
    ax[1, 1].text(
        1, 0.15, f"largest error inside: {100 * err_o:.3f}%  (limit 1%)", color="w", fontsize=11
    )

    taus = np.array([0.1, 1.0, 10.0])
    got = []
    for tau in taus:
        t2 = texture_from_fields(BOX, np.full(BOX.shape, tau / 60.0), np.full(BOX.shape, T0))
        got.append(march_segment([None, t2], thin, (0, 0, -30), (0, 0, 30))[0] / s0)
    tt = np.geomspace(0.01, 30, 200)
    ax[0, 2].semilogx(tt, -np.expm1(-tt), "w-", lw=3, label="exact: S (1 - e^-tau)")
    ax[0, 2].semilogx(taus, got, "o", color="C1", ms=10, label="marcher")
    for tau, g in zip(taus, got, strict=True):
        ax[0, 2].text(tau * 1.15, g - 0.07, f"{100 * (g / -np.expm1(-tau) - 1):+.3f}%", color="w")
    ax[0, 2].set_title("test 5: gas slab of optical thickness tau")
    ax[1, 2].axis("off")
    ax[1, 2].text(
        0.0,
        0.5,
        "All three are exact-answer tests with straight rays\n"
        "(no black hole yet; light bending is step 3).\n\n"
        "Orange dashes should sit on the white lines.",
        fontsize=13,
    )
    for a in ax.flat[[1, 2, 4]]:
        a.set_facecolor("k")
        a.legend(facecolor="0.2", labelcolor="w")
        a.set_xlabel("b (distance from centre)" if a is not ax[0, 2] else "tau")
    ax[0, 2].set_xlabel("optical thickness tau")
    out = ROOT / "outputs" / "checks" / "phase8_step2_test_images.png"
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    print(f"thin sphere max error {100 * err_t:.3f}%, opaque {100 * err_o:.4f}%, slab {got}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
