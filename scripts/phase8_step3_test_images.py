"""Phase 8 step 3 check images: gas marched along bent light rays on the GPU
(phases/phase-08-gas-rendering.md 8.6 tests 6 and 7).

    uv run python scripts/phase8_step3_test_images.py

Writes outputs/checks/phase8_step3_rings.png (a small glowing ball hidden directly behind the
hole, seen as rings, with the exact ring angles drawn on) and
outputs/checks/phase8_step3_gpu_vs_cpu.png (a real snapshot: GPU image, CPU reference image and
their difference).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.physics.geodesics import schw_shadow_angle
from nebula.render.bent_rays import render_reference
from nebula.render.camera import Camera
from nebula.render.gas_scenes import point_source_textures, ring_angle_deg
from nebula.render.gpu_march import render_gas, upload_scene
from nebula.render.march import GreyBlackbody
from nebula.render.snapshot_scene import snapshot_textures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
RINGS = [
    (50.0, 30.0, 11.701931, 5.849751),
    (50.0, 100.0, 14.958880, 5.850477),
    (30.0, 15.0, 15.114892, 9.639931),
]
FOV, N = 36.0, 512


def rings():
    fig, ax = plt.subplots(2, 3, figsize=(16, 11))
    for k, (d_cam, r_src, prim, sec) in enumerate(RINGS):
        scene = upload_scene(point_source_textures(r_src), GreyBlackbody(kappa=1e-3))
        cam = Camera(position=(d_cam, 0.0, 0.0), vfov_deg=FOV, width=N, height=N)
        img, tm = render_gas(cam, scene, n_sub=2)
        i = img[..., 0]
        sh = np.degrees(schw_shadow_angle(d_cam))
        got = ring_angle_deg(cam, i, sh + 0.3, 90.0)
        half = np.degrees(np.arctan(np.tan(np.radians(FOV / 2))))
        a = ax[0, k]
        a.imshow(np.sqrt(i / i.max()), cmap="inferno", extent=[-half, half, -half, half])
        t = np.linspace(0, 2 * np.pi, 400)
        for ang, style, lab in ((prim, "c--", "exact main ring"), (sh, "w:", "shadow edge")):
            rr = np.degrees(np.arctan(np.tan(np.radians(ang))))  # gnomonic: same at centre
            a.plot(rr * np.cos(t), rr * np.sin(t), style, lw=0.8, label=lab)
        a.set_title(
            f"camera r = {d_cam:g}, ball at r = {r_src:g} behind the hole\n"
            f"main ring: exact {prim:.4f} deg, ours {got:.4f} deg "
            f"({(got - prim) / (FOV / N):+.3f} pixel)"
        )
        a.set_xlabel("degrees")
        a.legend(loc="lower right", fontsize=8)
        # Radial profile through the zoomed second ring
        p0 = np.radians(sec)
        fwd = np.array([-np.cos(p0), 0.0, np.sin(p0)])
        zfov = 0.008
        zcam = Camera(
            position=(d_cam, 0.0, 0.0),
            look_at=tuple(np.array([d_cam, 0.0, 0.0]) + fwd),
            vfov_deg=zfov,
            width=4,
            height=1024,
        )
        z, _ = render_gas(zcam, scene, n_sub=2)
        rows = np.arange(1024) + 0.5
        dirs = zcam.directions(np.full(1024, 2.0), rows)
        psi = np.degrees(np.arccos(np.clip(dirs @ np.array([-1.0, 0, 0]), -1, 1)))
        zgot = ring_angle_deg(zcam, z[..., 0], sh, sh + 1.0)
        b = ax[1, k]
        b.plot((psi - sec) * 3600, z[:, 2, 0] / max(z[:, 2, 0].max(), 1e-30), "C1-", lw=1.5)
        b.axvline(0, color="c", ls="--", label="exact second ring")
        b.axvline((sh - sec) * 3600, color="k", ls=":", label="shadow edge")
        b.set_xlabel("angle from the exact second-ring position (arcseconds)")
        b.set_title(
            f"second ring, zoomed 4500x\nexact {sec:.6f} deg, ours {zgot:.6f} deg\n"
            f"off by {(zgot - sec) * 3600:+.3f} arcsec = {(zgot - sec) / (FOV / N):+.5f} pixel",
            fontsize=10,
        )
        b.legend(fontsize=8)
        print(f"D {d_cam:g} r {r_src:g}: main {got:.5f} vs {prim}, second {zgot:.6f} vs {sec}; "
              f"slowest tile {tm.max:.2f} s")  # fmt: skip
    fig.suptitle(
        "Test 6: a small glowing ball directly behind the black hole, seen through its gravity "
        "(GPU). Light bends around the hole and the ball appears as a ring;\n"
        "a much fainter second ring comes from light that loops once more around the hole. "
        "Limit: within 1 pixel (0.07 deg) of the exact angles.",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUT / "phase8_step3_rings.png", dpi=80)
    plt.close(fig)


def gpu_vs_cpu():
    textures, _ = snapshot_textures(ROOT / "runs" / "R6.1_100k", 150)
    model = GreyBlackbody(kappa=1e7)
    cam = Camera(position=(60.0, 0.0, 10.0), vfov_deg=90.0, width=64, height=64)
    gpu, _ = render_gas(cam, upload_scene(textures, model), n_sub=1)
    pix = [(i, j) for j in range(1, 64, 4) for i in range(2, 64, 4)]
    cpu = render_reference(cam, textures, model, pixels=pix)
    big = Camera(position=(60.0, 0.0, 10.0), vfov_deg=90.0, width=512, height=512)
    hi, tm = render_gas(big, upload_scene(textures, model), n_sub=2)
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    # Brightness spans many powers of ten (hot gas near the hole): show log10 of I.
    lo, top = np.percentile(np.log10(hi[..., 0][hi[..., 0] > 0]), [2, 99.9])
    ax[0].imshow(np.log10(np.maximum(hi[..., 0], 1e-30)), cmap="inferno", vmin=lo, vmax=top)
    ax[0].set_title(
        "real snapshot (day 273), GPU, 512 x 512, log brightness\ncamera r = 61 just above the "
        "orbit plane, 90 deg view, looking at the hole"
    )
    ax[1].imshow(np.log10(np.maximum(gpu[..., 0], 1e-30)), cmap="inferno", vmin=lo, vmax=top)
    rows = np.array([p[1] for p in pix])
    cols = np.array([p[0] for p in pix])
    ax[1].plot(cols, rows, "c.", ms=2)
    ax[1].set_title("same view, GPU 64 x 64 (test 7)\ncyan dots: pixels also done by the CPU")
    rel = np.abs(gpu[rows, cols, 0] / cpu[rows, cols, 0] - 1)
    sc = ax[2].scatter(cols, rows, c=np.log10(np.maximum(rel, 1e-8)), cmap="viridis", vmin=-7,
                       vmax=-3, s=40)  # fmt: skip
    ax[2].set_xlim(0, 64)
    ax[2].set_ylim(64, 0)
    ax[2].set_aspect("equal")
    ax[2].set_title(
        f"GPU vs CPU difference per pixel (log10)\nworst {rel.max():.1e}, limit 1e-3 (= -3 on "
        f"the colour bar)"
    )
    fig.colorbar(sc, ax=ax[2], label="log10 |GPU / CPU - 1|")
    plt.tight_layout()
    plt.savefig(OUT / "phase8_step3_gpu_vs_cpu.png", dpi=80)
    print(f"GPU vs CPU worst {rel.max():.2e}; 512x512 x 4 rays: slowest tile {tm.max:.2f} s, "
          f"total {tm.total:.1f} s")  # fmt: skip


if __name__ == "__main__":
    rings()
    gpu_vs_cpu()
    print("wrote outputs/checks/phase8_step3_rings.png, phase8_step3_gpu_vs_cpu.png")
