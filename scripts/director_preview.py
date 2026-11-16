"""Preview a shot (Phase 8 step 6, phases/phase-08-gas-rendering.md 8.5): the camera path as
a picture, and a quick low-resolution clip.

    uv run python scripts/director_preview.py configs/shots/orbit_pass.toml [scale=0.5]
        [every=1] [n_sub=1] [clip=1]

Writes outputs/checks/director_<shot>_path.png (top and side views of the camera path round
the hole, with the gas at the middle of the shot; distance from the hole and simulation day
against video time), outputs/checks/director_<shot>_plan.npz (the frame list) and, with
clip=1, outputs/checks/director_<shot>_preview.mp4 (every `every`-th frame at `scale` x the
shot's resolution, played at fps / every so the motion keeps its speed).
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nebula.render.director import CODE_DAY, Shot, Timeline, plan_frames
from nebula.render.shot_render import ShotRenderer, camera_gas_clearance
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks"
CLOSE_GAS = 10.0  # GM/c^2: closer gas shows as big blurry foreground blobs (orbit_pass draft 1)


def path_picture(shot, plan, timeline, out):
    mid = timeline.paths[int(np.searchsorted(timeline.times, plan.sim_t[len(plan.sim_t) // 2]))
                         - 1]  # fmt: skip
    gas = np.load(mid)["x"].astype(float)
    near = np.linalg.norm(gas, axis=1) < 3 * np.abs(plan.position).max()
    gas = gas[near][:: max(1, near.sum() // 20000)]
    p, vt = plan.position, plan.video_t
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for a, (i, j, lab) in zip(ax[0], ((0, 1, "top view (x, y)"), (0, 2, "side view (x, z)")),
                              strict=True):  # fmt: skip
        a.scatter(gas[:, i], gas[:, j], s=0.3, c="0.75", label="gas (middle of the shot)")
        sc = a.scatter(p[:, i], p[:, j], c=vt, s=6, cmap="viridis", label="camera")
        a.plot(plan.look_at[:, i], plan.look_at[:, j], "r-", lw=1, label="looks at")
        for k in range(0, len(vt), shot.fps):  # an arrow each second: where it looks
            d = plan.look_at[k] - p[k]
            d = d / np.linalg.norm(d) * 0.25 * np.linalg.norm(p[k])
            a.annotate("", p[k, [i, j]] + d[[i, j]], p[k, [i, j]],
                       arrowprops={"arrowstyle": "->", "color": "C1", "lw": 0.8})  # fmt: skip
        a.plot(0, 0, "k*", ms=14, label="black hole")
        lim = 1.15 * np.abs(p[:, [i, j]]).max()
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_title(lab)
        a.legend(loc="lower left", fontsize=8)
        fig.colorbar(sc, ax=a, label="video time (s)")
    ax[1, 0].plot(vt, np.linalg.norm(p, axis=1))
    ax[1, 0].axhline(6, color="r", ls=":", label="closest allowed (6)")
    ax[1, 0].set_xlabel("video time (s)")
    ax[1, 0].set_ylabel("camera distance from the hole (GM/c^2)")
    ax[1, 0].legend()
    ax[1, 1].plot(vt, plan.sim_t / CODE_DAY)
    ax[1, 1].set_xlabel("video time (s)")
    ax[1, 1].set_ylabel("simulation day")
    fig.suptitle(f"shot {shot.name}: {len(vt)} frames, {shot.duration:g} s at {shot.fps} fps")
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    plt.close(fig)


def main():
    shot_path = Path(sys.argv[1])
    o = dict(a.split("=") for a in sys.argv[2:])
    scale, every = float(o.get("scale", 0.5)), int(o.get("every", 1))
    shot = Shot.load(shot_path)
    timeline = Timeline.from_sources(shot.sources, ROOT / "runs")
    plan = plan_frames(shot, timeline)
    plan.save(OUT / f"director_{shot.name}_plan.npz")
    path_picture(shot, plan, timeline, OUT / f"director_{shot.name}_path.png")
    print(f"wrote outputs/checks/director_{shot.name}_path.png and _plan.npz "
          f"({len(plan.video_t)} frames)")  # fmt: skip
    dist, inside = camera_gas_clearance(plan, timeline)
    close = dist < CLOSE_GAS
    print(f"nearest gas to the camera: {dist.min():.1f} (at video t "
          f"{plan.video_t[np.argmin(dist)]:.2f} s)")  # fmt: skip
    if close.any() and not inside.any():
        ts = plan.video_t[close]
        print(f"NOTE: gas within {CLOSE_GAS:g} of the camera in {close.sum()} frames "
              f"({ts.min():.2f}-{ts.max():.2f} s): expect blurry foreground gas")  # fmt: skip
    if inside.any():
        ts = plan.video_t[inside]
        print(f"WARNING: camera inside gas in {inside.sum()} frames (video t {ts.min():.2f}-"
              f"{ts.max():.2f} s): raise or move the camera there")  # fmt: skip
    else:
        print("camera never inside gas")
    if o.get("clip", "1") != "1":
        return
    w = 2 * round(shot.width * scale / 2)
    h = 2 * round(shot.height * scale / 2)
    rend = ShotRenderer(shot, plan, timeline, width=w, height=h, n_sub=int(o.get("n_sub", 1)))
    t0 = time.perf_counter()
    rend.calibrate()
    frames = []
    idx = range(0, len(plan.video_t), every)
    for n, i in enumerate(idx):
        frames.append(rend.frame(i))
        if n % 24 == 0:
            print(f"  frame {i} (day {plan.sim_t[i] / CODE_DAY:.2f}): "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)  # fmt: skip
    out = OUT / f"director_{shot.name}_preview.mp4"
    write_video(frames, out, fps=max(1, round(shot.fps / every)))
    print(f"wrote {out.relative_to(ROOT)}: {len(frames)} frames at {w}x{h}, "
          f"{(time.perf_counter() - t0) / len(frames):.1f} s per frame")  # fmt: skip


if __name__ == "__main__":
    main()
