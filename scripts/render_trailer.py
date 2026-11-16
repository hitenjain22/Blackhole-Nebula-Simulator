"""Render a sequence of shots into one video with dissolves (Phase 8 step 6).

    uv run python scripts/render_trailer.py configs/trailer.toml [scale=1] [n_sub=] [tag=]
    uv run python scripts/render_trailer.py resume=outputs/renders/<folder>
    uv run python scripts/render_trailer.py configs/trailer.toml ... reuse=outputs/renders/<old>

Trailer file:

    name = "trailer_v1"
    dissolve = 0.5                      # seconds of cross-fade between shots
    shots = ["approach.toml", "orbit_pass.toml", "aftermath.toml"]   # in configs/shots/
    [match_exposure]                    # optional: shot -> shot whose base exposure it reuses
    aftermath = "orbit_pass"            # (keeps brightness continuous across that dissolve)

Writes outputs/renders/<name>_<tag>_<YYYYmmdd-HHMMSS>/ (never reuses a folder: rule 8):
manifest.json (git commit, the trailer and shot files' text, settings, library versions,
machine, start/end and every resume), plans/<shot>.npz (frame lists), exposures.json,
frames/<shot>/NNNNN.png and <name>.mp4. Every frame is written as it is finished, so a crash or
reboot loses at most one frame: run again with resume=<folder> to continue (overnight runs).

reuse=<old folder>: a shot whose text, scale and n_sub are exactly those of the old render (and
which the old render finished) is not rendered again: its frames are hard-linked (no copy, no
disk space) and its exposure taken over, so after changing one shot only that shot is
re-rendered. The old folder is not touched; the manifest records what was reused.
"""

import json
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nebula.render.director import Shot, Timeline, plan_frames
from nebula.render.frames import write_png
from nebula.render.gas_frames import Exposure
from nebula.render.shot_render import ShotRenderer
from nebula.render.video import write_video
from nebula.sim.runio import _git, _versions

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "configs" / "shots"
RENDERS = ROOT / "outputs" / "renders"


def new_folder(cfg, opts, trailer_text, shot_texts) -> Path:
    tag = opts.get("tag", f"x{opts.get('scale', '1')}")
    folder = RENDERS / f"{cfg['name']}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": cfg["name"], "git": _git(), "options": opts, "trailer": trailer_text,
        "shots": shot_texts, "versions": _versions(),
        "machine": {"platform": platform.platform(), "machine": platform.machine()},
        "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "end": None, "resumed": [],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


def _reusable(old, old_m, old_exp, name, text, opts, n) -> bool:
    """The old render made exactly this shot (same text, scale, n_sub) and finished it."""
    same = all(str(old_m["options"].get(k)) == str(opts.get(k)) for k in ("scale", "n_sub"))
    done = all((old / "frames" / name / f"{i:05d}.png").exists() for i in range(n))
    return same and old_m["shots"].get(name) == text and name in old_exp and done


def update(folder, **fields):
    path = folder / "manifest.json"
    m = json.loads(path.read_text())
    for k, v in fields.items():
        if k == "reused":
            m.setdefault("reused", {}).update(v)
        elif k == "resumed":
            m["resumed"].append({"time": v, "git": _git()})
        else:
            m[k] = v
    path.write_text(json.dumps(m, indent=2))


def main():
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if "resume" in opts:
        folder = Path(opts["resume"]).resolve()
        m = json.loads((folder / "manifest.json").read_text())
        opts = m["options"]
        cfg = tomllib.loads(m["trailer"])
        shot_texts = m["shots"]
        update(folder, resumed=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    else:
        tpath = Path(sys.argv[1])
        trailer_text = tpath.read_text()
        cfg = tomllib.loads(trailer_text)
        shot_texts = {Path(s).stem: (SHOTS / s).read_text() for s in cfg["shots"]}
        folder = new_folder(cfg, opts, trailer_text, shot_texts)
    print(f"render folder: {folder.relative_to(ROOT)}", flush=True)
    scale = float(opts.get("scale", 1))
    match = cfg.get("match_exposure", {})
    exp_path = folder / "exposures.json"
    exposures = json.loads(exp_path.read_text()) if exp_path.exists() else {}
    (folder / "plans").mkdir(exist_ok=True)
    order, fps = [], None
    t0 = time.perf_counter()
    old = Path(opts["reuse"]).resolve() if opts.get("reuse") else None
    old_m = json.loads((old / "manifest.json").read_text()) if old else None
    old_exp = json.loads((old / "exposures.json").read_text()) if old else {}
    for name, text in shot_texts.items():
        tmp = folder / "plans" / f"{name}.toml"
        tmp.write_text(text)  # the shot exactly as recorded in the manifest
        shot = Shot.load(tmp)
        if fps is not None and shot.fps != fps:
            raise SystemExit(f"shot {name}: fps {shot.fps} differs from {fps}")
        fps = shot.fps
        timeline = Timeline.from_sources(shot.sources, ROOT / "runs")
        plan = plan_frames(shot, timeline)
        plan.save(folder / "plans" / f"{name}.npz")
        w, h = 2 * round(shot.width * scale / 2), 2 * round(shot.height * scale / 2)
        n_sub = int(opts["n_sub"]) if opts.get("n_sub") else shot.n_sub
        rend = ShotRenderer(shot, plan, timeline, width=w, height=h, n_sub=n_sub)
        n = len(plan.video_t)
        fdir = folder / "frames" / name
        fdir.mkdir(parents=True, exist_ok=True)
        if old is not None and name not in exposures and _reusable(old, old_m, old_exp, name,
                                                                   text, opts, n):  # fmt: skip
            for i in range(n):
                os.link(old / "frames" / name / f"{i:05d}.png", fdir / f"{i:05d}.png")
            exposures[name] = old_exp[name]
            update(folder, reused={name: str(old.relative_to(ROOT))})
            print(f"  {name}: {n} frames reused from {old.name}", flush=True)
        if name in exposures:
            rend.base = Exposure(*exposures[name])
        elif name in match:
            if match[name] not in exposures:
                raise SystemExit(f"{name}: match_exposure {match[name]!r} must be rendered first")
            rend.base = Exposure(*exposures[match[name]])
        else:
            rend.calibrate()
        exposures[name] = [rend.base.gas, rend.base.stars]
        exp_path.write_text(json.dumps(exposures, indent=2))
        for i in range(n):
            out = fdir / f"{i:05d}.png"
            if out.exists():
                continue  # resumed: already rendered
            tmp_png = fdir / f"{i:05d}.tmp.png"
            write_png(tmp_png, rend.frame(i))
            tmp_png.rename(out)  # complete frames only (a crash mid-write leaves a .tmp)
            if i % 24 == 0:
                print(f"  {name}: frame {i + 1}/{n}, {time.perf_counter() - t0:.0f} s", flush=True)
        order.append((name, n))
    # assemble: dissolves of `dissolve` seconds overlapping the shots' ends; frames are read
    # one at a time (a 1080p trailer held in memory would be ~4 GB)
    d = int(round(float(cfg.get("dissolve", 0.5)) * fps))
    count = [0]

    def stream():
        for k, (name, n) in enumerate(order):
            path = folder / "frames" / name
            last = order[k + 1][1] if k + 1 < len(order) else 0
            keep = min(d, n, last)  # this shot's last `keep` frames dissolve into the next
            start = min(d, n, order[k - 1][1]) if k > 0 else 0
            for i in range(start, n - keep):
                count[0] += 1
                yield iio.imread(path / f"{i:05d}.png")
            nxt = folder / "frames" / order[k + 1][0] if keep else None
            for j in range(keep):
                w = (j + 1) / (keep + 1)
                a = iio.imread(path / f"{n - keep + j:05d}.png").astype(float)
                b = iio.imread(nxt / f"{j:05d}.png").astype(float)
                count[0] += 1
                yield ((1 - w) * a + w * b).round().astype(np.uint8)

    video = folder / f"{cfg['name']}.mp4"
    write_video(stream(), video, fps=fps)
    update(folder, end=time.strftime("%Y-%m-%d %H:%M:%S %Z"), frames=count[0])
    print(f"wrote {video.relative_to(ROOT)}: {count[0]} frames ({count[0] / fps:.1f} s), "
          f"{time.perf_counter() - t0:.0f} s this session")  # fmt: skip


if __name__ == "__main__":
    main()
