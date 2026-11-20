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

Several processes on one machine (a cloud GPU: the per-frame CPU work is single-threaded):

    ... configs/<trailer>.toml tag=... probes_only=1      # folder, plans, exposures; no frames
    ... resume=<folder> worker=0/4  (and 1/4, 2/4, 3/4)   # each draws frames i with i % 4 == k
    ... resume=<folder>                                   # all frames exist: writes the video

A worker never writes the manifest (they run at once); each records itself in
workers/<k>.json (start, end, git commit, frames drawn) and stops if exposures are missing.

exposures_from=<old folder>: take each shot's exposure (base and automatic curve) from that
render's exposures.json instead of running the probes. The caller vouches that nothing that
shapes the gas exposure changed (the probes measure the gas only; e.g. [stars] settings or a
higher scale / n_sub do not change it); the manifest records it.

reuse_frames=<old folder>:<shot>:<N>: frames 0..N-1 of that shot are hard-linked from the old
render when they would come out the same: the old and new frame plans agree there (times,
camera, stops) and, after this render's exposure, each frame's gas and star exposure is within
1e-3 stops of the old one. The caller vouches that nothing else that shapes those frames
changed (e.g. a [bricks] setting that only steps in later). Otherwise it stops with the reason.
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


def _link_frames(spec, folder, name, plan, rend, exposures):
    """reuse_frames=<old>:<shot>:<N> (docstring): link frames 0..N-1 if they would match."""
    old_s, shot_name, n_s = spec.rsplit(":", 2)
    if shot_name != name:
        return
    old, n = Path(old_s).resolve(), int(n_s)
    old_plan = np.load(old / "plans" / f"{name}.npz")
    new_plan = np.load(folder / "plans" / f"{name}.npz")
    for k in old_plan.files:
        a, b = old_plan[k], new_plan[k]
        if a.shape[1:] != b.shape[1:] or not np.array_equal(a[:n], b[:n]):
            raise SystemExit(f"reuse_frames: plan '{k}' differs in frames < {n}")
    oe = json.loads((old / "exposures.json").read_text())[name]
    ne = exposures[name]
    old_gas = np.log2(oe[0]) + np.asarray(oe[2][:n] if len(oe) > 2 else np.zeros(n))
    new_gas = np.log2(ne[0]) + np.asarray(ne[2][:n] if len(ne) > 2 else np.zeros(n))
    d_gas = float(np.abs(old_gas - new_gas).max())
    d_star = abs(float(np.log2(oe[1] / ne[1])))
    print(f"  reuse_frames: exposure differs by up to {d_gas:.2e} stops (gas), "
          f"{d_star:.2e} (stars) in frames < {n}", flush=True)  # fmt: skip
    if d_gas > 1e-3 or d_star > 1e-3:
        raise SystemExit("reuse_frames: exposure differs by more than 1e-3 stops; render them")
    fdir = folder / "frames" / name
    for i in range(n):
        src = old / "frames" / name / f"{i:05d}.png"
        if not src.exists():
            raise SystemExit(f"reuse_frames: {src} missing")
        if not (fdir / f"{i:05d}.png").exists():
            os.link(src, fdir / f"{i:05d}.png")
    update(folder, reused_frames={name: {"from": str(old.relative_to(ROOT)), "frames": n,
                                         "max_exposure_diff_stops": d_gas}})  # fmt: skip
    print(f"  {name}: frames 0-{n - 1} linked from {old.name}", flush=True)


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
    worker = None
    if "worker" in opts:  # "k/N": this process draws frames i with i % N == k
        k_s, n_s = opts["worker"].split("/")
        worker = (int(k_s), int(n_s))
        if not 0 <= worker[0] < worker[1]:
            raise SystemExit(f"worker={opts['worker']}: need 0 <= k < N")
        if "resume" not in opts:
            raise SystemExit("worker= needs resume=<folder> (run probes_only=1 first)")
    if "resume" in opts:
        folder = Path(opts["resume"]).resolve()
        m = json.loads((folder / "manifest.json").read_text())
        # probes_only describes the run that made the folder, not this one: carried over, it
        # made the final "resume=<folder>" stop after the probes instead of writing the video
        # (AI_LOG 2026-11-20)
        opts = {k: v for k, v in m["options"].items() if k != "probes_only"}
        cfg = tomllib.loads(m["trailer"])
        shot_texts = m["shots"]
        if worker is None:
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
        if opts.get("exposures_from") and name not in exposures:
            src = ROOT / opts["exposures_from"] / "exposures.json"
            exposures[name] = json.loads(src.read_text())[name]
            done = json.loads((folder / "manifest.json").read_text()).get("exposures_from", {})
            update(folder, exposures_from={**done, name: opts["exposures_from"]})
            print(f"  {name}: exposure taken from {opts['exposures_from']}", flush=True)
        if worker is not None and name not in exposures:
            raise SystemExit(f"worker: shot {name} has no exposure yet (run probes_only=1)")
        if name in exposures:
            rend.base = Exposure(*exposures[name][:2])
            if len(exposures[name]) > 2:
                rend.adapt = np.asarray(exposures[name][2])
        elif name in match:
            if match[name] not in exposures:
                raise SystemExit(f"{name}: match_exposure {match[name]!r} must be rendered first")
            rend.base = Exposure(*exposures[match[name]])
        elif shot.auto_exposure is not None:
            a = shot.auto_exposure
            rend.adapt_exposure(float(a.get("every", 1.0)), float(a.get("smooth", 4.0)),
                                float(a.get("strength", 0.75)))  # fmt: skip
            print(f"  {name}: automatic exposure from probes, stops "
                  f"{rend.adapt.min():+.1f} to {rend.adapt.max():+.1f}", flush=True)  # fmt: skip
        else:
            rend.calibrate()
        exposures[name] = [rend.base.gas, rend.base.stars]
        if shot.auto_exposure is not None:
            exposures[name].append([float(v) for v in rend.adapt])
        exp_path.write_text(json.dumps(exposures, indent=2))
        if opts.get("reuse_frames") and worker is None:
            _link_frames(opts["reuse_frames"], folder, name, plan, rend, exposures)
        if opts.get("probes_only") and worker is None:
            print(f"  {name}: exposures written (probes_only)", flush=True)
            continue
        drawn = 0
        for i in range(n):
            if worker is not None and i % worker[1] != worker[0]:
                continue
            out = fdir / f"{i:05d}.png"
            if out.exists():
                continue  # resumed: already rendered
            tmp_png = fdir / f"{i:05d}.tmp{'' if worker is None else worker[0]}.png"
            write_png(tmp_png, rend.frame(i))
            tmp_png.rename(out)  # complete frames only (a crash mid-write leaves a .tmp)
            drawn += 1
            if i % 24 == 0 or (worker is not None and drawn % 6 == 0):
                print(f"  {name}: frame {i + 1}/{n}, {time.perf_counter() - t0:.0f} s", flush=True)
        order.append((name, n))
    if opts.get("probes_only") and worker is None:
        update(folder, probes_done=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        print("probes_only: done; start the workers with resume=<folder> worker=k/N", flush=True)
        return
    if worker is not None:
        wdir = folder / "workers"
        wdir.mkdir(exist_ok=True)
        (wdir / f"{worker[0]}.json").write_text(json.dumps({
            "worker": f"{worker[0]}/{worker[1]}", "git": _git(), "machine": platform.node(),
            "end": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": time.perf_counter() - t0},
            indent=2))  # fmt: skip
        print(f"worker {worker[0]}/{worker[1]} done, {time.perf_counter() - t0:.0f} s", flush=True)
        return
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
