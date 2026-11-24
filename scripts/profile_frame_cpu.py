"""Time the CPU stages of one story frame (ShotRenderer.picture without the GPU), for the
optimization review (2026-11-23). Mirrors picture() step by step.

    uv run python scripts/profile_frame_cpu.py FRAME [h_scale=1.5] [shot=configs/shots/full_story.toml]
"""

import sys
import time
from pathlib import Path

import numpy as np

import nebula.render.shot_render as SR
from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.physics.redshift import gn_static_frame_velocity
from nebula.render.director import CODE_DAY, GAS_SETTINGS, Shot, Timeline, plan_frames
from nebula.render.junctions import eased_state
from nebula.render.point_stars import gauss_stars
from nebula.render.snapshot_scene import textures_from_particles

ROOT = Path(__file__).resolve().parents[1]
args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
i = int(args[0])
shot = Shot.load(ROOT / opts.get("shot", "configs/shots/full_story.toml"))
tl = Timeline.from_sources(shot.sources, ROOT / "runs")
plan = plan_frames(shot, tl)
T = {}


def tick(name, t0):
    T[name] = T.get(name, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


t0 = time.perf_counter()
r = SR.ShotRenderer(shot, plan, tl, width=640, height=360, n_sub=1, sky=1)  # no GPU use
t0 = tick("renderer setup (once per process)", t0)
t = float(plan.sim_t[i])
k = min(max(int(np.searchsorted(tl.times, t, side="right")) - 1, 0), len(tl.times) - 2)
a, b = r._snap(k), r._snap(k + 1)
t0 = tick("load 2 snapshots", t0)
ids, x, p, s = eased_state(a, b, min(max(t, a.t), b.t), r.easer.at(t), r_drop=4.0, with_p=True)
t0 = tick("position gas (eased_state)", t0)
v = gn_static_frame_velocity(x, p)[1]
eos, mass = r._eos_mass(k)
temp = snapshot_temperature(eos, s["rho"], s["u"])
t0 = tick("velocities + temperatures", t0)
look = {**GAS_SETTINGS, **(shot.gas or {})}
h = s["h"] * float(opts.get("h_scale", look["h_scale"]))
mass = np.full(len(h), float(mass))
dp = r.disc(t)
t0 = tick("inner disc (incl. first-time edge feeds)", t0)
dp = r.disc(t)
t0 = tick("inner disc (again, cached feeds)", t0)
if dp is not None:
    x, h = np.concatenate([x, dp.x]), np.concatenate([h, dp.h * float(look["h_scale"])])
    mass, temp = np.concatenate([mass, dp.m]), np.concatenate([temp, dp.temp])
    v = np.concatenate([v, dp.gamma_beta])
texs = textures_from_particles(x, h, mass, temp, v, r.grid, r.plan.boxes(i, shot),
                               shot.brick_lattice(t / CODE_DAY))  # fmt: skip
t0 = tick("deposit onto grids (1st call, incl. Numba compile)", t0)
texs = textures_from_particles(x, h, mass, temp, v, r.grid, r.plan.boxes(i, shot),
                               shot.brick_lattice(t / CODE_DAY))  # fmt: skip
t0 = tick("deposit onto grids (2nd call)", t0)
node = np.zeros((362, 642, 4), np.float32)
node[..., 0], node[..., 3] = 1.0, 1.0
d = np.random.default_rng(0).normal(size=(362, 642, 3))
node[..., :3] = d / np.linalg.norm(d, axis=-1, keepdims=True)
size = np.radians(float(r.stars["size_deg"]))
gauss_stars(node[..., :3], node[..., 3], r.star_index, size)
t0 = tick("pinpoint stars (1st, incl. compile; random sky)", t0)
gauss_stars(node[..., :3], node[..., 3], r.star_index, size)
t0 = tick("pinpoint stars (2nd)", t0)
n_part = len(h)
bricks = texs[1] if len(texs) > 1 else None
print(f"frame {i} (t {i / shot.fps:.1f} s, day {t / CODE_DAY:.1f}): {n_part} particles, "
      f"h_scale {opts.get('h_scale', look['h_scale'])}, "
      f"bricks {0 if bricks is None else len(getattr(bricks, 'keys', []))}")
for name, sec in T.items():
    print(f"  {name:52s} {sec:7.2f} s")
