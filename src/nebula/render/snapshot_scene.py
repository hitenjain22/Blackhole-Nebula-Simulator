"""Render textures from a saved run snapshot (phases/phase-08-gas-rendering.md 8.3): load the
particles, take temperatures from the run's equation of state, deposit onto the log grid
(and optional detail boxes) and build march.Texture objects."""

import json
from pathlib import Path

import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.gasgrid import CartesianBox, LogSphericalGrid, deposit_box, deposit_log
from nebula.render.march import Texture, texture_from_gasgrid


def snapshot_textures(
    run_dir: Path, snap: int, grid: LogSphericalGrid | None = None, boxes=()
) -> tuple[list[Texture], float]:
    """([log-grid texture, box textures...], snapshot time in code units).

    Particle mass: the cloud is 1 code mass unit split equally (sim/cloud.py), so
    m = 1 / N(snapshot 0)."""
    run_dir = Path(run_dir)
    eos, mass = run_eos_and_mass(run_dir)
    d = np.load(run_dir / "snapshots" / f"snap_{snap:05d}.npz")
    x, h = d["x"].astype(float), d["h"].astype(float)
    t = snapshot_temperature(eos, d["rho"].astype(float), d["u"].astype(float))
    m = np.full(len(h), mass)
    v = d["p"].astype(float)
    texs = textures_from_particles(x, h, m, t, v, grid, boxes)
    return texs, float(d["t"])


def run_eos_and_mass(run_dir: Path) -> tuple[str, float]:
    """(equation of state name, particle mass) of a run: the manifest's particle_mass (written
    by scripts/ph_to_snapshots.py from Phantom's massoftype) if present, else 1 / N(snapshot 0)
    (our runs: the cloud is 1 code mass unit split equally and snapshot 0 is the start)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    if "particle_mass" in config:
        return config.get("eos", "gasrad"), float(config["particle_mass"])
    n0 = len(np.load(run_dir / "snapshots" / "snap_00000.npz")["h"])
    return config.get("eos", "gasrad"), 1.0 / n0


def textures_from_particles(x, h, m, t, v, grid=None, boxes=()) -> list[Texture]:
    """[log-grid texture, box textures...] from particle arrays (code units; t in K)."""
    grid = grid or LogSphericalGrid()
    m = np.broadcast_to(np.asarray(m, float), (len(h),)).copy()
    texs = [texture_from_gasgrid(deposit_log(x, h, m, t, v, grid))]
    for b in boxes:
        texs.append(texture_from_gasgrid(deposit_box(x, h, m, t, v, CartesianBox(**b))))
    return texs
