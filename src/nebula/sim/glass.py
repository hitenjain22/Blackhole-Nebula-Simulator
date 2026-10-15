"""Relaxed "glass" particle layouts (phases/phase-03-sph-core.md 3.2, initial conditions).

Random points give ~10% density noise and spurious pressure kicks. A glass is made once:
random particles in a periodic unit box, moved by SPH pressure forces (isothermal, sound
speed 1) with strong velocity damping, until the density noise std(rho)/mean(rho) is below
the target (1%). It is stored in data/ and tiled/stretched by ``tile_glass`` as needed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nebula.physics.eos import isothermal_pressure
from nebula.sim.ic import smoothing_length_uniform
from nebula.sim.sph import compute_density, forces_numba

DATA = Path(__file__).resolve().parents[3] / "data"


@dataclass
class GlassLog:
    steps: int
    noise: list[float]  # std(rho)/mean(rho) per step


def density_noise(pos: np.ndarray, box: float = 1.0) -> float:
    """std(rho)/mean(rho) of equal-mass particles in a periodic box."""
    n = len(pos)
    h0 = np.full(n, smoothing_length_uniform(n, box**3))
    d = compute_density(pos, np.full(n, 1.0 / n), h0, box=box)
    return float(d.rho.std() / d.rho.mean())


def make_glass(
    n: int, seed: int = 1, target_noise: float = 0.01, max_steps: int = 2000, settle: int = 50
) -> tuple[np.ndarray, GlassLog]:
    """Relax n random particles in the periodic unit box into a glass. Runs until the noise is
    below target_noise, then ``settle`` more steps. Returns positions in [0, 1)^3."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n, 3))
    vel = np.zeros_like(pos)
    mass = np.full(n, 1.0 / n)
    h = np.full(n, smoothing_length_uniform(n, 1.0))
    noise, below = [], None
    for step in range(max_steps):
        d = compute_density(pos, mass, h, box=1.0)
        h = d.h
        noise.append(float(d.rho.std() / d.rho.mean()))
        if below is None and noise[-1] < target_noise:
            below = step
        if below is not None and step >= below + settle:
            break
        f = forces_numba(
            pos, vel, mass, d.h, d.rho, d.omega, isothermal_pressure(d.rho, 1.0), d.neighbours
        )
        dt = 0.2 * h.min()  # Courant step for sound speed 1
        vel = (vel + dt * f.acc) * 0.8  # strong damping: the particles settle, they do not ring
        pos = np.mod(pos + dt * vel, 1.0)
    else:
        raise RuntimeError(f"glass did not relax below {target_noise} in {max_steps} steps")
    return pos, GlassLog(step + 1, noise)


def glass_path(n: int) -> Path:
    return DATA / f"glass_{n}.npy"


def load_glass(n: int = 4096) -> np.ndarray:
    """The stored glass of n particles in [0, 1)^3 (made by scripts/make_glass.py)."""
    path = glass_path(n)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run uv run python scripts/make_glass.py")
    return np.load(path)


def tile_glass(glass: np.ndarray, reps: tuple[int, int, int], size: tuple[float, float, float]):
    """Repeat a unit-box glass reps times along each axis and stretch it to fill [0, size).
    The glass is periodic, so the copies join seamlessly."""
    reps_arr = np.asarray(reps)
    shifts = np.stack(np.meshgrid(*[np.arange(k) for k in reps], indexing="ij"), -1).reshape(-1, 3)
    pos = (glass[None, :, :] + shifts[:, None, :]).reshape(-1, 3) / reps_arr
    return pos * np.asarray(size, float)
