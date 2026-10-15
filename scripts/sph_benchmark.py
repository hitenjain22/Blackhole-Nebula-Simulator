"""SPH speed measurement (phases/phase-03-sph-core.md 3.4).

Particle-updates per second for ~10k and ~50k particles in a uniform glass with random motions
(Wendland C4, eta = 1.6), measured three ways:
  passes   one density pass + one force pass on ready-made neighbour lists (the plan's measure)
  build    the neighbour-list build on its own
  step     a full hydro step (density with list reuse, switch gradients, forces, update)
Writes outputs/checks/sph_benchmark.txt.

Run: uv run python scripts/sph_benchmark.py
"""

import time
from pathlib import Path

import numpy as np

from nebula.sim.glass import load_glass, tile_glass
from nebula.sim.hydro import HydroConfig, initialise, stable_dt, step
from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import density_numba, forces_numba

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "sph_benchmark.txt"


def best_of(fn, reps=3):
    fn()  # warm-up / compilation
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def measure(tiles) -> str:
    x = tile_glass(load_glass(4096), tiles, tiles)
    n = len(x)
    box = np.array(tiles, float)
    mass = np.full(n, np.prod(tiles) / n)
    h = np.full(n, 1.6 * (np.prod(tiles) / n) ** (1 / 3))
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.1, x.shape)
    u = np.ones(n)
    nb = find_neighbours_numba(x, 1.25 * h, box=box)
    t_build = best_of(lambda: find_neighbours_numba(x, 1.25 * h, box=box))

    def passes():
        d = density_numba(x, mass, h, nb, 1.25 * h)
        p = (2 / 3) * d.rho * u
        cs = np.sqrt(10 / 9 * u)
        forces_numba(x, v, mass, d.h, d.rho, d.omega, p, nb, u=u, c_s=cs, alpha=np.zeros(n))

    t_pass = best_of(passes)
    cfg = HydroConfig(box=tuple(box))
    st = initialise(x, v, u, mass, h, cfg)
    dt = stable_dt(st, cfg)
    st = step(st, cfg, dt)  # warm-up
    builds0 = st.neighbour_cache.builds
    k = 10
    t0 = time.perf_counter()
    for _ in range(k):
        st = step(st, cfg, dt)
    t_step = (time.perf_counter() - t0) / k
    rebuilds = st.neighbour_cache.builds - builds0
    return (
        f"{n:>7,} {t_build:>8.3f} {t_pass:>9.3f} {n / t_pass:>19,.0f} {t_step:>7.3f} "
        f"{n / t_step:>22,.0f} {rebuilds:>9} of {k}"
    )


def main() -> None:
    lines = [
        "SPH speed (Wendland C4, eta 1.6, ~137 neighbours; list search radius 1.25 h).",
        "Plan target (phase-03 3.4): >= 2e5 particle-updates/s at 10k",
        "(one density + one force pass).",
        "",
        f"{'N':>7} {'build s':>8} {'passes s':>9} {'updates/s (passes)':>19} "
        f"{'step s':>7} {'updates/s (full step)':>22} {'list rebuilds':>14}",
    ]
    lines += [measure(tiles) for tiles in ((2, 1, 1), (3, 2, 2))]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
