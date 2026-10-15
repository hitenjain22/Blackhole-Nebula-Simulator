"""Numba speed calibration (Phase 0, Step 0.3).

Measures neighbour-pair evaluations per second for the stand-in SPH sum in
nebula._bench.pairs, at N = 10k and 50k (~60 neighbours each) on 1, 2 and 4 threads.
Writes outputs/checks/numba_bench.txt. Phases 3-7 recalibrate their time estimates from it.

Run: uv run python scripts/numba_bench.py
"""

import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
from scipy.spatial import cKDTree

from nebula._bench.pairs import R_CUT, _pair_sum_cells, build_cells, uniform_cube

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "numba_bench.txt"
SIZES = [10_000, 50_000]
THREADS = [1, 2, 4]
REPEATS = 7
NEIGHBOURS = 60


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - t0, result


def main() -> None:
    rng = np.random.default_rng(2026)
    lines = [
        f"Numba benchmark, {datetime.now():%Y-%m-%d %H:%M}",
        f"numba {numba.__version__}, max threads {numba.config.NUMBA_NUM_THREADS}",
        f"Kernel: S_i = sum_j exp(-|x_i - x_j|^2) over |x_i - x_j| < {R_CUT}, cell grid, "
        f"parallel over i. Median of {REPEATS} runs, first (compile) call excluded.",
        "",
        f"{'N':>7} {'threads':>7} {'pairs':>12} {'kernel s':>10} {'pairs/s':>12} "
        f"{'speed-up':>8} {'cells s':>8}",
    ]
    for n in SIZES:
        x = uniform_cube(n, NEIGHBOURS, rng)
        # Exact count of interacting (ordered) pairs, self included, as the kernel visits them.
        pairs = cKDTree(x).count_neighbors(cKDTree(x), np.nextafter(R_CUT, 0.0))
        build_cells(x, R_CUT)  # compile / load cache
        t_cells = np.median([timed(build_cells, x, R_CUT)[0] for _ in range(REPEATS)])
        cells = build_cells(x, R_CUT)
        base_rate = None
        results = {}
        for k in THREADS:
            numba.set_num_threads(k)
            _pair_sum_cells(x, *cells, R_CUT, R_CUT)  # warm-up
            runs = [timed(_pair_sum_cells, x, *cells, R_CUT, R_CUT) for _ in range(REPEATS)]
            t = float(np.median([r[0] for r in runs]))
            results[k] = runs[-1][1]
            rate = pairs / t
            base_rate = base_rate or rate
            lines.append(
                f"{n:>7,} {k:>7} {pairs:>12,} {t:>10.4f} {rate:>12.3e} "
                f"{rate / base_rate:>7.2f}x {t_cells:>8.4f}"
            )
        same = all(np.array_equal(results[1], results[k]) for k in THREADS)
        lines.append(f"        results identical for 1, 2, 4 threads: {same}")
        lines.append(f"        mean neighbours per point: {pairs / n:.1f}")
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
