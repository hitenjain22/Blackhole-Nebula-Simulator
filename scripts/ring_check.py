"""Is the fast-cooling ring where physics says? Ring radius vs r_circ = 50.89 (expected-values:
circular orbit with the cloud's angular momentum L = 7.3537), and a better thickness measure.

    uv run --no-sync --with sarracen --with scikit-learn python ~/ring_check.py OUT.png DUMP...

For each dump of R6.6 (beta 1) and R6.1_100k_ph_b (beta 230) where present: gas within r 400
that is bound; 'round' = eccentricity < 0.3 (as cool_compare). Prints the mass-weighted median
radius of the round gas, the median circularisation radius from each particle's own L, and the
share of near gas with |z|/R > 0.1 and > 0.2 (puffed-up gas). Analysis only."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # cool_compare.py beside it
from cool_compare import RUNS, load  # noqa: E402

from nebula.physics.gravity import gn_hamiltonian  # noqa: E402
from nebula.physics.orbits import circularization_radius  # noqa: E402

R_CIRC = circularization_radius(7.35374434)  # expected-values: 50.88962


def r_circ_of(ang):
    """physics.orbits.circularization_radius per particle; nan where no circular orbit."""
    return np.array([circularization_radius(a) if a * a >= 12 else np.nan for a in ang])


def analyse(x, p):
    r = np.linalg.norm(x, axis=1)
    h = gn_hamiltonian(x, p)
    sel = (r < 400) & (h < 0)
    xs, ps, rs = x[sel], p[sel], r[sel]
    lv = np.cross(xs, ps)
    nz = lv.sum(axis=0)
    nz /= np.linalg.norm(nz)
    z = xs @ nz
    big_r = np.sqrt(np.maximum(rs**2 - z**2, 1e-30))
    l2 = np.sum(lv**2, axis=1)
    e = np.sqrt(np.clip(1 + 2 * h[sel] * l2, 0, None))
    rnd = e < 0.3
    tilt = np.abs(z) / big_r
    return {
        "n_round": int(rnd.sum()),
        "ring_R_median": float(np.median(big_r[rnd])) if rnd.any() else np.nan,
        "ring_R_q25_q75": tuple(np.percentile(big_r[rnd], [25, 75]))
        if rnd.any()
        else (np.nan,) * 2,
        "rcirc_from_L_median_round": float(np.nanmedian(r_circ_of(np.sqrt(l2[rnd]))))
        if rnd.any()
        else np.nan,
        "puffed_gt_0.1": float(np.mean(tilt > 0.1)),
        "puffed_gt_0.2": float(np.mean(tilt > 0.2)),
        "thick_q90": float(np.percentile(tilt, 90)),
    }, big_r[rnd]


def main():
    out = sys.argv[1]
    dumps = [int(d) for d in sys.argv[2:]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    rows = []
    for name, run in RUNS.items():
        for d in dumps:
            f = run / f"main100k_{d:05d}"
            if not f.exists():
                continue
            t, x, p, _ = load(f)
            m, ring_r = analyse(x, p)
            day = t / 4079.27
            rows.append((name, day, m))
            print(f"{name:26s} day {day:6.1f} {m}", flush=True)
            ax = axes[0] if "fast" in name else axes[1]
            if len(ring_r):
                ax.hist(
                    ring_r,
                    bins=np.linspace(0, 200, 81),
                    histtype="step",
                    label=f"day {day:.0f} ({len(ring_r)})",
                )
    for ax, title in zip(axes, ("fast cooling (beta 1)", "slow cooling (beta 230)"), strict=True):
        ax.axvline(R_CIRC, color="k", ls="--", label="predicted r_circ = 50.9")
        ax.set_title(f"{title}: radius of gas on round orbits")
        ax.set_xlabel("distance from hole (code units, G = c = M = 1)")
        ax.set_ylabel("particles")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("wrote", out)


if __name__ == "__main__":
    main()
