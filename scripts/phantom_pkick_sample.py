"""Sample for the rule-2 test of the patch's nebula_pkick (tests/sim/test_phantom_patch.py).

    uv run --with sarracen --with scikit-learn python scripts/phantom_pkick_sample.py \
        DUMP_PKICK_ON DUMP_PKICK_OFF tests/data/phantom_pkick_sample.npz

The two dumps are Phantom's start-up derivative pass (nebula_dumpderivs = T, start file
renamed NAME_00000.tmp so Phantom writes dump 00000) on the same state, nebula_pkick T and F,
all other options as production. Keeps the 1,500 particles nearest the hole and 500 others
(seed 0), with both hydro accelerations (Phantom's fx, fy, fz; float64 in full dumps).
"""

import sys

import numpy as np
import sarracen


def load(path):
    s = sarracen.read_phantom(path)
    return s[0] if isinstance(s, tuple) else s


def main():
    on, off = load(sys.argv[1]), load(sys.argv[2])
    assert np.array_equal(on["iorig"].values, off["iorig"].values)
    x = on[["x", "y", "z"]].values.astype(float)
    r = np.linalg.norm(x, axis=1)
    near = np.argsort(r)[:1500]
    rest = np.setdiff1d(np.arange(len(r)), near)
    pick = np.sort(np.r_[near, np.random.default_rng(0).choice(rest, 500, replace=False)])
    np.savez_compressed(
        sys.argv[3], x=x[pick], acc_pkick_on=on[["fx", "fy", "fz"]].values[pick].astype(float),
        acc_pkick_off=off[["fx", "fy", "fz"]].values[pick].astype(float),
        time=float(on.params["time"]),
    )  # fmt: skip
    print(f"saved {len(pick)} particles, r = {r[pick].min():.1f} to {r[pick].max():.1f}")


if __name__ == "__main__":
    main()
