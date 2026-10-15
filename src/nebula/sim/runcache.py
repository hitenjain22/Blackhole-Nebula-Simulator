"""Cache for expensive deterministic runs shared by a slow test and its plot script.

A result is stored under a hash of the source files the run depends on (and the stored
glasses), so it is reused only while none of them changed. (sim/testproblems.py keeps its own
copy for the Phase 3 runs; its hash covers the Phase 3 files.)
"""

import hashlib
import pickle
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]  # src/nebula
ROOT = SRC.parents[1]

PHASE3_FILES = (
    "physics/kernels.py", "physics/eos.py", "physics/viscosity.py", "sim/sph.py", "sim/hydro.py",
    "sim/neighbours.py", "sim/kernels_nb.py", "sim/glass.py",
)  # fmt: skip
PHASE4_FILES = PHASE3_FILES + (
    "physics/gravity.py", "physics/orbits.py", "sim/integrator.py", "sim/coupled.py",
    "sim/blocks.py", "sim/timesteps.py", "sim/blob.py", "sim/cloud.py", "units.py",
)  # fmt: skip


def fingerprint(files) -> str:
    h = hashlib.sha256()
    paths = [SRC / f for f in files] + sorted((ROOT / "data").glob("glass_*.npy"))
    for f in paths:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def cached(name: str, compute, files=PHASE4_FILES):
    """compute()'s (pickleable) result, cached in outputs/cache under the files' hash."""
    cache_dir = ROOT / "outputs" / "cache"
    path = cache_dir / f"{name}_{fingerprint(files)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(result))
    tmp.rename(path)  # atomic: a crash never leaves a half-written entry
    return result
