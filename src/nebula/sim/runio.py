"""Run folders, manifests, checkpoints and snapshots (CLAUDE.md rules 5 and 8; "This machine":
checkpoint at least every 15 minutes and resume cleanly).

A run lives in runs/<name>/ and is never overwritten. manifest.json records the git commit
(and whether the tree had uncommitted changes), the full configuration, library versions,
machine, and start/end times. checkpoint.pkl (float64, the whole state) is written atomically,
so a crash never leaves a half-written file. Snapshots for plots and rendering are float32.
"""

import json
import pickle
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"


def _git() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "uncommitted_changes": bool(run("status", "--porcelain")),
    }


def _versions() -> dict:
    import numba
    import scipy

    return {"python": platform.python_version(), "numpy": np.__version__,
            "numba": numba.__version__, "scipy": scipy.__version__}  # fmt: skip


def _jsonable(obj):
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def new_run(name: str, config: dict) -> Path:
    """Create runs/<name>/ (refuses an existing one) and write its manifest."""
    folder = RUNS / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "snapshots").mkdir()
    manifest = {
        "name": name, "git": _git(), "config": _jsonable(config), "versions": _versions(),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "machine": platform.machine()},
        "seed": None,  # deterministic: glass initial conditions, no random numbers
        "start": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "end": None, "resumed": [],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


def update_manifest(folder: Path, **fields) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    for k, v in fields.items():
        if k == "resumed":
            manifest["resumed"].append({"time": v, "git": _git()})  # code may be newer
        else:
            manifest[k] = _jsonable(v)
    path.write_text(json.dumps(manifest, indent=2))


def save_checkpoint(folder: Path, payload) -> None:
    """Pickle ``payload`` (the full float64 state) atomically."""
    tmp = folder / "checkpoint.tmp"
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(folder / "checkpoint.pkl")


def load_checkpoint(folder: Path):
    return pickle.loads((folder / "checkpoint.pkl").read_bytes())


def save_snapshot(folder: Path, index: int, t: float, **arrays) -> None:
    """float32 copy of the given per-particle arrays (rule 5: snapshots for plots/rendering)."""
    out = {k: np.asarray(v, np.float32) if np.issubdtype(np.asarray(v).dtype, np.floating)
           else np.asarray(v) for k, v in arrays.items()}  # fmt: skip
    np.savez_compressed(folder / "snapshots" / f"snap_{index:05d}.npz", t=np.float64(t), **out)
