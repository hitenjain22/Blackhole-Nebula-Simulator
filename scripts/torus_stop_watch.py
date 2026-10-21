"""Stop a Phase 6 run at the plan's alternative end: "torus formed + 10 ring orbits"
(phases/phase-06-shocks-cooling-ring.md 6.4: "Stop condition for 'torus formed': >= 50% of the
remaining bound mass has orbital eccentricity < 0.3 and r < 200. Record that time as t_circ").

    uv run python scripts/torus_stop_watch.py RUN [once]     (run on the machine of the run)

Every 30 minutes it checks the newest snapshot. When the torus has formed it records t_circ in
the run's log and in runs/RUN/torus_stop.json; once a snapshot is 10 ring orbits (10 x 2,303,
expected-values §2) past t_circ it waits for the next checkpoint, stops the run with SIGTERM and
writes "finished at ..." to the log (so the resume helper leaves it alone). "once": report the
ring fraction of the newest snapshot and exit (no stopping).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from nebula.analysis.torus_metrics import orbit_elements, ring_measures

ROOT = Path(__file__).resolve().parents[1]
RING_ORBIT = 2303.26  # orbital period at r_circ = 50.89 (expected-values §2)
FORMED = 0.5


def newest(folder: Path):
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    return np.load(files[-1]) if files else None


def ring_fraction(d) -> float:
    x, p = d["x"].astype(float), d["p"].astype(float)
    return ring_measures(x, np.ones(len(x)), orbit_elements(x, p)).mass_fraction


def run_pids(run: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    key = f".venv/bin/python scripts/cloud_run.py {run}"
    return [int(ln.split()[0]) for ln in out.splitlines() if key in ln]


def log(folder: Path, text: str) -> None:
    with open(folder / "log.txt", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")


def main():
    run = sys.argv[1]
    folder = ROOT / "runs" / run
    if len(sys.argv) > 2 and sys.argv[2] == "once":
        d = newest(folder)
        print(
            f"{run}: t = {float(d['t']):,.0f}, ring fraction of bound mass {ring_fraction(d):.3f}"
        )
        return
    state_path = folder / "torus_stop.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"t_circ": None}
    while True:
        if "finished at" in (folder / "log.txt").read_text():
            return
        d = newest(folder)
        t = float(d["t"])
        if state["t_circ"] is None and ring_fraction(d) >= FORMED:
            state["t_circ"] = t
            state_path.write_text(json.dumps(state))
            log(
                folder,
                f"torus formed (>= 50% of bound mass with e < 0.3, r < 200) at t_circ = {t:.1f}",
            )
        if state["t_circ"] is not None and t >= state["t_circ"] + 10 * RING_ORBIT:
            ckpt = folder / "checkpoint.pkl"
            m0 = ckpt.stat().st_mtime
            while ckpt.stat().st_mtime == m0 and run_pids(run):
                time.sleep(10)
            for pid in run_pids(run):
                os.kill(pid, signal.SIGTERM)
            t_circ = state["t_circ"]
            msg = (f"finished at t = {t:.1f}: torus formed at t_circ = {t_circ:.1f} + 10 ring "
                   "orbits (phase-06 stop condition); stopped after a checkpoint")  # fmt: skip
            log(folder, msg)
            return
        time.sleep(1800)


if __name__ == "__main__":
    main()
