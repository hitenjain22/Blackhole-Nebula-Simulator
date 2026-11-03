"""Restart a finished run from one of its snapshots for a few days, to test what causes its
ledger drift (diagnosis step 3-4, HANDOFF 2026-11-02).

    uv run python scripts/restart_experiment.py NAME from=R6.1_50k_v2 day=416 days=8 [max_gap=2]
        [removal=sync|legacy] [dt_max=512] [cfl=0.5]
    uv run python scripts/restart_experiment.py NAME --resume

The run's own config (manifest) is used; max_gap (neighbouring time-step levels differ by at
most this; the run used 2, i.e. a factor 4) and removal (sync: the energy-consistent removal of
DECISIONS 2026-11-03; default the source run's own) are the settings an experiment may change, and,
for time-step convergence checks only, smaller steps: dt_max (longest step) and cfl (factor on
the Courant and force criteria, <= 1; CLAUDE.md rule 6 allows only smaller steps). The
snapshot is float32 and does not hold the integrator's mid-step state, so the restart starts
every particle fresh at that time: it follows the original statistically, not bit for bit.

Records in runs/NAME/: manifest.json, log.txt, checkpoint.pkl (every 15 minutes),
energy.csv (after every system step: t, ledger energy, particles removed so far and in this
step, active count), criteria.csv (every 50 steps: which time-step criterion is the smallest
for active gas within r < 300: Courant, force, viscous, orbit).
"""

import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from nebula.physics.navier_stokes import viscous_dt
from nebula.sim import blocks
from nebula.sim.runio import RUNS, load_checkpoint, new_run, save_checkpoint, update_manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloud_run  # noqa: E402  (block_config: the exact settings of the original run)

CHECKPOINT_EVERY = 15 * 60  # s (CLAUDE.md)
CRITERIA_EVERY = 50  # system steps
NEAR = 300.0  # "near the hole" for the criterion counts


def log(folder, text):
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(folder / "log.txt", "a") as f:
        f.write(line + "\n")


def start(name, opts):
    src = opts["from"]
    src_cfg = json.loads((RUNS / src / "manifest.json").read_text())["config"]
    day = float(opts["day"])
    cfg = dict(src_cfg, restart_from=src, restart_day=day, days=float(opts.get("days", 8)),
               max_gap=int(opts.get("max_gap", 2)))  # fmt: skip
    if "removal" in opts:
        cfg["removal_sync"] = opts["removal"] == "sync"
    if float(opts.get("dt_max", cfg["dt_max"])) > cfg["dt_max"] or float(opts.get("cfl", 1)) > 1:
        raise ValueError("experiments may only take smaller steps than the run (rule 6)")
    cfg["dt_max"] = float(opts.get("dt_max", cfg["dt_max"]))
    cfg["cfl"] = float(opts.get("cfl", 1.0))
    files = sorted((RUNS / src / "snapshots").glob("snap_*.npz"))
    day_t = cloud_run_day()
    times = np.array([float(np.load(f)["t"]) for f in files])
    f = files[int(np.searchsorted(times, day * day_t))]
    d = np.load(f)
    cfg["restart_snapshot"] = f.name
    folder = new_run(name, cfg)
    bcfg = experiment_config(cfg)
    x, p = d["x"].astype(float), d["p"].astype(float)
    v = blocks._velocity(bcfg.coupled, x, p)
    log_rm = json.loads((RUNS / src / "removed.json").read_text())
    mass = log_rm[0]["m"]
    t0 = float(d["t"])
    t_end = t0 + cfg["days"] * day_t
    st = blocks.initialise(x, v, d["u"].astype(float), mass, d["h"].astype(float), bcfg,
                           t_stop=t_end, t0=t0)  # fmt: skip
    st.ids = d["ids"].astype(np.int64)
    state = {"config": cfg, "st": st, "led0": blocks.ledger(st, bcfg), "steps": 0, "wall": 0.0}
    log(folder, f"restart of {src} from {f.name} (t = {t0:.1f}, day {t0 / day_t:.1f}), "
                f"N = {len(x)}, max_gap {cfg['max_gap']}, to day {t_end / day_t:.1f}")  # fmt: skip
    return folder, state


def cloud_run_day():
    from nebula import units

    return units.DAY / units.BlackHoleUnits().time_s


def experiment_config(cfg):
    bcfg = replace(cloud_run.block_config(cfg), max_gap=cfg["max_gap"])
    f = cfg.get("cfl", 1.0)
    if f != 1.0:
        hc = bcfg.coupled.hydro
        hc = replace(hc, courant=f * hc.courant, c_force=f * hc.c_force)
        bcfg = replace(bcfg, coupled=replace(bcfg.coupled, hydro=hc))
    return bcfg


def criteria_counts(st, bcfg, rows):
    """Which per-particle criterion of blocks.wanted_dt is the smallest, for active gas near
    the hole (same formulas, evaluated separately)."""
    rows = rows[np.linalg.norm(st.x[rows], axis=1) < NEAR]
    if len(rows) == 0:
        return None
    hc, cc = bcfg.coupled.hydro, bcfg.coupled
    h = st.h[rows]
    amag = np.linalg.norm(st.a_hydro[rows], axis=1)
    with np.errstate(divide="ignore"):
        c = np.stack([
            hc.courant * h / st.vsig_max[rows],
            np.where(amag > 0, hc.c_force * np.sqrt(h / amag), np.inf),
            viscous_dt(h, st.nu[rows]),
            cc.eta_orb * np.linalg.norm(st.x[rows], axis=1) ** 1.5,
        ])  # fmt: skip
    win = np.argmin(c, axis=0)
    dt_now = st.tick * (st.n_end[rows] - st.n_beg[rows])
    return [
        len(rows),
        *np.bincount(win, minlength=4).tolist(),
        float(np.median(dt_now)),
        float(np.median(np.min(c, axis=0))),
    ]


def main():
    name = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    folder = RUNS / name
    if "--resume" in sys.argv:
        state = load_checkpoint(folder)
        update_manifest(folder, resumed=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        log(folder, f"resumed at t = {state['st'].t:.1f}")
    else:
        folder, state = start(name, opts)
    cfg, st = state["config"], state["st"]
    bcfg = experiment_config(cfg)
    day_t = cloud_run_day()
    e_csv, c_csv = folder / "energy.csv", folder / "criteria.csv"
    if not e_csv.exists():
        e_csv.write_text("t,step,energy,removed_total,removed_now,active\n")
        c_csv.write_text("t,step,n_near,courant,force,viscous,orbit,dt_median,crit_median\n")
    w0 = time.perf_counter() - state["wall"]
    last_ckpt = time.perf_counter()
    buf_e, buf_c = [], []
    while st.n < st.n_stop:
        before = st.removed.count
        rows = blocks.advance(st, bcfg)
        state["steps"] += 1
        led = blocks.ledger(st, bcfg)
        buf_e.append((st.t, state["steps"], led["energy"], st.removed.count,
                      st.removed.count - before, len(rows)))  # fmt: skip
        if state["steps"] % CRITERIA_EVERY == 0:
            row = criteria_counts(st, bcfg, rows)
            if row is not None:
                buf_c.append((st.t, state["steps"], *row))
        if time.perf_counter() - last_ckpt > CHECKPOINT_EVERY or st.n >= st.n_stop:
            for path, buf in ((e_csv, buf_e), (c_csv, buf_c)):
                with open(path, "a", newline="") as fh:
                    csv.writer(fh).writerows(buf)
                buf.clear()
            state["wall"] = time.perf_counter() - w0
            save_checkpoint(folder, state)
            last_ckpt = time.perf_counter()
            de = (led["energy"] - state["led0"]["energy"]) / state["led0"]["abs_h"]
            log(folder, f"day {st.t / day_t:.2f}  steps {state['steps']}  N {len(st.x)}  "
                        f"removed {st.removed.count}  ledger change {de:+.3e} (of sum m|H| at "
                        f"restart)  wall {state['wall']:.0f} s")  # fmt: skip
    (folder / "removed.json").write_text(json.dumps(st.removed.log, indent=1))
    update_manifest(folder, end=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    log(folder, "finished")


if __name__ == "__main__":
    main()
