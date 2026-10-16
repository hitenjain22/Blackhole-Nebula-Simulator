"""The cloud on block steps with swallowing (Phase 5; review items 4 and 5).

    uv run python scripts/cloud_run.py NAME n=10000 radius=25 tend=400000
    uv run python scripts/cloud_run.py NAME --resume        (continue from the checkpoint)
    uv run python scripts/cloud_run.py NAME --resume tend=500000   (extend a finished run)

Options: n (particles), radius (cloud radius; D4: 25), tend, r_acc (4), recon (on/off:
velocity reconstruction, D2); Phase 6: eos (ideal | gasrad), beta (cooling; inf = none),
r_escape (50000), gravity (gn | newton: control run), seed (random glass carving).

Snapshots (Hiten, 2026-10-15: Phase 10.2a's rule from Phase 5 on): every
clamp(T_orb(r_med)/40, 10, 10^4), T_orb(r) = 2 pi r^(3/2), r_med the median radius of the gas
inside r = 2,000 (all gas if none is inside), plus an event snapshot when the centre of mass
passes pericentre.

Writes runs/NAME/: manifest.json, checkpoint.pkl (every 15 minutes), snapshots/ (float32),
diagnostics.csv (time, particles, removed, ledger errors, the D5 hole-resolution numbers and
the gas temperatures near the hole), removed.json (every swallowed particle), log.txt.
"""

import csv
import json
import sys
import time
from dataclasses import replace

import numpy as np

from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.cloud import cloud_ic
from nebula.sim.coupled import CoupledConfig
from nebula.sim.diagnostics import hole_resolution
from nebula.sim.runio import (
    RUNS,
    load_checkpoint,
    new_run,
    save_checkpoint,
    save_snapshot,
    update_manifest,
)

CHECKPOINT_EVERY = 15 * 60  # s (CLAUDE.md: at least every 15 minutes)


def parse(argv):
    name = argv[1]
    opts = dict(a.split("=", 1) for a in argv[2:] if "=" in a)
    return name, "--resume" in argv, opts


def make_config(opts):
    cfg = {
        "n": int(opts.get("n", 10_000)),
        "radius": float(opts.get("radius", 25.0)),
        "tend": float(opts.get("tend", 400_000.0)),
        "r_acc": float(opts.get("r_acc", 4.0)),
        "recon": opts.get("recon", "on") == "on",
        "dt_max": 4096.0,
        # Phase 6 options (defaults: Phase 5 behaviour)
        "eos": opts.get("eos", "ideal"),  # "gasrad": gas + radiation (phase-06 6.2)
        "beta": float(opts.get("beta", "inf")),  # beta-cooling (inf: none; scenario 230)
        "r_escape": float(opts["r_escape"]) if "r_escape" in opts else None,  # 50000 in Phase 6
        "gravity": opts.get("gravity", "gn"),  # "newton": the control run R6.4
        "seed": int(opts["seed"]) if "seed" in opts else None,  # random glass carving (R6.5)
    }
    return cfg


def block_config(cfg):
    cc = CoupledConfig()
    hc = replace(cc.hydro, reconstruction=cfg["recon"], eos=cfg.get("eos", "ideal"))
    cc = replace(cc, hydro=hc, cooling_beta=cfg.get("beta", float("inf")),
                 gravity=cfg.get("gravity", "gn"))  # fmt: skip
    return BlockConfig(coupled=cc, dt_max=cfg["dt_max"], r_acc=cfg["r_acc"],
                       r_escape=cfg.get("r_escape"))  # fmt: skip


def snapshot_interval(x: np.ndarray) -> float:
    """Phase 10.2a: clamp(T_orb(r_med)/40, 10, 10^4)."""
    r = np.linalg.norm(x, axis=1)
    inner = r[r < 2000.0]
    r_med = float(np.median(inner if len(inner) else r))
    return float(np.clip(2.0 * np.pi * r_med**1.5 / 40.0, 10.0, 1.0e4))


def log(folder, text):
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(folder / "log.txt", "a") as f:
        f.write(line + "\n")


def diagnostics_row(st, bcfg, led0):
    led = blocks.ledger(st, bcfg)
    hr = hole_resolution(st.x, st.h, st.u)
    de = abs(led["energy"] - led0["energy"]) / led0["abs_h"]
    dl = np.linalg.norm(led["angmom"] - led0["angmom"]) / np.linalg.norm(led0["angmom"])
    return {
        "t": st.t, "n_present": len(st.x), "removed": st.removed.count,
        "removed_first_pass": len(st.removed.first_pass()), "energy_err": de, "angmom_err": dl,
        "n_near60": hr.n_near, "two_h_over_r_med": hr.two_h_over_r_median,
        "two_h_over_r_max": hr.two_h_over_r_max, "across_hole_frac": hr.across_fraction,
        "n_gas300": hr.n_gas, "u_median": hr.u_median, "u_p99": hr.u_p99, "u_max": hr.u_max,
        "level_min": int(st.level.min()), "level_max": int(st.level.max()),
        "particle_steps": st.particle_steps,
    }, hr  # fmt: skip


def main():
    name, resume, opts = parse(sys.argv)
    folder = RUNS / name
    if resume:
        state = load_checkpoint(folder)
        blocks.upgrade_state(state["st"])  # a checkpoint from older code
        cfg = state["config"]
        update_manifest(folder, resumed=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        log(folder, f"resumed at t = {state['st'].t:.1f}")
        if "tend" in opts and float(opts["tend"]) > cfg["tend"]:
            # extend: every particle is at a step boundary at the old end (or mid-step before
            # it); moving the stop time only lets later steps run further (blocks.run logic)
            st = state["st"]
            cfg["tend"] = float(opts["tend"])
            st.n_stop = int(round((cfg["tend"] - st.t0) / st.tick))
            update_manifest(folder, config=cfg)
            log(folder, f"extended to t = {cfg['tend']:.0f}")
    else:
        cfg = make_config(opts)
        folder = new_run(name, cfg)
        ic = cloud_ic(cfg["n"], cfg["radius"], cfg.get("seed"))
        bcfg0 = block_config(cfg)
        st = blocks.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, bcfg0, t_stop=cfg["tend"])
        save_snapshot(
            folder,
            0,
            st.t,
            x=st.x,
            p=st.p,
            u=st.u,
            h=st.h,
            rho=st.rho,
            ids=st.ids,
            level=st.level,
            visc_heating=st.visc_heating,
            luminosity=st.luminosity,
        )  # fmt: skip  (the exact starting state)
        state = {"config": cfg, "st": st, "led0": blocks.ledger(st, bcfg0), "next_t": 0.0,
                 "snap": 1, "wall": 0.0, "com_vr_neg": False, "events": []}  # fmt: skip
        log(folder, f"started: N = {len(ic.x)}, radius {cfg['radius']}, recon {cfg['recon']}")
    bcfg = block_config(cfg)
    st = state["st"]
    csv_path = folder / "diagnostics.csv"
    w0 = time.perf_counter() - state["wall"]
    last_ckpt = time.perf_counter()
    n_removed = st.removed.count
    while st.n < st.n_stop:
        blocks.advance(st, bcfg)
        com = np.average(st.x, axis=0, weights=st.mass)
        com_v = np.average(st.p, axis=0, weights=st.mass)  # ~ velocity far out; sign is all we need
        event = state["com_vr_neg"] and float(com @ com_v) > 0.0
        if event:
            state["events"].append(("pericentre (centre of mass)", st.t))
            log(folder, f"event: centre of mass passed pericentre at t = {st.t:.1f}")
        state["com_vr_neg"] = float(com @ com_v) < 0.0
        if st.removed.count > n_removed:
            for e in st.removed.log[n_removed:]:
                tag = "  FIRST PASS (plan alarm)" if e["first_pass"] else ""
                log(folder, f"swallowed particle {e['id']} at t = {e['t']:.1f}: L {e['L0']:.3f}"
                            f" -> {e['L']:.3f}, u {e['u']:.2e}{tag}")  # fmt: skip
            n_removed = st.removed.count
        if st.t >= state["next_t"] or st.n >= st.n_stop or event:
            row, hr = diagnostics_row(st, bcfg, state["led0"])
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    w.writeheader()
                w.writerow(row)
            save_snapshot(folder, state["snap"], st.t, x=st.x, p=st.p, u=st.u, h=st.h,
                          rho=st.rho, ids=st.ids, level=st.level,
                          visc_heating=st.visc_heating, luminosity=st.luminosity)  # fmt: skip
            state["snap"] += 1
            dt_snap = snapshot_interval(st.x)
            state["next_t"] = st.t + dt_snap
            if state["snap"] % 20 == 0 or event:
                log(folder, f"t = {st.t:9.1f}  N = {len(st.x)}  removed {st.removed.count}  "
                            f"dE {row['energy_err']:.1e}  dL {row['angmom_err']:.1e}  "
                            f"wall {time.perf_counter() - w0:.0f} s | {hr.line()}")  # fmt: skip
        if time.perf_counter() - last_ckpt > CHECKPOINT_EVERY:
            state["wall"] = time.perf_counter() - w0
            save_checkpoint(folder, state)
            last_ckpt = time.perf_counter()
    state["wall"] = time.perf_counter() - w0
    save_checkpoint(folder, state)
    (folder / "removed.json").write_text(json.dumps(st.removed.log, indent=1))
    update_manifest(folder, end=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    led = blocks.ledger(st, bcfg)
    log(folder, f"finished at t = {st.t:.1f}: {st.removed.count} of {len(st.x) + st.removed.count}"
                f" swallowed ({len(st.removed.first_pass())} on the first pass); ledger dE "
                f"{abs(led['energy'] - state['led0']['energy']) / state['led0']['abs_h']:.1e}; "
                f"wall {state['wall']:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
