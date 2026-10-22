"""The SPH ring-spreading test (phases/phase-07-viscosity-drain.md 7.4; Lodato & Price 2010).

    uv run python scripts/ring_run.py NAME n=10000 visc=B alpha=0.1 tend=0.25
    uv run python scripts/ring_run.py NAME --resume

Options: n (particles), visc (none | A | B: no disk viscosity, Navier-Stokes nu = alpha
c_s^2/Omega_K, or the disc viscosity through the AV form), alpha (the Shakura-Sunyaev alpha;
Option B converts it with Eq. 124 using the measured <h>/H at r0: alpha_AV = 10 alpha H/<h>,
or give alpha_av directly), tend (in viscous times t_nu = r0^2/nu with nu = alpha (H/R)^2
r0^(1/2)), gravity (newton | gn), respa (on | off: gravity sub-stepping, sim/blocks.py; cap
respa_eta, default 0.02, from the hot-blob accuracy test), r_acc (4), seed (Halton scramble).

The ring: r0 = 50, Gaussian width 5, H/R = 0.1 at r0, locally isothermal c_s ~ r^(-3/4)
(sim/ringtest.py). Writes runs/NAME/: manifest.json, checkpoint.pkl (every 15 minutes),
snapshots/ every 0.005 t_nu (float32), diagnostics.csv (time, mass and angular-momentum ledgers,
mean h/H at r0), log.txt.
"""

import csv
import sys
import time
from dataclasses import replace

import numpy as np

from nebula.analysis.disk_profile import disk_profile
from nebula.physics.disc_viscosity import alpha_av_for
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.ringtest import H_OVER_R, R0, ring_setup
from nebula.sim.runio import (
    RUNS,
    load_checkpoint,
    new_run,
    save_checkpoint,
    save_snapshot,
    update_manifest,
)

CHECKPOINT_EVERY = 15 * 60  # s (CLAUDE.md)
SNAP_EVERY = 0.005  # t_nu


def parse(argv):
    name = argv[1]
    opts = dict(a.split("=", 1) for a in argv[2:] if "=" in a)
    return name, "--resume" in argv, opts


def make_config(opts):
    alpha = float(opts.get("alpha", 0.1))
    nu = alpha * H_OVER_R**2 * R0**0.5
    return {
        "n": int(opts.get("n", 10_000)),
        "visc": opts.get("visc", "B"),
        "alpha": alpha,
        "alpha_av": float(opts["alpha_av"]) if "alpha_av" in opts else None,
        "nu": nu,
        "t_nu": R0**2 / nu,
        "tend_tnu": float(opts.get("tend", 0.25)),
        "gravity": opts.get("gravity", "newton"),
        "respa": opts.get("respa", "on") == "on",
        "respa_eta": float(opts.get("respa_eta", 0.02)),
        "r_acc": float(opts.get("r_acc", 4.0)),
        "seed": int(opts.get("seed", 0)),
        "dt_max": 4096.0,
    }


def block_config(cfg, c_s0):
    cc = CoupledConfig(gravity=cfg["gravity"])
    hc = replace(cc.hydro, isothermal_cs=c_s0, iso_r0=R0)
    if cfg["visc"] == "A":
        hc = replace(hc, ns_alpha=cfg["alpha"])
    elif cfg["visc"] == "B":
        hc = replace(hc, disc_alpha_av=cfg["alpha_av"])
    elif cfg["visc"] != "none":
        raise ValueError(f"visc must be none, A or B, not {cfg['visc']!r}")
    return BlockConfig(coupled=replace(cc, hydro=hc), dt_max=cfg["dt_max"], r_acc=cfg["r_acc"],
                       respa=cfg["respa"], respa_eta=cfg["respa_eta"])  # fmt: skip


def h_over_H_at_r0(st):
    """Mean h / H in the annulus 48 < R < 52 (Option B's resolution; Eq. 124)."""
    rr = np.hypot(st.x[:, 0], st.x[:, 1])
    sel = np.abs(rr - R0) < 2.0
    prof = disk_profile(st.x[sel], st.p[sel], st.mass[sel], np.zeros(sel.sum()),
                        np.array([R0 - 2.0, R0 + 2.0]))  # fmt: skip
    return float(np.mean(st.h[sel]) / prof.height[0])


def log(folder, text):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}"
    print(line, flush=True)
    with open(folder / "log.txt", "a") as f:
        f.write(line + "\n")


def main():
    name, resume, opts = parse(sys.argv)
    folder = RUNS / name
    if resume:
        state = load_checkpoint(folder)
        blocks.upgrade_state(state["st"])
        cfg = state["config"]
        update_manifest(folder, resumed=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        log(folder, f"resumed at t = {state['st'].t:.1f}")
    else:
        cfg = make_config(opts)
        ring = ring_setup(cfg["n"], seed=cfg["seed"])
        cfg["c_s0"] = ring.c_s0
        cfg["tend"] = cfg["tend_tnu"] * cfg["t_nu"]
        if cfg["visc"] == "B" and cfg["alpha_av"] is None:
            # Eq. 124 at r0 from the initial smoothing lengths (a first density pass)
            probe = blocks.initialise(ring.x, ring.v, ring.u, ring.mass, ring.h,
                                      block_config({**cfg, "visc": "none"}, ring.c_s0),
                                      t_stop=1.0)  # fmt: skip
            ratio = h_over_H_at_r0(probe)
            cfg["alpha_av"] = float(alpha_av_for(cfg["alpha"], ratio, 1.0))
            cfg["h_over_H_r0_initial"] = ratio
        folder = new_run(name, cfg)
        bcfg0 = block_config(cfg, ring.c_s0)
        st = blocks.initialise(ring.x, ring.v, ring.u, ring.mass, ring.h, bcfg0,
                               t_stop=cfg["tend"])  # fmt: skip
        save_snapshot(folder, 0, st.t, x=st.x, p=st.p, h=st.h, rho=st.rho, ids=st.ids)
        state = {"config": cfg, "st": st, "led0": blocks.ledger(st, bcfg0),
                 "next_t": SNAP_EVERY * cfg["t_nu"], "snap": 1, "wall": 0.0}  # fmt: skip
        log(folder, f"started: N = {cfg['n']}, visc {cfg['visc']}, alpha {cfg['alpha']}, "
                    f"alpha_AV {cfg['alpha_av']}, t_nu {cfg['t_nu']:.0f}, tend "
                    f"{cfg['tend']:.0f}, respa {cfg['respa']}")  # fmt: skip
    bcfg = block_config(cfg, cfg["c_s0"])
    st = state["st"]
    csv_path = folder / "diagnostics.csv"
    w0 = time.perf_counter() - state["wall"]
    last_ckpt = [time.perf_counter()]
    led0 = state["led0"]

    def checkpoint(st):  # called after every system step
        if time.perf_counter() - last_ckpt[0] > CHECKPOINT_EVERY:
            state["wall"] = time.perf_counter() - w0
            save_checkpoint(folder, state)
            last_ckpt[0] = time.perf_counter()

    n_final = int(round((cfg["tend"] - st.t0) / st.tick))
    while st.n < n_final:
        t_next = min(state["next_t"], cfg["tend"])
        blocks.run(st, bcfg, t_next, callback=checkpoint)  # to a full sync point at t_next
        led = blocks.ledger(st, bcfg)
        row = {
            "t": st.t, "t_over_tnu": st.t / cfg["t_nu"], "n": len(st.x),
            "removed_mass": led["removed_mass"],
            "mass_err": abs(led["mass"] - led0["mass"]) / led0["mass"],
            "angmom_err": float(np.linalg.norm(led["angmom"] - led0["angmom"])
                                / np.linalg.norm(led0["angmom"])),
            "h_over_H_r0": h_over_H_at_r0(st),
            "particle_steps": st.particle_steps,
            "wall": time.perf_counter() - w0,
        }  # fmt: skip
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                w.writeheader()
            w.writerow(row)
        save_snapshot(folder, state["snap"], st.t, x=st.x, p=st.p, h=st.h, rho=st.rho,
                      ids=st.ids)  # fmt: skip
        log(folder, f"t = {st.t:9.0f} ({row['t_over_tnu']:.3f} t_nu)  removed mass "
                    f"{row['removed_mass']:.4f}  dJ/J {row['angmom_err']:.2e}  <h>/H "
                    f"{row['h_over_H_r0']:.3f}  wall {row['wall']:.0f} s")  # fmt: skip
        state["snap"] += 1
        state["next_t"] += SNAP_EVERY * cfg["t_nu"]
    state["wall"] = time.perf_counter() - w0
    save_checkpoint(folder, state)
    update_manifest(folder, end=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    log(folder, f"finished at t = {st.t:.0f}; wall {state['wall']:.0f} s")


if __name__ == "__main__":
    main()
