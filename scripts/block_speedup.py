"""Block time steps vs one global step (Phase 4, test 7): the 1k-particle cloud from apocentre
until its leading edge has reached pericentre (t = 385,000; the centre arrives at 396,224),
eta_orb = 0.00125, Courant 0.3. It stops there because at t = 386,646 one particle of the
leading edge, violently heated as the stream passes pericentre, loses its angular momentum
and falls in, identically with block and shared steps (a Phase 5 question; DECISIONS).

outputs/checks/block_speedup.txt   particle-steps, wall times, speed-up, ledgers, agreement
outputs/checks/block_levels.png    level histogram over time and active particles per step

Run: uv run python scripts/block_speedup.py      (~15-30 minutes)
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula.sim import blocks, coupled  # noqa: E402
from nebula.sim.blocks import BlockConfig  # noqa: E402
from nebula.sim.cloud import cloud_ic  # noqa: E402
from nebula.sim.coupled import CoupledConfig  # noqa: E402

CHECKS = Path(__file__).resolve().parents[1] / "outputs" / "checks"
T_END = 385_000.0
DT_MAX = 4096.0


def rel_ledger(led, led0):
    return abs(led["energy"] - led0["energy"]) / led0["abs_h"], float(
        np.linalg.norm(led["angmom"] - led0["angmom"]) / np.linalg.norm(led0["angmom"])
    )


def run_blocks(ic, cc):
    cfg = BlockConfig(coupled=cc, dt_max=DT_MAX)
    w0 = time.perf_counter()
    st = blocks.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, cfg, t_stop=T_END)
    led0 = blocks.ledger(st, cfg)
    hist_t, hist, step_t = [], [], []
    worst_e = worst_l = 0.0
    while st.n < st.n_stop:
        blocks.advance(st, cfg)
        step_t.append(st.t)
        if st.synced():
            e, ell = rel_ledger(blocks.ledger(st, cfg), led0)
            worst_e, worst_l = max(worst_e, e), max(worst_l, ell)
        if len(st.active_counts) % 50 == 0:
            hist_t.append(st.t)
            hist.append(np.bincount(st.level, minlength=41)[:41])
    wall = time.perf_counter() - w0
    return st, wall, worst_e, worst_l, np.array(hist_t), np.array(hist), np.array(step_t)


def run_global(ic, cc):
    w0 = time.perf_counter()
    st = coupled.initialise(ic.x, ic.v, ic.u, ic.mass, ic.h, cc)
    led0 = coupled.ledger(st, cc)
    steps = 0
    worst_e = worst_l = 0.0
    dts = []
    while st.t < T_END * (1 - 1e-14):
        dt = min(coupled.stable_dt(st, cc), T_END - st.t)
        st = coupled.step(st, cc, dt)
        steps += 1
        dts.append((st.t, dt))
        e, ell = rel_ledger(coupled.ledger(st, cc), led0)
        worst_e, worst_l = max(worst_e, e), max(worst_l, ell)
    wall = time.perf_counter() - w0
    return st, steps, wall, worst_e, worst_l, np.array(dts)


def main():
    CHECKS.mkdir(parents=True, exist_ok=True)
    ic = cloud_ic(1000)
    n = len(ic.x)
    cc = CoupledConfig()
    print(f"cloud: {n} particles; block steps ...", flush=True)
    bst, bwall, be, bl, ht, hh, step_t = run_blocks(ic, cc)
    d = blocks.diagnostics(bst)
    print(f"  done in {bwall:.0f} s; global step ...", flush=True)
    gst, gsteps, gwall, ge, gl, dts = run_global(ic, cc)
    print(f"  done in {gwall:.0f} s", flush=True)
    com_b = np.average(bst.x, axis=0, weights=bst.mass)
    com_g = np.average(gst.x, axis=0, weights=gst.mass)
    dx = np.linalg.norm(bst.x - gst.x, axis=1)
    lines = [
        f"1k cloud (N = {n}) from apocentre to t = {T_END:,.0f} (leading edge at pericentre), "
        f"eta_orb = {cc.eta_orb}, Courant {cc.hydro.courant}",
        f"block steps:  {d['particle_steps']:,} particle-steps in {d['system_steps']:,} system "
        f"steps (mean active {d['mean_active']:.0f}), wall {bwall:.0f} s, woken {d['woken']}",
        f"   time per stage: flow {d['timers']['flow']:.0f} s, hydro {d['timers']['hydro']:.0f} s, "
        f"levels {d['timers']['levels']:.0f} s",
        "   final level histogram (level: count): "
        + ", ".join(f"{k}: {c}" for k, c in enumerate(d["level_histogram"]) if c),
        f"global step:  {gsteps * n:,} particle-steps in {gsteps:,} steps, wall {gwall:.0f} s",
        f"speed-up: particle-steps x{gsteps * n / d['particle_steps']:.1f}, "
        f"wall time x{gwall / bwall:.1f}",
        f"ledger max |dE| / sum m|H|: blocks {be:.2e}, global {ge:.2e}; max |dL|/|L|: blocks "
        f"{bl:.1e}, global {gl:.1e}",
        f"centre of mass, blocks vs global: {np.linalg.norm(com_b - com_g):.3e} (r = "
        f"{np.linalg.norm(com_g):.2f}); particle positions differ by median {np.median(dx):.3e}, "
        f"max {dx.max():.3e}",
    ]
    text = "\n".join(lines)
    print(text)
    (CHECKS / "block_speedup.txt").write_text(text + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    levels = np.arange(hh.shape[1])
    used = levels[hh.sum(axis=0) > 0]
    # samples are every 50 system steps, i.e. uneven in time: draw each over its own interval
    edges = np.concatenate([[0.0], 0.5 * (ht[1:] + ht[:-1]), [ht[-1]]])
    lev_edges = np.arange(used[0], used[-1] + 2) - 0.5
    mesh = axes[0].pcolormesh(
        edges, lev_edges, hh[:, used[0] : used[-1] + 1].T, cmap="viridis", shading="flat"
    )
    cax = axes[0].inset_axes([1.01, 0.0, 0.015, 1.0])  # keeps both panels' time axes aligned
    fig.colorbar(mesh, cax=cax, label="particles")
    axes[0].set_ylabel(f"time-step level (step = {DT_MAX:g} / 2^level)")
    axes[0].set_title("Block time steps: how many particles sit on each level")
    counts = np.asarray(bst.active_counts)
    axes[1].semilogy(
        dts[:, 0], dts[:, 1], color="#c92a2a", lw=1, label="global step (all particles)"
    )
    axes[1].set_ylabel("global time step")
    axes[1].set_xlabel("time (GM/c^3)")
    ax2 = axes[1].twinx()
    ax2.plot(step_t, counts, color="#1f5fbf", lw=0.5, alpha=0.6)
    ax2.set_ylabel("active particles per block step (blue)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(CHECKS / "block_levels.png", dpi=130)


if __name__ == "__main__":
    main()
