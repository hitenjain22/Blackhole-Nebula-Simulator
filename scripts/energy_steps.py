"""Where in each particle step the ledger drift is made (diagnosis after the removal fix,
HANDOFF 2026-11-03). Restarts a run from a snapshot like restart_experiment.py and splits every
completed particle step (kick-flow-kick, blocks.py) into its three parts:

    open  = H(x_beg, p_half) - H(x_beg, p_beg) + u_half - u_beg      opening kick (incl. wake-up
                                                                      corrections)
    flow  = H(x_end, p_flow) - H(x_beg, p_half)                       gravity flow (GLF), no gas
    close = H(x_end, p_end) - H(x_end, p_flow) + u_end - (u_half - c) closing kick (c: cooling)

Summed over all particles and steps up to a sync point (every particle at a step boundary)
these add up to the ledger change (the bookkeeping check, column sum_parts vs ledger). In exact
SPH the kicks' first-order parts, v.dp + du ("open1", "close1"), cancel pair by pair between
particles kicked together and the flow conserves H; the rest of each kick is its second-order
part (for p: about |dp|^2 / 2). Sums are binned by the distance from the hole at the step end.

    uv run python scripts/energy_steps.py NAME from=R6.1_50k_v2 day=416 days=1 [removal=sync]

Writes runs/NAME/energy_steps.csv (cumulative sums / sum m|H| at the restart, at every sync
point and every CSV_EVERY system steps) and log.txt.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from nebula.sim import blocks, coupled

sys.path.insert(0, str(Path(__file__).resolve().parent))
import restart_experiment as rx  # noqa: E402  (same restart set-up)

BINS = np.array([0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, np.inf])  # r at step end
PARTS = ("open", "open1", "flow", "close", "close1")
CSV_EVERY = 2000


class StepAudit:
    """Per-step energy parts of a block-step run; call advance() instead of blocks.advance."""

    def __init__(self, st, bcfg, per_particle=False):
        self.st, self.bcfg, self.cc = st, bcfg, bcfg.coupled
        n_id = int(st.ids.max()) + 1
        self.n_id = n_id
        self.m_of = np.zeros(n_id)
        self.m_of[st.ids] = st.mass
        self.p_beg, self.u_beg = np.zeros((n_id, 3)), np.zeros(n_id)
        self.p_beg[st.ids], self.u_beg[st.ids] = st.p, st.u  # closed states at step starts
        self.sums = {k: np.zeros(len(BINS) - 1) for k in PARTS}
        # optional: each particle's own sum of open + flow + close and of the second-order parts
        self.by_id = {k: np.zeros(n_id) for k in ("total", "second")} if per_particle else None

    def total(self) -> float:
        return float(sum(self.sums[k].sum() for k in ("open", "flow", "close")))

    def advance(self):
        st, cc = self.st, self.cc
        rows = np.flatnonzero(st.n_end == st.n_end.min())
        ids = st.ids[rows].copy()
        x_beg, p_half = st.x_beg[rows].copy(), st.p_half[rows].copy()
        u_half, cool0 = st.u_half[rows].copy(), st.heat_cool[rows].copy()
        length = (st.n_end[rows] - st.n_beg[rows]) * st.tick
        n_log, e_rad0 = len(st.removed.log), st.e_radiated
        blocks.advance(st, self.bcfg)
        x_f, p_f = blocks._flow(cc, x_beg, p_half, length)  # the integrator's own flow
        pos = np.full(self.n_id, -1)
        pos[st.ids] = np.arange(len(st.ids))
        j = pos[ids]
        here = j >= 0
        m = self.m_of[ids]
        h_end, u_end, cool = np.empty(len(ids)), np.empty(len(ids)), np.zeros(len(ids))
        p_end = p_f.copy()
        h_end[here] = coupled.hamiltonian(cc, st.x[j[here]], st.p[j[here]])
        u_end[here], p_end[here] = st.u[j[here]], st.p[j[here]]
        cool[here] = st.heat_cool[j[here]] - cool0[here]
        gone = {e["id"]: e for e in st.removed.log[n_log:]}
        out = np.flatnonzero(~here)
        if len(out):  # removed now: their booked (closed) state; their last cooling from e_rad
            for k in out:
                h_end[k], u_end[k] = gone[int(ids[k])]["H"], gone[int(ids[k])]["u"]
            c_gone = st.e_radiated - e_rad0 - float(np.sum(m * cool))
            cool[out] = c_gone / np.sum(m[out])
        p_b, u_b = self.p_beg[ids], self.u_beg[ids]
        h_beg = coupled.hamiltonian(cc, x_beg, p_b)
        h_half = coupled.hamiltonian(cc, x_beg, p_half)
        h_flow = coupled.hamiltonian(cc, x_f, p_f)
        v_beg, v_flow = coupled.velocity(cc, x_beg, p_b), coupled.velocity(cc, x_f, p_f)
        du_close = u_end - (u_half - cool)
        part = {
            "open": h_half - h_beg + u_half - u_b,
            "open1": np.sum(v_beg * (p_half - p_b), axis=1) + u_half - u_b,
            "flow": h_flow - h_half,
            "close": h_end - h_flow + du_close,
        }
        close1 = np.sum(v_flow * (p_end - p_f), axis=1) + du_close
        part["close1"] = np.where(here, close1, part["close"])  # removed: momentum not kept
        b = np.clip(np.searchsorted(BINS, np.linalg.norm(x_f, axis=1)) - 1, 0, len(BINS) - 2)
        for k in PARTS:
            self.sums[k] += np.bincount(b, weights=m * part[k], minlength=len(BINS) - 1)
        if self.by_id is not None:
            tot = part["open"] + part["flow"] + part["close"]
            np.add.at(self.by_id["total"], ids, m * tot)
            two = part["open"] + part["close"] - part["open1"] - part["close1"]
            np.add.at(self.by_id["second"], ids, m * two)
        self.p_beg[ids[here]] = st.p[j[here]]  # the next steps start from the closed state
        self.u_beg[ids[here]] = st.u[j[here]]


def main():
    name = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    folder, state = rx.start(name, opts)
    st = state["st"]
    bcfg = rx.experiment_config(state["config"])
    audit = StepAudit(st, bcfg)
    day = rx.cloud_run_day()
    scale, e0 = state["led0"]["abs_h"], state["led0"]["energy"]
    out = folder / "energy_steps.csv"
    head = ["t", "day", "step", "synced", "ledger", "sum_parts"]
    head += [f"{k}_{i}" for k in PARTS for i in range(len(BINS) - 1)]
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerow(head)
    step, w0 = 0, time.perf_counter()
    while st.n < st.n_stop:
        audit.advance()
        step += 1
        synced = st.synced()
        if synced or step % CSV_EVERY == 0 or st.n >= st.n_stop:
            led = (blocks.ledger(st, bcfg)["energy"] - e0) / scale
            row = [st.t, st.t / day, step, synced, led, audit.total() / scale]
            row += [s / scale for k in PARTS for s in audit.sums[k]]
            with open(out, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
            if synced:
                wall = time.perf_counter() - w0
                rx.log(folder, f"day {st.t / day:.3f} (sync) ledger {led:+.4e}  parts sum "
                       f"{audit.total() / scale:+.4e}  wall {wall:.0f} s")  # fmt: skip
    rx.log(folder, "finished")


if __name__ == "__main__":
    main()
