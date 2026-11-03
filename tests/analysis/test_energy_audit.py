"""Energy audit between snapshots (analysis/energy_audit.py): a hand-built case with known
channel values, and a real block-step run where the audit's summed channels must reproduce the
run's own energy ledger."""

from dataclasses import replace

import numpy as np

from nebula.analysis.energy_audit import energy_channels
from nebula.physics.gravity import gn_hamiltonian, newton_hamiltonian
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig, hamiltonian
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def _snap(ids, x, p, u, visc=0.0, pdv=0.0, cool=0.0):
    n = len(ids)
    z = np.zeros(n)
    return {
        "ids": np.array(ids), "x": x, "p": p, "u": np.asarray(u, float),
        "heat_visc": z + visc, "heat_cond": z, "heat_pdv": z + pdv, "heat_cool": z + cool,
    }  # fmt: skip


def test_channels_known_answer():
    """Two particles, Newtonian H = |p|^2/2 - 1/r. Snapshot b: particle 1 slows so H falls by
    0.001 and that is booked as viscous heating with u up by 0.001 (balanced: both residuals 0);
    particle 2's u rises by 0.002 with nothing booked (r_heat = m 0.002); particle 3 is only in
    b (ignored); particle 0 only in a (removed: removal channel)."""
    x = np.array([[100.0, 0, 0], [0, 200.0, 0], [0, 0, 300.0]])
    p_a = np.array([[0, 0.1, 0], [0.05, 0, 0], [0.01, 0, 0]])
    a = _snap([0, 1, 2], x, p_a, [1e-3, 1e-3, 1e-3])
    p1 = np.sqrt(0.05**2 - 2 * 0.001)  # |p|^2/2 lower by 0.001 at the same position
    xb = np.array([[0, 200.0, 0], [0, 0, 300.0], [5.0, 5.0, 5.0]])
    pb = np.array([[p1, 0, 0], [0.01, 0, 0], [0, 0, 0]])
    b = _snap([1, 2, 3], xb, pb, [1e-3 + 0.001, 1e-3 + 0.002, 0.0])
    b["heat_visc"][0] = 0.001
    # particle 0 (H + u = 0.005 - 0.01 + 0.001 = -0.004 at a) was removed booked at -0.006
    r = energy_channels(a, b, 0.5, newton_hamiltonian, removed={0: -0.006})
    check("audit known answer: particles compared", 2, r["n_common"])
    check("audit known answer: r_removed = m (-0.006 + 0.004)", -0.001, r["r_removed"],
          rtol=1e-12)  # fmt: skip
    check("audit known answer: dH = -m 0.001", -0.0005, r["dH"], rtol=1e-10)
    check("audit known answer: r_motion = 0 (balanced)", 0.0, r["r_motion"], atol=1e-15)
    check("audit known answer: r_heat = m 0.002 (unbooked)", 0.001, r["r_heat"], rtol=1e-12)


def test_channels_reproduce_the_ledger(rng):
    """Warm stirred ball on its orbit, GN gravity, gas + radiation EOS and beta-cooling, block
    steps; 'snapshots' taken after arbitrary advances (not sync points, as in real runs). The
    audit's r_heat + r_motion summed over the intervals equals the ledger change
    E(t) - E(0) (E includes the radiated energy) to round-off, relative to sum m |H|."""
    x, v, u, m, h = warm_ball(rng)
    cc = CoupledConfig(cooling_beta=2.0)
    cc = replace(cc, hydro=replace(cc.hydro, eos="gasrad"))
    cfg = BlockConfig(coupled=cc, dt_max=64.0, r_acc=4.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 8)
    fields = ("ids", "x", "p", "u", "heat_visc", "heat_cond", "heat_pdv", "heat_disk", "heat_cool")

    def snap():
        return {k: np.array(getattr(st, k), copy=True) for k in fields}

    led0 = blocks.ledger(st, cfg)
    snaps = [snap()]
    for k in range(60):
        blocks.advance(st, cfg)
        if k % 7 == 3:
            snaps.append(snap())
    led1 = blocks.ledger(st, cfg)
    snaps.append(snap())
    rows = [
        energy_channels(a, b, m[0], gn_hamiltonian) for a, b in zip(snaps, snaps[1:], strict=False)
    ]
    total = sum(r["r_heat"] + r["r_motion"] for r in rows)
    scale = led0["abs_h"]
    check("audit run: nothing removed", 0, st.removed.count)
    same = np.array_equal(hamiltonian(cc, st.x, st.p), gn_hamiltonian(st.x, st.p))
    check("audit run: GN Hamiltonian is the run's", True, bool(same))
    check("audit run: cooling acted", True, st.e_radiated > 0)
    check("audit run: sum of channels = ledger change (/ sum m|H|)",
          (led1["energy"] - led0["energy"]) / scale, total / scale, atol=1e-13)  # fmt: skip


def test_channels_close_the_books_with_removals(rng):
    """As above without cooling, with an escape radius just inside the ball's outer edge so
    particles are removed during the run (mid-step for their neighbours). With the removal
    channel (booked H + u at removal minus the energy at the earlier snapshot) the three
    channels summed over the intervals equal the ledger change to round-off."""
    x, v, u, m, h = warm_ball(rng)
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0, r_acc=4.0, r_escape=502.4)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 8)
    fields = ("ids", "x", "p", "u", "heat_visc", "heat_cond", "heat_pdv", "heat_disk", "heat_cool")

    def snap():
        return {k: np.array(getattr(st, k), copy=True) for k in fields}

    led0 = blocks.ledger(st, cfg)
    snaps = [snap()]
    for k in range(60):
        blocks.advance(st, cfg)
        if k % 7 == 3:
            snaps.append(snap())
    led1 = blocks.ledger(st, cfg)
    snaps.append(snap())
    removed = {e["id"]: e["H"] + e["u"] for e in st.removed.log}
    pairs = zip(snaps, snaps[1:], strict=False)
    rows = [energy_channels(a, b, m[0], gn_hamiltonian, removed) for a, b in pairs]
    total = sum(r["r_heat"] + r["r_motion"] + r["r_removed"] for r in rows)
    scale = led0["abs_h"]
    print(f"   removed {st.removed.count}; removal channel "
          f"{sum(r['r_removed'] for r in rows) / scale:+.2e}")  # fmt: skip
    check("audit removals: particles were removed", True, st.removed.count >= 3)
    check("audit removals: counted in the audit", st.removed.count,
          sum(r["n_removed"] for r in rows))  # fmt: skip
    check("audit removals: three channels = ledger change (/ sum m|H|)",
          (led1["energy"] - led0["energy"]) / scale, total / scale, atol=1e-13)  # fmt: skip
