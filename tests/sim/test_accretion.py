"""Swallowing at r_acc with exact ledgers (DECISIONS D6), the hole-resolution diagnostic (D5),
and run folders / checkpoints (CLAUDE.md rule 8, resume after a crash)."""

import copy
import pickle

import numpy as np
import pytest

from nebula.physics.gravity import gn_p_from_v
from nebula.sim import blocks, runio
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.diagnostics import hole_resolution
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def plunging_and_safe(n_plunge=3, n_safe=3):
    """Gravity-only particles: some with L < 4 (captured on their first approach), some on
    safe orbits; spread out so they never interact."""
    x, v = [], []
    for k in range(n_plunge + n_safe):
        ang = 2 * np.pi * k / (n_plunge + n_safe)
        pos = 60.0 * np.array([np.cos(ang), np.sin(ang), 0.0])
        tang = np.array([-np.sin(ang), np.cos(ang), 0.0])
        vt = 0.03 if k < n_plunge else 0.12  # L = 1.8 (plunges) or 7.2 (safe)
        x.append(pos)
        v.append(vt * tang)
    return np.array(x), np.array(v)


def _plunge_run(eta):
    x, v = plunging_and_safe()
    cfg = BlockConfig(coupled=CoupledConfig(eta_orb=eta), dt_max=64.0, hydro=False, r_acc=4.0)
    n = len(x)
    st = blocks.initialise(x, v, np.full(n, 1e-6), np.ones(n), np.ones(n), cfg, t_stop=64.0 * 20)
    led0 = blocks.ledger(st, cfg)
    while st.n < st.n_stop:
        blocks.advance(st, cfg)
    return st, led0, blocks.ledger(st, cfg)


def test_swallowing_keeps_the_ledger_exact():
    """Gravity only: three particles with L < 4 plunge and are swallowed, three stay. The
    removal itself is exact (next test); what remains is the orbits' own time-step error
    (largest for the plunging ones near r = 4), so it must fall ~4x when the step is halved."""
    st, led0, led = _plunge_run(0.00125)
    _, led0f, ledf = _plunge_run(0.000625)
    check("plunging particles swallowed (ids 0-2), safe ones kept", [0, 1, 2],
          sorted(e["id"] for e in st.removed.log))  # fmt: skip
    check("all swallowed on their first approach (plan alarm raised)", 3,
          len(st.removed.first_pass()))  # fmt: skip
    check("ledger: mass (present + swallowed)", led0["mass"], led["mass"], rtol=1e-15)
    de = abs(led["energy"] - led0["energy"]) / led0["abs_h"]
    de_f = abs(ledf["energy"] - led0f["energy"]) / led0f["abs_h"]
    print(f"   energy (present + swallowed): {de:.2e} -> {de_f:.2e} with half the step")
    check("ledger: energy error is time-step error (falls >= 3x per halving)", True, de / de_f >= 3)
    check("ledger: angular momentum (present + swallowed)", led0["angmom"], led["angmom"],
          rtol=1e-12, per_vector=True)  # fmt: skip
    check("removed particles were inside r_acc", True,
          all(e["r"] < 4.0 for e in st.removed.log))  # fmt: skip


def test_removal_itself_moves_totals_exactly(rng):
    """At the instant of removal the totals do not change: what leaves the gas enters the
    removed ledger (hydro on, warm ball; one particle removed by hand)."""
    x, v, u, m, h = warm_ball(rng)
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0, r_acc=4.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 4)
    for _ in range(3):
        blocks.advance(st, cfg)
    before = blocks.ledger(st, cfg)
    st.x[7] = np.array([3.0, 0.0, 0.0])  # pretend particle 7 is inside r_acc at its step end
    st.x_beg[7] = st.x[7]
    before = blocks.ledger(st, cfg)
    rows = np.flatnonzero(st.n_end == st.n_end.min())
    rows = np.union1d(rows, [7])
    blocks._swallow(st, cfg, rows)
    after = blocks.ledger(st, cfg)
    check("one particle removed", 1, st.removed.count)
    check("energy unchanged by the removal", before["energy"], after["energy"], rtol=1e-14)
    check("angular momentum unchanged", before["angmom"], after["angmom"], rtol=1e-14,
          per_vector=True)  # fmt: skip
    check("momentum unchanged", before["momentum"], after["momentum"], rtol=1e-13,
          per_vector=True)  # fmt: skip
    check("arrays pruned consistently", len(st.x), len(st.ids))


def test_resume_from_checkpoint_is_bit_identical(rng):
    x, v, u, m, h = warm_ball(rng)
    u = u * np.where(np.arange(len(u)) % 5 == 0, 30.0, 1.0)
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=64.0, r_acc=4.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=64.0 * 8)
    for _ in range(15):
        blocks.advance(st, cfg)
    resumed = pickle.loads(pickle.dumps(copy.deepcopy(st)))
    for _ in range(25):
        blocks.advance(st, cfg)
        blocks.advance(resumed, cfg)
    check("resumed run: positions identical", st.x, resumed.x, rtol=0.0, per_vector=True)
    check("resumed run: u identical", st.u, resumed.u, rtol=0.0)
    check("resumed run: tick identical", st.n, resumed.n)


def test_run_folder_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(runio, "RUNS", tmp_path)
    folder = runio.new_run("demo", {"n": 10})
    manifest = (folder / "manifest.json").read_text()
    keys = ("commit", "config", "versions", "machine", "start")
    check("manifest has commit, config, versions", True, all(k in manifest for k in keys))
    with pytest.raises(FileExistsError):
        runio.new_run("demo", {"n": 10})


def test_hole_resolution_flags_kernels_across_the_hole():
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    x = np.column_stack([20 * np.cos(ang), 20 * np.sin(ang), np.zeros(40)])
    u = np.full(40, 1e-6)
    small = hole_resolution(x, np.full(40, 2.0), u)
    big = hole_resolution(x, np.full(40, 30.0), u)  # 2h = 60 > the ring's diameter 40
    check("ring at r = 20, h = 2: no pairs across the hole", 0.0, small.across_fraction)
    check("h = 30: pairs across the hole, warning", (True, True),
          (big.across_fraction > 0.3, big.warning))  # fmt: skip
    check("2h/r reported", 3.0, big.two_h_over_r_median, rtol=1e-12)
    assert gn_p_from_v is not None
