"""Energy-consistent removal of gas (blocks.py module docstring "Removal"; DECISIONS
2026-11-03). Warm stirred ball (tests/sim/test_blocks.py) with an escape radius inside its
outer edge, so gas leaves through it during the run while its neighbours are still pushing on
it. Old removal: the survivors kept a half-kick from each removed particle that nothing paid
for, an energy and angular momentum error of order dt per removal (the R6.1 v2 ledger drift).
Each case is compared with the same ball with nothing removed, which has only the ordinary
time-integration error."""

import numpy as np

from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def _errors(r_escape, removal_sync, max_gap, mixed, dt_max, t_stop=128.0):
    """Ledger energy change / sum m|H| and |angular momentum change| / |L| over the run."""
    x, v, u, m, h = warm_ball(np.random.default_rng(7))
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=dt_max, r_acc=4.0, r_escape=r_escape,
                      max_gap=max_gap, removal_sync=removal_sync)  # fmt: skip
    offsets = np.random.default_rng(3).integers(0, 4, len(x)) if mixed else None
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=t_stop, level_offset=offsets)
    a = blocks.ledger(st, cfg)
    blocks.run(st, cfg, t_stop)
    b = blocks.ledger(st, cfg)
    de = (b["energy"] - a["energy"]) / a["abs_h"]
    dl = np.linalg.norm(b["angmom"] - a["angmom"]) / np.linalg.norm(a["angmom"])
    return de, dl, st.removed.count


def test_removal_on_one_level_adds_no_error():
    """All particles on one level (dt_max 2, no level gap): with removal_sync each pair gets the
    same kicks up to the removal and none after, so angular momentum (exact for GN gravity,
    a central force) is conserved to round-off with removals, and the energy error is that of
    the run with nothing removed (measured 0.7% apart; the old removal 330%)."""
    e_none, _, n_none = _errors(None, True, 0, False, 2.0)
    e_sync, l_sync, n_sync = _errors(502.4, True, 0, False, 2.0)
    e_old, l_old, _ = _errors(502.4, False, 0, False, 2.0)
    print(f"   removed {n_sync}; energy: none {e_none:+.4e}  sync {e_sync:+.4e}  old {e_old:+.4e}"
          f"; L: sync {l_sync:.1e}  old {l_old:.1e}")  # fmt: skip
    check("removal one level: nothing removed in the reference", 0, n_none)
    check("removal one level: gas removed", True, n_sync >= 100)
    check("removal one level: angular momentum change (sync)", 0.0, l_sync, atol=1e-13)
    check("removal one level: old removal leaks angular momentum", True, l_old > 1e-6)
    check("removal one level: energy error = no-removal run's", e_none, e_sync, rtol=0.02)
    check("removal one level: old removal adds energy error", True,
          abs(e_old - e_none) > 0.5 * abs(e_none))  # fmt: skip


def test_removal_on_mixed_levels():
    """Random forced level offsets (0-3) with the limiter (max_gap 2): neighbours of a particle
    about to be removed are woken to end their steps with it. The energy error the removals add
    (difference from the no-removal run) is under 2% of what the old removal added (measured
    0.5%), and the angular momentum error under 20% of the old one (measured 8%; mixed levels
    are not exactly conservative even without removals: 1.6e-6 here)."""
    e_none, l_none, _ = _errors(None, True, 2, True, 1.0)
    e_sync, l_sync, n_sync = _errors(502.8, True, 2, True, 1.0)
    e_old, l_old, _ = _errors(502.8, False, 2, True, 1.0)
    print(f"   removed {n_sync}; energy: none {e_none:+.3e}  sync {e_sync:+.3e}  old {e_old:+.3e}"
          f"; L: none {l_none:.1e}  sync {l_sync:.1e}  old {l_old:.1e}")  # fmt: skip
    check("removal mixed: gas removed", True, n_sync >= 100)
    check("removal mixed: added energy error / old's < 0.02", True,
          abs(e_sync - e_none) < 0.02 * abs(e_old - e_none))  # fmt: skip
    check("removal mixed: L error / old's < 0.2", True, l_sync < 0.2 * l_old)
