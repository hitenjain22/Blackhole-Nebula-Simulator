"""Step 4.4: time-step limiter and wake-up (phases/phase-04-timestepping.md 4.2 test 3; Saitoh &
Makino 2009; Durier & Dalla Vecchia 2012 App. C)."""

import numpy as np
import pytest

from nebula.physics.sedov import sedov_solution
from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.hydro import HydroConfig
from nebula.sim.testproblems import SEDOV_RADII, sedov_exact_radius, sedov_radius, sedov_setup
from nebula.sim.timesteps import ticks
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def test_limiter_and_wake_up_invariants():
    """A very hot particle in a cold ball (its steps ~100x shorter than the rest). After
    every system step: each active particle and each of its neighbours are within 2 levels;
    every step ends on its level's grid; and every woken particle's opening kick was corrected
    by exactly a (L' - L) / 2 for its new step length L' (Durier & Dalla Vecchia Eq. C5)."""
    rng = np.random.default_rng(5)
    x, v, u, m, h = warm_ball(rng, u=1e-6, sigma_v=0.0)
    hot = int(np.argmin(np.linalg.norm(x - x.mean(axis=0), axis=1)))
    u[hot] = 10.0  # steps ~100x shorter than the orbit criterion sets for the rest
    cfg = BlockConfig(coupled=CoupledConfig(), dt_max=4096.0)
    st = blocks.initialise(x, v, u, m, h, cfg, t_stop=4096.0)
    gap_ok = grid_ok = kick_ok = True
    n_woken = 0
    for _ in range(60):
        p_half, u_half, n_end = st.p_half.copy(), st.u_half.copy(), st.n_end.copy()
        rows = blocks.advance(st, cfg)
        nb = st.last_pairs
        for i in rows:
            js = nb.index[nb.offset[i] : nb.offset[i + 1]]
            gap_ok &= bool(np.all(np.abs(st.level[js] - st.level[i]) <= 2))
        grid_ok &= bool(np.all(st.n_end % ticks(st.level) == 0))
        inactive = np.setdiff1d(np.arange(len(x)), rows)
        woken = inactive[st.n_end[inactive] != n_end[inactive]]
        n_woken += len(woken)
        if len(woken):
            d_len = (st.n_end[woken] - n_end[woken]) * st.tick
            kick_ok &= bool(
                np.allclose(
                    st.p_half[woken] - p_half[woken],
                    0.5 * d_len[:, None] * st.a_hydro[woken],
                    rtol=1e-12,
                    atol=1e-16 * np.max(np.abs(p_half)),
                )
                and np.allclose(
                    st.u_half[woken] - u_half[woken], 0.5 * d_len * st.dudt[woken], rtol=1e-12
                )
                and np.all(st.n_end[woken] < n_end[woken])
            )
    print(f"   levels {np.bincount(st.level).tolist()}, woken {n_woken}")
    check("wake-ups happened", True, n_woken > 0)
    check("active particles within 2 levels of every neighbour", True, gap_ok)
    check("every step ends on its level's grid", True, grid_ok)
    check("woken: steps only shortened, kicks corrected by a (L' - L)/2", True, kick_ok)


def _sedov_blocks(limiter: bool, tiles: int):
    prob = sedov_setup(tiles)
    hs = prob.state
    hc = HydroConfig(gamma=prob.cfg.gamma, box=prob.cfg.box)  # the usual Courant 0.3
    cfg = BlockConfig(coupled=CoupledConfig(hydro=hc, hole=False), dt_max=0.125, limiter=limiter)
    xi0 = sedov_solution(5 / 3).xi0
    times = [(r / xi0) ** 2.5 for r in SEDOV_RADII]
    st = blocks.initialise(hs.x, hs.v, hs.u, hs.mass, hs.h, cfg, t_stop=times[0])
    out = []
    for t in times:
        blocks.run(st, cfg, t)
        r_sph, _, _ = sedov_radius(st, prob.extra["h_bg"])
        energy = float(np.sum(st.mass * (0.5 * np.sum(st.p**2, axis=1) + st.u)))
        out.append((st.t, sedov_exact_radius(st.t), r_sph, energy))
    return out, st


@pytest.mark.slow
def test_sedov_needs_the_limiter():
    """Plan test 3: Sedov blast (110,592 particles) on block steps. With the limiter and
    wake-up the shock radius matches the self-similar law within 5% at three times; without
    them the cold particles on long steps let the blast through and the radius is visibly
    wrong (> 10% off at some time)."""
    with_lim, st = _sedov_blocks(True, 3)
    for t, r_exact, r_sph, energy in with_lim:
        check(f"limiter on: Sedov R at t={t:.4f}", r_exact, r_sph, rtol=0.05)
        print(f"   energy {energy:.6f}")
    print(f"   limiter on: woken {st.woken}, particle steps {st.particle_steps}")
    without, _ = _sedov_blocks(False, 3)
    worst = max(abs(r_sph / r_exact - 1) for _, r_exact, r_sph, _ in without)
    for t, r_exact, r_sph, energy in without:
        print(f"   limiter off: t={t:.4f} R {r_sph:.4f} vs {r_exact:.4f}, energy {energy:.4f}")
    check(
        "limiter off: radius error > 10% at some time (what the limiter fixes)", True, worst > 0.1
    )
