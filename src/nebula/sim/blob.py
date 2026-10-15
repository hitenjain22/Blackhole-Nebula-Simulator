"""The hot-blob energy-ledger experiment (phases/phase-04-timestepping.md 4.2 test 4;
expected-values §2 hot-blob orbit).

A warm gas blob (about 500 glass particles, radius 3, gamma 5/3, u = U_BLOB; no cooling, no
accretion) starts at apocentre x = (500, 0, 0), p = (0, ell/500, 0) on the
r_a = 500 -> r_p = 25 orbit and is followed through the pericentre squeeze that heats it
(pericentre at t ~ 13,500) and back out, to T_END_BLOB. Recorded: the relative ledger error
|delta sum m (H + u)| / sum m |H| and the relative angular-momentum error after every step
(block steps: at every full sync point).
"""

import numpy as np

from nebula.physics.gravity import gn_angmom, gn_v_from_p
from nebula.sim import blocks, coupled
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from nebula.sim.glass import load_glass
from nebula.sim.hydro import HydroConfig

ELL_BLOB = 7.1951756691  # expected-values §2: r_a = 500, r_p = 25
R_APO = 500.0
# The plan's u = 4e-6 (sound speed 2e-3) does not keep the blob together: with no self-gravity
# it expands freely at up to ~3 c_s, which changes some particles' angular momentum enough
# (7.2 -> 3.2) for them to plunge into the hole before pericentre. 100x cooler (c_s ~ 2e-4)
# keeps every particle on a safe orbit (DECISIONS, Step 4.2).
U_BLOB = 4e-8
# Not a whole radial orbit (~27,100): at r_p = 25 the tides tear the blob apart, and its most
# bound debris returns for a second pericentre from t ~ 19,000, collides with the stream, loses
# its angular momentum and falls in (DECISIONS, Step 4.2). 18,000 covers the whole first
# passage (including the largest gravity error, at pericentre) and stops before any return.
T_END_BLOB = 18_000.0


def blob_positions(radius: float = 3.0, n_target: int = 500) -> np.ndarray:
    g = load_glass(4096) - 0.5
    r_unit = (n_target / 4096 / (4 / 3 * np.pi)) ** (1 / 3)
    return g[np.linalg.norm(g, axis=1) < r_unit] * (radius / r_unit)


def _initial(cfg: CoupledConfig):
    pos = blob_positions() + np.array([R_APO, 0.0, 0.0])
    n = len(pos)
    p0 = np.tile([0.0, ELL_BLOB / R_APO, 0.0], (n, 1))
    rho = 1.0 / (4 / 3 * np.pi * 27.0)
    return (
        pos,
        gn_v_from_p(pos, p0),
        np.full(n, U_BLOB),
        np.full(n, 1.0 / n),
        np.full(n, 1.6 * (1 / n / rho) ** (1 / 3)),
    )


def _errors(led, led0):
    return abs(led["energy"] - led0["energy"]) / led0["abs_h"], float(
        np.linalg.norm(led["angmom"] - led0["angmom"]) / np.linalg.norm(led0["angmom"])
    )


def hot_blob(
    eta_orb: float, courant: float, wrong: bool = False, block_steps: bool = False, **hydro_kw
):
    """Returns a dict: t, energy and angular-momentum error series, max of each, the largest u
    reached, the lowest particle angular momentum and largest h seen (the blob must stay a
    blob), and the number of steps (particle-steps for block steps)."""
    cc = CoupledConfig(
        hydro=HydroConfig(courant=courant, conductivity_signal=coupled.COND_VELOCITY, **hydro_kw),
        eta_orb=eta_orb,
        wrong_coupling=wrong,
    )
    x, v, u, m, h = _initial(cc)
    t_series, e_series, l_series = [0.0], [0.0], [0.0]
    worst = {"L_min": np.inf, "h_max": 0.0, "u_max": 0.0}

    def watch(x, p, h, u):  # the blob must stay a blob: no particle near a plunging orbit
        worst["L_min"] = min(worst["L_min"], float(np.min(np.linalg.norm(gn_angmom(x, p), axis=1))))
        worst["h_max"] = max(worst["h_max"], float(np.max(h)))
        worst["u_max"] = max(worst["u_max"], float(np.max(u)))

    if not block_steps:
        st = coupled.initialise(x, v, u, m, h, cc)
        led0 = coupled.ledger(st, cc)
        steps = 0
        while st.t < T_END_BLOB * (1 - 1e-14):
            st = coupled.step(st, cc, min(coupled.stable_dt(st, cc), T_END_BLOB - st.t))
            steps += 1
            e, ell = _errors(coupled.ledger(st, cc), led0)
            watch(st.x, st.p, st.h, st.u)
            t_series.append(st.t)
            e_series.append(e)
            l_series.append(ell)
        work = steps
    else:
        if wrong:
            raise ValueError("the wrong-coupling comparison is for the shared step only")
        cfg = BlockConfig(coupled=cc, dt_max=64.0)
        st = blocks.initialise(x, v, u, m, h, cfg, t_stop=T_END_BLOB)
        led0 = blocks.ledger(st, cfg)
        while st.n < st.n_stop:
            blocks.advance(st, cfg)
            if not st.synced():
                continue
            e, ell = _errors(blocks.ledger(st, cfg), led0)
            watch(st.x, st.p, st.h, st.u)
            t_series.append(st.t)
            e_series.append(e)
            l_series.append(ell)
        work = st.particle_steps
    e_series, l_series = np.array(e_series), np.array(l_series)
    return {
        "t": np.array(t_series),
        "energy_error": e_series,
        "angmom_error": l_series,
        "max_energy_error": float(e_series.max()),
        "max_angmom_error": float(l_series.max()),
        "u_max": worst["u_max"],
        "steps": int(work),
        "min_particle_angmom": worst["L_min"],
        "max_h": worst["h_max"],
        "n": len(x),
    }


def cached_hot_blob(eta_orb: float, courant: float, wrong: bool = False, block_steps: bool = False):
    """``hot_blob`` through the run cache (shared by the slow tests and scripts/plot_ledger.py)."""
    from nebula.sim.runcache import cached

    name = f"hotblob_{eta_orb}_{courant}_{'wrong' if wrong else 'right'}"
    name += "_blocks" if block_steps else ""
    return cached(name, lambda: hot_blob(eta_orb, courant, wrong, block_steps))
