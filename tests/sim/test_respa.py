"""Gravity sub-stepping on block steps (sim/blocks.py cfg.respa; Price et al. 2018 §2.3.3):
with the gas-step cap equal to the orbit criterion every sub-step is the whole step, so the
scheme must reproduce the ordinary block steps result for result (wake-ups included)."""

import numpy as np
import pytest

from nebula.sim import blocks
from nebula.sim.blocks import BlockConfig
from nebula.sim.coupled import CoupledConfig
from tests.helpers import check
from tests.sim.test_blocks import warm_ball


def test_respa_with_equal_criteria_is_the_ordinary_scheme(rng):
    x, v, u, m, h = warm_ball(rng)
    offset = np.where(rng.random(len(u)) < 0.05, 5, 0)
    cc = CoupledConfig()
    out = {}
    for respa in (False, True):
        cfg = BlockConfig(coupled=cc, dt_max=64.0, r_acc=4.0, respa=respa, respa_eta=cc.eta_orb)
        st = blocks.initialise(x, v, u, m, h, cfg, t_stop=128.0, level_offset=offset)
        blocks.run(st, cfg, 128.0)
        out[respa] = st
    a, b = out[False], out[True]
    check("wake-ups exercised", True, b.woken > 0)
    check("RESPA (equal criteria): positions", a.x, b.x, rtol=0.0, per_vector=True)
    check("RESPA (equal criteria): momenta", a.p, b.p, rtol=0.0, per_vector=True)
    check("RESPA (equal criteria): internal energies", a.u, b.u, rtol=0.0)
    check("RESPA (equal criteria): system steps", len(a.active_counts), len(b.active_counts))


@pytest.mark.slow
def test_respa_hot_blob_ledger():
    """Plan test 4 (hot blob on its orbit) with gravity sub-stepping: the energy ledger meets the
    block-step requirement (< 1e-5 at (0.00125, 0.15)), and the angular-momentum drift, now set
    by the longer gas steps, is second order in the gas-step cap respa_eta. Measured 2026-10-21:
    energy 4.06e-6 (ordinary), 4.20e-6 (cap 0.02), 5.70e-6 (cap 0.05); |dL|/L 2.6e-11,
    8.5e-9, 5.1e-8 (ratio 6.0 for a 2.5x cap change; second order: 6.25); particle-steps 4.61M,
    0.47M, 0.37M."""
    from nebula.sim.blob import cached_hot_blob

    r02 = cached_hot_blob(0.00125, 0.15, block_steps=True, respa=True, respa_eta=0.02)
    r05 = cached_hot_blob(0.00125, 0.15, block_steps=True, respa=True, respa_eta=0.05)
    for name, r in (("0.02", r02), ("0.05", r05)):
        check(f"RESPA cap {name}: hot blob max |dE| / sum m|H|", 0.0, r["max_energy_error"],
              atol=1e-5)  # fmt: skip
    order = np.log(r05["max_angmom_error"] / r02["max_angmom_error"]) / np.log(2.5)
    print(f"   |dL|/L {r02['max_angmom_error']:.2e} (0.02), {r05['max_angmom_error']:.2e} (0.05); "
          f"steps {r02['steps']:,}, {r05['steps']:,}")  # fmt: skip
    check("angular-momentum drift: order in the cap (second order: 2)", 2.0, order, atol=0.5)
    check("blob on safe orbits (lowest L > 5)", True, r02["min_particle_angmom"] > 5)
