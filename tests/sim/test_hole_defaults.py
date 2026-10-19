"""The settings decided for runs with the black hole (DECISIONS 2026-10-19), and that older runs
and the test problems keep theirs."""

import importlib.util
from pathlib import Path

import numpy as np

from nebula.physics import viscosity as visc
from nebula.sim.coupled import H_MAX_HOLE, CoupledConfig
from nebula.sim.hydro import HydroConfig
from nebula.sim.sph import COND_PRESSURE, COND_VELOCITY
from tests.helpers import check

ROOT = Path(__file__).resolve().parents[2]


def _cloud_run():
    spec = importlib.util.spec_from_file_location("cloud_run", ROOT / "scripts" / "cloud_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hole_runs_use_magma2_conductivity_and_the_h_cap():
    hc = CoupledConfig().hydro
    check("with the hole: conductivity strength (MAGMA2, Rosswog 2020)", 0.05, hc.alpha_u)
    check("the constant matches the paper's value", 0.05, visc.ALPHA_U_MAGMA2)
    check("with the hole: velocity signal (Phantom Eq. 44)", COND_VELOCITY, hc.conductivity_signal)
    check("with the hole: h cap", 200.0, hc.h_max)
    check("H_MAX_HOLE", 200.0, H_MAX_HOLE)


def test_test_problems_keep_phantom_conductivity():
    """Phase 3 test problems (no hole) are unchanged: alpha_u = 1, pressure signal, no cap."""
    hc = HydroConfig()
    check("test problems: alpha_u", 1.0, hc.alpha_u)
    check("test problems: pressure signal", COND_PRESSURE, hc.conductivity_signal)
    check("test problems: no h cap", True, bool(np.isinf(hc.h_max)))


def test_cloud_run_new_and_legacy_configs():
    """New runs take the decided settings; a run started before them (no alpha_u / h_max in its
    manifest) resumes with what it ran with: alpha_u = 1, no cap."""
    cr = _cloud_run()
    new = cr.block_config(cr.make_config({})).coupled.hydro
    check("new run: alpha_u", 0.05, new.alpha_u)
    check("new run: h_max", 200.0, new.h_max)
    legacy = {"n": 10000, "radius": 60.0, "tend": 1.0, "r_acc": 4.0, "recon": True,
              "dt_max": 4096.0, "eos": "gasrad", "beta": 230.0}  # fmt: skip
    old = cr.block_config(legacy).coupled.hydro
    check("legacy run resumes with alpha_u = 1", 1.0, old.alpha_u)
    check("legacy run resumes without a cap", True, bool(np.isinf(old.h_max)))
