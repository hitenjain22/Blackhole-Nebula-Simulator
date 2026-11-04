"""Rule 2 for our Phantom patch (phantom/nebula_phantom.patch; DECISIONS 2026-11-04): the
formulas copied into Phantom's Fortran (reconstructed viscosity jump ``nebula_recon``,
conductivity ``nebula_cond``, gradient fallback and corrected div/curl ``nebula_grad``, the
h cap with Omega = 1 ``nebula_hmax``) give the same hydro acceleration and du/dt as our Python
code (sim/sph.py, itself tested against the readable reference) on the same particles.

Data: tests/data/phantom_force_sample.npz, written by scripts/phantom_force_check.py from one
Phantom derivative evaluation of its R6.1 50k day-300 state (ph_r2; cooling off, all nebula
options on, nebula_recon 1 and 0): all particles within 1,800 of the hole, of which those
within 600 are compared (their neighbours, neighbours' gradients and densities are complete;
2 h_max = 400). Phantom's own pressure is used (our sound speed scaled to it): the equation of
state's physical constants differ slightly between the codes (Phantom physcon.f90 vs our
CODATA units.py; ~2e-4 in P), which is a constants difference, not a formula difference.

Tolerances: median relative difference 1e-5, 99th percentile 1e-3, per particle relative to
its own |a| (or |du/dt|). Sources of honest difference: Phantom's pressure, h and alpha are
stored as float32 in the dump, and each code converges h to its own tolerance (~1e-6 here).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nebula import units
from nebula.sim.coupled import CoupledConfig
from nebula.sim.hydro import eos_state
from nebula.sim.neighbours import tighten
from nebula.sim.sph import compute_density, forces_numba, velocity_gradients_numba
from tests.helpers import check

DATA = Path(__file__).resolve().parents[1] / "data" / "phantom_force_sample.npz"
MEDIAN_TOL, P99_TOL = 1e-5, 1e-3


@pytest.fixture(scope="module")
def sample():
    d = dict(np.load(DATA))
    hc = CoupledConfig().hydro
    m = d["mass_phantom"] * units.BlackHoleUnits().mass_kg / hc.gas_units.mass_kg_per_code
    d["mass"] = np.full(len(d["x"]), m)
    d["mratio"] = d["mass_phantom"] / m  # Phantom mass unit / ours (pressure scales with it)
    d["inner"] = np.linalg.norm(d["x"] - d["center"], axis=1) < d["r_in"]
    dens = compute_density(d["x"], d["mass"], d["h_phantom"], kernel_name=hc.kernel,
                           eta=hc.eta, h_max=hc.h_max)  # fmt: skip
    d["dens"] = dens
    d["nb"] = tighten(dens.neighbours, d["x"], dens.h)
    return d


def _ours(d, recon: bool):
    hc = replace(CoupledConfig().hydro, eos="gasrad", reconstruction=recon)
    dens = d["dens"]
    p_ours, cs_ours = eos_state(hc, dens.rho, d["u"])
    p = d["pressure"] / d["mratio"]
    cs = cs_ours * np.sqrt(p / p_ours)
    gv = velocity_gradients_numba(d["x"], d["v"], d["mass"], dens.h, d["nb"], hc.kernel)
    f = forces_numba(d["x"], d["v"], d["mass"], dens.h, dens.rho, dens.omega, p, d["nb"],
                     hc.kernel, u=d["u"], c_s=cs, alpha=d["alpha"], shock=hc.shock(),
                     grad_v=gv if recon else None)  # fmt: skip
    return f.acc, f.dudt


def _rel(ours, theirs):
    if ours.ndim == 2:
        mag = np.linalg.norm(ours, axis=1)
        diff = np.linalg.norm(ours - theirs, axis=1)
    else:
        mag, diff = np.abs(ours), np.abs(ours - theirs)
    return diff / np.maximum(mag, 1e-6 * np.median(mag))


def test_sample_contents(sample):
    d = sample
    inner = d["inner"]
    capped = d["dens"].h[inner] >= CoupledConfig().hydro.h_max * (1 - 1e-12)
    check("Phantom sample: particles compared (r < 600)", 2203, int(inner.sum()))
    check("... of which at the h cap (h = 200)", 93, int(capped.sum()))
    rel_h = np.abs(d["dens"].h - d["h_phantom"]) / d["dens"].h
    check("h: ours vs Phantom, median rel. diff < 1e-6", True,
          bool(np.median(rel_h[inner]) < 1e-6))  # fmt: skip


@pytest.mark.parametrize("recon", [True, False], ids=["recon_on", "recon_off"])
def test_phantom_patch_matches_our_forces(sample, recon):
    d = sample
    inner = d["inner"]
    acc, dudt = _ours(d, recon)
    tag = "on" if recon else "off"
    ra = _rel(acc, d[f"acc_{tag}"])[inner]
    ru = _rel(dudt, d[f"dudt_{tag}"])[inner]
    check(f"recon {tag}: |a| median rel. diff", 0.0, float(np.median(ra)), atol=MEDIAN_TOL)
    check(f"recon {tag}: |a| 99th pct rel. diff", 0.0, float(np.percentile(ra, 99)), atol=P99_TOL)
    check(f"recon {tag}: du/dt median rel. diff", 0.0, float(np.median(ru)), atol=MEDIAN_TOL)
    check(f"recon {tag}: du/dt 99th pct rel. diff", 0.0, float(np.percentile(ru, 99)),
          atol=P99_TOL)  # fmt: skip


def test_reconstruction_changes_the_result(sample):
    """The comparison is sensitive to the copied formula: switching reconstruction changes
    Phantom's own result by far more than the tolerance in this hot, shocked region."""
    d = sample
    inner = d["inner"]
    r = _rel(d["acc_on"], d["acc_off"])[inner]
    check("Phantom recon on vs off: |a| 90th pct rel. diff > 100 x P99_TOL", True,
          bool(np.percentile(r, 90) > 100 * P99_TOL))  # fmt: skip


# --------------------------------------------------------------------------- nebula_pkick
#
# The patch's nebula_pkick (force.F90) turns Phantom's hydro acceleration into the force on the
# GN canonical momentum, as our integrator does (phase-04 4.1): stored fxyz = Minv(x) a_hydro,
# a Fortran copy of physics/gravity.gn_mass_matrix_inv. Data: tests/data/phantom_pkick_sample.npz
# (scripts/phantom_pkick_sample.py): Phantom's start-up pass on its 50k state at day 96 (closest
# approach, r down to 22), pkick on and off, 2,000 particles. Both are float64; the only
# honest difference is round-off, so the tolerance is 1e-12.

PKICK = Path(__file__).resolve().parents[1] / "data" / "phantom_pkick_sample.npz"


def test_phantom_pkick_matches_gn_mass_matrix_inv():
    from nebula.physics.gravity import gn_mass_matrix_inv

    d = np.load(PKICK)
    ours = np.einsum("nij,nj->ni", gn_mass_matrix_inv(d["x"]), d["acc_pkick_off"])
    scale = np.linalg.norm(d["acc_pkick_off"], axis=1)
    rel = np.linalg.norm(d["acc_pkick_on"] - ours, axis=1) / scale
    change = np.linalg.norm(d["acc_pkick_on"] - d["acc_pkick_off"], axis=1) / scale
    check("pkick: Phantom Minv a vs gn_mass_matrix_inv, max rel. diff", 0.0, float(rel.max()),
          atol=1e-12)  # fmt: skip
    check("pkick: the transform itself changes a by > 1% (test is sensitive)", True,
          bool(change.max() > 1e-2))  # fmt: skip
