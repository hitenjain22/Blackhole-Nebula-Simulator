"""The Numba copies in sim/fastphys.py against the physics/ functions they mirror (rule 2)."""

import numpy as np

from nebula import units
from nebula.physics.eos import T_MAX_SI, T_MIN_SI, gasrad_state, gasrad_temperature_si, gasrad_u_si
from nebula.physics.gravity import gn_v_from_p
from nebula.sim.fastphys import gasrad_state_nb, gasrad_temperature_nb, gn_v_from_p_nb
from tests.helpers import check


def test_gn_v_from_p_matches_physics(rng):
    r = rng.uniform(2.5, 6000.0, 5000)
    x = rng.normal(0.0, 1.0, (5000, 3))
    x *= (r / np.linalg.norm(x, axis=1))[:, None]
    p = rng.normal(0.0, 0.3, (5000, 3))
    check("GN v from p: Numba vs physics", gn_v_from_p(x, p), gn_v_from_p_nb(x, p), rtol=1e-14,
          per_vector=True)  # fmt: skip


def test_gasrad_temperature_matches_physics(rng):
    """Over the whole range the runs see and beyond (rho 1e-15..1e3 kg/m^3, T 10..1e11 K)."""
    rho = 10.0 ** rng.uniform(-15.0, 3.0, 20000)
    t_true = 10.0 ** rng.uniform(1.0, 11.0, 20000)
    u = gasrad_u_si(rho, t_true)
    ref = gasrad_temperature_si(rho, u)
    fast = gasrad_temperature_nb(rho, u)
    # both stop once a log-T step is below EOS_TOL = 1e-12, so each is within 1e-12 of the
    # root (measured: 1.00e-12 for both) and the two can differ by up to 2e-12
    check("gas + radiation T: Numba vs the T that made u", t_true, fast, rtol=1.1e-12)
    check("gas + radiation T: physics vs the T that made u", t_true, ref, rtol=1.1e-12)
    check("gas + radiation T: Numba vs physics", ref, fast, rtol=2.5e-12)


def test_gasrad_state_matches_physics(rng):
    rho = 10.0 ** rng.uniform(-9.0, -3.0, 3000)  # code density (cloud scale)
    u = 10.0 ** rng.uniform(-9.0, -3.0, 3000)  # code u
    gu = units.GasUnits()
    for name, a, b in zip(("P", "c_s", "T", "beta"), gasrad_state(rho, u, gu),
                          gasrad_state_nb(rho, u, gu), strict=True):  # fmt: skip
        # P and beta scale at most as T^4 (radiation), so 4x the T difference
        check(f"gas + radiation state {name}: Numba vs physics", a, b, rtol=1e-11)


def test_gasrad_temperature_rejects_out_of_range():
    import pytest

    rho = np.array([1e-6])
    with pytest.raises(ValueError):
        gasrad_temperature_nb(rho, gasrad_u_si(rho, 0.5 * T_MIN_SI))
    with pytest.raises(ValueError):
        gasrad_temperature_nb(rho, gasrad_u_si(rho, 2.0 * T_MAX_SI))
