"""Disc viscosity through the AV form (physics/disc_viscosity.py, Phantom Eqs. 120-124) on
hand-worked values."""

from nebula.physics.disc_viscosity import alpha_av_for, alpha_ss_from_av, disc_q, nu_av
from tests.helpers import check


def test_disc_q_hand_value():
    """q = -rho h/(2 r) alpha c_s vdr = -2 x 0.5/(2 x 0.25) x 0.1 x 0.3 x (-0.04) = +2.4e-3
    (approaching: positive, like a pressure); receding flips the sign."""
    check("q, approaching pair", 2.4e-3, float(disc_q(2.0, 0.5, 0.25, 0.1, 0.3, -0.04)),
          rtol=1e-14)  # fmt: skip
    check("q, receding pair", -2.4e-3, float(disc_q(2.0, 0.5, 0.25, 0.1, 0.3, 0.04)), rtol=1e-14)


def test_equivalent_viscosity_and_alpha_mapping():
    check("nu_AV = alpha c_s h / 10", 0.1 * 0.3 * 0.5 / 10, float(nu_av(0.1, 0.3, 0.5)),
          rtol=1e-14)  # fmt: skip
    # Phantom Eq. 124: alpha_AV = 1, <h>/H = 0.5 -> alpha_SS = 0.05
    check("alpha_SS = alpha_AV <h> / (10 H)", 0.05, float(alpha_ss_from_av(1.0, 0.5, 1.0)),
          rtol=1e-14)  # fmt: skip
    check("alpha_av_for inverts it", 0.7, float(alpha_ss_from_av(alpha_av_for(0.7, 0.3, 2.0),
          0.3, 2.0)), rtol=1e-14)  # fmt: skip
