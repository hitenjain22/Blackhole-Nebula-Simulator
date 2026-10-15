"""Step 2.1: float64 reference light rays (phases/phase-02-blackhole-render.md; expected-values §3).

Deflection and shadow values carry extra digits from reference/scripts/21 (mpmath), so the
printed rounding is far below the rtol 1e-7 tests; they agree with the §3 values as printed.
"""

import numpy as np

from nebula.physics.geodesics import (
    B_CRIT,
    schw_deflection,
    schw_einstein_ring_angle,
    schw_impact_parameter,
    schw_ray_escape_angle,
    schw_ray_path,
    schw_shadow_angle,
)
from tests.helpers import check


def test_deflection():
    table = {  # b: (deflection rad, rtol) ; §3
        1000.0: (0.004011823809925, 1e-7),
        50.0: (0.08508345038325, 1e-7),
        10.0: (0.5903957876058, 1e-7),
        6.0: (1.71938831023, 1e-7),
        5.3: (3.55793804246, 1e-6),
    }
    for b, (expected, rtol) in table.items():
        check(f"§3 deflection b={b:g} (rad)", expected, schw_deflection(b), rtol=rtol)
    check("weak field: b=1000 deflection ~ 4/b", 4e-3, schw_deflection(1000.0), rtol=0.01)


def test_capture():
    check("§3 b=5.19 captured", True, bool(np.isnan(schw_deflection(5.19))))
    check("§3 b=5.20 escapes", True, bool(np.isfinite(schw_deflection(5.20))))
    check("critical impact parameter 3 sqrt(3)", 5.196152, B_CRIT, rtol=1e-7)


def test_shadow():
    table = {20.0: 14.2690273279, 50.0: 5.84417829399, 100.0: 2.94855519768, 1000.0: 0.297421072645}
    for d, expected in table.items():
        check(
            f"§3 shadow radius D={d:g} (deg)", expected, np.degrees(schw_shadow_angle(d)), rtol=1e-7
        )


def test_shadow_edge_is_capture_boundary():
    """Rays just inside the shadow angle are captured, just outside escape (D = 50)."""
    psi_sh = schw_shadow_angle(50.0)
    inside = schw_ray_escape_angle(50.0, psi_sh * (1 - 1e-6))
    outside = schw_ray_escape_angle(50.0, psi_sh * (1 + 1e-6))
    check("D=50: psi_sh (1 - 1e-6) captured", True, bool(np.isnan(inside)))
    check("D=50: psi_sh (1 + 1e-6) escapes", True, bool(np.isfinite(outside)))
    check(
        "D=50: b at shadow edge = 3 sqrt(3)",
        B_CRIT,
        schw_impact_parameter(50.0, psi_sh),
        rtol=1e-12,
    )


def test_escape_angles():
    table = {  # (D, psi deg): phi_inf deg ; §3
        (50.0, 10.0): 210.86090,
        (50.0, 20.0): 175.52870,
        (50.0, 45.0): 140.94777,
        (50.0, 80.0): 102.85233,
        (20.0, 20.0): 219.43474,
        (20.0, 45.0): 151.84667,
        (20.0, 80.0): 107.64767,
    }
    for (d, psi), expected in table.items():
        got = np.degrees(schw_ray_escape_angle(d, np.radians(psi)))
        check(f"§3 escape angle D={d:g}, psi={psi:g} (deg)", expected, got, atol=1e-5)
    check(
        "§3 D=20, psi=10 captured",
        True,
        bool(np.isnan(schw_ray_escape_angle(20.0, np.radians(10)))),
    )
    check("looking straight away (psi=180) sees e1", 0.0, schw_ray_escape_angle(50.0, np.pi))
    check(
        "looking straight at the hole: captured",
        True,
        bool(np.isnan(schw_ray_escape_angle(50.0, 0.0))),
    )


def test_einstein_rings():
    table = {(50.0, 0): 17.872702, (50.0, 1): 5.850814, (20.0, 0): 29.828318, (20.0, 1): 14.283091}
    for (d, order), expected in table.items():
        name = "primary" if order == 0 else "secondary"
        got = np.degrees(schw_einstein_ring_angle(d, order))
        check(f"§3 Einstein ring D={d:g} {name} (deg)", expected, got, atol=1e-5)


def test_ray_path_obeys_first_integral():
    """Along the path, (du/dphi)^2 = 1/b^2 - u^2 (1 - 2u) (finite differences of the samples)."""
    d, psi = 50.0, np.radians(20.0)
    phi, pos = schw_ray_path(d, psi, 4001)
    u = 1.0 / np.linalg.norm(pos, axis=1)
    dudphi = np.gradient(u, phi)[1:-1]
    b = schw_impact_parameter(d, psi)
    rhs = 1 / b**2 - u[1:-1] ** 2 * (1 - 2 * u[1:-1])
    check("ray path starts at the camera (r = D)", d, np.linalg.norm(pos[0]), rtol=1e-12)
    check("ray path: first integral holds", rhs, dudphi**2, atol=1e-8)
    check("ray path closest approach > 3 (escaping ray)", True, bool(u.max() < 1 / 3))
