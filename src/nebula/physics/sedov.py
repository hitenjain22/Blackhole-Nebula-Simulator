"""Sedov-Taylor point explosion: exact self-similar solution (phases/phase-03-sph-core.md 3.2
test 7; expected-values §4, verified by reference/scripts/22 with the same method).

Energy E released at a point in uniform gas of density rho0 and negligible pressure:
    R(t) = xi0 (E t^2 / rho0)^(1/5)
With xi = r / R(t): v = (R/t) phi(xi), rho = rho0 psi(xi), p = rho0 (R/t)^2 pi(xi), where
    (phi - 2xi/5) psi' + psi phi'          = -2 psi phi / xi
    (phi - 2xi/5) phi' + pi' / psi         =  (3/5) phi
    pi' - gamma (pi/psi) psi'              =  (6/5) pi / (phi - 2xi/5)
with strong-shock values at xi = 1: psi = (g+1)/(g-1), phi = (4/5)/(g+1), pi = (8/25)/(g+1);
and xi0 = J^(-1/5), J = int_0^1 (psi phi^2/2 + pi/(g-1)) 4 pi xi^2 dxi (Sedov 1959;
Landau & Lifshitz, Fluid Mechanics §106; Kamm & Timmes 2007). Dimensionless.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SedovSolution:
    gamma: float
    xi0: float
    sol: object  # dense solution in xi from 1 down to ~0: y = (psi, phi, pi, -J(xi))

    def profile(self, xi):
        """(rho/rho0, v/(R/t), p/(rho0 (R/t)^2)) at xi = r/R in [1e-6, 1]; zero ahead of the shock
        is replaced by the undisturbed gas (rho/rho0 = 1, v = p = 0) for xi > 1."""
        xi = np.atleast_1d(np.asarray(xi, float))
        inside = xi <= 1.0
        psi, phi, pi = np.ones_like(xi), np.zeros_like(xi), np.zeros_like(xi)
        if np.any(inside):
            y = self.sol.sol(np.clip(xi[inside], 1e-6, 1.0))
            psi[inside], phi[inside], pi[inside] = y[0], y[1], y[2]
        return psi, phi, pi

    def radius(self, t, energy: float = 1.0, rho0: float = 1.0):
        return self.xi0 * (energy * t * t / rho0) ** 0.2


@lru_cache(maxsize=8)
def sedov_solution(gamma: float) -> SedovSolution:
    g = gamma

    def rhs(xi, y):
        psi, phi, pi, _ = y
        w = phi - 0.4 * xi
        a = np.array([[w, psi, 0.0], [0.0, w, 1.0 / psi], [-g * pi / psi, 0.0, 1.0]])
        b = np.array([-2.0 * psi * phi / xi, 0.6 * phi, 1.2 * pi / w])
        dpsi, dphi, dpi = np.linalg.solve(a, b)
        dj = (0.5 * psi * phi**2 + pi / (g - 1.0)) * 4.0 * np.pi * xi**2
        return [dpsi, dphi, dpi, dj]

    y0 = [(g + 1) / (g - 1), 0.8 / (g + 1), 0.32 / (g + 1), 0.0]
    sol = solve_ivp(
        rhs, (1.0, 1e-6), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True
    )
    j = -sol.y[3, -1]
    return SedovSolution(g, float(j ** (-0.2)), sol)
