"""Script 22: the Sedov-Taylor blast-wave constant xi0, computed independently (added during
Phase 3, Step 3.7; expected-values §4 had quoted 1.1517 from the literature, unverified).

Strong point explosion of energy E in uniform gas (density rho0, negligible pressure), 3D:
    R(t) = xi0 (E t^2 / rho0)^(1/5)
Self-similar ansatz with xi = r / R(t), R ∝ t^(2/5):
    v = (R/t) phi(xi),  rho = rho0 psi(xi),  p = rho0 (R/t)^2 pi(xi)
Euler equations (mass, momentum, entropy) become ODEs in xi:
    (phi - 2xi/5) psi' + psi phi'           = -2 psi phi / xi
    (phi - 2xi/5) phi' + pi' / psi          =  (3/5) phi
    pi' - gamma (pi/psi) psi'               =  (6/5) pi / (phi - 2xi/5)
Strong-shock (Rankine-Hugoniot) values at xi = 1, with shock speed D = (2/5) R/t:
    psi = (g+1)/(g-1),  phi = (2/5) 2/(g+1),  pi = 2 (2/5)^2 / (g+1)
Energy: E = rho0 R^5 / t^2 * J,  J = int_0^1 (psi phi^2 / 2 + pi/(g-1)) 4 pi xi^2 dxi,
so xi0 = J^(-1/5). Integrated inward from the shock with DOP853.

Expected: gamma = 5/3 -> 1.1517 (Kamm & Timmes 2007); gamma = 1.4 -> 1.033 (the classic
value for air, e.g. Landau & Lifshitz, Fluid Mechanics §106). Needs numpy, scipy.
"""

import numpy as np
from scipy.integrate import solve_ivp


def xi0(gamma: float, xi_end: float = 1e-6) -> tuple[float, object]:
    g = gamma

    def rhs(xi, y):
        psi, phi, pi, _ = y
        w = phi - 0.4 * xi
        a = np.array([[w, psi, 0.0], [0.0, w, 1.0 / psi], [-g * pi / psi, 0.0, 1.0]])
        b = np.array([-2.0 * psi * phi / xi, 0.6 * phi, 1.2 * pi / w])
        dpsi, dphi, dpi = np.linalg.solve(a, b)
        dj = (0.5 * psi * phi**2 + pi / (g - 1.0)) * 4.0 * np.pi * xi**2
        return [dpsi, dphi, dpi, dj]

    y0 = [(g + 1) / (g - 1), 0.4 * 2 / (g + 1), 2 * 0.4**2 / (g + 1), 0.0]
    sol = solve_ivp(rhs, (1.0, xi_end), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    j = -sol.y[3, -1]  # integrated from 1 down to ~0
    return j ** (-0.2), sol


if __name__ == "__main__":
    for g in (5 / 3, 1.4):
        x, sol = xi0(g)
        print(f"gamma {g:.4f}: xi0 = {x:.6f}   (shock density ratio {(g + 1) / (g - 1):.3f})")
    x, sol = xi0(5 / 3)
    for xi in (0.9, 0.8, 0.6, 0.4):
        psi, phi, pi, _ = sol.sol(xi)
        print(f"  gamma 5/3, xi = {xi}: rho/rho0 = {psi:.5f}, v/(R/t) = {phi:.5f}, p/(rho0 (R/t)^2) = {pi:.5f}")
