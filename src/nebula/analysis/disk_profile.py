"""Radial profile of a torus or disk from particle data, for the 1D model
(phases/phase-07-viscosity-drain.md 7.3: "find the torus plane from the gas's total angular
momentum, bin mass into annuli -> Sigma(r); measure H/r and c_s per annulus -> nu(r)").

Arrays in, arrays out (snapshot reading stays in scripts). The plane is the one perpendicular to
the total angular momentum sum m x cross p of the gas passed in (choose the torus gas, e.g.
bound with e < 0.3, before calling). In that frame, R is the cylindrical radius and z the height:

    Sigma(R) = (mass in the annulus) / (pi (R_out^2 - R_in^2))
    H(R)     = sqrt(<z^2>)  (mass-weighted)
    c_s(R)   = mass-weighted mean sound speed
    nu(R)    = alpha c_s H = alpha c_s^2 / Omega_K  when H = c_s / Omega_K (Shakura & Sunyaev 1973)

Here nu = alpha c_s H with the measured H (the two forms agree only in vertical equilibrium).
"""

from dataclasses import dataclass

import numpy as np


def disk_frame(x: np.ndarray, p: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) whose third row is the unit total angular momentum: x_disk = R @ x."""
    ell = np.sum(np.asarray(mass, float)[:, None] * np.cross(x, p), axis=0)
    ez = ell / np.linalg.norm(ell)
    trial = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = trial - ez * (trial @ ez)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.vstack([ex, ey, ez])


@dataclass
class DiskProfile:
    edges: np.ndarray  # (n + 1,) annulus edges in R
    centres: np.ndarray  # (n,)
    sigma: np.ndarray  # (n,) surface density
    height: np.ndarray  # (n,) H = rms z (nan where empty)
    c_s: np.ndarray  # (n,) mass-weighted sound speed (nan where empty)
    mass: np.ndarray  # (n,) mass per annulus
    frame: np.ndarray  # (3, 3) rotation to the disk frame


def disk_profile(x, p, mass, c_s, edges) -> DiskProfile:
    """Sigma(R), H(R), c_s(R) in annuli ``edges`` of the plane perpendicular to the total J."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    m, cs = np.asarray(mass, float), np.asarray(c_s, float)
    frame = disk_frame(x, p, m)
    xd = x @ frame.T
    rr = np.hypot(xd[:, 0], xd[:, 1])
    k = np.digitize(rr, edges) - 1
    n = len(edges) - 1
    ok = (k >= 0) & (k < n)
    mbin = np.bincount(k[ok], weights=m[ok], minlength=n)
    z2 = np.bincount(k[ok], weights=m[ok] * xd[ok, 2] ** 2, minlength=n)
    csb = np.bincount(k[ok], weights=m[ok] * cs[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        height = np.sqrt(z2 / mbin)
        cs_mean = csb / mbin
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return DiskProfile(np.asarray(edges, float), centres, mbin / area, height, cs_mean, mbin, frame)


def viscosity_profile(profile: DiskProfile, alpha: float) -> np.ndarray:
    """nu(R) = alpha c_s H from the measured profile (nan where an annulus is empty)."""
    return alpha * profile.c_s * profile.height
