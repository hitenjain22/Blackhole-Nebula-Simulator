"""Point sources seen through a lens map: solid angles and pixel radiance (final-render
pinpoint stars; DECISIONS 2026-11-19).

A camera pixel looks, through the bent rays, at a patch of the sky. For a point source (a
star) of flux F inside that patch, the image is magnified by mu = dOmega_image / dOmega_source
(Schneider, Ehlers & Falco 1992, *Gravitational Lenses*, Sec. 2.3 and 5.1: lensing conserves
surface brightness, so the image flux is mu F). A pixel of screen solid angle Omega_pix whose
patch covers Omega_sky of the sky holding the star therefore has mean radiance

    I_pix = mu F / Omega_pix = F / Omega_sky    (mu = Omega_pix / Omega_sky for that patch)

so lensing needs no separate magnification formula: the area of sky each pixel sees carries
it (a patch squeezed near the hole -> bright image; parity-flipped images count with |area|).

Solid angles of patches come from their corner directions, split into spherical triangles
(Van Oosterom & Strackee 1983, IEEE Trans. Biomed. Eng. BME-30, 125, Eq. 8).

Units: flux in any unit (the star catalogue's 10^(-0.4 m)); solid angles in sr; radiance in
flux per sr (the unit of the sky cube map, render/starfield.py). Pure functions.
"""

import numpy as np


def spherical_triangle_solid_angle(a, b, c):
    """Solid angle [sr] of the spherical triangle with vertex unit vectors a, b, c (..., 3):

        tan(Omega / 2) = |a . (b x c)| / (1 + a.b + b.c + c.a)

    Van Oosterom & Strackee 1983 Eq. 8 (atan2 form, valid for Omega up to 2 pi)."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    num = np.abs(np.sum(a * np.cross(b, c), axis=-1))
    den = 1.0 + np.sum(a * b, axis=-1) + np.sum(b * c, axis=-1) + np.sum(c * a, axis=-1)
    return 2.0 * np.arctan2(num, den)


def quad_solid_angle(d00, d10, d11, d01):
    """Solid angle [sr] of the patch with corner unit vectors in order around it (a convex
    or mildly bent quadrilateral; two triangles split along d00-d11)."""
    return spherical_triangle_solid_angle(d00, d10, d11) + spherical_triangle_solid_angle(
        d00, d11, d01
    )


def point_source_radiance(flux, sky_solid_angle):
    """Mean radiance [flux / sr] of a pixel whose sky patch of solid angle Omega_sky [sr]
    holds a point source of flux F: I = F / Omega_sky (Schneider, Ehlers & Falco 1992
    Sec. 5.1: image flux mu F with mu = Omega_pix / Omega_sky; module docstring)."""
    return np.asarray(flux, float) / np.asarray(sky_solid_angle, float)


def point_lens_magnification(u):
    """Total magnification of both images of a point lens at source offset u (in Einstein
    radii), weak field: mu = (u^2 + 2) / (u sqrt(u^2 + 4)) (Paczynski 1986, ApJ 304, 1,
    Eq. 4). Used to test the pixel photometry of lensed stars."""
    u = np.asarray(u, float)
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))
