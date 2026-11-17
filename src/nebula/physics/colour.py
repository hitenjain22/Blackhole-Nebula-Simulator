"""Blackbody colour: CIE 1931 XYZ of a blackbody, its lookup table, and sRGB
(phases/phase-09-colour-physics.md 9.1; expected values in reference/expected-values.md §7).

Sources: Planck's law B_lambda(T) = 2 h c^2 / lambda^5 / (exp(h c / (lambda k T)) - 1)
(Rybicki & Lightman 1979 Eq. 1.54); CIE 1931 2-degree colour-matching functions (CIE 018:2019,
table passed in by the caller: physics functions do no I/O, the table is read by
nebula.render.colour_data); XYZ -> linear sRGB matrix and the sRGB transfer function from
IEC 61966-2-1:1999.

Units: wavelengths in nm where the CMF table is concerned (1 nm sampling), metres inside the
Planck function; temperatures in K; intensities in W m^-2 sr^-1 (XYZ are the CMF-weighted
integrals of B_lambda over wavelength, so they share that unit). Functions with `_si` names
compute in SI by rule 4's exception (blackbody colour is naturally written in SI).

Pure functions; NumPy arrays or scalars.
"""

import numpy as np

from nebula import units

# IEC 61966-2-1 (sRGB), XYZ (D65) -> linear sRGB; as printed in phase-09 9.1 / expected §7
XYZ_TO_SRGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)
SRGB_KNEE = 0.0031308  # linear value where the sRGB curve switches from linear to power law


def planck_lambda_si(lam_m, t_k):
    """Planck spectral radiance B_lambda(T) [W m^-2 sr^-1 m^-1]; lambda in m, T in K.

    B_lambda = 2 h c^2 / lambda^5 / (exp(h c / (lambda k T)) - 1); Rybicki & Lightman 1979
    Eq. 1.54. expm1 keeps precision in the Rayleigh-Jeans limit (h c << lambda k T).
    """
    lam = np.asarray(lam_m, float)
    x = units.H_PLANCK * units.C / (lam * units.K_B * np.asarray(t_k, float))
    with np.errstate(over="ignore"):  # exp overflow -> inf -> B = 0 (deep Wien tail)
        return 2.0 * units.H_PLANCK * units.C**2 / lam**5 / np.expm1(x)


def blackbody_xyz_si(t_k, lam_nm, cmf):
    """CIE XYZ of a blackbody [W m^-2 sr^-1]: X = sum_lambda B_lambda(T) xbar(lambda) dlambda.

    t_k: scalar or array of temperatures (K), result has shape t_k.shape + (3,).
    lam_nm: (N,) wavelengths in nm with a uniform 1 nm step; cmf: (N, 3) xbar, ybar, zbar.
    Riemann sum at 1 nm, as phase-09 9.1 and expected-values §7 (script 17) specify.
    """
    lam_nm = np.asarray(lam_nm, float)
    dlam_m = (lam_nm[1] - lam_nm[0]) * 1e-9
    t = np.asarray(t_k, float)
    b = planck_lambda_si(lam_nm * 1e-9, t[..., None])  # (..., N)
    return (b @ np.asarray(cmf, float)) * dlam_m


def bolometric_intensity_si(t_k):
    """Frequency-integrated blackbody intensity sigma T^4 / pi [W m^-2 sr^-1], T in K.

    Rybicki & Lightman 1979 Eq. 1.43.
    """
    return units.SIGMA_SB * np.asarray(t_k, float) ** 4 / np.pi


def blackbody_chroma(t_k, lam_nm, cmf):
    """chroma(T) = XYZ(T) / (sigma T^4 / pi): colour per unit bolometric intensity
    (dimensionless). phase-09 9.1; the lookup table stores its logarithm."""
    t = np.asarray(t_k, float)
    return blackbody_xyz_si(t, lam_nm, cmf) / bolometric_intensity_si(t)[..., None]


def chromaticity(xyz):
    """CIE (x, y) = (X, Y) / (X + Y + Z) of XYZ with shape (..., 3); CIE 015:2018 Eq. 7.1."""
    xyz = np.asarray(xyz, float)
    s = xyz.sum(axis=-1)
    return xyz[..., 0] / s, xyz[..., 1] / s


def xyz_from_xyy(x, y, big_y):
    """XYZ (..., 3) from chromaticity (x, y) and luminance Y; CIE 015:2018 Eq. 7.2."""
    x, y, big_y = (np.asarray(v, float) for v in (x, y, big_y))
    return np.stack([x * big_y / y, big_y, (1.0 - x - y) * big_y / y], axis=-1)


def luminous_efficacy(t_k, lam_nm, cmf):
    """Luminous efficacy of blackbody light [lm/W]: K_m Y / (sigma T^4 / pi), K_m = 683.002 lm/W
    (units.K_M_LUMEN). Phase 9 test 3."""
    return units.K_M_LUMEN * blackbody_chroma(t_k, lam_nm, cmf)[..., 1]


def xyz_to_linear_srgb(xyz):
    """Linear sRGB (..., 3) from XYZ (..., 3) with the IEC 61966-2-1 matrix (D65 white).

    Values may be negative or above 1 (out of gamut / brighter than white); gamut handling and
    tone mapping are separate steps (phase-09 9.1, 9.5)."""
    return np.asarray(xyz, float) @ XYZ_TO_SRGB.T


def srgb_encode(linear):
    """sRGB transfer function (linear -> encoded, both in [0, 1]); IEC 61966-2-1 Eq. 5:
    12.92 L for L <= 0.0031308, else 1.055 L^(1/2.4) - 0.055."""
    lin = np.asarray(linear, float)
    return np.where(
        lin <= SRGB_KNEE, 12.92 * lin, 1.055 * np.maximum(lin, SRGB_KNEE) ** (1.0 / 2.4) - 0.055
    )


def srgb_decode(encoded):
    """Inverse of srgb_encode (encoded -> linear); IEC 61966-2-1 Eq. 4 (threshold 12.92 x knee,
    which keeps the inverse exact at the knee rather than at the standard's rounded 0.04045)."""
    enc = np.asarray(encoded, float)
    knee = 12.92 * SRGB_KNEE
    return np.where(enc <= knee, enc / 12.92, ((np.maximum(enc, knee) + 0.055) / 1.055) ** 2.4)


def chroma_lut(t_min, t_max, n, lam_nm, cmf):
    """Lookup table of log chroma on a log-T grid: (log_t (n,), log_chroma (n, 3)).

    phase-09 9.1: interpolating log(chroma) linearly in log T with 4,096 entries from 1e3 to
    1e8 K keeps the relative error below ~2.4e-5 (expected-values §7, script 21). Natural logs.
    """
    log_t = np.linspace(np.log(t_min), np.log(t_max), n)
    return log_t, np.log(blackbody_chroma(np.exp(log_t), lam_nm, cmf))


def lut_chroma(t_k, log_t, log_chroma):
    """chroma(T) from the table of chroma_lut by linear interpolation of log chroma in log T.

    T outside the table is clamped to its ends (the renderer's temperatures lie inside it; the
    caller keeps T in range). Returns shape t_k.shape + (3,)."""
    lt = np.clip(np.log(np.asarray(t_k, float)), log_t[0], log_t[-1])
    step = (log_t[-1] - log_t[0]) / (len(log_t) - 1)
    pos = (lt - log_t[0]) / step
    i = np.minimum(pos.astype(int), len(log_t) - 2)
    f = (pos - i)[..., None]
    return np.exp(log_chroma[i] * (1.0 - f) + log_chroma[i + 1] * f)


def blackbody_xyz_lut(t_k, log_t, log_chroma):
    """XYZ of a blackbody [W m^-2 sr^-1] from the lookup table: chroma(T) sigma T^4 / pi
    (phase-09 9.1). With T the observed temperature g T_em this is the observed light of gas
    emitting as a blackbody at T_em (9.4: g^3 B_(nu/g)(T) = B_nu(g T), so no extra g^4)."""
    t = np.asarray(t_k, float)
    return lut_chroma(t, log_t, log_chroma) * bolometric_intensity_si(t)[..., None]
