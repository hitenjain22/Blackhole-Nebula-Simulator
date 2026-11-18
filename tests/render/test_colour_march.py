"""Relativistic colour radiative transfer on the CPU (phase-09 9.4; tests 11 and 12 of 9.7).

March readable reference (render.march with march.RelativisticBlackbody) along bent rays
(render.bent_rays.march_bent_ray). Expected values: phase-09 9.7 and expected-values §6-§7."""

import numpy as np
import pytest

from nebula.physics.colour import (
    blackbody_xyz_si,
    bolometric_intensity_si,
    chroma_lut,
    chromaticity,
)
from nebula.render.bent_rays import RAY_ESCAPED, march_bent_ray
from nebula.render.colour_data import load_cmf
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.march import RelativisticBlackbody, texture_from_fields
from tests.helpers import check

LAM, CMF = load_cmf()
T_GAS = 1.0e4


@pytest.fixture(scope="module")
def lut():
    return chroma_lut(1e3, 1e8, 4096, LAM, CMF)


def toward(cam, target):
    d = np.asarray(target, float) - cam
    return d / np.linalg.norm(d)


def test_11_opaque_static_shell(lut):
    """An opaque, static shell whose outer surface is at r = 10, T = 1e4 K, seen from a static
    camera at r = 1,000: every ray that hits it shows sigma (g T)^4 / pi with
    g = sqrt(1 - 2/10) / sqrt(1 - 2/1000) (0.5%), coloured as a blackbody at g T. The radial
    cells are 0.005 wide at r = 10, so the surface sits within ~0.005 of r = 10 (g changes by
    6e-5 there)."""
    g = LogSphericalGrid(r_min=9.0, r_max=12.0, nr=600, nth=8, nph=16)
    rc = g.r_centres()[:, None, None] * np.ones(g.shape)
    rho = np.where(rc <= 10.0, 1.0, 0.0)  # filled up to r = 10 (cell centres)
    tex = texture_from_fields(g, rho, np.full(g.shape, T_GAS))  # no velocity: static gas
    d_cam = 1000.0
    cam = np.array([d_cam, 0.0, 0.0])
    model = RelativisticBlackbody(kappa=1e4, r_cam=d_cam, log_t=lut[0], log_chroma=lut[1])
    g_exp = np.sqrt(1 - 2 / 10.0) / np.sqrt(1 - 2 / d_cam)
    want_bol = float(bolometric_intensity_si(g_exp * T_GAS))
    xe, ye = chromaticity(blackbody_xyz_si(g_exp * T_GAS, LAM, CMF))
    print(f"   g expected {g_exp:.8f}; observed T {g_exp * T_GAS:.1f} K")
    for b in (0.0, 4.0, 8.0):  # aim points across the disc of the shell
        i, trans, status, n = march_bent_ray([tex], model, cam, toward(cam, (0.0, b, 0.3 * b)))
        check(f"shell, aim b = {b:g}: opaque (transmission)", 0.0, trans, atol=1e-4)
        check(f"shell, aim b = {b:g}: bolometric sigma (g T)^4 / pi", want_bol, float(i[3]),
              rtol=5e-3)  # fmt: skip
        x, y = chromaticity(i[:3])
        check(f"shell, aim b = {b:g}: colour x = blackbody at g T", xe, x, atol=1e-4)
        check(f"shell, aim b = {b:g}: colour y = blackbody at g T", ye, y, atol=1e-4)


def ring_textures(r0, half_width, half_height, nr, nth, nph, r_lo, r_hi, kappa_rho=1.0):
    """Opaque ring of gas on circular orbits about the z axis, counterclockwise seen from +z,
    with static-frame speed beta = 1/sqrt(r0 - 2) (expected-values §7) everywhere in it."""
    g = LogSphericalGrid(r_min=r_lo, r_max=r_hi, nr=nr, nth=nth, nph=nph)
    r = g.r_centres()[:, None, None]
    th = ((np.arange(nth) + 0.5) * g.dth)[None, :, None]
    ph = ((np.arange(nph) + 0.5) * g.dph)[None, None, :]
    z = r * np.cos(th)
    inside = (np.abs(r - r0) <= half_width) & (np.abs(z) <= half_height)
    rho = np.where(inside, kappa_rho, 0.0) * np.ones(g.shape)
    beta = 1.0 / np.sqrt(r0 - 2.0)
    gb = beta / np.sqrt(1 - beta**2)
    vel = np.zeros((*g.shape, 3))
    vel[..., 0] = -gb * np.sin(ph) * np.ones(g.shape)
    vel[..., 1] = gb * np.cos(ph) * np.ones(g.shape)
    return texture_from_fields(g, rho, np.full(g.shape, T_GAS), vel), beta


def test_12a_doppler_ratio_far_ring(lut):
    """phase-09 test 12a: thin opaque ring at r = 1,000, edge-on from D = 1e5 on +x. The end at
    y = -1,000 moves toward the camera (+x), the end at y = +1,000 away. Bolometric ratio
    ((1 + beta)/(1 - beta))^4 = 1.288296 (expected-values §7), within 2%."""
    tex, beta = ring_textures(1000.0, 5.0, 40.0, nr=40, nth=128, nph=1024, r_lo=980.0,
                              r_hi=1020.0)  # fmt: skip
    d_cam = 1e5
    cam = np.array([d_cam, 0.0, 0.0])
    model = RelativisticBlackbody(kappa=10.0, r_cam=d_cam, log_t=lut[0], log_chroma=lut[1])
    app, ta, _, _ = march_bent_ray([tex], model, cam, toward(cam, (0.0, -1000.0, 0.0)))
    rec, tr, _, _ = march_bent_ray([tex], model, cam, toward(cam, (0.0, 1000.0, 0.0)))
    check("far ring: approaching end opaque", 0.0, ta, atol=1e-4)
    check("far ring: receding end opaque", 0.0, tr, atol=1e-4)
    ratio = float(app[3] / rec[3])
    check("far ring: approaching / receding bolometric (§7)", 1.288296, ratio, rtol=0.02)
    check("far ring: ratio = ((1+beta)/(1-beta))^4 with this beta", ((1 + beta) / (1 - beta)) ** 4,
          ratio, rtol=0.02)  # fmt: skip
    xa, _ = chromaticity(app[:3])
    xr, _ = chromaticity(rec[:3])
    check("far ring: approaching end bluer (smaller x)", True, xa < xr)


def test_12b_bright_side_near_hole(lut):
    """phase-09 test 12b: the same kind of ring at r = 10, edge-on from r = 1,000: rays aimed
    symmetrically at the two ends (y = -11 approaching, y = +11 receding) both reach the ring
    (light bending changes the exact ratio, so only the sign is tested): the approaching side
    is brighter and bluer. Getting this backwards is the classic sign error."""
    tex, _ = ring_textures(10.0, 1.5, 1.5, nr=200, nth=96, nph=512, r_lo=6.0, r_hi=16.0)
    d_cam = 1000.0
    cam = np.array([d_cam, 0.0, 0.0])
    model = RelativisticBlackbody(kappa=10.0, r_cam=d_cam, log_t=lut[0], log_chroma=lut[1])
    app, ta, sa, _ = march_bent_ray([tex], model, cam, toward(cam, (0.0, -11.0, 0.0)))
    rec, tr, sr, _ = march_bent_ray([tex], model, cam, toward(cam, (0.0, 11.0, 0.0)))
    print(
        f"   near ring: bolometric ratio {app[3] / rec[3]:.3f}; transmissions {ta:.2e}, "
        f"{tr:.2e}; statuses {sa}, {sr} ({RAY_ESCAPED} = escaped)"
    )
    check("near ring: both rays end in the gas (transmission < 1e-3)", True,
          ta < 1e-3 and tr < 1e-3)  # fmt: skip
    check("near ring: approaching side brighter", True, app[3] > rec[3])
    xa, ya = chromaticity(app[:3])
    xr, yr = chromaticity(rec[:3])
    check("near ring: approaching side bluer (smaller x and y)", True, xa < xr and ya < yr)


def test_thermal_opacity_semitransparent_shell(lut):
    """opacity = "thermal" (phase-09 9.4 option 1): a static shell of uniform gas (r 950-1050,
    T = 1e5 K) with density set so its radial optical depth is ~1, seen along a radial ray from
    r = 1e5. Transmission = exp(-alpha L), alpha = physics.opacity.thermal_absorption_code and
    L = integral dr / sqrt(1 - 2/r) (static radial length); bolometric = sigma (g T)^4 / pi
    (1 - transmission), g = sqrt(1 - 2/1000) / sqrt(1 - 2/1e5) (g varies by 1e-4 over the
    shell). 2%: the shell's edges are smoothed over one cell by the trilinear sampling."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    from nebula.physics.opacity import thermal_absorption_code

    t_gas = 1e5
    length = quad(lambda r: 1.0 / np.sqrt(1.0 - 2.0 / r), 950.0, 1050.0)[0]
    rho0 = brentq(lambda rho: thermal_absorption_code(rho, t_gas) * length - 1.0, 1e-14, 1.0)
    g = LogSphericalGrid(r_min=900.0, r_max=1100.0, nr=400, nth=8, nph=16)
    rc = g.r_centres()[:, None, None] * np.ones(g.shape)
    rho = np.where((rc > 950.0) & (rc < 1050.0), rho0, 0.0)
    tex = texture_from_fields(g, rho, np.full(g.shape, t_gas))
    d_cam = 1e5
    cam = np.array([d_cam, 0.0, 0.0])
    model = RelativisticBlackbody(kappa=1.0, r_cam=d_cam, log_t=lut[0], log_chroma=lut[1],
                                  opacity="thermal")  # fmt: skip
    i, trans, _, _ = march_bent_ray([tex], model, cam, np.array([-1.0, 0.0, 0.0]))
    want_t = np.exp(-thermal_absorption_code(rho0, t_gas) * length)
    gfac = np.sqrt(1 - 2 / 1000.0) / np.sqrt(1 - 2 / d_cam)
    want_i = float(bolometric_intensity_si(gfac * t_gas)) * (1 - want_t)
    print(f"   shell density {rho0:.3e} (code), alpha {thermal_absorption_code(rho0, t_gas):.3e}")
    check("thermal shell: transmission exp(-alpha L)", want_t, trans, rtol=0.02)
    check("thermal shell: bolometric sigma (g T)^4 / pi (1 - transmission)", want_i, float(i[3]),
          rtol=0.02)  # fmt: skip
    with pytest.raises(ValueError, match="opacity"):
        RelativisticBlackbody(kappa=1.0, r_cam=d_cam, log_t=lut[0], log_chroma=lut[1],
                              opacity="fog")  # fmt: skip
