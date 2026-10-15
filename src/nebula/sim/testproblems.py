"""Standard SPH test problems (phases/phase-03-sph-core.md 3.2): set-ups and measurements,
shared by tests/sim/test_standard_problems.py and scripts/plot_standard_problems.py.

Sod shock tube (test 5): gamma = 1.4, (rho, P) = (1, 1) | (0.125, 0.1), equal-mass particles
from glass (8x more on the dense side), 3D tube periodic in y and z. It is also periodic in x
(simplest boundary), which adds a second, mirrored interface at x = 0 = L; the box is long
enough that its waves never reach the comparison window [x0 - 0.3 L/2, x0 + 0.4 L/2].
Sound wave (test 4): periodic box, gamma = 5/3, rho0 = 1, c_s = 1, amplitude 1e-4.
"""

from dataclasses import dataclass

import numpy as np

from nebula.physics.riemann import sample, solve_riemann
from nebula.sim.glass import load_glass, tile_glass
from nebula.sim.hydro import HydroConfig, HydroState, initialise, run

SOD_GAMMA = 1.4
SOUND_GAMMA = 5.0 / 3.0


@dataclass
class Problem:
    state: HydroState
    cfg: HydroConfig
    t_end: float
    x0: float  # diaphragm position (Sod) or 0 (sound wave)
    extra: dict


# --------------------------------------------------------------------------- Sod


def sod_setup(resolution: str = "quick") -> Problem:
    """quick: tube length 1, cross-section 1/4, left spacing 1/64 (9,216 particles), t = 0.1.
    full (the plan's test): length 2, cross-section 1/8, left spacing 1/128 (36,864
    particles), t = 0.2."""
    if resolution == "full":
        length, width, t_end = 2.0, 0.125, 0.2
    else:
        length, width, t_end = 1.0, 0.25, 0.1
    copies = int(round(length / 2 / width))
    left = tile_glass(load_glass(4096), (copies, 1, 1), (length / 2, width, width))
    right = tile_glass(load_glass(512), (copies, 1, 1), (length / 2, width, width))
    right[:, 0] += length / 2
    x = np.concatenate([left, right])
    n_l = len(left)
    mass = 1.0 * (length / 2) * width**2 / n_l  # equal masses: right side has 1/8 the particles
    rho = np.where(np.arange(len(x)) < n_l, 1.0, 0.125)
    p = np.where(np.arange(len(x)) < n_l, 1.0, 0.1)
    u = p / ((SOD_GAMMA - 1.0) * rho)
    cfg = HydroConfig(gamma=SOD_GAMMA, box=(length, width, width))
    h0 = cfg.eta * (mass / rho) ** (1 / 3)
    st = initialise(x, np.zeros_like(x), u, mass, h0, cfg)
    return Problem(st, cfg, t_end, length / 2, {"length": length, "width": width})


@dataclass
class SodMeasurement:
    rho_star_l: float
    rho_star_r: float
    p_star: float
    u_star: float
    shock_x: float
    shock_x_exact: float
    h_shock: float  # smoothing length just behind the shock
    oscillation: float  # max |v - u*| / u* in the post-shock plateau (binned)


def sod_measure(prob: Problem, st: HydroState) -> SodMeasurement:
    sol = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, SOD_GAMMA)
    t, x0 = st.t, prob.x0
    x = st.x[:, 0]
    tail, contact, shock = x0 + sol.s_l_tail * t, x0 + sol.u_star * t, x0 + sol.s_r_head * t
    h_l, h_r = np.median(st.h[st.rho > 0.35]), np.median(st.h[(st.rho > 0.2) & (st.rho < 0.35)])
    star_l = (x > tail + 3 * h_l) & (x < contact - 3 * h_l)
    star_r = (x > contact + 3 * h_r) & (x < shock - 3 * h_r)
    star = star_l | star_r
    # shock position: rightmost point where density crosses midway between rho*R and 0.125
    mid = 0.5 * (sol.rho_star_r + 0.125)
    # Window ends at 3/4 of the tube: midway between this shock and the mirrored one coming
    # from the periodic end (both move at the same speed, from x0 and from L).
    edges = np.arange(contact, 0.75 * prob.extra["length"], 0.25 * h_r)
    centres, rho_bin = _bin(x, st.rho, edges)
    above = np.where(rho_bin > mid)[0]
    k = above[-1]
    shock_x = centres[k] + (rho_bin[k] - mid) / (rho_bin[k] - rho_bin[k + 1]) * (
        centres[k + 1] - centres[k]
    )
    v_edges = np.arange(contact + 3 * h_r, shock - 3 * h_r, h_r)
    if len(v_edges) >= 3:
        _, v_bin = _bin(x, st.v[:, 0], v_edges)
        oscillation = float(np.nanmax(np.abs(v_bin - sol.u_star)) / sol.u_star)
    else:  # post-shock plateau narrower than its margins: not measurable at this resolution
        oscillation = float("nan")

    def mean(a, sel):
        return float(np.mean(a[sel])) if np.any(sel) else float("nan")

    return SodMeasurement(
        mean(st.rho, star_l), mean(st.rho, star_r), mean(st.pressure, star),
        mean(st.v[:, 0], star), float(shock_x), float(shock), float(h_r), oscillation,
    )  # fmt: skip


def _bin(x, y, edges):
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[ok], weights=y[ok], minlength=len(edges) - 1)
    c = np.bincount(idx[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return 0.5 * (edges[1:] + edges[:-1]), s / c


def sod_exact(prob: Problem, x, t):
    sol = solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, SOD_GAMMA)
    return sample(sol, x, t, prob.x0)


# --------------------------------------------------------------------------- sound wave

AMPLITUDE = 1e-4


def sound_setup(resolution: str = "low") -> Problem:
    """low: 32 particles per wavelength (2,048); high: 64 per wavelength (16,384). Box
    (1, 1/4, 1/4), one wavelength along x, right-moving wave of amplitude 1e-4."""
    glass = load_glass(512 if resolution == "low" else 4096)
    width = 0.25
    base = tile_glass(glass, (4, 1, 1), (1.0, width, width))
    k = 2 * np.pi
    x = base.copy()
    x[:, 0] = np.mod(base[:, 0] - AMPLITUDE / k * np.sin(k * base[:, 0]), 1.0)  # rho ~ 1 + A cos kx
    n = len(x)
    mass = width * width / n
    rho = 1.0 + AMPLITUDE * np.cos(k * base[:, 0])
    u0 = 1.0 / (SOUND_GAMMA * (SOUND_GAMMA - 1.0))  # c_s = 1 at rho = 1
    u = u0 * rho ** (SOUND_GAMMA - 1.0)  # adiabatic
    v = np.zeros_like(x)
    v[:, 0] = AMPLITUDE * np.cos(k * base[:, 0])  # c_s A cos kx: travels toward +x
    cfg = HydroConfig(gamma=SOUND_GAMMA, box=(1.0, width, width))
    h0 = np.full(n, cfg.eta * mass ** (1 / 3))
    st = initialise(x, v, u, mass, h0, cfg)
    # Baseline: the same glass without the wave, evolved with the same time steps; subtracting
    # it removes the glass's own density noise (0.3%, larger than the wave) and its small
    # jiggling, which are identical in both runs to first order in the amplitude.
    flat = initialise(base, np.zeros_like(x), np.full(n, u0), mass, h0, cfg)
    return Problem(st, cfg, 1.0, 0.0, {"baseline": flat, "k": k})


@dataclass
class WaveMeasurement:
    amplitude: float  # fitted density amplitude / A
    phase_cycles: float  # phase shift in wavelengths (0 = back where it started)
    l1: float  # mean |<delta rho>_slice - exact| / A over 32 slices along x
    profile_x: np.ndarray
    profile: np.ndarray  # slice-averaged delta rho / A


def run_sound(prob: Problem) -> tuple[HydroState, HydroState]:
    """Evolve the wave and its no-wave baseline with identical fixed steps (Courant step of the
    baseline: 0.3 h / c_s) to t_end. Returns (wave state, baseline state)."""
    base = prob.extra["baseline"]
    dt = prob.cfg.courant * float(np.min(base.h / base.vsig_max))
    n_steps = int(np.ceil(prob.t_end / dt))
    dt = prob.t_end / n_steps
    return run(prob.state, prob.cfg, prob.t_end, dt), run(base, prob.cfg, prob.t_end, dt)


def sound_measure(prob: Problem, st: HydroState, baseline: HydroState) -> WaveMeasurement:
    k = prob.extra["k"]
    delta = st.rho - baseline.rho
    xx = st.x[:, 0] - st.t  # comoving with the wave (c_s = 1); after one period, xx = x - 1
    design = np.column_stack([np.cos(k * xx), np.sin(k * xx)])
    (a, b), *_ = np.linalg.lstsq(design, delta, rcond=None)
    amp = np.hypot(a, b) / AMPLITUDE
    phase = np.arctan2(b, a) / (2 * np.pi)
    # L1 on the slice-averaged profile: particle-by-particle, the wave and no-wave runs drift
    # apart slightly (chaotic particle motion), which is noise in the subtraction, not wave error.
    edges = np.linspace(0.0, 1.0, 33)
    idx = np.digitize(np.mod(st.x[:, 0], 1.0), edges) - 1
    prof = np.array([delta[idx == q].mean() for q in range(32)]) / AMPLITUDE
    centres = 0.5 * (edges[1:] + edges[:-1])
    l1 = np.mean(np.abs(prof - np.cos(k * (centres - st.t))))
    return WaveMeasurement(float(amp), float(phase), float(l1), centres, prof)


def run_problem(prob: Problem, callback=None) -> HydroState:
    return run(prob.state, prob.cfg, prob.t_end, callback=callback)


# --------------------------------------------------------------------------- Gresho vortex
#
# Gresho & Chan (1990), as used by Springel (2010) and Cullen & Dehnen (2010); expected-values
# §4: rho = 1, gamma = 5/3, a steady vortex (pressure gradient balances rotation exactly):
#     v_phi = 5r (r < 0.2); 2 - 5r (0.2 <= r < 0.4); 0 (r >= 0.4)
#     P = 5 + 12.5 r^2 (r < 0.2); 9 + 12.5 r^2 - 20 r + 4 ln(5r) (0.2-0.4); 3 + 4 ln 2 (r >= 0.4)
# It is 2D: run as a slab periodic in z (unit square periodic in x, y; centre at (0.5, 0.5)).

GRESHO_GAMMA = 5.0 / 3.0


def gresho_vphi(r):
    r = np.asarray(r, float)
    return np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))


def gresho_pressure(r):
    r = np.asarray(r, float)
    rs = np.maximum(r, 1e-12)
    inner = 5 + 12.5 * r**2
    mid = 9 + 12.5 * r**2 - 20 * r + 4 * np.log(5 * rs)
    return np.where(r < 0.2, inner, np.where(r < 0.4, mid, 3 + 4 * np.log(2)))


def gresho_setup(resolution: str = "low", switch: bool = True) -> Problem:
    """low: 16 particles per unit length (glass_512, 2,048 particles); high: 32 per unit
    (glass_4096, 16,384). Slab thickness 1/2. switch=False: constant alpha = 1."""
    glass = load_glass(512 if resolution == "low" else 4096)
    depth = 0.5
    x = tile_glass(glass, (2, 2, 1), (1.0, 1.0, depth))
    n = len(x)
    mass = depth / n
    d = x[:, :2] - 0.5
    r = np.linalg.norm(d, axis=1)
    vphi = gresho_vphi(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        ehat = np.where(r[:, None] > 0, np.column_stack([-d[:, 1], d[:, 0]]) / r[:, None], 0.0)
    v = np.zeros_like(x)
    v[:, :2] = vphi[:, None] * ehat
    u = gresho_pressure(r) / ((GRESHO_GAMMA - 1.0) * 1.0)
    cfg = HydroConfig(
        gamma=GRESHO_GAMMA, box=(1.0, 1.0, depth), switch=switch, alpha_const=1.0, alpha_init=0.0
    )
    st = initialise(x, v, u, mass, np.full(n, cfg.eta * mass ** (1 / 3)), cfg)
    return Problem(st, cfg, 1.0, 0.0, {})


def gresho_l1(st: HydroState) -> tuple[float, np.ndarray, np.ndarray]:
    """L1 error of v_phi over particles with r < 0.5, and binned profile (r, <v_phi>)."""
    d = st.x[:, :2] - 0.5
    r = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vphi = (d[:, 0] * st.v[:, 1] - d[:, 1] * st.v[:, 0]) / r
    sel = (r < 0.5) & (r > 0)
    l1 = float(np.mean(np.abs(vphi[sel] - gresho_vphi(r[sel]))))
    edges = np.linspace(0, 0.5, 26)
    centres, prof = _bin(r[sel], vphi[sel], edges)
    return l1, centres, prof


# --------------------------------------------------------------------------- Sedov blast
#
# Energy E = 1 deposited in the internal energy of the particles within reach of the box
# centre (kernel-weighted, background smoothing length) in a uniform glass of density 1,
# gamma = 5/3; background u = 1e-6 of the peak blast value (negligible pressure). Exact:
# shock radius R = xi0 (E t^2 / rho)^(1/5), xi0 = 1.151666 (expected-values §4, script 22).

SEDOV_GAMMA = 5.0 / 3.0
SEDOV_XI0 = 1.151666


def sedov_setup(tiles: int = 3) -> Problem:
    """Periodic unit box tiled with tiles^3 copies of glass_4096 (3 -> 110,592 particles)."""
    from nebula.physics.kernels import get_kernel

    x = tile_glass(load_glass(4096), (tiles, tiles, tiles), (1.0, 1.0, 1.0))
    n = len(x)
    mass = 1.0 / n
    cfg = HydroConfig(gamma=SEDOV_GAMMA, box=(1.0, 1.0, 1.0))
    h_bg = cfg.eta * mass ** (1 / 3)
    r = np.linalg.norm(x - 0.5, axis=1)
    w = get_kernel(cfg.kernel).w(r, h_bg)
    u_blast = w / np.sum(w) / mass  # sum(m u) = E = 1
    u = np.maximum(u_blast, 1e-6 * u_blast.max())
    st = initialise(x, np.zeros_like(x), u, mass, np.full(n, h_bg), cfg)
    return Problem(st, cfg, 0.0, 0.0, {"h_bg": h_bg, "energy": 1.0})


def sedov_radius(st: HydroState, h_bg: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Shock radius = where the radially binned density peaks (parabola-refined)."""
    r = np.linalg.norm(st.x - 0.5, axis=1)
    edges = np.arange(0.0, 0.5, 0.25 * h_bg)
    centres, prof = _bin(r, st.rho, edges)
    k = int(np.nanargmax(prof))
    a, b, _ = np.polyfit(centres[k - 1 : k + 2] - centres[k], prof[k - 1 : k + 2], 2)
    return float(centres[k] - b / (2 * a)), centres, prof


def sedov_exact_radius(t: float, energy: float = 1.0) -> float:
    return SEDOV_XI0 * (energy * t * t) ** 0.2
