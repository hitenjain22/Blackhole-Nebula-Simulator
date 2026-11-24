"""CPU reference ray march through the render grids (phases/phase-08-gas-rendering.md 8.3-8.4).

This is the readable float64 reference that the GPU marcher (step 3) must match (CLAUDE.md
rule 2; 8.6 test 7). It is plain Python on purpose: clarity over speed.

Textures (8.3, changed 2026-11-13, DECISIONS): each grid holds rho and rho (T_em / T_REF)^4 at
cell centres, linear (not log), float32 as the GPU will see them; empty cells hold 0. Linear
interpolation keeps the column density (so alpha ds) and the emission (j = kappa rho sigma T^4 /
pi, linear in rho T^4) of the deposited gas along any ray; interpolating logs shrank every gas
edge by about half a cell (thin emissive sphere: -1 to -2% at the centre, -21% at b = 0.95 R)
and would break thin streams into dots. The temperature at a sample is T = T_REF (rho T^4 /
rho)^(1/4). Sampling is trilinear in index space between cell centres, as GPU texture filtering
does: log grid in (ln r, theta, phi) with theta clamped at the poles and phi wrapping across 0 /
2 pi; outside r_min..r_max the log grid is empty. A detail box is trilinear in (x, y, z),
clamped at its faces, and blends into the log grid over its outer ``BLEND_CELLS`` cells so its
edges leave no seam (8.3).

Marching (8.4): straight sub-steps no longer than half the local cell size of the finest grid
there (log grid: r min(dln r, dth); box: dx), sampled at each sub-step's midpoint, combined with
physics.radiative_transfer.transfer_step. Stops when the transmission falls below ``T_STOP``.
Path lengths are flat, or with ``curved=True`` the static observer's proper length around the
hole (step 3; render/bent_rays.py marches bent rays chord by chord).

The emission model is a separate object (``GreyBlackbody``) so Phase 9 can swap it without
touching the march loop (8 "Done when").
"""

import math
from dataclasses import dataclass

import numpy as np

from nebula.physics.colour import blackbody_xyz_lut, bolometric_intensity_si
from nebula.physics.emission_lines import line_emission_code, lines_to_xyz
from nebula.physics.geodesics import schw_static_proper_length
from nebula.physics.kerr import metric_parts, redshift_factor_kerr, static_frame_direction_kerr
from nebula.physics.opacity import thermal_absorption_code, thin_emission_fraction
from nebula.physics.radiative_transfer import (
    blackbody_intensity_si,
    grey_absorption,
    transfer_step,
)
from nebula.physics.redshift import (
    gas_frame_path_factor,
    redshift_factor,
    static_frame_direction,
)
from nebula.render.bricks import (
    EXIT_EPS,
    BrickTexture,
    entry_distance,
    exit_distance,
    home_allocated,
    sample_bricks,
    sample_bricks_rho_gb,
)
from nebula.render.gasgrid import CartesianBox, GasGrid, LogSphericalGrid

T_REF = 1.0e4  # K; rho T^4 is stored as rho (T / T_REF)^4 to stay well inside float32
BLEND_CELLS = 2.0  # detail boxes fade into the log grid over this many cells at their faces
T_STOP = 1e-4  # stop marching when the transmission falls below this (8.4)
STEP_FRACTION = 0.5  # sub-step <= this fraction of the local cell size (8.4)
# Largest |gamma beta| used by the colour model (beta = 0.995). Real gas stays far below it
# (|gamma beta| ~ 1 at r ~ 4); it only guards rho_gb / rho where both are float32 rounding
# residues at the thin edge of the gas, which on the GPU could give any value.
GB_MAX = 10.0
# Smallest density the colour model treats as gas (code units; real gas here is ~1e-6 and
# above). The colour model divides by rho (temperature, velocity); deposited float32 grids
# can hold subnormal residues (~1e-45) that GPUs flush to zero in some operations but not
# others, giving 0/0 (AI_LOG 2026-11-17). Absorption at 1e-30 is ~1e-23 per unit length:
# nothing visible is lost. The GPU uses the same value (GasParams.colour2.z).
RHO_MIN_COLOUR = 1e-30


@dataclass
class Texture:
    """One grid's fields as the renderer samples them (float32): rho and rho (T_em/T_REF)^4."""

    grid: LogSphericalGrid | CartesianBox
    rho: np.ndarray
    rho_t4: np.ndarray
    rho_gb: np.ndarray | None = None  # (..., 3): rho x mass-weighted gamma beta (Phase 9)


def texture_from_fields(grid, rho, t_em, vel=None) -> Texture:
    """Texture from cell fields (rho in code units, T_em in K, optional mass-weighted velocity
    (..., 3) (Phase 9: gamma beta); empty cells rho = 0)."""
    rho = np.asarray(rho, float)
    rt4 = rho * (np.asarray(t_em, float) / T_REF) ** 4
    if not (np.all(np.isfinite(rt4)) and rt4.max(initial=0.0) < 1e37 and rho.max() < 1e37):
        raise OverflowError("texture values would overflow float32; change T_REF or units")
    rgb = None if vel is None else (rho[..., None] * np.asarray(vel, float)).astype(np.float32)
    return Texture(grid, rho.astype(np.float32), rt4.astype(np.float32), rgb)


def temperature_from_sample(rho, rho_t4):
    """T [K] at a sample from interpolated rho and rho (T/T_REF)^4 (0 where rho = 0)."""
    return T_REF * (rho_t4 / rho) ** 0.25 if rho > 0.0 else 0.0


def texture_from_gasgrid(gg: GasGrid) -> Texture:
    return texture_from_fields(gg.grid, gg.rho, gg.t_em, gg.vel)


@dataclass(frozen=True)
class GreyBlackbody:
    """Phase 8 placeholder emission (8.4): S = sigma T^4 / pi [W m^-2 sr^-1],
    alpha = kappa rho [1/code length], kappa in code units (length^2 / mass)."""

    kappa: float

    def source_and_absorption(self, rho, rho_t4):
        """(S, alpha) from the sampled texture values."""
        t = temperature_from_sample(rho, rho_t4)
        return blackbody_intensity_si(t), grey_absorption(rho, self.kappa)


@dataclass(frozen=True)
class LineLook:
    """Nebula emission lines on top of the blackbody (Phase 14 N1; physics/emission_lines.py),
    with the look knobs (labelled cheats, DECISIONS 2026-11-23):

    gain: multiplies the line glow; continuum_mix: 0 = today's continuum source S = B, 1 = the
    thin-gas form S = sqrt(q/(1+q)) B (physics.opacity.thin_emission_fraction), in between a
    linear blend of the two factors; x_h, x_o3, x_ns: the ionization tags (uniform until N4
    deposits per-particle tags); n_cap, t_fade, fade_power: as line_emissivities_si.
    tables: render.colour_data.load_emission_tables(); lam, cmf: load_cmf()."""

    tables: dict
    lam: np.ndarray
    cmf: np.ndarray
    gain: float = 1.0
    continuum_mix: float = 0.0
    x_h: float = 1.0
    x_o3: float = 0.03
    x_ns: float = 0.3
    n_cap: float = 1e3
    t_fade: float = 4e4
    fade_power: float = 8.0


@dataclass(frozen=True)
class RelativisticBlackbody:
    """Phase 9 emission (phase-09 9.4): gas emits as a blackbody at its temperature T_em in its
    own frame, with grey absorption alpha_gas = kappa rho in its own frame. Seen from a static
    camera at r_cam, a step contributes the blackbody at the observed temperature g T_em
    (physics.redshift.redshift_factor; no extra g^4) and has optical depth
    kappa rho ds_static (gamma - gamma beta . n) (physics.redshift.gas_frame_path_factor).
    Channels: CIE X, Y, Z and the bolometric intensity sigma (g T)^4 / pi [W m^-2 sr^-1].

    log_t, log_chroma: the colour table of physics.colour.chroma_lut.

    opacity: "constant" (kappa rho, phase-09 9.4 option 2; Phase 8's tuned look) or "thermal"
    (option 1: physics.opacity.thermal_absorption_code, free-free + electron scattering at the
    gas's own temperature; kappa is then unused). Hiten 2026-11-17: "thermal" clears the glow of
    thin gas round the ring (DECISIONS)."""

    kappa: float
    r_cam: float
    log_t: np.ndarray
    log_chroma: np.ndarray
    opacity: str = "constant"
    # spinning hole (Phase 11): the Kerr static lapse in the redshift (physics.kerr notes);
    # f_cam = Kerr-Schild f at the camera. spin 0: the Phase 9 formulas unchanged.
    spin: float = 0.0
    f_cam: float = 0.0
    # Phase 14 N1: emission lines (needs opacity "thermal"); None = Phase 9 colours unchanged
    lines: LineLook | None = None
    relativistic = True

    def __post_init__(self):
        if self.opacity not in ("constant", "thermal"):
            raise ValueError(
                f'opacity = {self.opacity!r}: use "constant" or "thermal"'
            )  # march_segment: sample velocities and pass the light direction
        if self.lines is not None and self.opacity != "thermal":
            raise ValueError('emission lines need opacity = "thermal" (their continuum blend)')

    channels = 4

    def source_and_dtau(self, mid, e, rho, rho_t4, rho_gb, path_static):
        """(source (4,), optical depth) of a step at mid along the outward direction e."""
        t_em = temperature_from_sample(rho, rho_t4)
        gb = np.asarray(rho_gb, float) / rho
        gbl = float(np.sqrt(gb @ gb))
        if gbl > GB_MAX:
            gb = gb * (GB_MAX / gbl)
        if self.spin > 0.0:  # approximate Doppler near a spinning hole (physics.kerr notes)
            n_light = -static_frame_direction_kerr(mid, e, self.spin)
            g = float(redshift_factor_kerr(mid, gb, n_light, self.spin, self.f_cam))
        else:
            n_light = -static_frame_direction(mid, e)  # light travels back toward the camera
            r = float(np.sqrt(mid @ mid))
            g = float(redshift_factor(r, gb, n_light, self.r_cam))
        t_obs = g * t_em
        src = np.zeros(4)
        if t_obs > 0.0:
            src[:3] = blackbody_xyz_lut(t_obs, self.log_t, self.log_chroma)
            src[3] = bolometric_intensity_si(t_obs)
        if self.opacity == "thermal":
            alpha = float(thermal_absorption_code(rho, t_em))
        else:
            alpha = float(grey_absorption(rho, self.kappa))
        doppler = float(gas_frame_path_factor(gb, n_light))
        dtau = alpha * path_static * doppler
        if self.lines is not None and dtau > 0.0:
            src = src * self._continuum_factor(rho, t_em)
            src = src + self._line_source(rho, t_em, g, path_static * doppler) / dtau
        return src, dtau

    def _continuum_factor(self, rho, t_em):
        mix = self.lines.continuum_mix
        return (1.0 - mix) + mix * float(thin_emission_fraction(rho, t_em))

    def _line_source(self, rho, t_em, g, path_gas):
        """Observed line radiance of the step (X, Y, Z, bolometric) [W m^-2 sr^-1]: g^4 j over
        the gas-frame path (physics/emission_lines.py), times the gain. The caller divides by
        dtau so that transfer_step's S (1 - e^-dtau) gives j path (1 - e^-dtau) / dtau."""
        lk = self.lines
        j = line_emission_code(rho, t_em, lk.x_h, lk.x_o3, lk.x_ns, lk.tables, n_cap=lk.n_cap,
                               t_fade=lk.t_fade, fade_power=lk.fade_power)  # fmt: skip
        out = np.zeros(4)
        out[:3] = lines_to_xyz(j, g, lk.lam, lk.cmf)
        out[3] = g**4 * float(np.sum(j))
        return lk.gain * path_gas * out


# ---------------------------------------------------------------- sampling


def _trilinear(arr, fi, fj, fk, wrap_k):
    """Trilinear interpolation of arr at fractional indices (cell centres at integers).

    i and j are clamped to the array; k wraps if wrap_k (phi), else is clamped.
    """
    n0, n1, n2 = arr.shape
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    k0 = int(np.floor(fk))
    ti, tj, tk = fi - i0, fj - j0, fk - k0
    out = 0.0
    for di, wi in ((0, 1.0 - ti), (1, ti)):
        ii = min(max(i0 + di, 0), n0 - 1)
        for dj, wj in ((0, 1.0 - tj), (1, tj)):
            jj = min(max(j0 + dj, 0), n1 - 1)
            for dk, wk in ((0, 1.0 - tk), (1, tk)):
                kk = (k0 + dk) % n2 if wrap_k else min(max(k0 + dk, 0), n2 - 1)
                out += wi * wj * wk * float(arr[ii, jj, kk])
    return out


def sample_log(tex: Texture, p):
    """(rho, rho (T/T_REF)^4) of the log grid at point p; zero outside r_min..r_max."""
    g = tex.grid
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    r = math.sqrt(x * x + y * y + z * z)
    if not (g.r_min <= r <= g.r_max):
        return 0.0, 0.0
    fi = math.log(r / g.r_min) / g.dlnr - 0.5
    fj = math.acos(min(max(z / r, -1.0), 1.0)) / g.dth - 0.5
    fk = (math.atan2(y, x) % (2.0 * math.pi)) / g.dph - 0.5
    return _trilinear(tex.rho, fi, fj, fk, True), _trilinear(tex.rho_t4, fi, fj, fk, True)


def _box_weight(b: CartesianBox, p):
    """Weight of the box at p: 1 inside, falling linearly to 0 over BLEND_CELLS at the faces."""
    inner = np.min(0.5 * b.size - np.abs(p - np.asarray(b.centre, float)))  # distance to a face
    return float(np.clip(inner / (BLEND_CELLS * b.dx), 0.0, 1.0))


def sample_box(tex: Texture, p):
    b = tex.grid
    f = (p - b.lo) / b.dx - 0.5
    return (
        _trilinear(tex.rho, f[0], f[1], f[2], False),
        _trilinear(tex.rho_t4, f[0], f[1], f[2], False),
    )


def _split(textures):
    """(log-grid texture, boxes, brick texture or None) of a texture list."""
    bricks = [t for t in textures[1:] if isinstance(t, BrickTexture)]
    boxes = [t for t in textures[1:] if not isinstance(t, BrickTexture)]
    if len(bricks) > 1:
        raise ValueError("at most one brick texture")
    return textures[0], boxes, (bricks[0] if bricks else None)


def sample(textures, p):
    """Fields at p from the finest grid there: boxes (listed finest first) blended into the log
    grid, plus the sparse bricks if present. ``textures``: the log-grid texture first (or
    None), then box textures and/or one bricks.BrickTexture. Bricks hold particles the log grid
    does not (each particle is deposited once), so their fields simply add (no blending)."""
    base, boxes, bricks = _split(textures)
    lr, lt = sample_log(base, p) if base is not None else (0.0, 0.0)
    if bricks is not None:
        br, bt = sample_bricks(bricks, p)
        lr, lt = lr + br, lt + bt
    for tb in boxes:
        w = _box_weight(tb.grid, p)
        if w > 0.0:
            br, bt = sample_box(tb, p)
            return w * br + (1.0 - w) * lr, w * bt + (1.0 - w) * lt
    return lr, lt


def _trilinear_vec(arr, fi, fj, fk, wrap_k):
    """_trilinear of each component of arr (..., 3)."""
    return np.array([_trilinear(arr[..., c], fi, fj, fk, wrap_k) for c in range(arr.shape[-1])])


def sample_log_rho_gb(tex: Texture, p):
    g = tex.grid
    if tex.rho_gb is None:
        return np.zeros(3)
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    r = math.sqrt(x * x + y * y + z * z)
    if not (g.r_min <= r <= g.r_max):
        return np.zeros(3)
    fi = math.log(r / g.r_min) / g.dlnr - 0.5
    fj = math.acos(min(max(z / r, -1.0), 1.0)) / g.dth - 0.5
    fk = (math.atan2(y, x) % (2.0 * math.pi)) / g.dph - 0.5
    return _trilinear_vec(tex.rho_gb, fi, fj, fk, True)


def sample_box_rho_gb(tex: Texture, p):
    if tex.rho_gb is None:
        return np.zeros(3)
    f = (p - tex.grid.lo) / tex.grid.dx - 0.5
    return _trilinear_vec(tex.rho_gb, f[0], f[1], f[2], False)


def sample_rho_gb(textures, p):
    """rho x gamma beta (3,) at p, combined from the grids exactly as ``sample`` combines rho
    (so rho_gb / rho is the mass-weighted gamma beta of the gas there)."""
    base, boxes, bricks = _split(textures)
    lv = sample_log_rho_gb(base, p) if base is not None else np.zeros(3)
    if bricks is not None:
        lv = lv + sample_bricks_rho_gb(bricks, p)
    for tb in boxes:
        w = _box_weight(tb.grid, p)
        if w > 0.0:
            return w * sample_box_rho_gb(tb, p) + (1.0 - w) * lv
    return lv


def cell_size(textures, p):
    """Local cell size of the finest grid at p (box dx; the bricks' dx where p's home brick is
    allocated, if finer; else r min(dln r, dth) on the log grid)."""
    base, boxes, bricks = _split(textures)
    for tb in boxes:
        b = tb.grid
        c, half = b.centre, 0.5 * b.size
        if all(abs(float(p[a]) - c[a]) <= half for a in range(3)):
            return b.dx
    g = base.grid
    r = max(math.sqrt(float(p[0]) ** 2 + float(p[1]) ** 2 + float(p[2]) ** 2), g.r_min)
    size = r * min(g.dlnr, g.dth)
    if bricks is not None and home_allocated(bricks, p):
        size = min(size, bricks.lattice.dx)
    return size


def brick_step_limit(textures, p, e):
    """Longest step from p along e allowed by the bricks (inf where there are none): inside the
    box holding the bricks, a point whose home brick is empty steps no further than just past
    that brick's far face, so no allocated brick is jumped over (bricks.py); from outside the
    box, the step ends just inside its face."""
    bricks = _split(textures)[2]
    if bricks is None or not len(bricks.keys):
        return math.inf
    dx = bricks.lattice.dx
    if np.all(p >= bricks.box_lo) and np.all(p < bricks.box_hi):
        if home_allocated(bricks, p):
            return math.inf  # cell_size already holds the step to STEP_FRACTION dx
        return exit_distance(p, e, dx)
    t = entry_distance(p, e, bricks.box_lo, bricks.box_hi)
    return t + EXIT_EPS * dx if math.isfinite(t) else math.inf


# ---------------------------------------------------------------- marching


KERR_F_MAX = 0.99  # spinning hole: gas only where f < this (static lapse > 0.1)


def march_segment(textures, model, a, b, intensity=0.0, transmission=1.0, curved=False,
                  spin: float = 0.0):  # fmt: skip
    """March the straight segment a -> b (front to back) and return (I, T, n_steps).

    Each sub-step is at most STEP_FRACTION x the local cell size at both its start and its
    end (cells shrink toward the hole), sampled at its midpoint. ``curved``: optical depth
    uses the static observer's proper length of each sub-step around the hole
    (physics.geodesics.schw_static_proper_length) instead of its flat length.

    With a Phase 9 model (``model.relativistic``) the intensity is an array of
    ``model.channels`` (start it at zeros), and each step also samples the gas velocity and
    passes the step's direction to the model.
    """
    rel = getattr(model, "relativistic", False)
    if rel and np.ndim(intensity) == 0:
        intensity = np.full(model.channels, float(intensity))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    seg = b - a
    length = float(np.sqrt(seg @ seg))
    if length == 0.0:
        return intensity, transmission, 0
    e = seg / length
    s = 0.0
    n = 0
    while s < length and transmission >= T_STOP:
        p = a + s * e
        ds = min(STEP_FRACTION * cell_size(textures, p), length - s)
        ds = min(ds, STEP_FRACTION * cell_size(textures, p + ds * e))
        ds = min(ds, brick_step_limit(textures, p, e))
        mid = p + 0.5 * ds * e
        lr, lt = sample(textures, mid)
        r_mid = float(np.sqrt(mid @ mid))
        # A step with no gas changes nothing; skipping it also avoids 0 x infinity when a
        # captured ray's last step ends exactly on the horizon, where the proper length
        # diverges (r_mid = 2 happens in float32 on the GPU).
        deep = spin > 0.0 and float(metric_parts(mid, spin)[1]) >= KERR_F_MAX  # ergosphere
        if lr > (RHO_MIN_COLOUR if rel else 0.0) and (not curved or r_mid > 2.0) and not deep:
            path = ds
            if curved:
                q = p + ds * e
                dr = float(np.sqrt(q @ q) - np.sqrt(p @ p))
                path = float(schw_static_proper_length(ds, dr, r_mid))
            if rel:
                src, dtau = model.source_and_dtau(mid, e, lr, lt, sample_rho_gb(textures, mid),
                                                  path)  # fmt: skip
            else:
                src, alpha = model.source_and_absorption(lr, lt)
                dtau = alpha * path
            if dtau > 0.0:
                intensity, transmission = transfer_step(intensity, transmission, src, dtau)
        s += ds
        n += 1
    if rel:
        return np.asarray(intensity, float), float(transmission), n
    return float(intensity), float(transmission), n


def march_polyline(textures, model, points):
    """March a ray given as points from the camera outward (front to back); returns (I, T).

    Step 3 feeds the points of a bent light ray; here they may be any polyline."""
    intensity, transmission = 0.0, 1.0
    for a, b in zip(points[:-1], points[1:], strict=True):
        intensity, transmission, _ = march_segment(textures, model, a, b, intensity, transmission)
        if transmission < T_STOP:
            break
    return intensity, transmission
