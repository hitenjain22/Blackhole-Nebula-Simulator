"""Procedural star field baked into a cube map (phases/phase-02-blackhole-render.md Step 2.4).

Artistic background, not data: ~100,000 stars plus an optional faint Milky Way band, all
procedural (no catalogue, so no licence questions). Labelled as procedural in the videos.

Stars: directions uniform on the sky, plus a fraction concentrated toward the band;
magnitudes with number counts rising steeply toward faint stars, N(< m) ∝ 10^(0.5 m) (roughly
the slope of real star counts); temperatures 3,000-30,000 K with a placeholder blackbody
colour (three-wavelength Planck ratio; Phase 9 replaces it with CIE colour matching).
Each star is a Gaussian spot of fixed angular size, deposited by true angular distance, so
spots crossing a cube-face edge continue seamlessly on the neighbour face; each star's
weights are normalized so its total light (radiance x solid angle) is exactly its flux.

Cube map layout follows WebGPU / Vulkan / D3D (face order +X, -X, +Y, -Y, +Z, -Z; ``face_uv``
below); tests/render/test_starfield.py checks it against GPU sampling.

Stored as rgba16float. Radiances are scaled so the brightest texel is PEAK_VALUE (float16
overflows at 65,504); the absolute scale is artistic and absorbed by the exposure.
"""

from dataclasses import dataclass

import numba
import numpy as np
import wgpu
from scipy.ndimage import map_coordinates

from nebula.render.device import get_device, load_shader, run_items
from nebula.units import H_PLANCK, K_B, C

PEAK_VALUE = 1000.0
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])  # Rec. 709 luminance of linear RGB
FACE_AXES = (  # (major axis index, sign) for faces +X, -X, +Y, -Y, +Z, -Z
    (0, 1.0),
    (0, -1.0),
    (1, 1.0),
    (1, -1.0),
    (2, 1.0),
    (2, -1.0),
)


@dataclass(frozen=True)
class StarFieldConfig:
    n_stars: int = 100_000
    seed: int = 2026
    face_size: int = 2048
    mag_bright: float = -1.0  # brightest magnitude drawn
    mag_faint: float = 8.0  # faintest magnitude drawn
    count_slope: float = 0.5  # d log10 N(<m) / dm
    star_sigma_texels: float = 1.5  # Gaussian spot size, in texels at a face centre
    band: bool = True  # the Milky Way band (Hiten's choice, 2026-10-14)
    band_pole: tuple[float, float, float] = (0.35, -0.55, 0.76)  # normal of the band's plane
    band_width_deg: float = 9.0
    band_brightness: float = (
        15.0  # band radiance relative to the sky-averaged starlight (chosen by eye)
    )
    band_star_fraction: float = 0.35  # extra stars concentrated toward the band


# --------------------------------------------------------------------------- geometry


def face_uv(face: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For directions d (..., 3): (u, v, |major component|) on ``face`` in the
    WebGPU/Vulkan/D3D convention, u, v in [0, 1] with v = 0 at the top row of the face."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    if face == 0:
        sc, tc, ma = -z, -y, x
    elif face == 1:
        sc, tc, ma = z, -y, -x
    elif face == 2:
        sc, tc, ma = x, z, y
    elif face == 3:
        sc, tc, ma = x, -z, -y
    elif face == 4:
        sc, tc, ma = x, -y, z
    else:
        sc, tc, ma = -x, -y, -z
    return 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0), ma


def texel_directions(face: int, size: int) -> np.ndarray:
    """Unit directions (size, size, 3) of the texel centres of one face (row = v, col = u)."""
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # sc, tc in (-1, 1)
    tc, sc = np.meshgrid(c, c, indexing="ij")
    one = np.ones_like(sc)
    d = {
        0: (one, -tc, -sc),
        1: (-one, -tc, sc),
        2: (sc, one, tc),
        3: (sc, -one, -tc),
        4: (sc, -tc, one),
        5: (-sc, -tc, -one),
    }[face]
    v = np.stack(d, axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def texel_solid_angles(size: int) -> np.ndarray:
    """Solid angle (sr) of each texel of a face (same for all faces): exact, from the
    area of a cube-face rectangle projected onto the sphere."""
    edges = np.linspace(-1.0, 1.0, size + 1)

    def corner(a, b):
        return np.arctan2(a * b, np.sqrt(a * a + b * b + 1.0))

    a, b = np.meshgrid(edges, edges, indexing="ij")
    f = corner(a, b)
    return f[1:, 1:] - f[:-1, 1:] - f[1:, :-1] + f[:-1, :-1]


# --------------------------------------------------------------------------- stars


def placeholder_blackbody_rgb(t_kelvin: np.ndarray) -> np.ndarray:
    """Rough linear RGB of a blackbody: Planck B_lambda at 610 / 550 / 465 nm, normalized to
    unit luminance (Rec. 709 weights). Placeholder until Phase 9's CIE colour pipeline."""
    lam = np.array([610e-9, 550e-9, 465e-9])
    t = np.asarray(t_kelvin, float)[..., None]
    b = 1.0 / lam**5 / np.expm1(H_PLANCK * C / (lam * K_B * t))
    rgb = b / b[..., 1:2]
    return rgb / (rgb @ LUMA_WEIGHTS)[..., None]


@dataclass
class Stars:
    direction: np.ndarray  # (n, 3) unit vectors
    magnitude: np.ndarray  # (n,)
    temperature: np.ndarray  # (n,) K
    flux: np.ndarray  # (n,) 10^(-0.4 m)
    rgb: np.ndarray  # (n, 3) linear colour x flux


def _band_pole(cfg: StarFieldConfig) -> np.ndarray:
    p = np.asarray(cfg.band_pole, float)
    return p / np.linalg.norm(p)


def generate_stars(cfg: StarFieldConfig) -> Stars:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_stars
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    if cfg.band:  # pull a fraction of the stars toward the band's plane
        k = int(cfg.band_star_fraction * n)
        pole = _band_pole(cfg)
        lat = np.radians(cfg.band_width_deg) * rng.normal(size=k)
        along = d[:k] - (d[:k] @ pole)[:, None] * pole
        along /= np.linalg.norm(along, axis=1, keepdims=True)
        d[:k] = np.cos(lat)[:, None] * along + np.sin(lat)[:, None] * pole
    # N(<m) ∝ 10^(s m): inverse-CDF sampling between mag_bright and mag_faint.
    s = cfg.count_slope
    lo, hi = 10 ** (s * cfg.mag_bright), 10 ** (s * cfg.mag_faint)
    mag = np.log10(lo + rng.uniform(size=n) * (hi - lo)) / s
    # Temperatures: log-uniform 3,000-30,000 K, weighted toward cool stars (p ∝ 1/T in log T).
    u = rng.uniform(size=n)
    temp = 3000.0 / (1.0 - u * (1.0 - 3000.0 / 30000.0))
    flux = 10 ** (-0.4 * mag)
    return Stars(d, mag, temp, flux, placeholder_blackbody_rgb(temp) * flux[:, None])


@numba.njit(cache=True)
def _face_coords(face, x, y, z):
    if face == 0:
        return -z, -y, x
    if face == 1:
        return z, -y, -x
    if face == 2:
        return x, z, y
    if face == 3:
        return x, -z, -y
    if face == 4:
        return x, -y, z
    return -x, -y, -z


@numba.njit(cache=True)
def _texel_vector(face, scc, tcc):
    if face == 0:
        return 1.0, -tcc, -scc
    if face == 1:
        return -1.0, -tcc, scc
    if face == 2:
        return scc, 1.0, tcc
    if face == 3:
        return scc, -1.0, -tcc
    if face == 4:
        return scc, -tcc, 1.0
    return -scc, -tcc, -1.0


@numba.njit(cache=True)
def _spot(face, size, sigma, radius, omega, x, y, z, out_i, out_j, out_w):
    """Texels of ``face`` covered by a Gaussian spot at direction (x, y, z): fills out_* and
    returns how many. Weight = exp(-angle^2 / (2 sigma^2)) x texel solid angle."""
    sc, tc, ma = _face_coords(face, x, y, z)
    if ma <= 0.3:
        return 0
    cu = 0.5 * (sc / ma + 1.0) * size
    cv = 0.5 * (tc / ma + 1.0) * size
    i0 = int(np.floor(cv)) - radius
    j0 = int(np.floor(cu)) - radius
    m = 0
    cut2 = (4.0 * sigma) ** 2
    for i in range(max(i0, 0), min(i0 + 2 * radius + 2, size)):
        tcc = (i + 0.5) / size * 2.0 - 1.0
        for j in range(max(j0, 0), min(j0 + 2 * radius + 2, size)):
            scc = (j + 0.5) / size * 2.0 - 1.0
            tx, ty, tz = _texel_vector(face, scc, tcc)
            cosang = (tx * x + ty * y + tz * z) / np.sqrt(tx * tx + ty * ty + tz * tz)
            ang2 = 2.0 * (1.0 - cosang)  # chord^2, equal to angle^2 to 1e-7 at these sizes
            if ang2 > cut2:
                continue
            out_i[m] = i
            out_j[m] = j
            out_w[m] = np.exp(-0.5 * ang2 / (sigma * sigma)) * omega[i, j]
            m += 1
    return m


@numba.njit(cache=True)
def _weights_one_face(star_dir, face, size, sigma, radius, omega):
    n = star_dir.shape[0]
    total = np.zeros(n)
    cap = (2 * radius + 2) ** 2
    ii = np.empty(cap, np.int64)
    jj = np.empty(cap, np.int64)
    ww = np.empty(cap)
    for k in range(n):
        m = _spot(
            face,
            size,
            sigma,
            radius,
            omega,
            star_dir[k, 0],
            star_dir[k, 1],
            star_dir[k, 2],
            ii,
            jj,
            ww,
        )
        for q in range(m):
            total[k] += ww[q]
    return total


@numba.njit(cache=True)
def _deposit(face_img, star_dir, star_scaled_rgb, face, size, sigma, radius, omega):
    """face_img (size, size, 3) += radiance of all stars' spots on this face. star_scaled_rgb
    is each star's rgb flux divided by its total kernel weight, so sum(L x omega) = flux."""
    cap = (2 * radius + 2) ** 2
    ii = np.empty(cap, np.int64)
    jj = np.empty(cap, np.int64)
    ww = np.empty(cap)
    for k in range(star_dir.shape[0]):
        m = _spot(
            face,
            size,
            sigma,
            radius,
            omega,
            star_dir[k, 0],
            star_dir[k, 1],
            star_dir[k, 2],
            ii,
            jj,
            ww,
        )
        for q in range(m):
            w = ww[q] / omega[ii[q], jj[q]]  # radiance = flux x kernel / solid angle
            for c in range(3):
                face_img[ii[q], jj[q], c] += w * star_scaled_rgb[k, c]


# --------------------------------------------------------------------------- band


def _value_noise(d: np.ndarray, rng_seed: int, octaves: int = 5) -> np.ndarray:
    """Smooth fractal noise in [0, 1] on the sphere: trilinear value noise on a 3D grid,
    summed over octaves (deterministic for a seed)."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros(d.shape[:-1])
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 8 * 2**o
        grid = rng.uniform(size=(n + 1, n + 1, n + 1))
        coords = ((d.reshape(-1, 3) + 1.0) * 0.5 * n).T
        out += amp * map_coordinates(grid, coords, order=1, mode="nearest").reshape(out.shape)
        total += amp
        amp *= 0.55
    return out / total


def band_radiance(d: np.ndarray, cfg: StarFieldConfig, mean_star_radiance: float) -> np.ndarray:
    """Milky Way band: Gaussian in latitude from the band's plane, mottled by fractal noise,
    with a soft brighter 'core' region (artistic). Linear RGB (..., 3), slightly warm."""
    pole = _band_pole(cfg)
    lat = np.arcsin(np.clip(d @ pole, -1, 1))
    w = np.radians(cfg.band_width_deg)
    profile = np.exp(-0.5 * (lat / w) ** 2)
    noise = _value_noise(d, cfg.seed + 1)
    core_dir = np.cross(pole, [0.0, 0.0, 1.0])
    core_dir /= np.linalg.norm(core_dir)
    core = 1.0 + 1.5 * np.exp(-0.5 * (np.arccos(np.clip(d @ core_dir, -1, 1)) / 0.5) ** 2)
    level = cfg.band_brightness * mean_star_radiance * profile * core * (0.25 + 1.5 * noise**2)
    return level[..., None] * np.array([1.0, 0.93, 0.85])


# --------------------------------------------------------------------------- cube map


@dataclass
class CubeMap:
    """Linear RGB radiance, faces (6, size, size, 3) float32 at full resolution, plus mips
    (list of (6, s, s, 3) for s = size/2 ... 1). ``scale`` multiplies stored values back to
    the stars' flux units (radiance in flux / sr)."""

    faces: np.ndarray
    mips: list[np.ndarray]
    scale: float
    total_star_flux: float

    @property
    def gpu_bytes(self) -> int:
        texels = sum(m.shape[0] * m.shape[1] * m.shape[2] for m in [self.faces, *self.mips])
        return texels * 8  # rgba16float


def _deposit_stars(direction, rgb_flux, size: int, sigma_texels: float) -> np.ndarray:
    """Radiance faces (6, size, size, 3) float64 of Gaussian star spots, each star's light
    exactly conserved (weights normalized over all faces)."""
    omega = texel_solid_angles(size)
    sigma = sigma_texels * np.pi / 2 / size
    radius = int(np.ceil(4 * sigma_texels)) + 1
    total_w = np.zeros(len(direction))
    for face in range(6):
        total_w += _weights_one_face(direction, face, size, sigma, radius, omega)
    scaled = rgb_flux / total_w[:, None]
    faces = np.zeros((6, size, size, 3), dtype=np.float64)
    for face in range(6):
        _deposit(faces[face], direction, scaled, face, size, sigma, radius, omega)
    return faces


def _finish(faces: np.ndarray, total_flux: float) -> CubeMap:
    """Scale to PEAK_VALUE, convert to float32 and build the mip chain (2x2 box averages)."""
    scale = faces.max() / PEAK_VALUE
    faces = (faces / scale).astype(np.float32)
    mips = []
    level = faces
    while level.shape[1] > 1:
        s = level.shape[1] // 2
        level = level.reshape(6, s, 2, s, 2, 3).mean(axis=(2, 4)).astype(np.float32)
        mips.append(level)
    return CubeMap(faces, mips, float(scale), float(total_flux))


def build_cube_map(cfg: StarFieldConfig | None = None) -> CubeMap:
    """The procedural sky of ``cfg``: stars plus (optionally) the Milky Way band."""
    cfg = cfg or StarFieldConfig()
    size = cfg.face_size
    stars = generate_stars(cfg)
    faces = _deposit_stars(stars.direction, stars.rgb, size, cfg.star_sigma_texels)
    if cfg.band:
        mean_radiance = stars.flux.sum() / (4 * np.pi)
        for face in range(6):
            faces[face] += band_radiance(texel_directions(face, size), cfg, mean_radiance)
    return _finish(faces, stars.flux.sum())


def single_star_cube_map(direction, sigma_deg: float, face_size: int = 512) -> CubeMap:
    """A black sky with one white Gaussian star (angular sigma in degrees) in ``direction``:
    the light source for the Einstein-ring validation image (phase-02 Step 2.5)."""
    d = np.asarray(direction, float)[None, :]
    d /= np.linalg.norm(d)
    sigma_texels = np.radians(sigma_deg) / (np.pi / 2 / face_size)
    faces = _deposit_stars(d, np.ones((1, 3)), face_size, sigma_texels)
    return _finish(faces, 1.0)


# --------------------------------------------------------------------------- GPU


@dataclass
class GPUSky:
    """A cube map on the GPU: texture, cube view, trilinear sampler, and the value scale."""

    texture: object
    view: object
    sampler: object
    scale: float

    def entries(self, first_binding: int = 3) -> list[dict]:
        """Bind-group entries: the cube view, then the sampler (render_stars.wgsl: 3 and 4)."""
        return [
            {"binding": first_binding, "resource": self.view},
            {"binding": first_binding + 1, "resource": self.sampler},
        ]


def upload_cube(levels: list[np.ndarray], scale: float = 1.0) -> GPUSky:
    """Upload mip levels (each (6, s, s, 3 or 4), largest first) as an rgba16float cube map."""
    device = get_device()
    size = levels[0].shape[1]
    tex = device.create_texture(
        size=(size, size, 6),
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        mip_level_count=len(levels),
    )
    for m, level in enumerate(levels):
        s = level.shape[1]
        rgba = np.ones((6, s, s, 4), dtype=np.float16)
        rgba[..., : level.shape[-1]] = level
        if not np.all(np.isfinite(rgba)):
            raise ValueError(f"mip {m}: values overflow float16 (max 65,504)")
        for face in range(6):
            device.queue.write_texture(
                {"texture": tex, "mip_level": m, "origin": (0, 0, face)},
                np.ascontiguousarray(rgba[face]).tobytes(),
                {"bytes_per_row": s * 8, "rows_per_image": s},
                (s, s, 1),
            )
    view = tex.create_view(dimension="cube")
    sampler = device.create_sampler(
        mag_filter="linear", min_filter="linear", mipmap_filter="linear"
    )
    return GPUSky(tex, view, sampler, scale)


def upload_cube_map(cube: CubeMap) -> GPUSky:
    return upload_cube([cube.faces, *cube.mips], cube.scale)


def sample_cube(sky: GPUSky, dirs: np.ndarray, lod: float = 0.0) -> np.ndarray:
    """Sample the GPU cube map in directions (n, 3) at mip level ``lod``: (n, 4) values as
    stored (multiply by sky.scale for flux radiance)."""
    dirs = np.asarray(dirs, dtype=np.float64)
    items = np.zeros((len(dirs), 4), dtype=np.float32)
    items[:, :3] = dirs
    items[:, 3] = lod
    raw = run_items(
        load_shader("cube_sample.wgsl"),
        np.array([len(dirs), 0, 0, 0], dtype=np.uint32).tobytes(),
        [items],
        len(dirs),
        16,
        extra=sky.entries(3),
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(len(dirs), 4)


def cube_gpu_bytes(face_size: int) -> int:
    """GPU memory of an rgba16float cube map with all mip levels."""
    total, s = 0, face_size
    while s >= 1:
        total += 6 * s * s * 8
        s //= 2
    return total
