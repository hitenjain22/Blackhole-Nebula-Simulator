"""GPU gas rendering along bent rays (phases/phase-08-gas-rendering.md 8.4): Python side of
shaders/gas_march.wgsl. The CPU reference it must match is render/bent_rays.py (8.6 test 7).

Textures go to the GPU as rg32float 3D textures (rho, rho (T/T_REF)^4), the same float32 values
the CPU reference samples. The log grid is required (it sets the step size away from boxes);
up to two detail boxes (finest first, as in march.sample).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import wgpu

from nebula import units
from nebula.render.camera import Camera
from nebula.render.device import TileTimings, get_device, load_shader, run_tiled, uniform_buffer
from nebula.render.march import BLEND_CELLS, STEP_FRACTION, T_REF, T_STOP, GreyBlackbody, Texture
from nebula.render.starfield import GPUSky, upload_cube

MAX_BOXES = 2
SHADER = ("schw_ray.wgsl", "camera_rays.wgsl", "gas_march.wgsl")
OUT_DIAGNOSTIC = 0  # (I, T of escaped rays, captured fraction, sub-steps): tests
OUT_PICTURE = 1  # (gas I, T x star radiance RGB): frames
OUT_SKY_DIR = 2  # (sky direction of the pixel-centre ray with n_sub = 1, escaped 0/1): tests


def _upload(tex: Texture | None) -> wgpu.GPUTexture:
    """rg32float 3D texture of a Texture (array [i, j, k] at texel (k, j, i)); 1 empty texel
    for None."""
    if tex is None:
        data = np.zeros((1, 1, 1, 2), np.float32)
    else:
        data = np.ascontiguousarray(np.stack([tex.rho, tex.rho_t4], axis=-1), dtype=np.float32)
    n0, n1, n2 = data.shape[:3]
    device = get_device()
    t = device.create_texture(
        size=(n2, n1, n0),
        dimension="3d",
        format=wgpu.TextureFormat.rg32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        {"texture": t},
        data.tobytes(),
        {"bytes_per_row": n2 * 8, "rows_per_image": n1},
        (n2, n1, n0),
    )
    return t


def _box_params(tex: Texture | None) -> tuple[np.ndarray, np.ndarray]:
    f = np.zeros(8, np.float32)
    u = np.zeros(4, np.uint32)
    if tex is not None:
        b = tex.grid
        f[0:3], f[3] = b.lo, b.dx
        f[4:7], f[7] = b.centre, 0.5 * b.size
        u[0], u[1] = b.n, 1
    return f, u


def gas_params_bytes(textures, model: GreyBlackbody, step_fraction: float = STEP_FRACTION) -> bytes:
    """The GasParams uniform of shaders/gas_march.wgsl."""
    g = textures[0].grid
    s_scale = units.SIGMA_SB * T_REF**4 / np.pi  # S = s_scale * rho T^4 / rho (GreyBlackbody)
    head = np.array(
        [
            g.r_min, g.r_max, g.dlnr, g.dth,
            g.dph, step_fraction, T_STOP, BLEND_CELLS,
            model.kappa, s_scale, 0.0, 0.0,
        ],
        np.float32,
    )  # fmt: skip
    out = head.tobytes() + np.array([g.nr, g.nth, g.nph, 0], np.uint32).tobytes()
    boxes = list(textures[1:]) + [None] * (MAX_BOXES - len(textures[1:]))
    for tb in boxes:
        f, u = _box_params(tb)
        out += f.tobytes() + u.tobytes()
    return out


@dataclass
class GpuScene:
    """Textures and parameters uploaded once, reused for many frames/cameras."""

    gpu_textures: list
    params: wgpu.GPUBuffer


def upload_scene(textures, model: GreyBlackbody, step_fraction: float = STEP_FRACTION) -> GpuScene:
    """Upload the textures (log grid first, then up to MAX_BOXES boxes). ``step_fraction``:
    march sub-step as a fraction of the local cell (8.4: 0.5; smaller only for test 10)."""
    if textures[0] is None:
        raise ValueError("the log grid is required (it sets the march step away from boxes)")
    if len(textures) - 1 > MAX_BOXES:
        raise ValueError(f"at most {MAX_BOXES} detail boxes")
    boxes = list(textures[1:]) + [None] * (MAX_BOXES - len(textures[1:]))
    gpu = [_upload(textures[0])] + [_upload(b) for b in boxes]
    return GpuScene(gpu, uniform_buffer(gas_params_bytes(textures, model, step_fraction)))


@cache
def _black_sky() -> GPUSky:
    """A 1-texel black cube map, bound when a render has no star field."""
    return upload_cube([np.zeros((6, 1, 1, 3), np.float32)])


def render_gas(
    camera: Camera,
    scene: GpuScene,
    n_sub: int = 1,
    tile: int | None = None,
    sky: GPUSky | None = None,
    out_mode: int = OUT_DIAGNOSTIC,
) -> tuple[np.ndarray, TileTimings]:
    """(H, W, 4) float32 image. OUT_DIAGNOSTIC: mean I [W m^-2 sr^-1], mean T of escaped
    sub-rays, captured fraction, mean sub-steps per ray (bent_rays.render_reference layout).
    OUT_PICTURE: mean gas I, then the mean over sub-rays of T x star radiance (RGB, the cube
    map's flux units; black without ``sky``). n_sub <= 4.

    Default tile: 64 px for 1 ray per pixel, shrinking with n_sub so each dispatch traces about
    the same number of rays (measured: 64 x 64 x 4 rays up to 0.86 s on the Iris Plus with a
    busy CPU, against the 1 s watchdog limit)."""
    if not 1 <= n_sub <= 4:
        raise ValueError("n_sub must be 1..4")
    if tile is None:
        tile = max(16, 64 // n_sub)
    sky = sky or _black_sky()
    cam_buf = uniform_buffer(camera.uniform_bytes(n_sub, out_mode, (sky.scale, 0.0)))
    extra = [{"binding": 2, "resource": {"buffer": cam_buf}}]
    for k, t in enumerate(scene.gpu_textures):
        extra.append({"binding": 3 + k, "resource": t.create_view()})
    extra.append({"binding": 6, "resource": {"buffer": scene.params}})
    extra.extend(sky.entries(7))
    return run_tiled(load_shader(*SHADER), camera.width, camera.height, tile=tile, extra=extra)
