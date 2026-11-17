"""GPU gas rendering along bent rays (phases/phase-08-gas-rendering.md 8.4): Python side of
shaders/gas_march.wgsl. The CPU reference it must match is render/bent_rays.py (8.6 test 7).

Textures go to the GPU as rg32float 3D textures (rho, rho (T/T_REF)^4), the same float32 values
the CPU reference samples. The log grid is required (it sets the step size away from boxes);
up to two detail boxes (finest first, as in march.sample) and optionally one sparse brick
texture (bricks.py): uploaded as an r32uint index texture of brick slots (slot + 1, 0 = empty)
over the allocated bricks' bounding range and an rg32float atlas of 9^3-texel bricks (8^3
cells plus a one-cell apron on the high side, copied from the neighbouring bricks, so the
trilinear sample at any point reads its home brick only).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import wgpu

from nebula import units
from nebula.render.bricks import BRICK, BrickTexture, pack_keys, unpack_keys
from nebula.render.camera import Camera
from nebula.render.device import TileTimings, get_device, load_shader, run_tiled, uniform_buffer
from nebula.render.march import (
    BLEND_CELLS,
    STEP_FRACTION,
    T_REF,
    T_STOP,
    GreyBlackbody,
    Texture,
    _split,
)
from nebula.render.starfield import GPUSky, upload_cube

MAX_BOXES = 2
APRON = BRICK + 1  # atlas texels per brick side
MAX_TEX_3D = 2048  # wgpu default maxTextureDimension3D
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


def _upload_u32(data: np.ndarray) -> wgpu.GPUTexture:
    """r32uint 3D texture of a (n0, n1, n2) array (element [i, j, k] at texel (k, j, i))."""
    data = np.ascontiguousarray(data, dtype=np.uint32)
    n0, n1, n2 = data.shape
    device = get_device()
    t = device.create_texture(
        size=(n2, n1, n0),
        dimension="3d",
        format=wgpu.TextureFormat.r32uint,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        {"texture": t}, data.tobytes(), {"bytes_per_row": n2 * 4, "rows_per_image": n1},
        (n2, n1, n0),
    )  # fmt: skip
    return t


def brick_arrays(tex: BrickTexture):
    """(index (n0, n1, n2) uint32 of slot + 1, origin brick (3,), atlas (A0 9, A1 9, A2 9, 2)
    float32, slots per axis (A0, A1, A2)). Slot s sits at atlas brick (s // (A1 A2),
    (s // A2) % A1, s % A2)."""
    nb = len(tex.keys)
    bi, bj, bk = unpack_keys(tex.keys)
    origin = np.array([bi.min(), bj.min(), bk.min()], np.int64)
    dims = np.array([bi.max(), bj.max(), bk.max()]) - origin + 1
    a = int(np.ceil(nb ** (1.0 / 3.0)))
    slots = np.array([int(np.ceil(nb / (a * a))), a, a])
    if dims.max() > MAX_TEX_3D or (slots * APRON).max() > MAX_TEX_3D:
        raise ValueError(
            f"brick textures too large ({dims} index, {slots * APRON} atlas): raise dx"
        )
    index = np.zeros(dims, np.uint32)
    index[bi - origin[0], bj - origin[1], bk - origin[2]] = np.arange(1, nb + 1)
    data = np.stack([tex.rho, tex.rho_t4], axis=-1)  # (nb, 8, 8, 8, 2)
    padded = np.zeros((nb, APRON, APRON, APRON, 2), np.float32)
    padded[:, :BRICK, :BRICK, :BRICK] = data
    for o0 in (0, 1):
        for o1 in (0, 1):
            for o2 in (0, 1):
                if o0 == o1 == o2 == 0:
                    continue
                key = pack_keys(bi + o0, bj + o1, bk + o2)
                at = np.clip(np.searchsorted(tex.keys, key), 0, nb - 1)
                sel = np.nonzero(tex.keys[at] == key)[0]
                dst = tuple(BRICK if o else slice(0, BRICK) for o in (o0, o1, o2))
                src = tuple(0 if o else slice(0, BRICK) for o in (o0, o1, o2))
                padded[(sel, *dst)] = data[at[sel]][(slice(None), *src)]
    grid = np.zeros((int(np.prod(slots)), APRON, APRON, APRON, 2), np.float32)
    grid[:nb] = padded
    del padded
    atlas = grid.reshape(*slots, APRON, APRON, APRON, 2).transpose(0, 3, 1, 4, 2, 5, 6)
    atlas = np.ascontiguousarray(atlas).reshape(*(slots * APRON), 2)
    return index, origin, atlas, slots


def _brick_params(bricks: BrickTexture | None, slots=None, origin=None, dims=None) -> bytes:
    f = np.zeros(4, np.float32)
    i = np.zeros(4, np.int32)
    u = np.zeros(4, np.uint32)
    a = np.zeros(4, np.uint32)
    lo = np.zeros(4, np.float32)
    hi = np.zeros(4, np.float32)
    if bricks is not None:
        f[0], f[2] = bricks.lattice.dx, 1.0
        i[:3], u[:3], a[:3] = origin, dims, slots
        lo[:3], hi[:3] = bricks.box_lo, bricks.box_hi
    return (f.tobytes() + i.tobytes() + u.tobytes() + a.tobytes() + lo.tobytes()
            + hi.tobytes())  # fmt: skip


def _box_params(tex: Texture | None) -> tuple[np.ndarray, np.ndarray]:
    f = np.zeros(8, np.float32)
    u = np.zeros(4, np.uint32)
    if tex is not None:
        b = tex.grid
        f[0:3], f[3] = b.lo, b.dx
        f[4:7], f[7] = b.centre, 0.5 * b.size
        u[0], u[1] = b.n, 1
    return f, u


def gas_params_bytes(textures, model: GreyBlackbody, step_fraction: float = STEP_FRACTION,
                     brick_layout=None) -> bytes:  # fmt: skip
    """The GasParams uniform of shaders/gas_march.wgsl (``brick_layout``: (slots, origin,
    dims) of the uploaded brick textures, if any)."""
    g = textures[0].grid
    _, box_list, bricks = _split(textures)
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
    boxes = list(box_list) + [None] * (MAX_BOXES - len(box_list))
    for tb in boxes:
        f, u = _box_params(tb)
        out += f.tobytes() + u.tobytes()
    out += _brick_params(bricks, *(brick_layout or (None, None, None)))
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
    _, box_list, bricks = _split(textures)
    if len(box_list) > MAX_BOXES:
        raise ValueError(f"at most {MAX_BOXES} detail boxes")
    boxes = list(box_list) + [None] * (MAX_BOXES - len(box_list))
    gpu = [_upload(textures[0])] + [_upload(b) for b in boxes]
    layout = None
    if bricks is not None and len(bricks.keys):
        index, origin, atlas, slots = brick_arrays(bricks)
        gpu += [_upload_u32(index), _upload_atlas(atlas)]
        layout = (slots, origin, np.array(index.shape))
    else:
        bricks = None
        gpu += [_upload_u32(np.zeros((1, 1, 1))), _upload_atlas(np.zeros((1, 1, 1, 2)))]
    tex_for_params = [textures[0], *box_list] + ([bricks] if bricks is not None else [])
    params = gas_params_bytes(tex_for_params, model, step_fraction, layout)
    return GpuScene(gpu, uniform_buffer(params))


def _upload_atlas(atlas: np.ndarray) -> wgpu.GPUTexture:
    return _upload(Texture(None, atlas[..., 0], atlas[..., 1]))


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
    for k, t in enumerate(scene.gpu_textures[:3]):
        extra.append({"binding": 3 + k, "resource": t.create_view()})
    for k, t in enumerate(scene.gpu_textures[3:]):
        extra.append({"binding": 9 + k, "resource": t.create_view()})
    extra.append({"binding": 6, "resource": {"buffer": scene.params}})
    extra.extend(sky.entries(7))
    return run_tiled(load_shader(*SHADER), camera.width, camera.height, tile=tile, extra=extra)
