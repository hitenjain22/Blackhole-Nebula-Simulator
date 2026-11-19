"""GPU device and tiled compute dispatch (Phase 0, Step 0.4).

Images are computed in tiles so no single GPU dispatch runs long enough to trip the macOS
GPU watchdog or freeze the screen (CLAUDE.md, "This machine"). Results go to a storage
*buffer* of vec4<f32> per pixel (buffers avoid texture-format limits) and are read back
to NumPy one tile at a time.

Shader contract for ``run_tiled``: entry point ``main`` with @workgroup_size(8, 8), and

    struct Tile { origin: vec2<u32>, image: vec2<u32>, size: vec2<u32>, _pad: vec2<u32> };
    @group(0) @binding(0) var<uniform> tile: Tile;
    @group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

writing pixel (origin + local) to out[local.y * size.x + local.x].

Shader contract for ``run_items`` (one invocation per item, e.g. per test ray): entry point
``main`` with @workgroup_size(64), binding 0 a uniform struct (any layout; its bytes are
passed in), bindings 1..k read-only storage inputs, binding k+1 the read_write output.
Shader libraries (e.g. schw_ray.wgsl) are combined with an entry shader by ``load_shader``.
"""

import os
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"
WORKGROUP = 8


@cache
def get_device() -> wgpu.GPUDevice:
    """The high-performance GPU (Intel Iris Plus via Metal on this Mac), created once."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No GPU adapter found")
    return adapter.request_device_sync()


def load_shader(*names: str) -> str:
    """WGSL source of one or more files in render/shaders, concatenated in order (WGSL has no
    #include: libraries come first, then the entry shader that calls them)."""
    return "\n".join((SHADER_DIR / name).read_text() for name in names)


def run_items(
    shader_code: str,
    uniform: bytes,
    inputs: list[np.ndarray],
    n_items: int,
    out_item_bytes: int,
    workgroup: int = 64,
    extra: list[dict] | None = None,
) -> bytes:
    """Run a compute shader once per item (see the module docstring for the binding layout)
    and return the raw output buffer (n_items * out_item_bytes bytes).

    For small jobs (tests, lookup tables); images go through ``run_tiled``-style tiling.
    """
    device = get_device()
    pipeline = get_pipeline(shader_code)
    usage = wgpu.BufferUsage
    uni = device.create_buffer_with_data(
        data=uniform + bytes(-len(uniform) % 16), usage=usage.UNIFORM
    )
    entries = [{"binding": 0, "resource": {"buffer": uni}}]
    for k, arr in enumerate(inputs, start=1):
        buf = device.create_buffer_with_data(
            data=np.ascontiguousarray(arr).tobytes(), usage=usage.STORAGE
        )
        entries.append({"binding": k, "resource": {"buffer": buf}})
    out = device.create_buffer(size=n_items * out_item_bytes, usage=usage.STORAGE | usage.COPY_SRC)
    entries.append({"binding": len(inputs) + 1, "resource": {"buffer": out}})
    entries.extend(extra or [])  # e.g. a texture view and sampler at the next bindings
    bind_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind_group)
    cpass.dispatch_workgroups(-(-n_items // workgroup), 1, 1)
    cpass.end()
    device.queue.submit([encoder.finish()])
    return bytes(device.queue.read_buffer(out))


@dataclass
class TileTimings:
    """Wall time per tile (dispatch + readback), in seconds."""

    per_tile: list[float] = field(default_factory=list)

    @property
    def max(self) -> float:
        return max(self.per_tile)

    @property
    def total(self) -> float:
        return sum(self.per_tile)


_PIPELINES: dict[str, wgpu.GPUComputePipeline] = {}


def get_pipeline(shader_code: str) -> wgpu.GPUComputePipeline:
    """Compiled compute pipeline for this WGSL source, cached (compiling takes ~0.3 s)."""
    if shader_code not in _PIPELINES:
        device = get_device()
        module = device.create_shader_module(code=shader_code)
        _PIPELINES[shader_code] = device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "main"}
        )
    return _PIPELINES[shader_code]


def uniform_buffer(data: bytes) -> wgpu.GPUBuffer:
    """A uniform buffer holding ``data`` (padded to a multiple of 16 bytes)."""
    return get_device().create_buffer_with_data(
        data=data + bytes(-len(data) % 16), usage=wgpu.BufferUsage.UNIFORM
    )


# Adaptive tiles (AI_LOG 2026-11-19: fixed 32-px tiles of heavy 1080p x 4-ray frames ran for
# seconds and the Mac's WindowServer watchdog killed the screen three times). Each tile is
# timed and the next one sized so it takes about TILE_TARGET_S. NEBULA_TILE_TARGET overrides it
# (a headless cloud GPU has no screen to starve: larger tiles, less overhead).
TILE_TARGET_S = float(os.environ.get("NEBULA_TILE_TARGET", "0.25"))
MIN_TILE = 8


def run_tiled(
    shader_code: str,
    width: int,
    height: int,
    tile: int = 256,
    extra: list[dict] | None = None,
    vec4_per_pixel: int = 1,
    target_s: float | None = None,
) -> tuple[np.ndarray, TileTimings]:
    """Run a compute shader over a width x height image in tiles; return (H, W, 4 k) float32
    with k = ``vec4_per_pixel`` (the shader writes out[k idx + j], j < k).

    ``tile`` is the largest tile side. Tiles are sized from the measured time per pixel of the
    previous tile so each takes about ``target_s`` (default TILE_TARGET_S): rows of tiles,
    the row height and the tile widths adapting (MIN_TILE .. tile). Pixels do not depend on
    the tiling (test_gpu.py).

    ``extra``: further bind-group entries (binding >= 2), e.g. a camera uniform buffer or a
    texture view and sampler, shared by all tiles.
    """
    k = int(vec4_per_pixel)
    target = TILE_TARGET_S if target_s is None else float(target_s)
    max_area = tile * tile
    device = get_device()
    pipeline = get_pipeline(shader_code)
    out_buf = device.create_buffer(
        size=max_area * 16 * k, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    uni_buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": uni_buf}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            *(extra or []),
        ],
    )
    image = np.empty((height, width, 4 * k), dtype=np.float32)
    timings = TileTimings()
    area = float(min(max_area, 32 * 32))  # start small (unknown cost), then adapt
    y0 = 0
    while y0 < height:
        side = int(np.clip(np.sqrt(area), MIN_TILE, tile))
        th = min(side, height - y0)
        x0 = 0
        while x0 < width:
            tw = min(int(np.clip(area / th, MIN_TILE, max_area // th)), width - x0)
            t0 = time.perf_counter()
            params = np.array([x0, y0, width, height, tw, th, 0, 0], dtype=np.uint32)
            device.queue.write_buffer(uni_buf, 0, params.tobytes())
            encoder = device.create_command_encoder()
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(pipeline)
            cpass.set_bind_group(0, bind_group)
            cpass.dispatch_workgroups(-(-tw // WORKGROUP), -(-th // WORKGROUP), 1)
            cpass.end()
            device.queue.submit([encoder.finish()])
            # read_buffer waits for the GPU, so the timing covers the whole tile.
            data = device.queue.read_buffer(out_buf, 0, tw * th * 16 * k)
            dt = time.perf_counter() - t0
            timings.per_tile.append(dt)
            image[y0 : y0 + th, x0 : x0 + tw] = np.frombuffer(data, np.float32).reshape(
                th, tw, 4 * k
            )
            # next tile: the area that would take `target` at this tile's rate (grow at most 2x)
            want = target / max(dt, 1e-6) * (tw * th)
            area = float(np.clip(want, MIN_TILE * MIN_TILE, min(max_area, 2.0 * tw * th)))
            x0 += tw
        y0 += th
    return image, timings
