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


def run_tiled(
    shader_code: str,
    width: int,
    height: int,
    tile: int = 256,
    extra: list[dict] | None = None,
) -> tuple[np.ndarray, TileTimings]:
    """Run a compute shader over a width x height image in tiles; return (H, W, 4) float32.

    ``extra``: further bind-group entries (binding >= 2), e.g. a camera uniform buffer or a
    texture view and sampler, shared by all tiles.
    """
    device = get_device()
    pipeline = get_pipeline(shader_code)
    out_buf = device.create_buffer(
        size=tile * tile * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
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
    image = np.empty((height, width, 4), dtype=np.float32)
    timings = TileTimings()
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            tw, th = min(tile, width - x0), min(tile, height - y0)
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
            data = device.queue.read_buffer(out_buf, 0, tw * th * 16)
            timings.per_tile.append(time.perf_counter() - t0)
            image[y0 : y0 + th, x0 : x0 + tw] = np.frombuffer(data, np.float32).reshape(th, tw, 4)
    return image, timings
