// GPU check pattern (Phase 0, Step 0.4). Not physics.
// Mirrors nebula.render.gpu_pattern.gpu_pattern_reference; tested in tests/test_gpu.py.
//
// Per pixel: logistic map x <- r x (1 - x) with r in [2.5, 2.95] and x0 in [0.05, 0.95].
// That range has a stable fixed point (|f'| = |2 - r| < 1), so float32 rounding is damped,
// not amplified, and the GPU result is checkable against a float32 CPU reference.
// The mean is summed in blocks: a plain running float32 sum of 2,000 terms carries ~3e-5
// rounding error on both GPU and CPU, which would hide real GPU differences.

struct Tile { origin: vec2<u32>, image: vec2<u32>, size: vec2<u32>, _pad: vec2<u32> };
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

const N_BLOCKS: u32 = 50u;
const BLOCK: u32 = 40u;  // N_BLOCKS * BLOCK = 2,000 iterations
const N_EARLY: u32 = 20u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tile.size.x || gid.y >= tile.size.y) {
        return;
    }
    let px = tile.origin + gid.xy;
    let u = (f32(px.x) + 0.5) / f32(tile.image.x);
    let v = (f32(px.y) + 0.5) / f32(tile.image.y);
    let r = 2.5 + 0.45 * u;
    var x = 0.05 + 0.9 * v;
    var early = 0.0;
    var total = 0.0;
    for (var b = 0u; b < N_BLOCKS; b++) {
        var block_sum = 0.0;
        for (var k = 0u; k < BLOCK; k++) {
            x = r * x * (1.0 - x);
            block_sum += x;
            if (b * BLOCK + k == N_EARLY - 1u) {
                early = x;
            }
        }
        total += block_sum;
    }
    out[gid.y * tile.size.x + gid.x] = vec4<f32>(x, early, total / f32(N_BLOCKS * BLOCK), 1.0);
}
