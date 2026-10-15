// Entry shader: sample a cube map in given directions (tests, sky previews).
// Input per item: (direction xyz, mip level). Output: sampled RGBA.

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dirs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var sky: texture_cube<f32>;
@group(0) @binding(4) var sky_sampler: sampler;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let d = dirs[i];
    out[i] = textureSampleLevel(sky, sky_sampler, d.xyz, d.w);
}
