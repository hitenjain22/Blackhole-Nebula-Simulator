// Entry shader: escape angles for a list of rays (tests, lookup). Needs schw_ray.wgsl first.
// Input per ray: (u0 = 1/D, cos psi, sin psi, unused). Output: (phi_inf, status, half_turns,
// steps) as f32.

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let ray = rays[i];
    var res: RayResult;
    if (ray.z < 1e-7) {
        // Exactly radial: inward is captured, outward sees e1 (phi_inf = 0).
        res.phi_inf = 0.0;
        res.steps = 0u;
        res.half_turns = 0u;
        res.status = select(RAY_ESCAPED, RAY_CAPTURED, ray.y > 0.0);
    } else {
        res = trace_ray(ray.x, ray_initial_slope(ray.x, ray.y, ray.z));
    }
    out[i] = vec4<f32>(res.phi_inf, f32(res.status), f32(res.half_turns), f32(res.steps));
}
