// Camera rays for image shaders (Phase 2). Library; needs schw_ray.wgsl first, and is
// followed by an entry shader (render_grid.wgsl, render_stars.wgsl). Tiled dispatch:
// nebula.render.device.run_tiled; bindings 0-2 here, entry shaders may add more.
//
// Per pixel the entry shaders use n_sub x n_sub sub-rays on a fixed stratified pattern
// (mirrors camera.sub_pixel_offsets). pixel_dir mirrors camera.Camera.directions; bend
// turns a camera direction into the sky direction it sees (trace_ray). Entry shaders output
// (mean linear RGB, fraction of sub-rays captured by the hole).

struct Tile { origin: vec2<u32>, image: vec2<u32>, size: vec2<u32>, _pad: vec2<u32> };
struct Camera {
    pos: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    lens: vec4<f32>,   // tan(vfov/2), aspect, sky parameters a, b
    dims: vec4<u32>,   // width, height, n_sub, mode
};
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> cam: Camera;

const MODE_FLAT_GNOMONIC_GRID: u32 = 0u;
const MODE_HOLE_LATLONG_GRID: u32 = 1u;
const MODE_FLAT_LATLONG_GRID: u32 = 2u;
const MODE_HOLE_STARS: u32 = 3u;
const MODE_FLAT_STARS: u32 = 4u;
const MODE_DEBUG_DIRECTION: u32 = 10u;
const MODE_DEBUG_GNOMONIC: u32 = 11u;

fn pixel_dir(px: f32, py: f32) -> vec3<f32> {
    let x = (2.0 * px / f32(cam.dims.x) - 1.0) * cam.lens.x * cam.lens.y;
    let y = (1.0 - 2.0 * py / f32(cam.dims.y)) * cam.lens.x;
    return normalize(cam.fwd.xyz + x * cam.right.xyz + y * cam.up.xyz);
}

struct SkyHit { escaped: bool, dir: vec3<f32> };

// Bend a ray leaving the camera in direction d around the hole (camera at cam.pos).
// Ray plane basis: e1 = outward radial at the camera, e2 = part of d perpendicular to e1.
fn bend(d: vec3<f32>) -> SkyHit {
    let r_cam = length(cam.pos.xyz);
    let e1 = cam.pos.xyz / r_cam;
    let cos_psi = -dot(d, e1);             // angle from the direction toward the hole
    let perp = d + cos_psi * e1;           // d - (d.e1) e1
    let sin_psi = length(perp);
    var hit: SkyHit;
    if (sin_psi < 1e-7) {                  // exactly radial
        hit.escaped = cos_psi < 0.0;
        hit.dir = e1;
        return hit;
    }
    let e2 = perp / sin_psi;
    let u0 = 1.0 / r_cam;
    let res = trace_ray(u0, ray_initial_slope(u0, cos_psi, sin_psi));
    hit.escaped = res.status == RAY_ESCAPED;
    hit.dir = cos(res.phi_inf) * e1 + sin(res.phi_inf) * e2;
    return hit;
}

