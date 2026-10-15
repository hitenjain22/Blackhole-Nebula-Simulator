// Image of the sky as seen by a static camera near the hole (Phase 2, Steps 2.3-2.5).
// Needs schw_ray.wgsl first. Tiled dispatch: nebula.render.device.run_tiled.
//
// Per pixel: n_sub x n_sub sub-rays on a fixed stratified pattern (mirrors
// camera.sub_pixel_offsets). Each sub-ray: direction from the pinhole camera (pixel_dir
// mirrors camera.Camera.directions), bent around the hole by trace_ray, sky looked up in the
// escape direction. Output: (mean linear RGB, fraction of sub-rays captured by the hole).

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

// Grid of straight lines on the sky: lines where s.x/s.z or s.y/s.z is a multiple of
// spacing a (half width b, same tangent units). Straight in any pinhole image looking along z.
fn sky_gnomonic_grid(s: vec3<f32>) -> vec3<f32> {
    if (s.z <= 0.0) {
        return vec3<f32>(0.02, 0.02, 0.03);
    }
    let g = s.xy / s.z / cam.lens.z;
    let dist = abs(g - round(g)) * cam.lens.z;
    if (min(dist.x, dist.y) < cam.lens.w) {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    return vec3<f32>(0.05, 0.08, 0.20);
}

// Latitude-longitude grid (z up), every a degrees, line half width b degrees; the equator
// and the longitude-zero meridian (+x) are coloured so orientation is visible.
fn sky_latlong_grid(s: vec3<f32>) -> vec3<f32> {
    let deg = 57.29578;
    let lat = asin(clamp(s.z, -1.0, 1.0)) * deg;
    let lon = atan2(s.y, s.x) * deg;
    let a = cam.lens.z;
    let w = cam.lens.w;
    let dlat = abs(lat - a * round(lat / a));
    let dlon = abs(lon - a * round(lon / a)) * cos(lat / deg);
    if (abs(lat) < w) {
        return vec3<f32>(1.0, 0.35, 0.1);
    }
    if (abs(lon) < w / max(cos(lat / deg), 0.05)) {
        return vec3<f32>(0.2, 1.0, 0.3);
    }
    if (min(dlat, dlon) < w) {
        return vec3<f32>(0.9, 0.9, 0.9);
    }
    // Faint colour by hemisphere so the lensed image shows which way is up.
    return select(vec3<f32>(0.05, 0.05, 0.12), vec3<f32>(0.10, 0.06, 0.04), s.z > 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tile.size.x || gid.y >= tile.size.y) {
        return;
    }
    let pix = tile.origin + gid.xy;
    let idx = gid.y * tile.size.x + gid.x;
    let mode = cam.dims.w;
    let px = f32(pix.x);
    let py = f32(pix.y);

    if (mode == MODE_DEBUG_DIRECTION) {
        out[idx] = vec4<f32>(pixel_dir(px + 0.5, py + 0.5), 0.0);
        return;
    }
    if (mode == MODE_DEBUG_GNOMONIC) {
        let d = pixel_dir(px + 0.5, py + 0.5);
        out[idx] = vec4<f32>(d.x / d.z, d.y / d.z, d.z, 0.0);
        return;
    }

    let n = cam.dims.z;
    var rgb = vec3<f32>(0.0);
    var captured = 0.0;
    for (var b = 0u; b < n; b++) {
        for (var a = 0u; a < n; a++) {
            let d = pixel_dir(px + (f32(a) + 0.5) / f32(n), py + (f32(b) + 0.5) / f32(n));
            var hit: SkyHit;
            if (mode == MODE_HOLE_LATLONG_GRID) {
                hit = bend(d);
            } else {
                hit.escaped = true;
                hit.dir = d;
            }
            if (!hit.escaped) {
                captured += 1.0;
            } else if (mode == MODE_FLAT_GNOMONIC_GRID) {
                rgb += sky_gnomonic_grid(hit.dir);
            } else {
                rgb += sky_latlong_grid(hit.dir);
            }
        }
    }
    let m = f32(n * n);
    out[idx] = vec4<f32>(rgb / m, captured / m);
}
