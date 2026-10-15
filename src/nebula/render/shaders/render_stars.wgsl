// Entry shader: the star-field cube map seen through the hole's gravity (Phase 2, Step 2.4).
// Load order: schw_ray.wgsl, camera_rays.wgsl, render_stars.wgsl. cam.lens.z = scale that
// turns stored values back into star-flux radiance (starfield.CubeMap.scale).
//
// Mip level from a simple ray differential (phase-02 Step 2.4): the angular spread of the
// escape directions of a pixel's sub-rays, divided by n_sub, is the sky patch each sub-ray
// stands for; sampling that patch's average stops stars sparkling where lensing magnifies.

@group(0) @binding(3) var sky: texture_cube<f32>;
@group(0) @binding(4) var sky_sampler: sampler;

const MAX_SUB: u32 = 16u;          // n_sub <= 4
const TEXEL_ANGLE_FACE0: f32 = 0.000766990;  // (pi/2) / 2048 rad, level-0 texel at face centre

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tile.size.x || gid.y >= tile.size.y) {
        return;
    }
    let pix = tile.origin + gid.xy;
    let idx = gid.y * tile.size.x + gid.x;
    let mode = cam.dims.w;
    let n = min(cam.dims.z, 4u);
    let px = f32(pix.x);
    let py = f32(pix.y);
    let face_size = f32(textureDimensions(sky).x);
    let texel_angle = TEXEL_ANGLE_FACE0 * 2048.0 / face_size;

    var dirs: array<vec3<f32>, MAX_SUB>;
    var n_esc = 0u;
    var mean = vec3<f32>(0.0);
    var captured = 0.0;
    for (var b = 0u; b < n; b++) {
        for (var a = 0u; a < n; a++) {
            let d = pixel_dir(px + (f32(a) + 0.5) / f32(n), py + (f32(b) + 0.5) / f32(n));
            var hit: SkyHit;
            if (mode == MODE_HOLE_STARS) {
                hit = bend(d);
            } else {
                hit.escaped = true;
                hit.dir = d;
            }
            if (hit.escaped) {
                dirs[n_esc] = hit.dir;
                mean += hit.dir;
                n_esc += 1u;
            } else {
                captured += 1.0;
            }
        }
    }
    var rgb = vec3<f32>(0.0);
    if (n_esc > 0u) {
        let m = normalize(mean);
        var spread = 0.0;
        for (var k = 0u; k < n_esc; k++) {
            spread = max(spread, acos(clamp(dot(dirs[k], m), -1.0, 1.0)));
        }
        // Each sub-ray stands for a patch ~ (full spread across the pixel) / n_sub.
        let footprint = 2.0 * spread / f32(n);
        let lod = log2(max(footprint / texel_angle, 1.0));
        for (var k = 0u; k < n_esc; k++) {
            rgb += textureSampleLevel(sky, sky_sampler, dirs[k], lod).rgb;
        }
    }
    let total = f32(n * n);
    out[idx] = vec4<f32>(rgb * cam.lens.z / total, captured / total);
}
