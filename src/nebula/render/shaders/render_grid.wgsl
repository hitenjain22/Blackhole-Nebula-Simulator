// Entry shader: synthetic sky grids for camera tests (Phase 2, Step 2.3).
// Load order: schw_ray.wgsl, camera_rays.wgsl, render_grid.wgsl.

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
