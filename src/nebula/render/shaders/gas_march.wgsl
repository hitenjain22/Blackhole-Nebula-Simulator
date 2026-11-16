// Gas along bent light rays (Phase 8, phases/phase-08-gas-rendering.md 8.4). Entry shader.
// Load order: schw_ray.wgsl, camera_rays.wgsl, gas_march.wgsl (bindings 0-2 come from
// camera_rays.wgsl; 3-6 gas; 7-8 the star cube map). Python side: nebula.render.gpu_march.
//
// Float32 twin of the CPU reference render/bent_rays.py + render/march.py (rule 2; 8.6 test
// 7 compares them). Formula copies, each mirroring a physics/ function:
//   transfer_step    -> physics.radiative_transfer.transfer_step (expm1 by series, below)
//   source / alpha   -> physics.radiative_transfer.blackbody_intensity_si, grey_absorption
//                       via render.march.GreyBlackbody: S = (sigma T_REF^4 / pi) (rho T^4 /
//                       rho) with rho T^4 stored as rho (T / T_REF)^4; alpha = kappa rho
//   proper_length    -> physics.geodesics.schw_static_proper_length
// Ray geometry: RK4 in phi from schw_ray.wgsl, exactly as bent_rays.ray_polyline.
//
// Textures are rg32float (r = rho, g = rho (T/T_REF)^4) sampled with 8 textureLoad calls
// (trilinear between cell centres in index space, as march._trilinear), not hardware filtering:
// hardware filter weights carry only ~8 bits (DECISIONS 2026-11-15). A texture holding array
// shape (n0, n1, n2) is width n2, height n1, depth n0: element [i, j, k] is at (k, j, i).

struct BoxParams {
    lo: vec4<f32>,      // lower corner xyz, cell size dx
    centre: vec4<f32>,  // centre xyz, half size
    n: vec4<u32>,       // cells per side, active (0/1), -, -
};
struct GasParams {
    grid: vec4<f32>,    // r_min, r_max, dln r, dtheta
    grid2: vec4<f32>,   // dphi, step fraction, T_STOP, blend cells
    model: vec4<f32>,   // kappa (code units), S scale = sigma T_REF^4 / pi (SI), -, -
    ngrid: vec4<u32>,   // nr, ntheta, nphi, -
    box0: BoxParams,
    box1: BoxParams,
};
@group(0) @binding(3) var log_tex: texture_3d<f32>;
@group(0) @binding(4) var box0_tex: texture_3d<f32>;
@group(0) @binding(5) var box1_tex: texture_3d<f32>;
@group(0) @binding(6) var<uniform> gas: GasParams;

const TWO_PI_F: f32 = 6.28318531;
const U_HORIZON: f32 = 0.5;
const RADIAL_EPS: f32 = 1e-7;

// Trilinear interpolation at fractional indices (cell centres at integers); i, j clamped,
// k wrapped if wrap_k (phi) else clamped. Mirrors render.march._trilinear.
fn trilinear(t: texture_3d<f32>, fi: f32, fj: f32, fk: f32, wrap_k: bool) -> vec2<f32> {
    let dims = vec3<i32>(textureDimensions(t));  // (n2, n1, n0)
    let i0 = i32(floor(fi));
    let j0 = i32(floor(fj));
    let k0 = i32(floor(fk));
    let ti = fi - f32(i0);
    let tj = fj - f32(j0);
    let tk = fk - f32(k0);
    var acc = vec2<f32>(0.0);
    for (var di = 0; di < 2; di++) {
        let ii = clamp(i0 + di, 0, dims.z - 1);
        let wi = select(1.0 - ti, ti, di == 1);
        for (var dj = 0; dj < 2; dj++) {
            let jj = clamp(j0 + dj, 0, dims.y - 1);
            let wj = select(1.0 - tj, tj, dj == 1);
            for (var dk = 0; dk < 2; dk++) {
                var kk = k0 + dk;
                if (wrap_k) {
                    kk = ((kk % dims.x) + dims.x) % dims.x;
                } else {
                    kk = clamp(kk, 0, dims.x - 1);
                }
                let wk = select(1.0 - tk, tk, dk == 1);
                acc += wi * wj * wk * textureLoad(t, vec3<i32>(kk, jj, ii), 0).rg;
            }
        }
    }
    return acc;
}

fn sample_log(p: vec3<f32>) -> vec2<f32> {
    let r = length(p);
    if (r < gas.grid.x || r > gas.grid.y) {
        return vec2<f32>(0.0);
    }
    let fi = log(r / gas.grid.x) / gas.grid.z - 0.5;
    let fj = acos(clamp(p.z / r, -1.0, 1.0)) / gas.grid.w - 0.5;
    var ph = atan2(p.y, p.x);
    if (ph < 0.0) {
        ph += TWO_PI_F;
    }
    let fk = ph / gas.grid2.x - 0.5;
    return trilinear(log_tex, fi, fj, fk, true);
}

// Box weight: 1 inside, falling linearly to 0 over BLEND_CELLS at the faces (march._box_weight).
fn box_weight(b: BoxParams, p: vec3<f32>) -> f32 {
    let inner = b.centre.w - abs(p - b.centre.xyz);
    return clamp(min(inner.x, min(inner.y, inner.z)) / (gas.grid2.w * b.lo.w), 0.0, 1.0);
}

fn box_inside(b: BoxParams, p: vec3<f32>) -> bool {
    return all(abs(p - b.centre.xyz) <= vec3<f32>(b.centre.w));
}

fn sample_box(t: texture_3d<f32>, b: BoxParams, p: vec3<f32>) -> vec2<f32> {
    let f = (p - b.lo.xyz) / b.lo.w - 0.5;
    return trilinear(t, f.x, f.y, f.z, false);
}

// Fields at p: the first active box covering p, blended into the log grid (march.sample).
fn sample_fields(p: vec3<f32>) -> vec2<f32> {
    let base = sample_log(p);
    if (gas.box0.n.y != 0u) {
        let w = box_weight(gas.box0, p);
        if (w > 0.0) {
            return w * sample_box(box0_tex, gas.box0, p) + (1.0 - w) * base;
        }
    }
    if (gas.box1.n.y != 0u) {
        let w = box_weight(gas.box1, p);
        if (w > 0.0) {
            return w * sample_box(box1_tex, gas.box1, p) + (1.0 - w) * base;
        }
    }
    return base;
}

// Local cell size of the finest grid at p (march.cell_size).
fn cell_size(p: vec3<f32>) -> f32 {
    if (gas.box0.n.y != 0u && box_inside(gas.box0, p)) {
        return gas.box0.lo.w;
    }
    if (gas.box1.n.y != 0u && box_inside(gas.box1, p)) {
        return gas.box1.lo.w;
    }
    return max(length(p), gas.grid.x) * min(gas.grid.z, gas.grid.w);
}

// 1 - e^-x without float32 cancellation for small x (series to x^4: relative error < 1e-8
// for x < 0.03). Mirrors the -expm1(-dtau) of physics.radiative_transfer.transfer_step.
fn one_minus_exp(x: f32) -> f32 {
    if (x < 0.03) {
        return x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0))));
    }
    return 1.0 - exp(-x);
}

// (1 - e^-x) / x, finite for every x >= 0 (series below 0.03, as one_minus_exp).
fn one_minus_exp_over_x(x: f32) -> f32 {
    if (x < 0.03) {
        return 1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0)));
    }
    return (1.0 - exp(-x)) / x;
}

// Mirrors physics.geodesics.schw_static_proper_length.
fn proper_length(dl: f32, dr: f32, r: f32) -> f32 {
    return sqrt(dl * dl + dr * dr * 2.0 / (r - 2.0));
}

struct MarchState { intensity: f32, transmission: f32, steps: u32 };

// March the chord a -> b front to back (march.march_segment with curved = True).
fn march_chord(a: vec3<f32>, b: vec3<f32>, st_in: MarchState) -> MarchState {
    var st = st_in;
    let seg = b - a;
    let len = length(seg);
    if (len == 0.0) {
        return st;
    }
    let e = seg / len;
    let frac = gas.grid2.y;
    var s = 0.0;
    loop {
        if (s >= len || st.transmission < gas.grid2.z) {
            break;
        }
        let p = a + s * e;
        var ds = min(frac * cell_size(p), len - s);
        ds = min(ds, frac * cell_size(p + ds * e));
        let mid = p + 0.5 * ds * e;
        let f = sample_fields(mid);
        let alpha = gas.model.x * f.x;
        let r_mid = length(mid);
        // No gas: nothing changes; also avoids 0 x infinity at r_mid = 2 exactly (march.py).
        if (alpha > 0.0 && r_mid > 2.0) {
            let q = p + ds * e;
            let path = proper_length(ds, length(q) - length(p), r_mid);
            let dtau = alpha * path;
            // S (1 - e^-dtau) with S = s_scale rho_t4 / rho and dtau = kappa rho path, written
            // as s_scale rho_t4 kappa path (1 - e^-dtau) / dtau: no division by rho, which in
            // float32 can be a rounding residue at the thin edge of the gas (gave inf pixels).
            let src_dtau = gas.model.y * f.y * gas.model.x * path;
            st.intensity += st.transmission * src_dtau * one_minus_exp_over_x(dtau);
            st.transmission *= exp(-dtau);
        }
        s += ds;
        st.steps += 1u;
    }
    return st;
}

struct RayOut {
    intensity: f32,
    transmission: f32,
    status: u32,
    steps: u32,
    sky_dir: vec3<f32>,  // direction on the sky seen by an escaped ray (Phase 2 bend)
};

// Continue an outgoing ray from state (u, w) at angle phi to infinity: the escape angle
// phi_inf. Same RK4 steps and Hermite root as trace_ray (schw_ray.wgsl), so a ray gets the
// same sky direction as in the Phase 2 star renderer (tested: test_gpu_march).
fn escape_angle(u_in: f32, w_in: f32, phi_in: f32) -> f32 {
    var u = u_in;
    var w = w_in;
    var phi = phi_in;
    loop {
        if (phi >= PHI_CAP) {
            break;
        }
        let h = ray_step_size(u);
        let k1u = w;
        let k1w = ray_accel(u);
        let k2u = w + 0.5 * h * k1w;
        let k2w = ray_accel(u + 0.5 * h * k1u);
        let k3u = w + 0.5 * h * k2w;
        let k3w = ray_accel(u + 0.5 * h * k2u);
        let k4u = w + h * k3w;
        let k4w = ray_accel(u + h * k3u);
        let u1 = u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let w1 = w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        if (u1 < 0.0) {
            return phi + ray_hermite_root(u, w, u1, w1, h) * h;
        }
        u = u1;
        w = w1;
        phi += h;
    }
    return phi;
}

fn ray_point(r: f32, ph: f32, e1: vec3<f32>, e2: vec3<f32>) -> vec3<f32> {
    return r * (cos(ph) * e1 + sin(ph) * e2);
}

// One bent ray from the camera in direction d (bent_rays.march_bent_ray). The geometry keeps
// going after the gas turns opaque so the escape/capture status is always known.
fn march_ray(d: vec3<f32>) -> RayOut {
    let cam_pos = cam.pos.xyz;
    let r_cam = length(cam_pos);
    let e1 = cam_pos / r_cam;
    let cos_psi = -dot(d, e1);
    let perp = d + cos_psi * e1;
    let sin_psi = length(perp);
    var st: MarchState;
    st.intensity = 0.0;
    st.transmission = 1.0;
    st.steps = 0u;
    var res: RayOut;
    let r_max = gas.grid.y;
    if (sin_psi < RADIAL_EPS) {
        if (cos_psi > 0.0) {
            st = march_chord(cam_pos, 2.0 * e1, st);
            res.status = RAY_CAPTURED;
        } else {
            st = march_chord(cam_pos, max(r_max, r_cam) * e1, st);
            res.status = RAY_ESCAPED;
            res.sky_dir = e1;
        }
        res.intensity = st.intensity;
        res.transmission = st.transmission;
        res.steps = st.steps;
        return res;
    }
    let e2 = perp / sin_psi;
    let u_out = 1.0 / r_max;
    var u = 1.0 / r_cam;
    var w = ray_initial_slope(u, cos_psi, sin_psi);
    var phi = 0.0;
    var prev = cam_pos;
    res.status = RAY_ORBITING;
    loop {
        if (phi >= PHI_CAP) {
            break;
        }
        let h = ray_step_size(u);
        let k1u = w;
        let k1w = ray_accel(u);
        let k2u = w + 0.5 * h * k1w;
        let k2w = ray_accel(u + 0.5 * h * k1u);
        let k3u = w + 0.5 * h * k2w;
        let k3w = ray_accel(u + 0.5 * h * k2u);
        let k4u = w + h * k3w;
        let k4w = ray_accel(u + h * k3u);
        let u1 = u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        let w1 = w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        let marching = st.transmission >= gas.grid2.z;
        if (u1 >= U_HORIZON) {
            let t = (U_HORIZON - u) / (u1 - u);
            if (marching) {
                st = march_chord(prev, ray_point(2.0, phi + t * h, e1, e2), st);
            }
            res.status = RAY_CAPTURED;
            break;
        }
        if (u1 < u_out && u1 < u) {
            let t = clamp((u - u_out) / (u - u1), 0.0, 1.0);
            if (marching && t > 0.0) {
                st = march_chord(prev, ray_point(r_max, phi + t * h, e1, e2), st);
            }
            res.status = RAY_ESCAPED;
            let phi_inf = escape_angle(u, w, phi);
            res.sky_dir = cos(phi_inf) * e1 + sin(phi_inf) * e2;
            break;
        }
        phi += h;
        u = u1;
        w = w1;
        let next = ray_point(1.0 / u, phi, e1, e2);
        if (marching) {
            st = march_chord(prev, next, st);
        }
        prev = next;
    }
    res.intensity = st.intensity;
    res.transmission = st.transmission;
    res.steps = st.steps;
    return res;
}

// Output modes (cam.dims.w):
//   OUT_DIAGNOSTIC: (mean I, mean T of escaped sub-rays (0 for captured), captured fraction,
//                   mean sub-steps) (bent_rays.render_reference; tests 6-7)
//   OUT_PICTURE:    (mean gas I, mean of T x star radiance RGB): the stars behind the gas,
//                   dimmed by it. Star sampling as render_stars.wgsl (cam.lens.z = cube scale;
//                   mip level from the spread of the pixel's escaped sub-ray directions).
//   OUT_SKY_DIR:    (sky direction xyz, 1 if escaped else 0) of the last sub-ray (with n_sub = 1:
//                   the pixel-centre ray), to test escape directions against physics.geodesics
const OUT_DIAGNOSTIC: u32 = 0u;
const OUT_PICTURE: u32 = 1u;
const OUT_SKY_DIR: u32 = 2u;
const MAX_SUB: u32 = 16u;  // n_sub <= 4
const TEXEL_ANGLE_FACE0: f32 = 0.000766990;  // as render_stars.wgsl

@group(0) @binding(7) var sky: texture_cube<f32>;
@group(0) @binding(8) var sky_sampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tile.size.x || gid.y >= tile.size.y) {
        return;
    }
    let pix = tile.origin + gid.xy;
    let idx = gid.y * tile.size.x + gid.x;
    let n = min(cam.dims.z, 4u);
    let total = f32(n * n);
    var acc = vec4<f32>(0.0);
    var dirs: array<vec3<f32>, MAX_SUB>;
    var trans: array<f32, MAX_SUB>;
    var n_esc = 0u;
    var mean = vec3<f32>(0.0);
    var last = vec4<f32>(0.0);
    for (var b = 0u; b < n; b++) {
        for (var a = 0u; a < n; a++) {
            let d = pixel_dir(f32(pix.x) + (f32(a) + 0.5) / f32(n),
                              f32(pix.y) + (f32(b) + 0.5) / f32(n));
            let r = march_ray(d);
            let escaped = r.status == RAY_ESCAPED;
            acc += vec4<f32>(r.intensity, select(0.0, r.transmission, escaped),
                             select(1.0, 0.0, escaped), f32(r.steps));
            last = vec4<f32>(r.sky_dir, select(0.0, 1.0, escaped));
            if (escaped) {
                dirs[n_esc] = r.sky_dir;
                trans[n_esc] = r.transmission;
                mean += r.sky_dir;
                n_esc += 1u;
            }
        }
    }
    if (cam.dims.w == OUT_DIAGNOSTIC) {
        out[idx] = acc / total;
        return;
    }
    if (cam.dims.w == OUT_SKY_DIR) {
        out[idx] = last;
        return;
    }
    var rgb = vec3<f32>(0.0);
    if (n_esc > 0u) {
        let m = normalize(mean);
        var spread = 0.0;
        for (var k = 0u; k < n_esc; k++) {
            spread = max(spread, acos(clamp(dot(dirs[k], m), -1.0, 1.0)));
        }
        let face_size = f32(textureDimensions(sky).x);
        let texel_angle = TEXEL_ANGLE_FACE0 * 2048.0 / face_size;
        let footprint = 2.0 * spread / f32(n);
        let lod = log2(max(footprint / texel_angle, 1.0));
        for (var k = 0u; k < n_esc; k++) {
            rgb += trans[k] * textureSampleLevel(sky, sky_sampler, dirs[k], lod).rgb;
        }
    }
    out[idx] = vec4<f32>(acc.x / total, rgb * cam.lens.z / total);
}
