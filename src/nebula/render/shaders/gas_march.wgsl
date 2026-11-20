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
// Phase 9 colour mode (render.march.RelativisticBlackbody). The march step holds a
// "// #if COLOUR / // #else / // #endif" block: gpu_march.shader_source keeps one branch, so
// the grey and colour shaders are compiled separately (one shader with both branches ran
// ~250x slower on the Iris Plus, even with the colour branch never taken: AI_LOG 2026-11-17).
//   static_dir       -> physics.redshift.static_frame_direction
//   g, path factor   -> physics.redshift.redshift_factor, gas_frame_path_factor
//   lut_chroma       -> physics.colour.lut_chroma (log-log table, clamped at its ends)
//   emission         -> physics.colour.blackbody_xyz_lut at g T_em, bolometric_intensity_si
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
    bricks: vec4<f32>,  // cell size dx, -, active (1/0), -
    brick_o: vec4<i32>, // index origin (brick coordinates), -
    brick_n: vec4<u32>, // index dimensions (bricks), -
    brick_a: vec4<u32>, // atlas slots per axis (A0, A1, A2), -
    brick_lo: vec4<f32>, // box holding the bricks' home regions (BrickTexture.box_lo/hi)
    brick_hi: vec4<f32>,
    colour: vec4<f32>,  // 1 = Phase 9 colour, LUT ln T_0, LUT d ln T, LUT entries
    colour2: vec4<f32>, // q_ff (0: constant kappa), max |gamma beta| (march.GB_MAX), min rho (RHO_MIN_COLOUR), c_es
};
@group(0) @binding(3) var log_tex: texture_3d<f32>;
@group(0) @binding(4) var box0_tex: texture_3d<f32>;
@group(0) @binding(5) var box1_tex: texture_3d<f32>;
@group(0) @binding(6) var<uniform> gas: GasParams;
// Sparse bricks (render/bricks.py, gpu_march.brick_arrays): slot + 1 per brick (0 = empty)
// and the atlas of 9^3-texel bricks (8^3 cells + one-cell apron on the high side).
@group(0) @binding(9) var brick_index: texture_3d<u32>;
@group(0) @binding(10) var brick_atlas: texture_3d<f32>;
// Phase 9: rho x gamma beta (rgba32float, a unused) of each grid, and the colour table
// (ln chroma XYZ per entry, physics.colour.chroma_lut). Dummies when not in colour mode.
@group(0) @binding(11) var log_vel: texture_3d<f32>;
@group(0) @binding(12) var box0_vel: texture_3d<f32>;
@group(0) @binding(13) var box1_vel: texture_3d<f32>;
@group(0) @binding(14) var brick_vel: texture_3d<f32>;
@group(0) @binding(15) var chroma_lut: texture_2d<f32>;  // (entries x 1) rgba32float
const BRICK_I: i32 = 8;
const APRON_I: i32 = 9;
const EXIT_EPS: f32 = 1e-4;  // bricks.EXIT_EPS (in cells)
const MAX_MARCH_STEPS: u32 = 400000u;  // per ray (normal rays take < 10^4; a hang guard)

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

// Home brick of p: the brick of cell floor(p / dx - 1/2) (bricks.home_brick). WGSL >> on i32
// is an arithmetic shift, i.e. floor division by 8 for negative indices too.
fn brick_home(p: vec3<f32>) -> vec3<i32> {
    let c = vec3<i32>(floor(p / gas.bricks.x - 0.5));
    return c >> vec3<u32>(3u);
}

// Slot + 1 of brick b, 0 if absent (bricks.home_allocated).
fn brick_slot(b: vec3<i32>) -> u32 {
    let q = b - gas.brick_o.xyz;
    let n = vec3<i32>(gas.brick_n.xyz);
    if (any(q < vec3<i32>(0)) || any(q >= n)) {
        return 0u;
    }
    return textureLoad(brick_index, vec3<i32>(q.z, q.y, q.x), 0).r;
}

fn brick_allocated(p: vec3<f32>) -> bool {
    return gas.bricks.z > 0.5 && brick_slot(brick_home(p)) != 0u;
}

// Trilinear between lattice cell centres (bricks.sample_bricks), all 8 corners read from the
// home brick's atlas block (the apron holds the neighbours' first cells). A point whose home
// brick is empty samples 0, which is exact: bricks are allocated so that a non-zero sample
// always has an allocated home brick.
fn sample_bricks(p: vec3<f32>) -> vec2<f32> {
    if (gas.bricks.z < 0.5) {
        return vec2<f32>(0.0);
    }
    let f = p / gas.bricks.x - 0.5;
    let fl = floor(f);
    let c = vec3<i32>(fl);
    let home = c >> vec3<u32>(3u);
    let slot = brick_slot(home);
    if (slot == 0u) {
        return vec2<f32>(0.0);
    }
    let s = slot - 1u;
    let a = gas.brick_a;
    let blk = vec3<i32>(i32(s / (a.y * a.z)), i32((s / a.z) % a.y), i32(s % a.z));
    let base = APRON_I * blk + (c - BRICK_I * home);
    let t = f - fl;
    var acc = vec2<f32>(0.0);
    for (var di = 0; di < 2; di++) {
        let wi = select(1.0 - t.x, t.x, di == 1);
        for (var dj = 0; dj < 2; dj++) {
            let wj = select(1.0 - t.y, t.y, dj == 1);
            for (var dk = 0; dk < 2; dk++) {
                let wk = select(1.0 - t.z, t.z, dk == 1);
                let q = base + vec3<i32>(di, dj, dk);
                acc += wi * wj * wk * textureLoad(brick_atlas, vec3<i32>(q.z, q.y, q.x), 0).rg;
            }
        }
    }
    return acc;
}

// Step limit from the bricks (march.brick_step_limit): inside the box holding the bricks, from
// a point whose home brick is empty, step only to just past that brick's far face
// (bricks.exit_distance); from outside the box, only to just inside its face
// (bricks.entry_distance).
fn brick_step_limit(p: vec3<f32>, e: vec3<f32>) -> f32 {
    if (gas.bricks.z < 0.5) {
        return 3.0e38;
    }
    let dx = gas.bricks.x;
    let lo = gas.brick_lo.xyz;
    let hi = gas.brick_hi.xyz;
    if (all(p >= lo) && all(p < hi)) {
        let b = brick_home(p);
        if (brick_slot(b) != 0u) {
            return 3.0e38;
        }
        var best = 3.0e38;
        for (var k = 0; k < 3; k++) {
            if (e[k] > 0.0) {
                best = min(best, ((f32(b[k] + 1) * 8.0 + 0.5) * dx - p[k]) / e[k]);
            } else if (e[k] < 0.0) {
                best = min(best, ((f32(b[k]) * 8.0 + 0.5) * dx - p[k]) / e[k]);
            }
        }
        return max(best, 0.0) + EXIT_EPS * dx;
    }
    var t0 = -3.0e38;
    var t1 = 3.0e38;
    for (var k = 0; k < 3; k++) {
        if (e[k] != 0.0) {
            let u = (lo[k] - p[k]) / e[k];
            let v = (hi[k] - p[k]) / e[k];
            t0 = max(t0, min(u, v));
            t1 = min(t1, max(u, v));
        } else if (p[k] < lo[k] || p[k] >= hi[k]) {
            return 3.0e38;
        }
    }
    if (t1 < max(t0, 0.0)) {
        return 3.0e38;
    }
    return max(t0, 0.0) + EXIT_EPS * dx;
}

// Fields at p: the first active box covering p, blended into the log grid plus the bricks
// (march.sample: the bricks hold particles the log grid does not, so their fields add).
fn sample_fields(p: vec3<f32>) -> vec2<f32> {
    let base = sample_log(p) + sample_bricks(p);
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

// #if COLOUR
// Colour shader: density, rho T^4 and rho x gamma beta from ONE trilinear pass per grid (the
// same corner indices and weights for the two textures), combined exactly like sample_fields
// (march.sample + march.sample_rho_gb). Sharing the index maths keeps the colour shader small:
// separate samplers for the two fields made it hang the Iris Plus (AI_LOG 2026-11-17).
struct Pair { f: vec2<f32>, v: vec3<f32> };

fn trilinear_pair(t: texture_3d<f32>, tv: texture_3d<f32>, fi: f32, fj: f32, fk: f32,
                  wrap_k: bool) -> Pair {
    let dims = vec3<i32>(textureDimensions(t));
    let i0 = i32(floor(fi));
    let j0 = i32(floor(fj));
    let k0 = i32(floor(fk));
    let ti = fi - f32(i0);
    let tj = fj - f32(j0);
    let tk = fk - f32(k0);
    var out: Pair;
    out.f = vec2<f32>(0.0);
    out.v = vec3<f32>(0.0);
    for (var c = 0; c < 8; c++) {
        let di = c & 1;
        let dj = (c >> 1u) & 1;
        let dk = (c >> 2u) & 1;
        let ii = clamp(i0 + di, 0, dims.z - 1);
        let jj = clamp(j0 + dj, 0, dims.y - 1);
        var kk = k0 + dk;
        if (wrap_k) {
            kk = ((kk % dims.x) + dims.x) % dims.x;
        } else {
            kk = clamp(kk, 0, dims.x - 1);
        }
        let w = select(1.0 - ti, ti, di == 1) * select(1.0 - tj, tj, dj == 1)
              * select(1.0 - tk, tk, dk == 1);
        let q = vec3<i32>(kk, jj, ii);
        out.f += w * textureLoad(t, q, 0).rg;
        out.v += w * textureLoad(tv, q, 0).xyz;
    }
    return out;
}

fn sample_pair_log(p: vec3<f32>) -> Pair {
    var out: Pair;
    out.f = vec2<f32>(0.0);
    out.v = vec3<f32>(0.0);
    let r = length(p);
    if (r < gas.grid.x || r > gas.grid.y) {
        return out;
    }
    let fi = log(r / gas.grid.x) / gas.grid.z - 0.5;
    let fj = acos(clamp(p.z / r, -1.0, 1.0)) / gas.grid.w - 0.5;
    var ph = atan2(p.y, p.x);
    if (ph < 0.0) {
        ph += TWO_PI_F;
    }
    return trilinear_pair(log_tex, log_vel, fi, fj, ph / gas.grid2.x - 0.5, true);
}

fn sample_pair_box(t: texture_3d<f32>, tv: texture_3d<f32>, b: BoxParams, p: vec3<f32>) -> Pair {
    let f = (p - b.lo.xyz) / b.lo.w - 0.5;
    return trilinear_pair(t, tv, f.x, f.y, f.z, false);
}

// Bricks: rho, rho T^4 from the atlas, rho x gamma beta from the velocity atlas (same layout).
fn sample_pair_bricks(p: vec3<f32>) -> Pair {
    var out: Pair;
    out.f = vec2<f32>(0.0);
    out.v = vec3<f32>(0.0);
    if (gas.bricks.z < 0.5) {
        return out;
    }
    let f = p / gas.bricks.x - 0.5;
    let fl = floor(f);
    let c = vec3<i32>(fl);
    let home = c >> vec3<u32>(3u);
    let slot = brick_slot(home);
    if (slot == 0u) {
        return out;
    }
    let s = slot - 1u;
    let a = gas.brick_a;
    let blk = vec3<i32>(i32(s / (a.y * a.z)), i32((s / a.z) % a.y), i32(s % a.z));
    let base = APRON_I * blk + (c - BRICK_I * home);
    let t = f - fl;
    for (var k = 0; k < 8; k++) {
        let d = vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1);
        let w = select(1.0 - t.x, t.x, d.x == 1) * select(1.0 - t.y, t.y, d.y == 1)
              * select(1.0 - t.z, t.z, d.z == 1);
        let q = base + d;
        let tq = vec3<i32>(q.z, q.y, q.x);
        out.f += w * textureLoad(brick_atlas, tq, 0).rg;
        out.v += w * textureLoad(brick_vel, tq, 0).xyz;
    }
    return out;
}

fn sample_pair(p: vec3<f32>) -> Pair {
    let lg = sample_pair_log(p);
    let bk = sample_pair_bricks(p);
    var base: Pair;
    base.f = lg.f + bk.f;
    base.v = lg.v + bk.v;
    if (gas.box0.n.y != 0u) {
        let w = box_weight(gas.box0, p);
        if (w > 0.0) {
            let bx = sample_pair_box(box0_tex, box0_vel, gas.box0, p);
            base.f = w * bx.f + (1.0 - w) * base.f;
            base.v = w * bx.v + (1.0 - w) * base.v;
            return base;
        }
    }
    if (gas.box1.n.y != 0u) {
        let w = box_weight(gas.box1, p);
        if (w > 0.0) {
            let bx = sample_pair_box(box1_tex, box1_vel, gas.box1, p);
            base.f = w * bx.f + (1.0 - w) * base.f;
            base.v = w * bx.v + (1.0 - w) * base.v;
            return base;
        }
    }
    return base;
}
// #endif

// physics.colour.lut_chroma: exp of ln chroma interpolated linearly in ln T, T clamped to the
// table (gas.colour.y = ln T_0, .z = d ln T, .w = entries).
fn lut_chroma(t: f32) -> vec3<f32> {
    let n = u32(gas.colour.w);
    let lt = clamp(log(t), gas.colour.y, gas.colour.y + gas.colour.z * f32(n - 1u));
    let pos = (lt - gas.colour.y) / gas.colour.z;
    let i = min(u32(pos), n - 2u);
    let f = pos - f32(i);
    let a = textureLoad(chroma_lut, vec2<i32>(i32(i), 0), 0).xyz;
    let b = textureLoad(chroma_lut, vec2<i32>(i32(i) + 1, 0), 0).xyz;
    return exp(a * (1.0 - f) + b * f);
}

// Local cell size of the finest grid at p (march.cell_size).
fn cell_size(p: vec3<f32>) -> f32 {
    if (gas.box0.n.y != 0u && box_inside(gas.box0, p)) {
        return gas.box0.lo.w;
    }
    if (gas.box1.n.y != 0u && box_inside(gas.box1, p)) {
        return gas.box1.lo.w;
    }
    var size = max(length(p), gas.grid.x) * min(gas.grid.z, gas.grid.w);
    if (brick_allocated(p)) {
        size = min(size, gas.bricks.x);
    }
    return size;
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

// intensity: grey I in .x (Phase 8), or X, Y, Z, bolometric (Phase 9 colour mode)
struct MarchState { intensity: vec4<f32>, transmission: f32, steps: u32 };

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
        ds = min(ds, brick_step_limit(p, e));
        let mid = p + 0.5 * ds * e;
        // #if COLOUR
        let pr = sample_pair(mid);
        let f = pr.f;
        // #else
        let f = sample_fields(mid);
        // #endif
        let alpha = gas.model.x * f.x;
        let r_mid = length(mid);
        // No gas: nothing changes; also avoids 0 x infinity at r_mid = 2 exactly (march.py).
        if (alpha > 0.0 && r_mid > 2.0) {
            let q = p + ds * e;
            let path = proper_length(ds, length(q) - length(p), r_mid);
            // #if COLOUR
            // march.RHO_MIN_COLOUR: subnormal float32 residues would give 0/0 below
            if (f.x > gas.colour2.z) {
                // Phase 9 (march.RelativisticBlackbody): blackbody at g T_em, optical depth
                // kappa rho path (gamma - gamma beta . n). S dtau is written without dividing
                // by rho (see the grey case): sigma (g T)^4 / pi = g^4 s_scale rho_t4 / rho.
                var gb = pr.v / f.x;
                let gbl = length(gb);
                if (gbl > gas.colour2.y) {
                    gb *= gas.colour2.y / gbl;  // march.GB_MAX: float32 residue guard
                }
                let rh = mid / r_mid;
                let er = dot(e, rh);
                // #if KERR
                let lapse = kerr_lapse(mid);  // physics.kerr.static_lapse (approximate Doppler)
                // #else
                let lapse = sqrt(1.0 - 2.0 / r_mid);
                // #endif
                let n_light = -normalize(e + (er / lapse - er) * rh);
                let doppler = sqrt(1.0 + dot(gb, gb)) - dot(gb, n_light);
                // camera factor sqrt(1 - 2/r_cam) from the camera itself (it moves per frame)
                // #if KERR
                let g = lapse / sqrt(1.0 - kerr.spin.z) / doppler;  // kerr.spin.z = f at the camera
                // #else
                let g = lapse / sqrt(1.0 - 2.0 / length(cam.pos.xyz)) / doppler;
                // #endif
                let t_em = T_REF_F * sqrt(sqrt(f.y / f.x));
                let g2 = g * g;
                // opacity "thermal" when gas.colour2.x > 0: physics.opacity.
                // thermal_absorption_code in the form of thermal_absorption_constants:
                // alpha = c_es rho sqrt(q (1 + q)), q = q_ff rho T^(-7/2), T in K. Written
                // without a branch or exp/log (a first version with both hung the Iris Plus, AI_LOG).
                let t_m35 = 1.0 / (t_em * t_em * t_em * sqrt(t_em));
                let q = gas.colour2.x * f.x * t_m35;
                let th = select(0.0, 1.0, gas.colour2.x > 0.0);
                let alpha_g = mix(alpha, gas.colour2.w * f.x * sqrt(q * (1.0 + q)), th);
                let dtau = alpha_g * path * doppler;
                let bol_dtau = g2 * g2 * gas.model.y * (f.y / f.x) * dtau;
                let w = st.transmission * bol_dtau * one_minus_exp_over_x(dtau);
                st.intensity += w * vec4<f32>(lut_chroma(g * t_em), 1.0);
                st.transmission *= exp(-dtau);
            }
            // #else
            {
                let dtau = alpha * path;
                // S (1 - e^-dtau) with S = s_scale rho_t4 / rho and dtau = kappa rho path,
                // written as s_scale rho_t4 kappa path (1 - e^-dtau) / dtau: no division by
                // rho, which in float32 can be a rounding residue at the thin edge of the gas
                // (gave inf pixels).
                let src_dtau = gas.model.y * f.y * gas.model.x * path;
                st.intensity.x += st.transmission * src_dtau * one_minus_exp_over_x(dtau);
                st.transmission *= exp(-dtau);
            }
            // #endif
        }
        // Always move on: s + ds rounds to s in float32 when ds (a brick-edge step of
        // EXIT_EPS cells) is below half an ulp of s (s in the thousands on wide shots); the ray
        // then marched in place forever and hung the GPU (NVIDIA Xid 109; AI_LOG 2026-11-19).
        let s_next = s + ds;
        s = select(s_next, s + max(abs(s), 1.0) * 4.0e-7, s_next <= s);
        st.steps += 1u;
        if (st.steps >= MAX_MARCH_STEPS) {  // backstop: no ray may run without end
            break;
        }
    }
    return st;
}

struct RayOut {
    intensity: vec4<f32>,
    transmission: f32,
    status: u32,
    steps: u32,
    sky_dir: vec3<f32>,  // direction on the sky seen by an escaped ray (Phase 2 bend)
};

// #if KERR
// Spinning hole (Phase 11, 11.2): Hamiltonian rays in Cartesian Kerr-Schild coordinates, the
// arriving photon followed backward, fixed RK4 steps. Float32 twin of render/kerr_rays.py
// (ray_polyline, step_size, sky_direction), which mirrors physics/kerr.py (backward_rhs,
// hamiltonian_grad_x, ks_radius, metric_parts); tests: tests/render/test_kerr_gpu.py.
struct KerrCam {
    spin: vec4<f32>,  // a, r_h, f at the camera, -
    c0: vec4<f32>,    // p_mu ~ c0 - (a0 d.x + a1 d.y + a2 d.z) (kerr_rays.camera_map)
    a0: vec4<f32>,
    a1: vec4<f32>,
    a2: vec4<f32>,
};
@group(0) @binding(17) var<uniform> kerr: KerrCam;
const KERR_H_FAR: f32 = 0.02;     // kerr_rays.H_FAR
const KERR_R_NEAR: f32 = 2.0;     // kerr_rays.R_NEAR
const KERR_H_MIN: f32 = 1.0e-4;   // kerr_rays.H_MIN
const KERR_R_SKY: f32 = 1.0e6;    // kerr_rays.R_SKY
const KERR_MAX_STEPS: u32 = 20000u;  // kerr_rays.MAX_STEPS
const KERR_MARGIN: f32 = 1.0e-3;  // physics.kerr.HORIZON_MARGIN

fn kerr_radius(x: vec3<f32>) -> f32 {  // physics.kerr.ks_radius
    let a2 = kerr.spin.x * kerr.spin.x;
    let b = dot(x, x) - a2;
    return sqrt(0.5 * b + sqrt(0.25 * b * b + a2 * x.z * x.z));
}

fn kerr_f(x: vec3<f32>) -> f32 {  // physics.kerr.metric_parts: f = 2 r^3 / (r^4 + a^2 z^2)
    let r = kerr_radius(x);
    let a = kerr.spin.x;
    return 2.0 * r * r * r / (r * r * r * r + a * a * x.z * x.z);
}

fn kerr_lapse(x: vec3<f32>) -> f32 {  // physics.kerr.static_lapse
    return sqrt(max(1.0 - kerr_f(x), 0.0));
}

struct KerrRhs { dx: vec3<f32>, dp: vec3<f32> };

// physics.kerr.backward_rhs: (-dH/dp, +dH/dx) with p_t = -1. df/dr is written as
// 2 r^2 (3 a^2 z^2 - r^4) / D^2 (equal to hamiltonian_grad_x's form; no cancellation of
// r^6 terms in float32).
fn kerr_rhs(x: vec3<f32>, p: vec3<f32>) -> KerrRhs {
    let a = kerr.spin.x;
    let r = kerr_radius(x);
    let r2 = r * r;
    let z2 = x.z * x.z;
    let dd = r2 * r2 + a * a * z2;
    let f = 2.0 * r2 * r / dd;
    let bb = r2 + a * a;
    let l = vec3<f32>((r * x.x + a * x.y) / bb, (r * x.y - a * x.x) / bb, x.z / r);
    let s = 1.0 + dot(l, p);
    let dr = vec3<f32>(x.x * r2 * r, x.y * r2 * r, x.z * r * bb) / dd;
    var df = (2.0 * r2 * (3.0 * a * a * z2 - r2 * r2) / (dd * dd)) * dr;
    df.z += -4.0 * r2 * r * a * a * x.z / (dd * dd);
    let aa = r * (x.x * p.x + x.y * p.y) + a * (x.y * p.x - x.x * p.y);
    let da = dr * (x.x * p.x + x.y * p.y) + r * vec3<f32>(p.x, p.y, 0.0)
           + a * vec3<f32>(-p.y, p.x, 0.0);
    var dlp = (da * bb - aa * 2.0 * r * dr) / (bb * bb) - (x.z * p.z / r2) * dr;
    dlp.z += p.z / r;
    let grad = -0.5 * s * s * df - f * s * dlp;
    var out: KerrRhs;
    out.dx = -(p - f * s * l);
    out.dp = grad;
    return out;
}

struct KerrState { x: vec3<f32>, p: vec3<f32> };

fn kerr_rk4(x: vec3<f32>, p: vec3<f32>, h: f32) -> KerrState {
    let k1 = kerr_rhs(x, p);
    let k2 = kerr_rhs(x + 0.5 * h * k1.dx, p + 0.5 * h * k1.dp);
    let k3 = kerr_rhs(x + 0.5 * h * k2.dx, p + 0.5 * h * k2.dp);
    let k4 = kerr_rhs(x + h * k3.dx, p + h * k3.dp);
    var out: KerrState;
    out.x = x + h / 6.0 * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    out.p = p + h / 6.0 * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);
    return out;
}

fn kerr_step_size(r: f32) -> f32 {  // kerr_rays.step_size
    return max(KERR_H_FAR * r * min(1.0, (r - kerr.spin.y) / KERR_R_NEAR), KERR_H_MIN);
}

fn kerr_sky(x_in: vec3<f32>, p_in: vec3<f32>) -> vec3<f32> {  // kerr_rays.sky_direction
    var st: KerrState;
    st.x = x_in;
    st.p = p_in;
    for (var k = 0u; k < 200u; k++) {
        let r = kerr_radius(st.x);
        if (r >= KERR_R_SKY) {
            break;
        }
        st = kerr_rk4(st.x, st.p, 0.1 * r);
    }
    return normalize(kerr_rhs(st.x, st.p).dx);
}

// One camera ray around the spinning hole (bent_rays.march_bent_ray with spin): the RK4
// points joined by chords, one march_chord call per step (a second call site inlines a second
// copy of the march; too big for the Iris Plus compiler, AI_LOG 2026-11-19).
fn march_ray(d: vec3<f32>) -> RayOut {
    var st: MarchState;
    st.intensity = vec4<f32>(0.0);
    st.transmission = 1.0;
    st.steps = 0u;
    var res: RayOut;
    res.status = RAY_ORBITING;
    res.sky_dir = vec3<f32>(0.0);
    let r_max = gas.grid.y;
    let pc = kerr.c0 - (kerr.a0 * d.x + kerr.a1 * d.y + kerr.a2 * d.z);
    var x = cam.pos.xyz;
    var p = pc.yzw / (-pc.x);
    for (var n = 0u; n < KERR_MAX_STEPS; n++) {
        let r = kerr_radius(x);
        let nxt = kerr_rk4(x, p, kerr_step_size(r));
        let r1 = kerr_radius(nxt.x);
        var end = nxt.x;
        var done = false;
        if (r1 < kerr.spin.y + KERR_MARGIN) {
            res.status = RAY_CAPTURED;
            done = true;
        } else if (r1 > r_max && r1 > r) {
            end = x + clamp((r_max - r) / (r1 - r), 0.0, 1.0) * (nxt.x - x);
            res.status = RAY_ESCAPED;
            res.sky_dir = kerr_sky(nxt.x, nxt.p);
            done = true;
        }
        if (st.transmission >= gas.grid2.z) {
            st = march_chord(x, end, st);
        }
        if (done) {
            break;
        }
        x = nxt.x;
        p = nxt.p;
    }
    res.intensity = st.intensity;
    res.transmission = st.transmission;
    res.steps = st.steps;
    return res;
}

// #else
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
    st.intensity = vec4<f32>(0.0);
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

// #endif
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
//   OUT_COLOUR:     two vec4 per pixel: mean (X, Y, Z, bolometric) of the gas, then (mean T of
//                   escaped sub-rays, captured fraction, mean sub-steps, 0)
//                   (bent_rays.render_reference_colour; Phase 9 test 13)
//   OUT_COLOUR_PICTURE: two vec4 per pixel: mean (X, Y, Z, bolometric), then (mean of T x
//                   star radiance RGB, 0)
const OUT_COLOUR: u32 = 3u;
const OUT_COLOUR_PICTURE: u32 = 4u;
//   OUT_SKY_DIR_T:  (sky direction xyz, gas transmission T, or -1 if captured) of the last
//                   sub-ray (n_sub = 1: the pixel-centre ray): the node grid of the pinpoint
//                   stars (render/point_stars.py)
const OUT_SKY_DIR_T: u32 = 5u;
const T_REF_F: f32 = 1.0e4;  // march.T_REF
const MAX_SUB: u32 = 16u;  // n_sub <= 4
const TEXEL_ANGLE_FACE0: f32 = 0.000766990;  // as render_stars.wgsl

@group(0) @binding(7) var sky: texture_cube<f32>;
@group(0) @binding(8) var sky_sampler: sampler;
// per-pixel sub-ray count for photon-ring pixels (0: use n_sub); a 1-element buffer = none
@group(0) @binding(16) var<storage, read> sub_map: array<u32>;
const MAX_RING_SUB: u32 = 8u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= tile.size.x || gid.y >= tile.size.y) {
        return;
    }
    let pix = tile.origin + gid.xy;
    let idx = gid.y * tile.size.x + gid.x;
    // Photon-ring pixels (render/ring_samples.py: |b - b_crit| within a few pixels) trace
    // m x m sub-rays instead of n x n: the ring is thinner than a pixel and 4 rays drew it as a
    // dotted line (DECISIONS 2026-11-19). One ray loop for both (a second inlined march loop
    // made the shader too big for the Iris Plus compiler: AI_LOG 2026-11-19).
    var m = min(cam.dims.z, 4u);
    if (arrayLength(&sub_map) == cam.dims.x * cam.dims.y) {
        m = max(m, min(sub_map[pix.y * cam.dims.x + pix.x], MAX_RING_SUB));
    }
    let total = f32(m * m);
    var acc = vec4<f32>(0.0);
    var col = vec4<f32>(0.0);
    var dirs: array<vec3<f32>, MAX_SUB>;
    var trans: array<f32, MAX_SUB>;
    var n_esc = 0u;  // escaped sub-rays kept for the sky sample (the first MAX_SUB)
    var n_esc_all = 0u;
    var mean = vec3<f32>(0.0);
    var last = vec4<f32>(0.0);
    for (var b = 0u; b < m; b++) {
        for (var a = 0u; a < m; a++) {
            let d = pixel_dir(f32(pix.x) + (f32(a) + 0.5) / f32(m),
                              f32(pix.y) + (f32(b) + 0.5) / f32(m));
            let r = march_ray(d);
            let escaped = r.status == RAY_ESCAPED;
            acc += vec4<f32>(r.intensity.x, select(0.0, r.transmission, escaped),
                             select(1.0, 0.0, escaped), f32(r.steps));
            col += r.intensity;
            if (cam.dims.w == OUT_SKY_DIR_T) {
                last = vec4<f32>(r.sky_dir, select(-1.0, r.transmission, escaped));
            } else {
                last = vec4<f32>(r.sky_dir, select(0.0, 1.0, escaped));
            }
            if (escaped) {
                n_esc_all += 1u;
                if (n_esc < MAX_SUB) {
                    dirs[n_esc] = r.sky_dir;
                    trans[n_esc] = r.transmission;
                    mean += r.sky_dir;
                    n_esc += 1u;
                }
            }
        }
    }
    if (cam.dims.w == OUT_DIAGNOSTIC) {
        out[idx] = acc / total;
        return;
    }
    if (cam.dims.w == OUT_COLOUR) {
        out[2u * idx] = col / total;
        out[2u * idx + 1u] = vec4<f32>(acc.y / total, acc.z / total, acc.w / total, 0.0);
        return;
    }
    if (cam.dims.w == OUT_SKY_DIR || cam.dims.w == OUT_SKY_DIR_T) {
        out[idx] = last;
        return;
    }
    // Milky Way band behind the gas: the kept escaped sub-rays (all of them unless a ring pixel
    // has more than MAX_SUB), level from their spread, scaled to all escaped rays
    var rgb = vec3<f32>(0.0);
    if (n_esc > 0u) {
        let mn = normalize(mean);
        var spread = 0.0;
        for (var k = 0u; k < n_esc; k++) {
            spread = max(spread, acos(clamp(dot(dirs[k], mn), -1.0, 1.0)));
        }
        let face_size = f32(textureDimensions(sky).x);
        let texel_angle = TEXEL_ANGLE_FACE0 * 2048.0 / face_size;
        let footprint = 2.0 * spread / f32(min(m, 4u));
        let lod = log2(max(footprint / texel_angle, 1.0));
        for (var k = 0u; k < n_esc; k++) {
            rgb += trans[k] * textureSampleLevel(sky, sky_sampler, dirs[k], lod).rgb;
        }
        rgb *= f32(n_esc_all) / f32(n_esc);
    }
    if (cam.dims.w == OUT_COLOUR_PICTURE) {
        out[2u * idx] = col / total;
        out[2u * idx + 1u] = vec4<f32>(rgb * cam.lens.z / total, 0.0);
        return;
    }
    out[idx] = vec4<f32>(acc.x / total, rgb * cam.lens.z / total);
}
