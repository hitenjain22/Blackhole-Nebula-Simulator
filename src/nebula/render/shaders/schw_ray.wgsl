// Light rays around a non-spinning black hole, float32 (Phase 2, Step 2.2). Library: no
// bindings; combined with an entry shader by nebula.render.device.load_shader.
//
// Mirrors nebula.physics.geodesics (schw_ray_escape_angle); tested against it in
// tests/render/test_schw_ray_gpu.py. Units G = c = M = 1, u = 1/r, phi = angle in the ray plane:
//     d^2u/dphi^2 = 3u^2 - u          (second-order form: no stall at turning points)
// RK4 in phi with step h = min(0.02, 0.1 |u - 1/3| + 0.002) (smaller near the photon sphere,
// phase-02 Step 2.2), rounded down to a multiple of 2^-12 so that the running sum phi is
// exact in float32 (it would otherwise lose ~1e-4 rad over ~1,000 steps).

const RAY_ESCAPED: u32 = 0u;
const RAY_CAPTURED: u32 = 1u;   // reached r = 2
const RAY_ORBITING: u32 = 2u;   // still circling after PHI_CAP: treated as captured
const PHI_CAP: f32 = 25.132741;  // 8 pi
const H_GRID: f32 = 0.000244140625;  // 2^-12
const PI_F: f32 = 3.14159265;

struct RayResult {
    phi_inf: f32,      // escape angle (rad); undefined unless status == RAY_ESCAPED
    status: u32,
    half_turns: u32,   // floor(phi at termination / pi)
    steps: u32,
};

// du/dphi at the camera for a ray at angle psi from the direction to the hole, seen by a
// static observer at u0 = 1/D. Equals +-sqrt(1/b^2 - u0^2 (1 - 2 u0)) with
// b = D sin(psi)/sqrt(1 - 2/D), written without the cancellation near psi = 90 deg.
// Mirrors geodesics._initial_state.
fn ray_initial_slope(u0: f32, cos_psi: f32, sin_psi: f32) -> f32 {
    return u0 * sqrt(1.0 - 2.0 * u0) * cos_psi / sin_psi;
}

fn ray_step_size(u: f32) -> f32 {
    let h = min(0.02, 0.1 * abs(u - 1.0 / 3.0) + 0.002);
    return max(floor(h / H_GRID), 1.0) * H_GRID;
}

fn ray_accel(u: f32) -> f32 {
    return 3.0 * u * u - u;
}

// Root in (0, 1) of the cubic Hermite interpolant of u over one step, where u goes from
// ua > 0 to ub < 0 with slopes wa, wb (per unit phi) over a step of length h.
fn ray_hermite_root(ua: f32, wa: f32, ub: f32, wb: f32, h: f32) -> f32 {
    var t = ua / (ua - ub);  // linear first guess
    for (var k = 0; k < 4; k++) {
        let t2 = t * t;
        let t3 = t2 * t;
        let p = (2.0 * t3 - 3.0 * t2 + 1.0) * ua + (t3 - 2.0 * t2 + t) * h * wa
              + (-2.0 * t3 + 3.0 * t2) * ub + (t3 - t2) * h * wb;
        let dp = (6.0 * t2 - 6.0 * t) * ua + (3.0 * t2 - 4.0 * t + 1.0) * h * wa
               + (-6.0 * t2 + 6.0 * t) * ub + (3.0 * t2 - 2.0 * t) * h * wb;
        t = clamp(t - p / dp, 0.0, 1.0);
    }
    return t;
}

fn trace_ray(u0: f32, du0: f32) -> RayResult {
    var u = u0;
    var w = du0;
    var phi = 0.0;
    var res: RayResult;
    res.status = RAY_ORBITING;
    res.phi_inf = 0.0;
    var steps = 0u;
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
        steps += 1u;
        if (u1 >= 0.5) {
            res.status = RAY_CAPTURED;
            phi += h;
            break;
        }
        if (u1 < 0.0) {
            res.status = RAY_ESCAPED;
            res.phi_inf = phi + ray_hermite_root(u, w, u1, w1, h) * h;
            phi = res.phi_inf;
            break;
        }
        u = u1;
        w = w1;
        phi += h;
    }
    res.half_turns = u32(floor(phi / PI_F));
    res.steps = steps;
    return res;
}
