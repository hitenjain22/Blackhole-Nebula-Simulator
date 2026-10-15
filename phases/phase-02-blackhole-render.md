# Phase 2: First black-hole render (detailed)

**Goal:** a GPU ray tracer that bends light around a non-spinning hole and shows a lensed star field, checked against exact numbers. This is the foundation the gas rendering (Phases 8–9) is built on, so the ray integrator is written the way Phase 8 needs it: it walks along each ray in space, not just a lookup of the final direction.

**Ends with:** a 1280×720 image of the black hole in front of a star field with a correct-size shadow and Einstein ring, and a short orbiting-camera video. The first "wow" moment.

---

## Background: light rays around a Schwarzschild hole

A light ray around a non-spinning hole stays in a plane through the hole. In that plane, with u = 1/r and angle φ, the ray obeys (M = 1):

```
d²u/dφ² = 3u² − u
(du/dφ)² = 1/b² − u²(1 − 2u)          (first integral; b = impact parameter)
```

**Camera setup.** A camera at distance D (static observer). For a pixel, let ψ be the angle between the pixel's viewing direction d and the direction *toward* the hole. Then:
- Ray plane basis: e₁ = x_cam/|x_cam| (outward radial), e₂ = normalized part of d perpendicular to e₁.
- Impact parameter: b = D sin ψ / √(1 − 2/D).
- Start: u₀ = 1/D, φ = 0, du/dφ = +√(1/b² − u₀²(1 − 2u₀)) if looking toward the hole (ψ < 90°), negative if looking away.
- Integrate in φ. **Captured** if u ≥ 1/2 (r ≤ 2). **Escaped** when u crosses 0: the sky direction seen is cos φ∞ e₁ + sin φ∞ e₂.
- Degenerate case d ∥ e₁ (exactly radial): looking inward is captured; looking outward sees e₁.
- 3D position along the ray at any step: r = 1/u, position = r (cos φ e₁ + sin φ e₂). Phase 8 uses this to sample the gas.

**Shadow size** (Synge 1966): rays with b < 3√3 are captured, so the shadow's angular radius ψ_sh satisfies sin ψ_sh = (3√3/D)√(1 − 2/D).

---

## Step 2.1: Float64 reference ray tracer (CPU)

**Do:** `src/nebula/physics/geodesics.py`:
- `schw_ray_escape_angle(D, psi)` → φ∞ (or NaN if captured), via SciPy DOP853 (rtol 1e-12) with terminal events at u = 0 (escape) and u = 0.5 (capture).
- `schw_deflection(b)` for a ray from infinity (for tests).
- `schw_shadow_angle(D)` (Synge formula).
- `schw_ray_path(D, psi, n)`: positions along the ray (for plots and Phase 8 tests).

**Tests** (expected-values §3):

| Test | Expected | Tolerance |
|---|---|---|
| Deflection, b = 1000 | 0.00401182381 rad (weak-field 4/b = 0.004) | rtol 1e-7 |
| Deflection, b = 50 | 0.085083450 rad | rtol 1e-7 |
| Deflection, b = 10 | 0.590395788 rad | rtol 1e-7 |
| Deflection, b = 6 | 1.719388310 rad | rtol 1e-7 |
| Deflection, b = 5.3 | 3.557938042 rad (loops > half-turn) | rtol 1e-6 |
| Capture | b = 5.19 captured; b = 5.20 escapes | exact |
| Shadow angular radius, D = 20 / 50 / 100 / 1000 | 14.2690273° / 5.8441783° / 2.9485552° / 0.29742107° | rtol 1e-7 |
| Escape angle, D = 50, ψ = 10°, 20°, 45°, 80° | φ∞ = 210.86090°, 175.52870°, 140.94777°, 102.85233° | abs 1e-5° |
| Escape angle, D = 20, ψ = 20°, 45°, 80° | 219.43474°, 151.84667°, 107.64767°; ψ = 10° captured | abs 1e-5° |
| Einstein ring (source exactly behind the hole, at infinity), D = 50 | primary ring at ψ = 17.872702°, secondary at 5.850814° | abs 1e-5° |
| Same, D = 20 | primary 29.828318°, secondary 14.283091° | abs 1e-5° |

---

## Step 2.2: GPU ray integrator (WGSL)

**Do:** `src/nebula/render/shaders/schw_ray.wgsl`:
- Integrates u(φ) with RK4 in float32, step Δφ adaptive: smaller near the photon sphere (u ≈ 1/3) and near the turning point. Start with Δφ = min(0.02, 0.1 · |u − 1/3| + 0.002) and tune by test, not by eye.
- Termination: escape (u < 0 after having been positive), capture (u > 0.5), or a cap on total φ (e.g. 8π: rays still orbiting are treated as captured and counted).
- Output per pixel: escape direction (vec3), status flag, and number of half-turns (for diagnostics and the photon ring).
- Python wrapper in `src/nebula/render/` that runs it in tiles (Phase 0 helper).

**Test:** for D = 20 and D = 50 and 2,000 ψ values from 0 to 180° (denser near ψ_sh), compare GPU φ∞ with the Step 2.1 reference:
- for b > 1.05 · 3√3: |Δφ∞| < 2 × 10⁻⁴ rad;
- within 5% of the critical b: errors grow (the deflection diverges logarithmically); report the error curve as a plot, require < 10⁻² rad for b > 1.001 · 3√3, and require the capture/escape classification to match for all b outside ±10⁻⁴ of critical.
- Fix the step control until these pass. Do not widen tolerances.

**What can go wrong:**
- *float32 near the photon sphere:* rays near the critical impact parameter circle many times and amplify rounding. Integrating u(φ) (rather than x, y, z in time) keeps this well-conditioned; keep it that way.
- *Turning point:* du/dφ changes sign; RK4 on the second-order form handles it naturally. Don't integrate the square-root first-order form (it stalls at the turning point).
- *Plane basis when d ≈ e₁:* normalize carefully; if |d − (d·e₁)e₁| < 1e-7 treat as radial.

---

## Step 2.3: Camera and image pipeline

**Do:** `src/nebula/render/camera.py` and `frames.py`:
- Pinhole camera: position (any point with r > 3; Phase 8 keeps cameras at r ≥ 6), look-at target, up vector, vertical field of view. Pixel → world direction.
- **The camera is a static observer.** (A moving camera adds aberration; add it later only if a shot needs it, with a test.)
- Supersampling: N rays per pixel with a fixed (frame-independent) jitter pattern, e.g. a 4×4 stratified pattern. Same pattern every frame to avoid flicker.
- HDR float32 accumulation in linear colour; tone mapping (start with a simple filmic/ACES curve) and conversion to sRGB with the correct sRGB transfer function, then 8-bit with tiny ordered dithering to avoid banding. Write PNG.

**Test:** an image with no hole (flat space, straight rays) of a synthetic sky with grid lines must show straight grid lines in the expected pixels (checks camera maths independently of gravity).

---

## Step 2.4: Star field

**Do:** `src/nebula/render/starfield.py`:
- Procedural sky, seeded: ~100,000 stars with a realistic brightness distribution (number of stars rising steeply toward faint magnitudes), colours from blackbody temperatures 3,000–30,000 K (use a rough blackbody→RGB table for now; Phase 9 replaces it with the proper colour physics), plus an optional faint Milky Way band (procedural noise along a great circle). Option for later: a real star catalogue or a public-domain star map (check its licence before using it).
- Baked into an HDR cube map, 2048×2048 per face, float16 RGBA with mip levels (≈200 MB, ≈270 MB including the mips; check it fits; drop to 1024 if not).
- Stars drawn as small Gaussian spots of fixed angular size so they don't vanish or alias.

**What can go wrong:**
- *Near the Einstein ring, magnification is huge:* one pixel's rays fan out over a large patch of sky. Supersampling handles most of it; if rings of stars sparkle between frames, select the mip level from the spread of directions among a pixel's sub-rays (a simple "ray differential").
- *Cube-map seams:* sample with proper filtering across faces (WebGPU cube textures do this) or use an equirectangular map with care at the poles.

---

## Step 2.5: Validation images

**Do and check:**
1. `outputs/checks/shadow_D50.png`: camera at D = 50, FOV chosen so the shadow spans ~200 px. Measure the shadow radius in the image (fit a circle to the edge). Expected angular radius 5.844178°; must match to within 1 pixel.
2. `outputs/checks/einstein_ring_D50.png`: one bright star placed exactly behind the hole: the primary ring at 17.8727° and secondary at 5.8508°, each within 1 pixel.
3. `outputs/checks/blackhole_starfield.png`: the full star field, a pleasing composition (camera at r ≈ 30–60, hole slightly off-centre).
4. `outputs/checks/orbit_cam.mp4`: 240 frames, camera circling at r = 40, 720p, 4 rays per pixel. Record ms/frame.

**Hiten looks at:** the shadow and ring images (with the measured vs expected numbers printed on them), the full image, and the video.

---

## Step 2.6: Performance note

Record in DECISIONS.md: time per 720p and 1080p frame at 1, 4, 16 rays per pixel. Estimate Phase 8–9 costs from it (the gas makes each ray ~10–50× more expensive).

---

## Done when

- [ ] Reference tests pass; GPU vs reference error plot reviewed.
- [ ] Shadow and Einstein-ring images match expected sizes within 1 px.
- [ ] Orbiting-camera video made; frame timing recorded.
- [ ] Hiten has seen the images and video; committed.
