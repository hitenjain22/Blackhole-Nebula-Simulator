# Phase 8: Rendering the gas (in-depth)

**Goal:** turn simulation snapshots into lensed images of glowing gas. Light rays from the camera bend around the hole exactly as in Phase 2; along each ray the renderer marches through the gas, adding light where gas glows and dimming light where gas blocks it. The back of the torus appears lifted over the top of the hole, the lensed image under it, the shadow in the middle: the physically correct version of the famous movie look.

Colour is kept simple here (a temperature colour map or grey). Phase 9 replaces it with real blackbody colour, Doppler beaming and gravitational redshift; the structure built here must make that a drop-in change.

**Ends with:** a test suite of synthetic scenes with exact answers; lensed frames of real snapshots (cloud approach, crash, torus); an orbiting-camera video of the torus; a measured cost per frame and a rendering budget for Phase 10.

**Primary sources:** Phase 2's ray tracer; Rybicki & Lightman 1979 (*Radiative Processes in Astrophysics*, ch. 1: emission and absorption along a ray); Price 2007 (SPLASH paper: interpolating SPH particles to grids); James et al. 2015 (Interstellar black-hole rendering: ray bundles, avoiding flicker); Luminet 1979 (first computed image of a disk around a black hole; a good visual reference).

---

## 8.1 Pipeline per frame

```
frame time t ─► gas state at t (8.2) ─► deposit onto grids (8.3) ─► upload as 3D textures
camera + settings ─► GPU: for every pixel and sub-ray: bend the ray (Phase 2), march through grids (8.4),
                     add background stars at escape ─► HDR image ─► tone map ─► PNG
```

Keep the stages separate and individually testable. The GPU stage gets a CPU float64 twin (Numba) used only for tests (CLAUDE.md: every fast kernel has a readable reference).

---

## 8.2 The gas state at the frame time

Frames are needed at times that are not snapshot times. Linear interpolation of positions cuts corners near the hole (gas at r = 6 sweeps 60° in a sixth of an orbit), so:

- **Default:** take each particle's state (x, p) from the snapshot just before the frame time and advance it with the **gravity-only generalized leapfrog** (Phase 1) to the frame time. Scalar fields (u, h, ρ, luminosity) are blended linearly between the two snapshots. Particles accreted before the next snapshot are advanced until they cross r_acc, then dropped.
- To avoid a visible jump when switching to the next snapshot, cross-fade positions over the last 20% of each interval between the forward-advanced state and the backward-advanced state from the next snapshot.
- **Hero shots option:** re-run a segment of the simulation from a checkpoint with snapshots written exactly at the frame times (`nebula resim --from <checkpoint> --times <file>`). Runs are deterministic (Phase 4 test 5), so this reproduces the run exactly with no interpolation at all.
- Snapshot cadence for production (Phase 10.2a: Δt_snap = clamp(T_orb(r_med)/40, 10, 10⁴)) keeps each interpolation interval to about 1/40 of an orbit of the gas in view.
- Snapshots store x and p in float32 (rounding ≈ 6 × 10⁻⁸ relative). That limits how exactly an interpolated frame can match the simulation; it is far below one pixel, but tests must not ask for more (test 8).

---

## 8.3 Depositing particles onto grids (`render/gasgrid.py`)

The gas spans r ≈ 4 to 5,000 while the torus is ~15 thick at r ≈ 50. No single uniform grid works, so use **nested grids**:

1. **Global log-spherical grid** centred on the hole: r from 2 to r_max (default 2,000; 5,000 for early shots), logarithmic in r; θ and φ uniform (option: θ cells concentrated toward the equator). Default 192 (r) × 96 (θ) × 192 (φ) ≈ 3.5 M cells: cells ≈ 1.4–1.6 wide at r = 50, so the torus thickness spans ~9 cells.
2. **Detail boxes**: Cartesian grids (e.g., 160³) defined per shot, each with a centre path and size: one following the cloud on its approach, one parked on the crash site r_SI ≈ 1,664 at 12°, etc. A single log grid can't resolve the cloud far out (at r = 5,000 its cells are ~100 wide, the cloud's whole radius).

When sampling along a ray, use the finest grid that contains the point (blend over a few cells at box edges to avoid seams).

**Deposited fields** (per grid, float32 during deposition, uploaded as **two float32 RGBA 3D textures**):
- Texture A: log₁₀ ρ, log₁₀ T (mass-weighted), log₁₀ T (T⁴-weighted), log₁₀ of the emission source (or a spare channel).
- Texture B: the three velocity components needed by Phase 9 (γβ in the static frame, see Phase 9) and one spare.

Why not float16 (half precision), which would halve memory: in code units the gas density is ~10⁻¹⁸ to 10⁻¹² (the cloud itself ≈ 6 × 10⁻¹⁵: 0.1 M☉ spread over a radius of 100 M, with M = 4.3 × 10⁶ M☉), far below float16's smallest number (6 × 10⁻⁸), so it would read as zero; and T up to ~10⁹ K overflows float16's maximum (65,504). Logs fix the range, but float16 then keeps only ~3 significant figures (about 1% error in T, 4% in T⁴). Use float32. If GPU memory is ever short, float16 is allowed **only** for log-encoded fields and only after a test shows the image changes by < 0.5%.

Filtering: hardware linear filtering of float32 textures needs the WebGPU feature `float32-filterable`. Phase 0 (Step 0.2) records whether the Iris Plus reports it. If it isn't available, do trilinear interpolation in the shader with 8 `textureLoad` calls (a few lines of WGSL; test it against the CPU twin).

Fields deposited (before the log is taken):
- mass density ρ;
- mass-weighted temperature T (from the EOS) and, optionally, a T⁴-weighted temperature (emission-weighted; avoids dimming where hot thin gas mixes with cool dense gas);
- mass-weighted 3-velocity (for Doppler in Phase 9; store the GN-to-Schwarzschild 4-velocity components defined in Phase 9);
- per-particle luminosity from cooling (optional emission source for the light-curve-consistent look).

**Deposition rule:** each particle spreads its mass over the cells its kernel overlaps, weights = kernel value at cell centres × cell volume, **normalized to sum to 1 per particle**, so mass is conserved exactly even when h is smaller than a cell (then it lands in ≤ 8 cells by trilinear weights).

**Parallel rule:** scattering to cells from many particles races. Use per-thread private grids summed in a fixed order (2 threads × one grid's size; affordable). Deterministic results; test it.

---

## 8.4 Marching along the bent ray

- Integrate u(φ) exactly as Phase 2 (WGSL RK4). Each RK4 step gives two points on the ray in 3D (r = 1/u, position = r(cos φ e₁ + sin φ e₂)).
- Between them, march in straight sub-steps no longer than **half the local cell size** of the finest grid there (log grid: cell size ≈ r Δ ln r). A step that would skip a cell is a bug that shows up as banding.
- **Transfer (front to back from the camera):**
  ```
  Δτ = α Δs                                   (α = absorption coefficient, Δs = path length)
  I += T_trans · S · (1 − e^(−Δτ))           (S = source function = emissivity / α; constant within the step)
  T_trans *= e^(−Δτ)
  ```
  Exact for constant S within a step, and well-behaved from transparent (Δτ → 0: I += T_trans · j Δs) to opaque. Stop early when T_trans < 10⁻⁴.
- At escape, add T_trans × the star-field colour in the escape direction (Phase 2). At capture, add nothing.
- **Phase 8 simplifications** (lifted in Phase 9): Δs is the static observer's proper length √(dr²/(1 − 2/r) + r² dΩ²); no Doppler or redshift; emission from a simple model (e.g., S = σT⁴/π, α = κ_grey ρ with κ_grey chosen so the torus is opaque and the stream partly transparent; document the value).

---

## 8.5 Camera and shots (`render/director.py`)

A **shot file** (TOML) describes one video segment:
- camera keyframes: position (spherical r, θ, φ around the hole, or relative to a named target), look-at target (`hole`, `cloud_com`, `crash_site`, or coordinates), vertical field of view;
- a **time warp**: a monotone spline mapping video time to simulation time, so a few seconds of video can cover the 97-day approach and many seconds the crash;
- exposure keyframes, render settings (resolution, sub-rays per pixel, grid sizes, detail boxes);
- the run id it renders.

Rules: camera stays at r ≥ 6; the camera is a static observer (a moving camera would add aberration; only add it with a test); camera paths are smooth (spline through keyframes, no sudden turns); the frame-time list is computed once and saved with the shot.

---

## 8.6 Tests (synthetic scenes with exact answers)

| # | Test | Requirement |
|---|---|---|
| 1 | Deposition mass | total grid mass = Σm to 10⁻¹² for random particles with h from 0.1 to 10 cells; also near grid edges and across the log grid's φ = 0 seam |
| 2 | Deposition determinism | 1 vs 2 threads: identical grids |
| 3 | Thin emissive sphere, flat space (straight rays, no hole) | intensity vs impact parameter b equals j · 2√(R² − b²) within 1% for b < 0.95 R |
| 4 | Opaque sphere, flat space | uniform disk of intensity S; edge sharp within 1 cell |
| 5 | Uniform slab | I = S(1 − e^(−τ)) for τ = 0.1, 1, 10 within 1% |
| 6 | **Lensed point source** (with the hole): small bright sphere on the axis behind the hole | ring angles: camera D = 50, source r = 30: primary 11.701931°, secondary 5.849751°; source r = 100: 14.958880°, 5.850477°; camera D = 30, source r = 15: 15.114892°, 9.639931°. Within 1 pixel. |
| 7 | GPU vs CPU reference (64 × 64 image of a real snapshot) | per-pixel relative difference < 10⁻³ in linear HDR, excluding pixels within 2 px of the shadow edge (check capture/escape classification separately: identical outside ±10⁻⁴ of critical b) |
| 8 | Interpolation | (a) the interpolation code fed float64 states matches a float64 gravity-only GLF advance to 10⁻¹² relative; (b) hydro-off run: positions interpolated from float32 snapshots at mid-snapshot times match snapshots written exactly at those times by `resim` to 10⁻⁵ of r (float32 rounding is ~6 × 10⁻⁸; the rest is time-step error, which must fall ~4× when η_orb is halved) |
| 9 | Temporal stability | same snapshot, camera moved by 10⁻⁶: image change < 10⁻³ (no per-frame randomness) |
| 10 | Step size | halving the march step changes a real frame by < 0.5% (in linear HDR, mean absolute) |

---

## 8.7 Performance and memory (measure and record)

- Rough budget: a 1080p frame is 2.1 M pixels × 4 sub-rays × ~300 march steps ≈ 2.5 × 10⁹ texture samples, plus CPU deposition (< 1 s for 10k–50k particles). Expect seconds to tens of seconds per frame on the Iris Plus GPU. **Measure**: ms/frame at 720p, 1080p and 4K for 1, 4 and 16 sub-rays; record in DECISIONS.md.
- Planning consequence: a 2-minute video at 30 fps is 3,600 frames. At 20 s/frame that's 20 hours (two nights) at 1080p. 4K costs ~4×: use it for short hero clips (10–20 s) or spread over several nights. Phase 10 handles resumable multi-night renders.
- Every GPU dispatch is tiled to finish in < 1 s (Phase 0: the macOS GPU watchdog kills long dispatches).
- Memory (two float32 RGBA textures = 32 bytes per cell): log grid 192 × 96 × 192 ≈ 3.5 M cells ≈ 113 MB; each 160³ detail box ≈ 131 MB; star cube map ≈ 270 MB with mips (Phase 2). Two detail boxes + log grid + stars ≈ 0.65 GB. Keep total GPU use < 1.5 GB (the Iris Plus shares the Mac's 8 GB). Per-thread deposition buffers on the CPU: 2 threads × 113 MB, fine.

---

## 8.8 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Steps skip cells | banding, stair-steps in the torus | step ≤ ½ cell everywhere (test 10) |
| Flicker between frames | the gas shimmers | kernel (not nearest-cell) deposition; fixed sub-ray pattern; same grid settings every frame; optional motion blur by averaging 2–4 sub-frame times for fast shots |
| Jumps at snapshot boundaries | gas "pops" every N frames | cross-fade (8.2), or `resim` for hero shots |
| Thin stream vanishes or looks dotted | stream thinner than cells | a detail box along it; or accept and say so; never fake a stream |
| Speckle in thin gas | individual particles visible as blobs | expected at low particle number; more particles; kernel deposition; slightly larger minimum smoothing in deposition (rendering only, documented) |
| Opaque torus hides everything | only the torus surface visible | physical for an optically thick torus; adjust κ_grey only in Phase 8's placeholder model; in Phase 9 the opacity is physics |
| Seams at grid boundaries | visible lines | blend zones at box edges; test 1 across the φ seam |
| Out of GPU memory / watchdog reset | crash or black frames | smaller tiles; smaller boxes; check sizes before upload |
| GPU result differs from CPU | test 7 fails | float32 issues near the photon sphere (Phase 2) or a sampling convention (cell centres vs corners); fix, don't loosen |
| Frames too slow | budget blown | fewer sub-rays for previews, early ray termination, coarser log grid far out; measure before optimizing |

---

## 8.9 What Hiten sees

1. The test images (with expected vs measured numbers printed on them).
2. **The first real lensed frame of the torus**: camera at r ≈ 40, slightly above the plane. The money shot.
3. Frames of the cloud approaching (with a detail box) and of the crash site.
4. A 10-second orbiting-camera video of the torus at 720p.
5. The timing table.

---

## Done when

- [ ] Tests 1–10 pass.
- [ ] Real-snapshot frames and the orbit video made and reviewed by Hiten.
- [ ] Per-frame costs recorded; a render budget written for Phase 10.
- [ ] Code structure lets Phase 9 swap the emission model and add redshift without touching the march loop; committed.
