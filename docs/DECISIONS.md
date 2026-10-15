# Decisions

Agreed changes and choices not fixed by the plan, with the reason (CLAUDE.md rule 7). Newest last.

---

## 2026-10-14: Python 3.12 pinned (Step 0.1)

Pinned with `uv python pin 3.12` (`.python-version`). 3.12 is the newest Python with Intel-Mac (x86_64) wheels for every package we need, in particular Numba/llvmlite (below). Fallback if a later step fails: 3.11.

## 2026-10-14: Numba pinned to 0.62.x (Step 0.1)

Numba 0.63 and later depend on llvmlite ≥ 0.46, and llvmlite published no `macosx_*_x86_64` wheels after **0.45.1** (checked on PyPI's file lists). Without a wheel, uv tries to build llvmlite from source, which needs LLVM and fails on this Mac. So `pyproject.toml` requires `numba>=0.62.1,<0.63` (llvmlite 0.45.1).

Consequence: Numba 0.62 supports NumPy only up to 2.3, so uv resolved **NumPy 2.3.5** (not the latest 2.5). Never run `uv lock --upgrade` without checking this pin still holds.

Resolved versions at the time: numba 0.62.1, llvmlite 0.45.1, numpy 2.3.5, scipy 1.18.1, matplotlib 3.11.2, wgpu 0.32.0, imageio 2.38.1, imageio-ffmpeg 0.6.0, pytest 9.1.1, ruff 0.17.0 (exact versions are locked in `uv.lock`).

## 2026-10-14: Numba speed calibration (Step 0.3)

Stand-in neighbour sum (`nebula._bench.pairs`, `scripts/numba_bench.py`), ~55 neighbours per point, short burst (not sustained; Phase 10 measures throttling over 60 minutes):

| N | 1 thread | 2 threads | 4 threads | speed-up at 4 |
|---|---|---|---|---|
| 10,000 | 1.67 × 10⁷ pairs/s | 3.33 × 10⁷ | **4.33 × 10⁷** | 2.59× |
| 50,000 | 1.31 × 10⁷ | 2.53 × 10⁷ | **3.08 × 10⁷** | 2.35× |

Results were bit-identical for 1, 2 and 4 threads. 50k is slower per pair, probably because its data no longer fits in the CPU cache (not investigated). Phases 3–7 recalibrate their time estimates from the 4-thread figures; the real SPH rate is measured in Phase 3 (§3.4).

## 2026-10-14: GPU check (Step 0.4)

wgpu 0.32 on the Iris Plus (Metal) works. Check pattern (2,000 loop iterations per pixel) at 1280 × 720 in 256 × 256 tiles: **~150 ms per image, ≤ 12 ms per tile** (first tile ~0.3 s including shader compilation). GPU matches the float32 CPU reference to 1.5 × 10⁻⁶ (tolerance 10⁻⁵). No fallback needed; Phase 2 proceeds on the GPU path. Readback uses `queue.read_buffer` once per tile.

## 2026-10-14: "Relative error" of a vector means per vector (Step 1.2)

For vector results (velocities, accelerations), a relative tolerance is applied to the length of the error vector divided by the length of the vector (`check(..., per_vector=True)`), not to each x, y, z component separately. Component-wise relative error depends on how the coordinate axes happen to point: a component that is nearly zero makes a perfect result look bad. Evidence from the first run: v → p → v round trip 4.0 × 10⁻¹⁶ per vector but 7.9 × 10⁻¹⁴ on one component 3,000× smaller than its vector; Hamilton's-equations check 1.1 × 10⁻⁸ per vector, 1.0 × 10⁻⁵ on one component 2,700× smaller than its vector. The tolerance numbers themselves are unchanged. Scalars are still compared element by element.

## 2026-10-14: GN formulas checked against the paper (Step 1.2)

`physics/gravity.py` was checked against Tejeda & Rosswog 2013 (arXiv:1303.4068): the Lagrangian (Eq. 2.11; Cartesian form A.4) and the Cartesian acceleration (Eq. A.5) match the plan's formulas with r_g = GM/c² = 1. The independent finite-difference test was confirmed to catch three deliberate typos (a 3 → 2 coefficient, a sign flip, a wrong power).

## 2026-10-14: Orbit time-step parameter η_orb = 0.00125 (Step 1.4)

Requirement (phase-01 Step 1.4, test 4): over 3 radial periods of the cloud orbit (r_p = 25, r_a = 5,000) with quantized adaptive steps η r^(3/2), |ΔH/H| < 10⁻⁴. Tested as the **maximum** over the run (the leapfrog's energy error swings up at each pericentre and returns; expected-values §2 warns to test the maximum).

| η | steps (3 orbits) | max \|ΔH/H\| | end \|ΔH/H\| | radial period error | precession |
|---|---|---|---|---|---|
| 0.005 (plan's suggestion) | 10,295 | 1.05 × 10⁻³ ✗ | 4.2 × 10⁻⁵ | 2.6 × 10⁻⁵ ✗ | 24.0395° |
| 0.0025 | 20,577 | 2.6 × 10⁻⁴ ✗ | 5.5 × 10⁻⁶ | 2.3 × 10⁻⁶ | 24.0397° |
| **0.00125** | 41,155 | **6.4 × 10⁻⁵** ✓ | 5.1 × 10⁻⁷ | 5.6 × 10⁻⁷ | 24.0398° |

The plan's "η = 0.005 gave 4 × 10⁻⁵" was the end-of-run value; the maximum is 25× larger. Relative to |H| the error is amplified because the orbit is nearly parabolic (|H| = 2 × 10⁻⁴ while the potential at pericentre is 0.04). **Chosen: η_orb = 0.00125** (also the value Phase 4's ledger test already specifies). Consequence: ~13,700 orbit steps per radial orbit per particle instead of the ~3,400 the plan assumed; Phase 4–7 cost estimates must use this. Option to revisit in Phase 4 (via rule 7, with tests): time-symmetric step selection (Hut, Makino & McMillan 1995), which may allow a larger η. Fixed-step tests use η = 0.02 for the circular orbit (plan's reference value; max |ΔH/H| 5.6 × 10⁻⁹ over 1,000 orbits).

## 2026-10-14: ISCO demo uses a radial push, not a slow-down (Step 1.5)

The plan's ISCO demo ("circular orbits at r = 6.2 and 5.8 with a 0.1% velocity nudge") doesn't say which way to nudge. A 0.1% slow-down makes **both** particles plunge: it lowers L² below 12, the minimum for any circular orbit (r = 6.2: L² 12.0125 → 11.9885), so the stable "valley" outside r = 6 disappears. Used instead: the same small inward radial push for both (radial velocity 0.1% of the orbital speed, L unchanged). Result: r = 6.2 oscillates between 6.165 and 6.234 to t = 2,500; r = 5.8 plunges at t ≈ 320. The plot title states the push.

## 2026-10-14: Star field look (Step 2.4; artistic, not physics)

Procedural sky (no catalogue): 100,000 stars, magnitudes −1 to 8 with N(<m) ∝ 10^(0.5 m), temperatures 3,000–30,000 K (placeholder colours until Phase 9), Gaussian spots of 1.5 texels; cube map 2048² × 6 faces, rgba16float with mips, 268 MB on the GPU (plan: ≈ 270 MB). **Milky Way band included (Hiten's choice).** Brightness chosen by eye from a comparison of 0.08 / 0.6 / 15 / 40: **15** (soft glow; 40 looks foggy). Preview exposure: the 99.98th-percentile pixel luminance maps to display 0.5 (the 99th percentile over-exposed faint stars into "snow"). Final shots set exposure by keyframes (Phase 9); star brightness is an artistic multiplier, labelled in the credits (phase-09 §9.5). Building the map takes ~25 s; worth caching once videos need it.

## 2026-10-14: Renderer timing and Phase 8–9 budget (Step 2.6)

Star field + black hole (no gas), Iris Plus GPU, `scripts/render_benchmark.py`:

| Resolution | Rays/px | GPU render | Tone map + encode (CPU) | Per frame | 2-min video (3,600 frames) |
|---|---|---|---|---|---|
| 1280×720 | 1 | 0.10 s | 0.41 s | 0.51 s | 31 min |
| 1280×720 | 4 | 0.16 s | 0.46 s | 0.62 s | 37 min |
| 1280×720 | 16 | 0.46 s | 0.30 s | 0.76 s | 46 min |
| 1920×1080 | 1 | 0.21 s | 0.76 s | 0.97 s | 58 min |
| 1920×1080 | 4 | 0.41 s | 0.71 s | 1.12 s | 67 min |
| 1920×1080 | 16 | 1.02 s | 0.71 s | 1.73 s | 104 min |

Slowest GPU tile 0.074 s (limit ~1 s): 256 × 256 tiles are fine. Not yet measured: 4K (~4× 1080p).

**Notes.** (1) At low ray counts the CPU tone-mapping step costs more than the GPU render; moving it to the GPU (Phase 9 or 10) would cut ~0.4–0.7 s per frame. (2) Phase 8–9 estimate: the plan expects gas to make each ray ~10–50× more expensive. At 1080p, 4 rays/px that is ~4–20 s per frame of GPU time, plus < 1 s of particle deposition: a 2-minute video takes ~4–20 hours, i.e. one or two overnight renders, consistent with the plan's budget (phase-08 §8.7: ~20 s/frame, two nights).

## 2026-10-14: Shock capturing and the energy-conservation test (Step 3.5)

**Formulas** copied from Phantom (Price et al. 2018) Eqs. 39–54: Monaghan signal-velocity viscosity with α on the sound-speed part only and β = 2; viscous heating on approaching pairs only (matches the viscous work exactly); Price 2008 conductivity with α_u = 1; Phantom's variant of the Cullen & Dehnen switch (α_max = 1, σ_decay = 0.1, Balsara-type limiter, backward-Euler decay), gradients with the linear-exact matrix correction.

**Plan test 8** (isolated box, full scheme, energy < 10⁻⁶ over 1,000 fixed steps). The equations conserve energy exactly at every instant (measured 7 × 10⁻¹⁸ relative); what remains is the second-order time-stepping error of the kick-drift-kick integrator with predicted velocities, falling 3.8× per halving of dt. Setup: an isolated sphere cut from the glass (738 particles), random velocities 0.3 c_s, u in 0.5–1.5. Max |ΔE/E| over 1,000 steps: 2.1 × 10⁻⁶ at dt = Courant/16 (fails), **5.0 × 10⁻⁷ at Courant/32 (passes)**. Chosen: Courant/32 for this test. Caveat: 1,000 such steps cover the time of ~31 normal steps. Production runs use Phase 4's block steps and their own ledger tests (phase-04 test 4).

**Open for Phase 4/5 (rule 7, to agree with Hiten):** Phantom uses a different conductivity signal speed when external forces act (|v_ab · r̂_ab|, its Eq. 44, after Wadsley et al. 2008) because the pressure-based one conducts heat spuriously in hydrostatic gas held up by gravity. Both are implemented (`COND_PRESSURE`, `COND_VELOCITY`); Phase 3 tests use the plan's pressure-based form.

## 2026-10-14: Default SPH kernel changed to Wendland C4 (Step 3.6; Hiten's decision)

**What the plan said:** cubic spline, η = 1.2 (≈ 58 neighbours) by default; Wendland C4 "as an alternative selected by config" (phase-03 §3.1).

**What happened:** plan test 4 (sound wave, amplitude 10⁻⁴, on a relaxed glass) requires the L1 error to decrease with resolution. With the cubic spline it did not: slice-averaged L1 6.2% at 32 particles per wavelength, 7.7% at 64. Diagnosis: (1) the loss is not time-step error (halving dt changed nothing) and not dissipation (≈ 6% height loss per period with viscosity and conductivity both off); (2) on a regular lattice the same wave keeps its height (99.3% → 99.9%), so it is scattering by the glass's disorder; (3) the cubic-spline glass cannot be relaxed below a noise floor of 0.34% (50, 400 or 1,500 extra settling steps: 0.34–0.39%), 34× the wave's amplitude. This is SPH's known resolution-independent ("E0") error, which depends on neighbour number, not particle count.

**Wendland C4, η = 1.6 (≈ 137 neighbours):** glass noise floor 0.13%; L1 2.3% → 1.3% (32 → 64 per wavelength; decreasing, as required); phase error 0.32% → 0.17% (limit 1%). Cost: ≈ 2.4× more neighbour pairs per step.

**Decision (Hiten, 2026-10-14):** Wendland C4 with η = 1.6 is the project default (`physics.kernels.DEFAULT_KERNEL`). Both glasses in `data/` were regenerated with it (noise 0.13%). The cubic spline stays available by name. The Sod shock tube passed with the cubic spline (plateaus within 1.5%) and is re-run with Wendland C4.

**Update to the Step 3.5 energy test after the kernel change (2026-10-14):** with Wendland C4 (larger h, so a longer Courant step) and the regenerated glass, the 1,000-step test gave max |ΔE/E| = 1.16 × 10⁻⁶ at Courant/32 (fails) and **2.6 × 10⁻⁷ at Courant/64 (passes)**; the step is now Courant/64. Same caveat as before: the equations conserve energy to round-off; this test bounds the time-stepping error only.
