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

## 2026-10-15: Gresho criterion, Sedov constant, and Sedov after the speed-up (Step 3.7; decided overnight, Hiten asleep)

- **Gresho "substantially":** the plan asks the switch to "cut the error substantially". Made quantitative as: velocity L1 at t = 1 with the switch ≤ 70% of the L1 with constant α = 1, and falling with resolution. Measured: 0.143 / 0.255 = 0.56 (16 per unit length), 0.091 / 0.195 = 0.47 (32 per unit); switch L1 0.143 → 0.091 with resolution. The vortex is 2D, run as a slab periodic in z (thickness 1/2).
- **Sedov constant verified:** new script `reference/scripts/22_sedov_constant.py` integrates the self-similar equations: ξ₀ = 1.151666 for γ = 5/3 (quoted 1.1517), 1.032777 for γ = 1.4 (classic 1.033). Noted in expected-values §4 without changing the value.
- **Order:** the Sedov blast needs ~110,000 particles (R ≥ 10 h inside the periodic box) with one global time step; at the measured speed (~7 minutes per time unit for 16,000 particles) that is many hours. Step 3.8 (speed) is done first, then the Sedov test. Nothing is skipped.

## 2026-10-15: SPH speed work and measured speed (Step 3.8; done overnight)

Profiling first (plan: "measure before optimizing"): the cKDTree neighbour lists took ~4× longer than the density and force passes together. Changes, each tested for identical results:
1. **Numba cell-grid neighbour search** (`find_neighbours_numba`), identical lists to the cKDTree reference in 6 cases (open/periodic, a 16:1 periodic box, isolated particles, a 100× smoothing-length range). ~1.5–2.7× faster than the cKDTree.
2. **List reuse between steps** (`NeighbourCache`): lists built with a search margin are reused while provably complete (each particle's usable margin shrinks by the largest displacement since the build; the density solver flags any particle that would exceed it, and the lists are rebuilt). Margin 1.4 (fastest of 1.25 / 1.4 / 1.6 on the Gresho and Sod set-ups); rebuilt on ~10–30% of steps.
3. **Distances computed once per particle** in the density solver (not per Newton iteration).
4. **Tightened lists** for the force and gradient passes (only r < 2 max(h_i, h_j)): bit-identical forces, ~half the pairs. Courant's v_sig,max is now taken over interacting neighbours only (as intended).
5. The density solver's list growth was made more robust (doubling, up to 12 rounds), after a test with 25× undersized starting h failed with the old 1.5× / 8 rounds.

**Measured** (`outputs/checks/sph_benchmark.txt`; varies ±15% with the Mac's temperature): density + force passes ≈ 70,000–83,000 particle-updates/s at 8k and ≈ 58,000–67,000 at 49k; a full hydro step (with the switch's gradient pass and list rebuilds) ≈ 30,000 updates/s at both sizes. Gresho vortex at 16k: 0.53 s per step (was ~1.8 s).

**Against the plan's target (≥ 2 × 10⁵ updates/s at 10k for density + force):** not met; ≈ 2.5–3× short. Most of the gap is the kernel: Wendland C4 has ≈ 2.4× the cubic spline's neighbours (Hiten's Step 3.6 decision, for accuracy). Not pursued further now (plan: "if far below, profile and fix before Phase 5"; the plan's main rule is never to trade accuracy for speed). Remaining options, in order of payoff: (a) Phase 4's block time steps (most particles in the real problem take far fewer steps; the plan's main speed lever); (b) merge the gradient pass into the force pass; (c) a faster list build (single pass, per-thread buffers). Phase 4–7 time estimates must use these measured rates.

## 2026-10-15: Sedov blast settings and the run cache (Step 3.7b; decided overnight, Hiten asleep)

- **Background energy:** the plan says "u_background ~10⁻⁶ of the blast". The first set-up gave the cold gas u = 10⁻⁶ × the hottest particle's u, which added up to 1.2% of the blast energy (it is spread over 110,592 particles). Now the background is u = 10⁻⁶ per unit mass, i.e. total background energy 10⁻⁶ of E = 1.
- **Step size for the Sedov test: Courant 0.075 (a quarter of 0.3).** With the usual 0.3 the shock radius passed easily (−0.5% to −0.6%, limit 5%), but total energy fell by 0.26% (my own extra check asks for < 0.1%; the plan only asks for the radius). The loss happens in the first few steps of the explosion and is time-step error, not a bug: on 32,768 particles it was 0.16%, 0.064% and 0.020% at Courant 0.3, 0.15 and 0.075 (falling ~3× per halving), and constant afterwards. Following the Step 3.5 precedent (conservation tests use smaller steps; tolerances are never loosened), the test runs at 0.075. Phase 4's block steps give the hot centre its own short steps, which is the real fix in production.
- **Run cache:** the 110,592-particle run (~8 minutes at Courant 0.3, ~4× that at 0.075) is shared by the slow test and the plot script through `testproblems.cached`, keyed by a hash of the source files the run depends on (`FINGERPRINT_FILES`) and the stored glasses. Any change to those files makes the test recompute; a cached result is never reused for different code.

## 2026-10-15: Conductivity signal speed with the hole (Phase 4; decided overnight)

Open question from Step 3.5. **Decision:** runs with the black hole use Phantom's velocity-based conductivity signal speed |v_ab · r̂_ab| (`COND_VELOCITY`, Phantom Eq. 44), the form Phantom itself uses whenever external forces act; the pressure-based one conducts heat in gas held up against gravity. The Phase 3 test problems (no gravity) keep the plan's pressure-based form. `CoupledConfig` defaults to `COND_VELOCITY`.

## 2026-10-15: Block time-step design choices (Steps 4.1–4.4; decided overnight)

- **Steps stored as [n_beg, n_end] on the integer timeline** (int64 ticks, K_MAX = 40), not as "level + phase". Every particle step is a plain kick–flow–kick of length L = n_end − n_beg; with one level for all particles the scheme reproduces the Step 4.2 shared-step scheme bit-for-bit (tested).
- **Inactive neighbours are predicted from their step start** with a gravity-only GLF flow over the elapsed time (plan 4.1), not by drifting everybody step by step. Momentum and internal energy use the standard predictors (p_flow + a(s − L/2), u_half + du/dt(s − L/2)).
- **Density, h and Ω of inactive neighbours are computed exactly**, every system step, for every neighbour of an active particle (forces still only for the active ones). First tried: GADGET-2's predictor ρ e^(−∇·v s). It fixed the plan's mixed-level sound wave (error 35% above the one-level run → 4% below), but a stricter check showed the scheme was still only *first order* in energy on mixed levels: an isolated ball with random motions, half the particles 4× shorter steps, energy error falling only 2× per halving of the step. Cause, found by switching pieces off: (1) the gradient-pass ∇·v is not exactly the rate at which the SPH density sum changes, and (2) the neighbours' grad-h factor Ω is stale. With exact neighbour densities and Ω, pure pressure is cleanly second order (3.9×, 3.95×, 3.97× per halving), as is the full scheme with constant α. With the Cullen–Dehnen switch on, a first-order residue remains (an inactive neighbour's α is held at its last value while its own heating uses the updated value); this is inherent to individual steps with a switch that jumps, and is small (measured below).
- **Only active particles are computed:** the density, gradient and force passes and the list tightening take an optional `rows` argument; row for row they are bit-identical to the full passes (tested), and with `rows` omitted the old results are reproduced bit-for-bit (checked on Sod, Gresho and Sedov before the change was kept). Reused neighbour lists are also checked to be complete for every inactive neighbour.
- **Limiter and wake-up:** neighbouring levels differ by at most 2 (Saitoh & Makino 2009). An inactive neighbour more than 4× slower than an active particle is woken by Durier & Dalla Vecchia's App. C rule (end no later than n + T′, opening kick corrected by a(L′ − L)/2). Their App. D ("time-step update" for instantaneous energy injection) is not needed until a feedback-like event exists (none in this project so far); noted for later phases.
- **Mixed-level Sod and sound wave tests** (plan test 2): half the particles forced 2 levels shorter. Marked slow (~70 s together), like Phase 3's Sod test.

## 2026-10-15: Hot-blob ledger test: blob 100× cooler than the plan's (Step 4.2; decided overnight — please review)

**What the plan says** (phase-04 test 4): u = 4 × 10⁻⁶ (sound speed ≈ 2 × 10⁻³), "enough pressure to matter, but not so much that the blob blows apart during the orbit". The expected-values numbers for this orbit come from a gravity-only calculation, so the second half of that sentence was never checked.

**What happened:** with u = 4 × 10⁻⁶ the blob has no self-gravity to hold it, so it expands freely at up to ~3 c_s. By t ≈ 10,000 its smoothing lengths had grown from ~1 to ~20–60 (a diffuse cloud, not a blob), and the expansion changed some particles' angular momentum from 7.20 to as low as 3.18. Below 4 the hole captures a particle: one did, and the run stopped at t = 12,595 with "particle inside r = 2.5" (the integrator's refusal, working as intended). Eight of 505 particles had dropped below 5.

**Decision (recommended option, taken while Hiten was asleep):** keep the plan's intent, change the number: u = 4 × 10⁻⁸ (sound speed ≈ 2 × 10⁻⁴). Free expansion then changes angular momentum by at most ~4%, and the blob only grows a few times its size by pericentre. Every requirement of the test (ledger < 10⁻⁵, factor 3–5 per halving, angular momentum < 10⁻¹⁰, wrong coupling flat and much larger) is unchanged, and each run now records the lowest particle angular momentum and the largest smoothing length so that it proves the blob held together. The alternative, removing particles that come within r_acc = 4 (Phase 5's accretion, with ledger entries), was not chosen because the plan says this test runs without accretion.

**Update, same night (the cooler blob alone was not enough):** a tracked run with u = 4 × 10⁻⁸ passed pericentre cleanly (t ≈ 13,500; lowest particle angular momentum 6.69, gas heated ~2,000×), but then failed at t = 22,720 in the same way. Cause: at r_p = 25 the tides tear a blob this size apart. The spread of orbital energy across it (~R/r_p² ≈ 0.005) exceeds its binding energy (0.0019), so its most bound debris returns for a second pericentre from t ≈ 19,000 while the centre is still outbound. That debris collides with the stream (u jumps 100×), the collision cancels its angular momentum (6.7 → 0.15), and it falls in. This is real physics (the stream collisions of Phase 6), not a code error. With the plan's u = 4 × 10⁻⁶ the swollen blob (size ~100 at pericentre) already collides with itself on the way in, which explains the first failure at t = 12,595.

**Revised decision:** keep u = 4 × 10⁻⁸ and measure from apocentre through pericentre to **t = 18,000** (`blob.T_END_BLOB`), instead of one full radial orbit. This includes the whole squeeze-and-heating episode and the largest gravity error (at pericentre), and stops before any debris returns. Each run checks that no particle's angular momentum falls below 5 (capture happens below 4). Requirements and tolerances are unchanged. **Hiten:** if you would rather keep the plan's full orbit, the alternative is to add accretion (r_acc = 4, with ledger entries) to this test; it is a Phase 5 feature anyway.

## 2026-10-15: Block-step hot-blob ledger: angular momentum (Step 4.5; overnight — please review)

With block steps the hot-blob ledger passes the plan's energy requirements (max error 4.06 × 10⁻⁶ < 10⁻⁵; 1.63 × 10⁻⁵ → 4.06 × 10⁻⁶, ratio 4.0). Total angular momentum drifts by 3.1 × 10⁻¹⁰ (coarse 1.1 × 10⁻⁹): exact conservation (the shared step gives 2 × 10⁻¹⁵) needs every pair force to act on both particles at the same instant, which individual time steps give up by design (GADGET-2 and Phantom have the same property). The plan's 10⁻¹⁰ requirement belongs to its test 4 (shared step) and is kept there. The block-step variant (added in the Phase 4 breakdown) instead checks that the drift is time-step error, falling ≥ 3× per halving (measured 3.5×). Before the exact-neighbour-density fix the drift was 7.5 × 10⁻⁷, so this check is what caught that problem.

**Sedov energy on block steps (same night):** with exact neighbour densities the block-step Sedov energy error converges with the step (32,768 particles, t = 0.002, Courant and force criteria scaled together: +1.05%, +0.27%, +0.085% at Courant 0.3, 0.15, 0.075; similar with the switch off). It is larger than with one shared step (−0.16% at 0.3) because the limiter lets cold neighbours of the hot centre take steps up to 4× the hot particles' steps, and time-step error grows with the square of the step. Plan test 3 checks the shock radius (within 0.2% with the limiter); the energy figure is reported, not tested.

## 2026-10-15: The 1k cloud loses a particle into the hole at t = 386,646 (found in Step 4.5; for Phase 5)

Running the minimal Phase 5 cloud (1k glass particles, radius 100, 10⁴ K, r_a = 5,000 → r_p = 25) for the speed-up test, one particle fell inside r = 2.5 at t = 386,646, before the centre's pericentre (396,224). Tracked: (1) during the approach the stream is stretched (its leading edge reaches pericentre ~11,000 before the centre), compressed and warmed (u 2.3 × 10⁻⁹ → 3 × 10⁻⁷), and pressure forces slowly lower the leading particles' angular momentum (7.25 → 6.3); (2) as the leading edge swings through pericentre (t ≈ 385,000–386,000) it is violently heated (u → 4 × 10⁻³ to 2.6 × 10⁻²), and one particle's angular momentum collapses (5 → 0.26), so it plunges. **The same particle falls in at the same time with one shared time step** (angular momenta agree to 0.1% at t = 386,134), so this is not a block-step error. The plan says a crash before t = 500,000 "means a bug, or a pressure effect worth investigating": this is the first thing Phase 5 must investigate (candidates: crossing of trajectories with different precession, viscosity in the converging stream, resolution of a stream thinner than h). Phase 4's speed-up test stops at t = 385,000, when the leading edge has reached pericentre.
