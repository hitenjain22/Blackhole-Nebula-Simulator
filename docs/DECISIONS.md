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

**Update, same night (the cooler blob alone was not enough):** a tracked run with u = 4 × 10⁻⁸ passed pericentre cleanly (t ≈ 13,500; lowest particle angular momentum 6.69, gas heated ~2,000×), but then failed at t = 22,720 in the same way. Cause: at r_p = 25 the tides tear a blob this size apart. The spread of orbital energy across it (~R/r_p² ≈ 0.005) exceeds its binding energy (0.0019), so its most bound debris returns for a second pericentre from t ≈ 19,000 while the centre is still outbound. That debris meets the stream (u jumps 100×), loses its angular momentum (6.7 → 0.15), and falls in. ~~This is real physics (the stream collisions of Phase 6), not a code error.~~ **Corrected 2026-10-15 (review `docs/review-2026-10-15-plunge.md` §2e): this is numerical, not real physics.** The returning debris is so sparse that its SPH smoothing lengths span the hole (2h/r = 5–18, 32–100% of neighbour pairs across the hole), and the artificial viscosity brakes gas moving in opposite directions against each other. With viscosity off the same blob runs to t = 26,000 with every particle's angular momentum above 7.03. The first pass (to t = 18,000) is fully resolved (2h/r ≤ 0.5, no pairs across the hole), so the Phase 4 ledger test stopped there remains valid. With the plan's u = 4 × 10⁻⁶ the swollen blob (size ~100 at pericentre) already collides with itself on the way in, which explains the first failure at t = 12,595.

**Revised decision:** keep u = 4 × 10⁻⁸ and measure from apocentre through pericentre to **t = 18,000** (`blob.T_END_BLOB`), instead of one full radial orbit. This includes the whole squeeze-and-heating episode and the largest gravity error (at pericentre), and stops before any debris returns. Each run checks that no particle's angular momentum falls below 5 (capture happens below 4). Requirements and tolerances are unchanged. **Hiten:** if you would rather keep the plan's full orbit, the alternative is to add accretion (r_acc = 4, with ledger entries) to this test; it is a Phase 5 feature anyway.

## 2026-10-15: Block-step hot-blob ledger: angular momentum (Step 4.5; overnight — please review)

With block steps the hot-blob ledger passes the plan's energy requirements (max error 4.06 × 10⁻⁶ < 10⁻⁵; 1.63 × 10⁻⁵ → 4.06 × 10⁻⁶, ratio 4.0). Total angular momentum drifts by 3.1 × 10⁻¹⁰ (coarse 1.1 × 10⁻⁹): exact conservation (the shared step gives 2 × 10⁻¹⁵) needs every pair force to act on both particles at the same instant, which individual time steps give up by design (GADGET-2 and Phantom have the same property). The plan's 10⁻¹⁰ requirement belongs to its test 4 (shared step) and is kept there. The block-step variant (added in the Phase 4 breakdown) instead checks that the drift is time-step error, falling ≥ 3× per halving (measured 3.5×). Before the exact-neighbour-density fix the drift was 7.5 × 10⁻⁷, so this check is what caught that problem.

**Sedov energy on block steps (same night):** with exact neighbour densities the block-step Sedov energy error converges with the step (32,768 particles, t = 0.002, Courant and force criteria scaled together: +1.05%, +0.27%, +0.085% at Courant 0.3, 0.15, 0.075; similar with the switch off). It is larger than with one shared step (−0.16% at 0.3) because the limiter lets cold neighbours of the hot centre take steps up to 4× the hot particles' steps, and time-step error grows with the square of the step. Plan test 3 checks the shock radius (within 0.2% with the limiter); the energy figure is reported, not tested.

## 2026-10-15: The 1k cloud loses a particle into the hole at t = 386,646 (found in Step 4.5; for Phase 5)

Running the minimal Phase 5 cloud (1k glass particles, radius 100, 10⁴ K, r_a = 5,000 → r_p = 25) for the speed-up test, one particle fell inside r = 2.5 at t = 386,646, before the centre's pericentre (396,224). Tracked: (1) during the approach the stream is stretched (its leading edge reaches pericentre ~11,000 before the centre), compressed and warmed (u 2.3 × 10⁻⁹ → 3 × 10⁻⁷), and pressure forces slowly lower the leading particles' angular momentum (7.25 → 6.3); (2) as the leading edge swings through pericentre (t ≈ 385,000–386,000) it is violently heated (u → 4 × 10⁻³ to 2.6 × 10⁻²), and one particle's angular momentum collapses (5 → 0.26), so it plunges. **The same particle falls in at the same time with one shared time step** (angular momenta agree to 0.1% at t = 386,134), so this is not a block-step error. The plan says a crash before t = 500,000 "means a bug, or a pressure effect worth investigating": this is the first thing Phase 5 must investigate (candidates: crossing of trajectories with different precession, viscosity in the converging stream, resolution of a stream thinner than h). Phase 4's speed-up test stops at t = 385,000, when the leading edge has reached pericentre.

## 2026-10-15: Hiten's review of the overnight decisions

1. **Hot-blob test:** approved as is (cooler blob, run to t = 18,000: the clean coupling check). Once Phase 5 has accretion (r_acc = 4, removed particles written into the ledger), a second version runs the plan's original blob (u = 4 × 10⁻⁶) for one full orbit with gas allowed to fall in, which also checks the accretion bookkeeping. Hiten: gas completing its orbit and then being swallowed is realistic and fine.
2. **Cloud particle falling in at t = 386,646:** investigated as the first step of Phase 5 (Step 5.0), before anything is built on top. Not to be rushed: the heating it shows (u ≈ 0.025, i.e. ~10¹¹ K) is not believable for this set-up and may point to a numerical problem.
3. **Conductivity signal speed with the hole (`COND_VELOCITY`, Phantom Eq. 44):** approved (plan change). Both forms are compared in Step 5.0, since conduction is one possible source of the heating.

Standing guidance from Hiten: take as long as needed (days or weeks) to find and fix problems properly; cross-checking against open-source codes is welcome (check licenses before copying any code: Phantom is GPL).

## 2026-10-15: External review of the plunge; decisions D1–D11 (review: `docs/review-2026-10-15-plunge.md`, scripts: `scripts/review-2026-10-15/`)

**Finding (replaces the planned Step 5.0 investigation):** the cloud plunge, the ~13% angular-momentum drift on the way in and the hot blob's late plunge are numerical, not physical. (1) On the long fall the cold gas (sound speed 5 × 10⁻⁵) is squeezed sideways supersonically; Phantom's viscosity mistakes the smooth squeeze for a shock (the switch is fully on from t ≈ 150,000) and heats and stirs the gas (25–34× too hot, angular momentum smeared 4–6% rms against ~1% from pressure). (2) Near the hole the cloud is under-resolved: at 1k the smoothing lengths span the hole even with viscosity off, so the viscosity brakes gas moving in opposite directions against each other. Gravity, pressure, conduction and block steps are not the cause.

**Confirmed on this Mac (review §2a, §2b), with the review's scripts unchanged:**

| Run (1k cloud, to t = 390,000) | Review | This Mac |
|---|---|---|
| A, gravity only | no plunge; lowest L 7.213; hottest u 2.3 × 10⁻⁹ | identical (7.213; 2.3 × 10⁻⁹) |
| B, viscosity off | no plunge; 7.077; 2.3 × 10⁻⁸ | identical (7.077; 2.3 × 10⁻⁸) |
| C, as in the repo | plunge t = 386,646 (particle 146); 0.26; 2.6 × 10⁻² | identical (t = 386,645.9, particle 146; 0.255; 2.6 × 10⁻²) |
| Across the hole, C (r < 60) | 2h/r 9.1 → 3.0; 90% → 35% of pairs | 9.07 → 3.04; 90% → 35% |
| Across the hole, B | 10.2 → 1.7; 87% → 21% | 9.64 → 1.66; 86% → 21% |

**Decisions (Hiten, 2026-10-15):**
- **D1** approved: the 1k cloud is a smoke test only (does it run, are the ledgers flat), never used for anything near the hole.
- **D2** approved as a method change: slope-limited velocity reconstruction in the artificial viscosity (Frontiere, Raskin & Owen 2017, J. Comput. Phys. 332, 160; Rosswog 2020, MNRAS 498, 4230, MAGMA2 Eqs. 17–18), written as a reference and a Numba version, and every Phase 3 and 4 test rerun with unchanged tolerances. If the block-step angular-momentum check cannot reach its 3× fall per halving, stop and ask Hiten.
- **D4** approved as a scenario change: **cloud radius 25 instead of 100** (phase-05 §5.1 table). A 4× smaller cloud has ~4× smaller smoothing lengths at the same particle count (as good as ~64× more particles near the hole). Trade-off: a narrower stream and less spread-out gas forming the disk; the real G2 cloud is far larger than either, so neither is "more realistic".
- **D5** approved: every run records, for particles at r < 60, the median and largest 2h/r and the percentage of neighbour pairs more than 90° apart around the hole, plus the typical and hottest-1% u at closest approach against the gravity-only ceiling (review §2d); a warning when the across-the-hole percentage exceeds 1%.
- **D6** approved: accretion at r_acc = 4 now, with full ledger entries (m, H, u, x × p); any particle swallowed on its first pass is listed with its history.
- **D7** approved: velocity-based conductivity signal with the hole (`COND_VELOCITY`).
- **D8** approved: the cooler blob stopped at t = 18,000 stays the Phase 4 ledger test (its pass is fully resolved); the warm-blob full-orbit variant with swallowing is **not** run (it would only measure this artefact). This replaces item 1 of the entry "Hiten's review of the overnight decisions".
- **D9** approved: keep Wendland C4, η = 1.6 (the cubic spline's particle fell in even earlier, review run H).
- **D11** approved: fix the gradient-pass crash when the correction matrix is near-singular (h_max capped, too few neighbours): fall back to the uncorrected gradient.
- D3 (resolution) and D10 (speed-up) follow from the runs below: 10k radius-25 with D2 and swallowing, then 50k overnight, temperatures at closest approach compared with the gravity-only ceiling (radius 25: typical ~2 × 10⁻⁸, hottest 1% ~1–3 × 10⁻⁶).

## 2026-10-15: D2 implemented: velocity reconstruction in the viscosity, with a continuous gate (please review the gate)

**What was built** (`physics/viscosity.py`, `sim/sph.py`): the pair's velocity jump is reconstructed to the midpoint with each particle's corrected velocity gradient and the van Leer limiter with the closeness factor (Frontiere, Raskin & Owen 2017 Eqs. 48–53 = the linear part of Rosswog 2020 MAGMA2 Eqs. 17, 21–23, both read from the papers; η_crit = 1/hfact = 0.625, η_fold = 0.2). A readable reference and a Numba version agree to 10⁻¹²; the readable formulas live in `physics/viscosity.py`. On by default (`HydroConfig.reconstruction`). In a smooth supersonic squeeze it removes the viscous heating (2.8 × 10⁻² → 1.6 × 10⁻¹⁹ without the closeness factor, 2.2 × 10⁻⁵ with it); pair forces stay antisymmetric (momentum, angular momentum and energy conserved to round-off).

**The gate: this project's choice, not the papers'.** Which velocity jump drives the viscous pressure q:
- the review's prototype used the reconstructed jump only when both the reconstructed and the actual jump say "approaching". q then snaps off when the actual jump crosses zero. On block steps (pair forces act at different times) this made the angular-momentum drift fall only **1.90×** per halving (the check needs ≥ 3×; reproduced here exactly as in the review);
- MAGMA2's own gate (reconstructed jump only) is continuous but allows negative viscous heating, and in the block-step blob it **drove one particle's internal energy below zero** (run stopped);
- **chosen: the "min-mod" gate**, μ = max(reconstructed, actual) when both are approaching, else 0: the less compressive of the two. It is continuous, never heats negatively, and never gives more viscosity than the plain form. Block-step blob: energy 1.63 × 10⁻⁵ → 4.06 × 10⁻⁶ (ratio 4.01), **angular momentum 1.49 × 10⁻¹⁰ → 2.6 × 10⁻¹¹ (ratio 5.7)**, 12× smaller than the plain viscosity's 3.1 × 10⁻¹⁰. So the review's stop condition (cannot reach 3×) did not arise. All three gates remain selectable (`ShockCapturing.recon_gate`) for the record.

**D11 implemented:** a near-singular correction matrix (fewer than 4 neighbours, or |det R| < 10⁻⁶ (tr R/3)³) falls back to the uncorrected gradient G/(tr R/3) in both gradient passes (switch and reconstruction), instead of crashing. Results elsewhere are bit-identical.

**D4 recorded (scenario change, approved by Hiten):** the cloud radius becomes 25 (was 100). Expected values that scale with the radius (tidal energy spread ±R/r_a², arrival-time spread ~21,800, expansion before pericentre) were computed for radius 100 in expected-values §2/§2b and are not changed; for radius 25 they scale down ~4× and are reported, not tested, until recomputed with a reference script. The 1k cloud is a smoke test only (D1).

**D6 implemented** (`sim/accretion.py`): particles inside r_acc = 4 are removed at the end of their own step (their state is then true, not predicted); mass, m(H + u), m(x × p) and m p go into a removed ledger that `blocks.ledger` adds back. The removal is exact (totals unchanged to 10⁻¹⁴); a plunge on the first approach is flagged (the plan's alarm). **D5 implemented** (`sim/diagnostics.py`): matches the review's across-the-hole numbers exactly on run C. Run folders, manifests and 15-minute checkpoints: `sim/runio.py`, `scripts/cloud_run.py` (resume is bit-identical).

**Hiten's approval of the min-mod gate (2026-10-15), conditional.** Our own choice, not from Frontiere et al. 2017 or Rosswog 2020:

    μ_ab = max(v*_ab · r̂_ab, v_ab · r̂_ab)   if both are negative (approaching), else 0
    q_a = −½ ρ_a v_sig,a μ_ab,   v_sig,a = α_a c_s,a + β |μ_ab|
    du_a/dt (shock heating) = −Σ_b m_b v_sig,a ½ μ_ab (v_ab · r̂_ab) F_ab(h_a) / (Ω_a ρ_a)

where v*_ab is the reconstructed jump (Frontiere Eqs. 48–53). Reason: q must change smoothly with time for the block-step angular-momentum error to shrink as fast as the steps (q switching off abruptly gave 1.9× per halving; this gives 5.7×), must never heat negatively (MAGMA2's gate drove a particle's u below 0), and should never exceed the plain viscosity. Approved on two conditions; if either fails, stop and tell Hiten: (1) the full slow suite passes with tolerances unchanged; (2) the 10k radius-25 run matches the review: 0 swallowed, lowest L about 5, 0% of pairs across the hole, typical gas near the hole about 1.5 × 10⁻⁶.

## 2026-10-15: Hiten's decisions on the Phase 5/6 update list

1. **Phase 6 runs use 10k, not 1k** (R6.1, R6.3a–d, R6.4, R6.5). 1k stays a smoke test only (D1).
2. **Snapshots follow Phase 10.2a's rule from Phase 5 on**: Δt_snap = clamp(T_orb(r_med)/40, 10, 10⁴), T_orb(r) = 2π r^(3/2), r_med = median radius of the gas inside r = 2,000 (all gas if none), plus event snapshots.
3. **Shock marker:** the viscous heating rate after reconstruction replaces "α_AV > 0.5" (the switch's α is ~1 for all the cold gas and no longer marks shocks).
4. After the slow suite passes: update the Phase 5 and 6 files for the other items of the list; no existing expected value is changed.
5. **On hold until Hiten picks the Phase 6 cloud size** (review §7: first crash vs radius; a radius-50 test is running in the review session): the crash timing, a new expected-values table, the tidal-spread check and the scenario-knob note.
6. The 10k radius-25 run goes ahead once the suite passes (it is the check of D2 against the review). The updated review (with §7 and `crash_by_radius.py`) replaced the old copy.

## 2026-10-15: Gresho test baseline after D2 (Hiten approved the recommended option)

With D2 the low-resolution Gresho ratio (switch / constant α = 1) came out at 0.713 against the 0.70 limit, although both errors fell: constant α = 1 from 0.255 to 0.174, the switch from 0.143 to 0.124 (review: 0.123). Reconstruction removes most of the viscosity in the vortex's smooth shear even at α = 1, so the comparison run improved most. That is expected from the sources: Frontiere et al. 2017 find their limited viscosity, without a switch, preserves the Gresho vortex best ("the dominant effect"), and Rosswog 2020 runs every test with α = 1 plus reconstruction. **Decision:** the yardstick stays the *standard* scheme (constant α = 1, no reconstruction), as in the plan's wording; the 0.70 limit is unchanged; and a new check requires that reconstruction does not make the vortex worse (switch + reconstruction ≤ switch alone). Cross-check of the implementation against open source: PySPH's MAGMA2 module has the same midpoint reconstruction, limiter, closeness factor and raw-velocity heating; the differences are its optional second-derivative term and its published gate (ours: min-mod, approved above with conditions).

**Min-mod gate: both of Hiten's conditions met (2026-10-15).** (1) Full slow suite passes with tolerances unchanged (with the Gresho baseline decision above). (2) 10k radius-25 run with D2 and swallowing (`runs/cloud10k_r25_D2`, commit 06b238a, 1 h 43 min), compared with the review's run:

| 10k, radius 25, with the fix | This run | Review | Match? |
|---|---|---|---|
| Swallowed by t = 400,000 | 0 of 9,992 | 0 | yes |
| Lowest angular momentum at the end | 5.13 | 5.2 | yes |
| Pairs across the hole | 0% (every snapshot) | 0% | yes |
| 2h/r near the hole at closest approach | median 0.12, max 0.54 | about 0.1–0.3 | roughly (our max is higher) |
| Typical u near the hole, t ≈ 396,9xx | 1.39 × 10⁻⁶ | 1.4 × 10⁻⁶ | yes |
| Hottest 1% | 8.9 × 10⁻⁶ | 8.6 × 10⁻⁶ | yes |
| Hottest | 4.9 × 10⁻⁴ | 4.6 × 10⁻⁴ | yes |
| Angular-momentum spread at the end | 3.0% rms, worst 1% −8.4% | 3% rms, worst 1% −8% | yes |

Also: centre of mass at pericentre at t = 396,236 (expected 396,224); energy ledger at most 3.3 × 10⁻⁵ during the passage (the gravity integrator's own pericentre error at η_orb = 0.00125 is up to 6.4 × 10⁻⁵), 1.8 × 10⁻⁵ at the end; angular momentum to 10⁻⁹. **Not visible in the review's table:** three hot outlier particles (u ≈ 10⁻³, ~300× the gravity-only ceiling, h ≈ 50 at r ≈ 200–330) had their angular momentum dip temporarily (lowest 1.18) and recover (5.6–8.0 at the end); none came near the hole. The gas temperatures at closest approach remain far above the gravity-only ceiling (typical ~70×), as the review expected at 10k; that is what the 50k run checks (D3).

## 2026-10-16 (night): Hiten's instructions for the night

Hiten (asleep): after the 50k analysis, finish Phase 5 and continue with as much of Phase 6 as possible; **do not start Phase 7**. Decisions are taken with the option judged best and most correct, recorded here; everything is fact-checked against the papers and cross-checked with open-source codes; any failure is diagnosed and fixed, never tolerated or hidden. Items that depend on the Phase 6 cloud size stay on hold (his decision of 2026-10-15).

## 2026-10-16 (night): Phase 6 steps 6.2–6.4 (EOS, opacity, cooling) — overnight decisions

- **Gas + radiation EOS** (`physics/eos.py`): u, P, β_gas, Γ₁, c_s = √(Γ₁P/ρ); T from (ρ, u) by Newton on log T inside a bisection bracket [1, 10¹³] K. Cross-checked with Phantom's open-source `eos_idealplusrad.f90`: the same u and P; Phantom solves for T by Newton without a bracket and uses 1 + P/(ρu) as the sound-speed index. We keep the plan's true adiabatic index Γ₁ (Chandrasekhar 1939; equal to Clayton 1968 Eq. 2-145, (32 − 24β − 3β²)/(24 − 21β), for γ = 5/3, checked algebraically); the two agree in both limits. Tests 1–3 pass first time (all §9 values to 10⁻⁸; round trip 10⁻¹² against the 10⁻¹⁰ limit).
- **Opacity** (`physics/opacity.py`): the plan's free–free coefficient (written from memory) is confirmed as the standard Kramers form 3.68 × 10²² g_ff (1 − Z)(1 + X) ρ T^(−7/2) cm²/g (ρ in g/cm³); SI 3.68 × 10¹⁸ … m²/kg. X = 0.70, Z = 0.02 added to `units.py` (consistent with κ_es = 0.02(1 + X) = 0.034). The plan-time torus gives τ_eff = 3.0 over its thickness, consistent with "a few to ~15". Metal bound-free absorption is not included (τ is a lower bound), as the plan states.
- **Cooling** (`physics/cooling.py`): exact β-cooling toward the floor u_floor(ρ) at 10⁴ K from the active EOS. **Decision:** gas below the floor relaxes *up* to it with the same law (the floor is photo-ionization heating), booked as negative radiated energy so the ledger stays exact; nothing else touches u. Applied in the middle of each particle's step (Strang split) in both the shared-step and block-step integrators; inactive neighbours get the cooling of their elapsed time in their predicted u. Tests 6, 7, 9 pass; with hydro, cooling adds no ledger error of its own (error with cooling / without = 1.09 shared step, 1.02 block steps; both fall ≥ 3.5× per halving).
- **The EOS is now chosen in `HydroConfig.eos`** ("ideal", "gasrad", "radiation" for test 4); one function serves all integrators. The ideal-gas path is bit-identical to before (Sod, Gresho, Sedov checked).

## 2026-10-16 (night): Phase 6 steps 6.1, 6.7 and the control gravity — overnight decisions

- **Escape radius** (`BlockConfig.r_escape`, phase-06 6.4: 50,000): removals logged as "escaped" alongside "accreted", with per-reason totals; ledger exact (test: energy error is the orbits' own, 4.0× per halving).
- **Seeded clouds for R6.5:** `cloud_ic(..., seed=k)` carves the ball from a randomly shifted (the glass is periodic) and randomly rotated glass; seed None keeps the standard cloud bit-for-bit.
- **Orbit elements** (`physics/orbits.turning_points`): the exact Schwarzschild turning points from (E, L) as the roots of 2L²u³ − L²u² + 2u + (E² − 1) = 0. **Bug caught and fixed before commit:** for nearly circular orbits the two outer roots nearly coincide and an eigen-solver can return them as a complex pair, which misclassified circular (ring!) gas as "plunging". Now the largest real root is taken and the other two follow exactly from Vieta's relations, with a round-off-level negative discriminant set to zero. Round trips to 10⁻¹³ (cloud orbit, blob orbit, near-ISCO, near-circular at r = 51).
- **Diagnostics** (`analysis/torus_metrics.py`): eccentricity histogram, ring/torus measures (e < 0.3, r < 200: mass fraction, radius, H/r, extent), unbound fraction, luminosity in W, accretion rate in M☉/yr and Eddington units (Ṁ_Edd = 0.0955 M☉/yr and L_Edd = 5.41 × 10³⁷ W reproduced; σ_T added to `units.py`), temperature histogram, optical-depth fractions.
- **Shock marker (Hiten's decision 3):** the force pass now also returns the viscous heating rate alone (bit-identical other outputs). **Threshold (overnight decision):** a particle is "in a shock" when that rate exceeds Ω_K(r) u, i.e. viscous heating would double its internal energy within one orbital time unit at its radius: scale-free, no tuning per run.
- **Newtonian control gravity** (`CoupledConfig.gravity = "newton"`, for R6.4): H = p²/2 − 1/r with an exact symplectic kick-drift-kick (reference + Numba, agree to 10⁻¹⁴); both integrators now get gravity through one set of helpers (GN path bit-identical). Kepler period to 5 × 10⁻⁸; energy error 4.0× per halving; precession −0.00001° per orbit against GN's 24.0398° from the same start.

## 2026-10-16 (night): Phase 6 tests 4–5 set-up decisions

- **Test 5 geometry:** slabs 1.0 long each (tube length 2, 32,768 particles), t_end = 0.6: the shocks (v/3 at most) would reach the slabs' receding outer ends only at t = 0.75. (A first version with 0.5-long slabs let the shocks overrun the slabs before the measurement; see AI_LOG.)
- **Shock speed** is measured between t_end/2 and t_end, so the shock's formation time is not counted. Result at the usual step (γ = 5/3): compression 3.930 (exact 4), post-shock u 0.509 (exact 0.5), |v| 0.012, shock speed 0.3341 (exact 1/3, +0.24%): D2 captures the strong shock correctly. γ = 4/3 (radiation limit): compression 6.91 (exact 7), u 0.506, |v| 0.018, shock speed 0.1648 (exact 1/6, −1.1%), all PASS.
- **Energy (1e-6):** this is the time-stepping error of kick-drift-kick through a strong shock (4.3 × 10⁻⁴ over the full run at the usual step; at t = 0.05: 1.5 × 10⁻⁵, 1.4 × 10⁻⁷, 2.6 × 10⁻⁸ at 1, 1/4, 1/16 of the step). As with Phase 3's energy test (Courant/64), the energy requirement is tested with small steps: 1/16 of the usual step, on a run to t = 0.15 (shocks fully formed): measured 3.5 × 10⁻⁸ (PASS, 63 min). The full-length run at 1/16 step would take several hours per test.

## 2026-10-16 (morning): 10k run extended past t = 400,000: tail particles swallowed (numerical)

- **Seen:** between t = 404,098 and 404,338 three particles (6019, 7881, 5791) were swallowed, each with L falling 7.39 → ~3.5 and u ~10⁻² (≈10¹¹ K).
- **Diagnosis:** they belong to the stream's sparse trailing tail (63 particles within r = 100 at t = 401,090, 0.6% of the mass). Density there is 10⁻⁹–10⁻¹⁰, so h grows to 35–140 at r = 27–83 (2h/r up to 3.6; the D5 warning fired: 23% of pairs across the hole). They exchange pressure forces with already-shocked gas (u ~10⁻⁴) up to ~200 away, lose angular momentum steadily over ~3,000 time units, and heat ~100×. Same failure the review identified (kernels larger than the distance to the hole), now in the under-resolved tail rather than the body of the stream. The 50k run's tail at t = 400,000 is better resolved (2h/r max 0.51, 0% across).
- **Decision (overnight, Claude):** let the extension run to t = 500,000 unchanged and account for every removal in the ledger. It measures how much mass this artifact affects (so far 0.03%). No method change without Hiten (rule 7). Options for him: accept it (ledgered, tiny mass); run Phase 6 at higher resolution; or a method change near the hole (e.g. a cap on h relative to r, or treating particles whose kernel spans the hole specially), which would need its own literature check and tests.

## 2026-10-16: D4 updated: Phase 6 cloud radius 60 (Hiten's decision)

Source: review §7 (`docs/review-2026-10-15-plunge.md`, updated 16 Oct; runs with `scripts/review-2026-10-15/run_cloud_acc.py`, crash times with `crash_by_radius.py`).

- **Phase 6 and later use cloud radius 60.** It is the smallest cloud whose first stream crash falls inside Phase 6 (passage 4, t ≈ 2.8 million, 10% of the stream still incoming at first contact). Radius 25 crashes only at passage 8 (t ≈ 5.9 million); radius 75 and 100 lose far more gas at 10k.
- **Radius 25 stays only for the 10k check run** (`runs/cloud10k_r25_D2`) that compares the min-mod D2 with the review. Phase 5's own expected values are not changed.
- **Known numerical effect: ~0.13% of the gas is lost at 10k, radius 60** (review: 3 swallowed + ~10 below the capture line L < 4 after the first pass, most in a late wave at t ≈ 401,500–403,000 as the back of the cloud rounds the hole). Same cause as the tail particles in the radius-25 extension (entry above): smoothing lengths larger than the distance to the hole where the gas is sparse. Every lost particle is in the removal ledger. Hiten accepted Claude's recommendation: accept it, track it, and check that it shrinks with resolution; no method change near the hole.
- **Check (resolution):** the lost fraction (swallowed + below L < 4) after the first pass must be smaller in a 50k radius-60 run than in the 10k one. Implemented as `scripts/loss_check.py` (see Phase 6 file). Every run's log reports the lost fraction; a run where it exceeds 1% of the gas is flagged for Hiten (threshold: Claude's choice, Hiten may change it).

## 2026-10-16: Phase 6 end time at radius 60: t = 3.8 × 10⁶ (Hiten approved Claude's suggestion)

- **Test-particle first contact** (expected-values §2c): radius 60 at t = 2,765,139 (passage 4); the plan's radius 100 at t = 1,964,230 (passage 3).
- **Time left before the planned end t = 3.0 × 10⁶:** radius 100 left 1,035,770 after first contact (1.3 cloud orbits: the rest of the crossing passage and the whole next passage, 4, arrive inside the run). Radius 60 leaves only 234,861: about 23% of that margin, under 2 orbits of the post-crash gas (period ~1.4 × 10⁵, §6.1), and passage 5 (t ≈ 3.56 × 10⁶) falls outside. Circularization takes several crash cycles, so **by the test-particle estimate 3.0 × 10⁶ is not enough to watch the disk start forming.**
- **Suggestion (Claude):** end at **t = 3.8 × 10⁶**, the plan's own margin after first contact (2,765,139 + 1,035,770 = 3,800,909), keeping the plan's early stop "torus formed + 10 ring orbits" and checkpoints (a run that forms its torus early stops early). Caveat in the other direction: pressure adds energy spread (Phase 5: ±14% at 50k), and at radius 60 passage 3 misses by only 8,700, so the hydro crash may well start at passage 3 (t ≈ 2.0 × 10⁶), when 3.0 × 10⁶ would already be enough. Cost: the extra 0.8 × 10⁶ lies in the expensive part (crashes, torus); measure on R6.1 before the rest.
- **Decision (Hiten, 2026-10-16):** end at t = 3.8 × 10⁶ ("better accurate than rushed"), with the early stop and checkpoints unchanged. Phase file updated (R6.1 and, through "same", R6.2–R6.5; run budget; risk table).
