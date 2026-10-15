# Phase 3: SPH core (in-depth)

**Goal:** a correct, tested smoothed-particle hydrodynamics (SPH) engine, with no black hole yet. Gas made of particles; density, pressure, shocks and heating computed from neighbours. This is the most important phase for correctness: everything later depends on it.

**Ends with:** the standard SPH test problems passing (Sod shock tube, sound wave, Gresho vortex, Sedov blast), plots of each against the exact solution, and a measured speed in particle-steps per second.

**Primary sources:** Price 2012 (J. Comput. Phys. 231, 759; the SPH equations used here), Price et al. 2018 (Phantom paper; the reference implementation choices), Monaghan 1997 and Cullen & Dehnen 2010 (artificial viscosity and its switch), Price 2008 (artificial conductivity), Springel 2010 review (ARA&A 48, 391) for context.

---

## 3.1 Design

**Formulation:** "variable smoothing length with grad-h terms" (Price 2012 §2–3; Phantom §2.2–2.3). It conserves momentum, angular momentum and energy exactly in the absence of dissipation, and is the standard modern choice.

**State per particle (float64):** position x, canonical momentum p (Phase 4 introduces the hole; until then p = v), mass m (all equal), specific internal energy u, smoothing length h, density ρ, Ω (grad-h term), pressure P, sound speed c_s, viscosity parameter α_i (per particle, for the switch), time-step level (Phase 4).

**Kernel:** cubic spline (M₄), support radius 2h, 3D normalization 1/π:
```
W(r,h) = 1/(π h³) × { 1 − 1.5q² + 0.75q³   (0 ≤ q < 1)
                    { 0.25(2 − q)³         (1 ≤ q < 2)
                    { 0                     (q ≥ 2),        q = r/h
```
Implement W, dW/dr and dW/dh. Wendland C4 (Dehnen & Aly 2012) as an alternative selected by config, for higher neighbour numbers without pairing instability. Kernel choice lives in one place.

**Density and smoothing length** (solve together per particle):
```
ρ_i = Σ_j m_j W(|x_i − x_j|, h_i),       h_i = η (m_i/ρ_i)^{1/3},   η = 1.2  (≈ 58 neighbours)
Ω_i = 1 − (∂h_i/∂ρ_i) Σ_j m_j ∂W_ij(h_i)/∂h_i
```
Newton–Raphson on h with bisection fallback; converge to |Δh/h| < 10⁻⁴; cap iterations; flag failures. Also clamp h between h_min (from the accretion radius region) and h_max (to stop isolated particles getting enormous h).

**Equation of state** (Phase 3 uses ideal gas; Phase 6 adds gas + radiation): P = (γ − 1) ρ u, c_s = √(γ P/ρ). Isothermal option: P = c_s² ρ.

**Acceleration from pressure (per unit mass):**
```
a_i = −Σ_j m_j [ P_i/(Ω_i ρ_i²) ∇_i W_ij(h_i) + P_j/(Ω_j ρ_j²) ∇_i W_ij(h_j) ] + a_visc
```
**Energy equation:**
```
du_i/dt = (P_i/(Ω_i ρ_i²)) Σ_j m_j (v_i − v_j)·∇_i W_ij(h_i) + (du/dt)_visc + (du/dt)_cond
```
**Artificial viscosity** (Monaghan 1997 signal-velocity form, as in Phantom): for approaching pairs only (v_ij·r̂_ij < 0),
```
q_ij^a = −½ ρ_i v_sig,i (v_ij·r̂_ij),   v_sig,i = α_i c_s,i + β_AV |v_ij·r̂_ij|,   β_AV = 2
```
(Price et al. 2018, the Phantom paper's artificial-viscosity section. Note where α sits: it multiplies **only** the sound-speed part; the β_AV term stays on even when the switch sets α_i ≈ 0, which is what stops particles streaming through each other in strong shocks. Putting α in front of everything is a common mistake. Open the paper and copy its exact symmetrized force and heating terms; derive and document the heating so energy is conserved to round-off in the inviscid-plus-viscous total.)

**Viscosity switch:** Cullen & Dehnen 2010 (α_i rises to α_max ≈ 1 at shocks, decays to α_min ≈ 0 elsewhere, using the time derivative of ∇·v). This matters enormously for disks (Phase 7): constant α_AV = 1 drains a disk much too fast for purely numerical reasons.

**Artificial conductivity** (Price 2008): smooths jumps in u at contact discontinuities (where the stream hits the torus). Implement with the signal velocity √(|P_i − P_j|/ρ̄); test it on the shock tube (should not hurt) and a contact discontinuity.

**Neighbour search:**
- Start with `scipy.spatial.cKDTree` (fast, correct, supports periodic boxes via `boxsize` for tests). Query radius per particle 2h_i, but **force sums need every j with |r_ij| < 2 max(h_i, h_j)**. Do a gather with radius 2 max(h_i, h_max,local) or a second pass; test that the neighbour set is symmetric.
- Later, if profiling shows the tree build/query dominates, write a Numba tree (or cell list per h-level). Not before.

**Code structure:** `physics/kernels.py`, `physics/eos.py`, `physics/viscosity.py` (pure formulas); `sim/neighbours.py`, `sim/sph.py` (density pass, force pass). Every Numba kernel has a slow, obvious Python reference and a test that they agree to 1e-12 relative on random particle sets.

**Parallel rule:** loops are "for each particle i, sum over its neighbours j"; write only to i. Each pair is computed twice (once from each side); that's the price of no race conditions and it's fine at this scale.

---

## 3.2 Test problems (expected values in reference/expected-values.md §4)

1. **Kernel:** ∫W dV = 1 (numerical quadrature, 1e-8); W(0,h) = 1/(πh³); W(h,h) = 0.25/(πh³); W(2h,h) = 0; ∇W matches finite differences (1e-7); ∂W/∂h matches finite differences.
2. **Density on a lattice:** particles on a cubic or close-packed lattice in a periodic box with uniform density 1: computed ρ within 0.5% (lattice), h converges, neighbour count ≈ 57.9 for η = 1.2.
3. **Pressure on a uniform lattice:** zero net acceleration to round-off (tests the force symmetry).
4. **Sound wave:** periodic 3D box, γ = 5/3, ρ₀ = 1, c_s = 1, amplitude 10⁻⁴: after one period the wave returns with L1 density error decreasing with resolution (expect roughly second order on a relaxed glass); phase error < 1%.
5. **Sod shock tube** (γ = 1.4, left ρ = 1, P = 1; right ρ = 0.125, P = 0.1; 3D tube periodic in y, z; equal-mass particles, so the left side has 8× the particle density): at t = 0.2 compare with the exact Riemann solution: post-shock pressure 0.30313, velocity 0.92745, density 0.42632 (left of contact) and 0.26557 (right), shock speed 1.75216, rarefaction head −1.18322 and tail −0.07027. Requirements: plateau values within 3%, shock position within 2 local h, no post-shock oscillations above 5%. Plot ρ, v, P, u vs x with the exact solution overlaid.
6. **Gresho–Chan vortex** (Gresho & Chan 1990; used in Springel 2010 and Cullen & Dehnen 2010): a steady rotating flow that should not change. Measure the velocity-profile L1 error at t = 1 with constant α_AV = 1 vs with the switch. **The switch must cut the error substantially** (this is the disk-relevant test).
7. **Sedov blast** (required: Phase 4's time-step limiter test reuses this setup): energy E = 1 put into the central few particles' u (spread with the kernel) in a uniform glass of ρ = 1, γ = 5/3, negligible background pressure (u_background ~10⁻⁶ of the blast). With one global time step here: the shock radius R(t) (where density peaks) matches ξ₀ (E t²/ρ)^{1/5} with ξ₀ ≈ 1.1517 (expected-values §4) within 5% at three times, once the blast spans ≥ 10 h. Plot ρ(r) at those times. (SPH smooths the density peak to ~2–3 of the exact jump value 4: that's normal at this resolution; test R, not the peak height.)
8. **Conservation:** in an isolated box with random velocities and the full scheme (viscosity on), total energy Σ m (½v² + u) conserved to < 10⁻⁶ relative over 1,000 steps (fixed global step for now), total momentum to round-off, total angular momentum to round-off.

Initial conditions for tests: **relaxed "glass" distributions**, not random points (random points give density noise of ~10% and spurious pressure forces). Make a glass once (random particles in a periodic box, evolved with SPH pressure and strong velocity damping until density noise is < 1%), store it, tile and stretch it as needed. Lattices are acceptable for the simplest tests.

---

## 3.3 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Neighbour lists not symmetric | momentum not conserved; particles drift in a uniform box | gather with 2·max(h_i, h_j); test symmetry |
| h iteration fails at sharp edges (cloud surface, vacuum) | NaNs or huge h | bisection fallback; h_max; count failures and log them |
| Pairing instability (particles clump in pairs) | grainy density, worse at high neighbour count | keep η ≤ 1.2 with cubic spline, or Wendland C4 |
| Tensile instability / clumping in tension | odd lines of particles | rarely an issue in compressive flows; Wendland helps |
| Too much viscosity in shear | Gresho vortex decays; disks drain too fast later | Cullen–Dehnen switch (+ Balsara factor), test 6 |
| Too little viscosity at shocks | post-shock noise in Sod | check the switch responds fast enough; α_max = 1 |
| Negative internal energy | crash or NaN in c_s | smaller time step; check heating terms; u floor only as a last resort, and logged |
| Energy not conserved | ledger drift | make the viscous heating exactly match the viscous force work; test 8 |
| Parallel race conditions | results change run to run | write only to i in prange loops; test determinism with 1 vs 4 threads (equal to ~1e-12) |
| Silent float32 | small errors everywhere | assert dtype float64 at function entry in debug mode |
| Slow neighbour search | most time in the tree | profile first; rebuild tree only for active particles (Phase 4); Numba tree if needed |

---

## 3.4 Performance target

Measure particle-updates per second for 10k and 50k particles (one density pass + one force pass). Budget from Phase 0's benchmark: aim for ≥ 2 × 10⁵ particle-updates/s on this laptop for 10k particles. If far below, profile and fix before Phase 5.

---

## Done when

- [ ] All tests above pass with plots in `outputs/checks/` (best ones copied to `docs/figures/`).
- [ ] Reference vs Numba agreement tests pass.
- [ ] Speed measured and recorded.
- [ ] Hiten has seen the shock tube and vortex plots; committed.
