# Phase 7: Viscosity, the drain and the plunge (in-depth)

**Goal:** stages 6 and 7. The torus slowly gives up angular momentum, spreads, and feeds gas inward until it passes the innermost stable orbit (r = 6) and plunges into the hole. Angular momentum transport is modelled with a Shakura–Sunyaev α viscosity, validated against a 1D viscous-disk model that is also a science tool of its own (it can run the slow, years-long drains that SPH can't afford).

**Ends with:** a tested 1D disk model; the SPH ring-spreading test matching it; a measured numerical-viscosity floor; a 10k run from the start until half the cloud has been swallowed, with its accretion-rate curve; a 50k overnight run; slow-drain variants from the 1D model.

**Primary sources:** Shakura & Sunyaev 1973 (α viscosity), Lynden-Bell & Pringle 1974 (spreading-ring solution), Pringle 1981 (ARA&A review of accretion discs; the diffusion equation), Lodato & Price 2010 (MNRAS 405, 1212; viscosity in SPH discs), Flebbe et al. 1994 (Navier–Stokes viscosity in SPH), Price et al. 2018 (Phantom §2: disc and physical viscosity), King, Pringle & Livio 2007 (what α is observed to be), Balbus & Hawley 1998 (why discs are turbulent: the magnetorotational instability; background only).

---

## 7.1 The physics, with numbers to expect

Real disks move angular momentum outward through magnetic turbulence (magnetorotational instability). We don't simulate magnetic fields; we use the standard stand-in, a viscosity

```
ν = α c_s H,    H = c_s / Ω_K,   so   ν = α c_s² / Ω_K,    Ω_K = r^(−3/2)
```
with α ≈ 0.01–0.4 from simulations and observations (King, Pringle & Livio 2007 argue observed values are ~0.1–0.4). Default α = 0.1.

**Expected time scales** (expected-values §5; plan-time 1D estimate with ν = α (H/r)² r^(1/2), a narrow ring at r_circ = 50.89, zero torque at r = 6):

| α | H/r | t_ν = r²/ν | Half the ring accreted after | Peak accretion at |
|---|---|---|---|---|
| 0.1 | 0.3 (thick torus, our default) | 40,300 (9.9 days, 17.7 ring orbits) | 26,600 (6.5 days, 11.6 orbits) | 0.07 t_ν |
| 0.03 | 0.3 | 134,500 (33 days) | 88,600 (22 days) | 0.07 t_ν |
| 0.1 | 0.1 (thin) | 363,000 (89 days) | 239,000 (59 days) | 0.07 t_ν |

The problem is self-similar: half-mass at 0.659 t_ν and peak Ṁ × t_ν = 1.945 (in units of the ring mass) for any α and H/r. **So a thick torus drains in days to weeks.** In our scenario the drain likely overlaps with the ongoing crashes (stages 4–6 blend): gas arriving late is still circularizing while the early torus is already draining. That's physically expected; the stage names are a story aid, not separate eras.

**The plunge.** No stable circular orbits exist inside r = 6. Gas reaching it spirals briefly and falls in within roughly an orbit. GN gravity reproduces this. Particles are removed at r_acc = 4 (the marginally bound radius), with full ledger entries.

**Radiative efficiency.** A thin disk radiates 5.7% of the accreted rest mass (binding energy at r = 6). A thick torus with cooling time ≈ 37 orbits but drain time ≈ 12 orbits swallows most of its heat instead: expect an efficiency well below 5.7% ("advective" or "slim" flow). Measure it.

---

## 7.2 Viscosity in SPH: two implementations, chosen by test

**Option A (primary): physical Navier–Stokes shear viscosity** with ν = α c_s²/Ω_K per particle (Flebbe et al. 1994; Phantom's "physical viscosity" computes it with two first derivatives rather than a noisy second derivative).
- Clean input: α is the physical α, independent of resolution.
- Needs a time-step limit dt ≤ C_visc h²/ν (Phase 4 criteria table; C_visc ≈ 0.1–0.2; tune by test).
- **Angular momentum:** the stress-tensor form is not guaranteed to conserve angular momentum exactly. Measure it in the ring test. If |ΔJ/J| > 10⁻⁶ per orbit, either fix the formulation (symmetrize) or switch to Option B.

**Option B (cross-check, and fallback): disc viscosity through the artificial viscosity** (Lodato & Price 2010): the AV term applied to approaching and receding pairs with a fixed α_AV, which acts like
```
α_SS ≈ (α_AV / 10) · ⟨h⟩ / H        (cubic spline kernel)
```
- Exactly conserves angular momentum (pair forces).
- But the effective α depends on resolution ⟨h⟩/H, which changes as the torus evolves and differs between 10k and 50k runs. Report the effective α_SS measured, not just α_AV.

**Both:** the viscous force is a hydro force, so it kicks **p** (Phase 4), and its heating uses v = ∂H/∂p, so the energy ledger stays exact. Ω_K = r^(−3/2) (state this choice; the GN circular frequency differs by 1–2% at r ≈ 50).

**Where viscosity acts.** Default: everywhere (outside the torus c_s is small and ν negligible: in the cloud at r = 5,000, ν ≈ 9 × 10⁻⁵ and the viscous time across the cloud is ~10⁸, far longer than the run). Config option `viscosity_mask = "circularized"`: only particles with orbital eccentricity < 0.5, ramped smoothly. Compare the two at 1k; record the choice in DECISIONS.md.

---

## 7.3 The 1D disk model (`src/nebula/physics/disk1d.py`)

**Equation** (Keplerian, axisymmetric, thin disk; Pringle 1981):
```
∂Σ/∂t = (1 / (2π r)) ∂Ṁ/∂r,         Ṁ(r) = 6π r^(1/2) ∂(ν Σ r^(1/2))/∂r      (Ṁ > 0 means inflow)
```
- **Grid:** cells uniform in √r from r_in = 6 to r_out = 10⁴ (≥ 400 cells; resolution test decides).
- **Form:** finite volume (cell masses change only by face fluxes Ṁ), so mass is conserved to round-off by construction.
- **Time stepping:** Crank–Nicolson (implicit; no stability limit). Start with 2–4 backward-Euler steps to damp the oscillations Crank–Nicolson produces from sharp initial rings (Rannacher start). If ν depends on Σ (optional thermal model), iterate each step (Picard) to 10⁻¹⁰.
- **Boundaries:** inner zero torque (ν Σ = 0 at r_in; whatever crosses is accreted and logged as Ṁ_acc(t)); outer zero flux.
- **ν(r):** default ν = α (H/r)² r^(1/2) with H/r a parameter; or ν(r) measured from an SPH snapshot (c_s(r), H(r) from the torus).
- **From a snapshot** (in `src/nebula/analysis/disk_profile.py`, not in `physics/`, because it reads files): find the torus plane from the gas's total angular momentum, bin mass into annuli → Σ(r); measure H/r and c_s per annulus → ν(r). It hands plain arrays to `disk1d.py`.
- **Outputs:** Σ(r, t), Ṁ_acc(t), cumulative accreted mass, angular momentum.

Note: the 1D model is Newtonian Keplerian; the zero-torque condition at r = 6 stands in for the relativistic inner edge. Say so in the report.

### 1D tests (expected values in expected-values §5)

| # | Test | Requirement |
|---|---|---|
| 1 | Lynden-Bell & Pringle ring, ν constant: evolve the exact profile from τ = 0.016 to τ = 0.064, 0.256, 1.0 (τ = 12νt/R₀²), inner boundary at 0.01 R₀, outer at 100 R₀ | L1 error in Σ < 1% (and decreasing with resolution); mass 0.999999995, 0.998280, 0.932079 within 2 × 10⁻³ |
| 2 | Same: angular momentum ∫2πr Σ r^(1/2) dr | equals 1 to 10⁻⁴ while negligible mass reaches the outer edge |
| 3 | Exact profile values (use `scipy.special.ive` to avoid overflow: Σ π R₀²/m = τ⁻¹ x^(−1/4) e^(−(1−x)²/τ) ive(¼, 2x/τ)) | the analytic function matches the §5 table to its printed precision (7 significant figures) and an independent high-precision evaluation (`mpmath`, using the unscaled Bessel function) to 10⁻¹² |
| 4 | Steady state: constant mass supply at r_feed = 50 | converges to ν Σ = (Ṁ / 3π)(1 − √(r_in/r)) inside r_feed within 0.5% |
| 5 | Self-similar drain (ν = α (H/r)² r^(1/2), Gaussian ring width 3% at r₀ = 50.89, r_in = 6) | half-mass at 0.659 t_ν and peak Ṁ t_ν = 1.945 within 3% (that's agreement between two independent solvers; if it fails, investigate both before changing anything) |
| 6 | Mass ledger | initial = disk + accreted + outer loss, to 10⁻¹² |

---

## 7.4 The SPH ring-spreading test (the key validation)

Following Lodato & Price 2010:
- A gas ring around the hole at r₀ = 50, Gaussian width 5, **H/r = 0.1 at r₀**, **locally isothermal** (c_s² ∝ r^(−3/2), so ν = α c_s²/Ω_K is constant in r and the LBP solution applies), α = 0.1 (Option A) or the matching α_AV (Option B). Then ν = α (H/r)² r₀^(1/2) = 7.07 × 10⁻³ and t_ν = r₀²/ν ≈ 3.5 × 10⁵.
- **Why these numbers:** the test must spread the ring measurably in an affordable time. A thinner ring (H/r = 0.05) has t_ν ≈ 1.4 × 10⁶ and would need ~55 hours at 50k particles to reach 1 t_ν. With H/r = 0.1 and comparisons at 0.05, 0.1 and 0.25 t_ν (LBP τ = 12νt/r₀² = 0.6, 1.2, 3) the ring spreads by √(νt) ≈ 11 to 18, two to four times its width, in 0.25 t_ν ≈ 8.8 × 10⁴ ≈ 50,000 orbit steps: about 40 minutes at 10k and 3–4 hours at 50k (at 2 × 10⁵ particle-updates/s; measure).
- Gravity: **Newtonian point mass** for the comparison with the Keplerian 1D model; then repeat with GN and report the (small) difference.
- Compare the azimuthally averaged Σ(r, t) with the 1D model started from the SPH Σ at the same t₀: L1 difference < 15% at t − t₀ = 0.05, 0.1, 0.25 t_ν. Fit α from the spreading rate: within 20% of the input.
- **Numerical floor:** the same ring with physical viscosity off (only the switch-controlled AV): fit the effective α_num. **Requirement: α ≥ 3 α_num at the production resolution.** Report α_num at 10k and 50k.
- Angular momentum: |ΔJ/J| over the test, for both options.

This test decides Option A vs B for production. Write the decision and evidence into DECISIONS.md.

---

## 7.5 Production runs

| Run | Particles | Notes | End |
|---|---|---|---|
| R7.1 | 10k | GN, β = 230, α = 0.1, chosen viscosity option | 50% of cloud mass accreted (or t_circ + 5 t_ν) |
| R7.2 | 50k | same; **overnight**, possibly two nights with checkpoints | same |
| R7.3a–d | 1k | α = 0.01, 0.03, 0.1, 0.3 | same |
| R7.4a–b | 1k | r_acc = 3 and 5 | same; Ṁ must not depend on r_acc beyond noise |
| R7.5 | 1D only | from the R7.1 torus at t_circ: α = 0.1 and the measured H/r | until 99% accreted |
| R7.6 | 1D only | **slow-drain variants**: H/r = 0.05, α = 0.01: drain over years | until 99% accreted |

**Cost estimate for R7.2** (measure on R7.1 first): torus particles at r ≈ 50 step at ~η r^(3/2) ≈ 1.8; three viscous times ≈ 1.2 × 10⁵ → ~70,000 steps each → 50k × 70,000 ≈ 3.5 × 10⁹ particle-steps, ~5 hours at 2 × 10⁵ updates/s, plus the crash phase. One to two nights. If the measured rate is far lower, report before launching.

**SPH vs 1D comparison:** Ṁ(t) of R7.1 vs R7.5 over the first 2 t_ν after t_circ: agreement within a factor ~1.5. A thick torus with ongoing feeding is not a thin disk, so exact agreement isn't expected; large disagreement means a bug or a physical difference to explain.

**α scaling:** drain time vs α on a log–log plot: slope ≈ −1.

**Slow drain (R7.6):** this is the "stage 6 takes far longer" regime Hiten's earlier notes mentioned. It is affordable only in 1D; show it as a figure and explain why thin, cool disks drain over years while our thick torus drains in days.

---

## 7.6 The plunge region

- Diagnostics inside r = 10: mass-weighted radial velocity v_r(r), density profile, sound speed. Expect a density drop and supersonic inflow inside r ≈ 6.
- Compare v_r(r) with a test particle released from the ISCO with a tiny nudge (Phase 1's ISCO demo).
- **Sonic point check:** the flow must be supersonic at r_acc, so the removal boundary can't affect the gas upstream. Report the sonic radius.

---

## 7.7 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Numerical viscosity rivals the physical one | drain time barely changes with α; ring test α_num close to α | more particles; check the switch decays in the torus; require α ≥ 3 α_num |
| Viscous time step dominates | run slows after the torus forms | expected only if ν h⁻² is large; check C_visc; don't raise it to speed up |
| Ṁ(t) noisy | spiky accretion curve | each removal is one particle's mass: bin in time, show √N error bars |
| Gas piling up just outside r_acc | density spike near r = 4 | boundary artifact: one-sided neighbour sets push gas; check the sonic point is outside r_acc; compare R7.4a–b |
| Torus vertically unresolved | ⟨h⟩/H > 0.5 | report ⟨h⟩/H vs time; Option B's mapping is invalid there; resolution limit for the report |
| Angular momentum drift (Option A) | J ledger drifts | measure; symmetrize or switch to Option B |
| Torus plane not the x–y plane | 1D binning smears the torus | find the plane from total J before binning |
| Heating–viscosity feedback | H/r grows, ν grows, drain speeds up | physical; the 1D comparison must use the measured H/r(t) |
| Crank–Nicolson ringing | negative Σ near a sharp ring | Rannacher start; never clip Σ silently (log any negative cell) |
| LBP evaluation overflow | NaN at small τ | the ive form in test 3 |

---

## 7.8 Physics questions this phase answers (for the report)

1. How long until half the cloud is swallowed, compared with the 1D estimate (6.5 days for α = 0.1, H/r = 0.3)?
2. What does Ṁ(t) look like? Peak value in Eddington units (Ṁ_Edd = 0.0955 M☉/yr at 10% efficiency). Fit the late-time decline with a power law and compare with slopes quoted in the literature (check the values in the papers before quoting them).
3. Where does the cloud end up: swallowed, escaped, still orbiting?
4. Radiative efficiency (radiated energy / accreted mass c²) vs the thin-disk 5.7%.
5. Does SPH agree with the 1D model? Where not, why?
6. How would a thin, cool disk differ (R7.6)?

---

## Done when

- [ ] 1D tests 1–6 pass.
- [ ] Ring-spreading test passes for the chosen option; α_num measured; decision logged.
- [ ] R7.1, R7.3, R7.4, R7.5, R7.6 complete; R7.2 complete or scheduled with a measured cost.
- [ ] Ledgers flat (|ΔE|/Σm|H| < 10⁻⁴, |ΔJ|/J < 10⁻⁶ for Option B; Option A's J drift reported).
- [ ] Plots (Ṁ(t), Σ(r, t) SPH vs 1D, α scaling, plunge profile) reviewed by Hiten; results-log entries; committed.
