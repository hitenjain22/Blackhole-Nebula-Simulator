# Phase 4: Block time stepping and coupling gas to the hole (in-depth)

**Goal:** let every particle take the time step its own situation needs (tiny near the hole or in a shock, huge far out), while keeping the Phase 1 orbit accuracy and the Phase 3 hydro accuracy, and couple SPH forces to the GN gravity so that energy and angular momentum are conserved exactly by construction.

**Ends with:** all Phase 1 and Phase 3 tests still passing under block steps; a Sedov test showing the time-step limiter works; an orbiting hot gas blob with an exact energy ledger; a measured speed-up.

**Primary sources:** Springel 2005 (GADGET-2, §5: block steps and KDK), Quinn et al. 1997 (time stepping), Saitoh & Makino 2009 (neighbour time-step limiter), Durier & Dalla Vecchia 2012 (MNRAS 419, 465; "wake-up" of inactive particles), Hairer, Lubich & Wanner, *Geometric Numerical Integration* (generalized leapfrog).

---

## 4.1 Design

**Integer timeline.** Time is t = t₀ + n · Δt_min, with n an int64 counter. Levels k = 0…K_max: dt_k = Δt_max / 2^k. A particle on level k is active when n is a multiple of 2^(K_max − k). Never accumulate time as a float (`t += dt` drifts and breaks synchronization).

**Per-particle step, Strang-split:**
1. Hydro half-kick: p ← p + (dt/2) a_hydro (computed at the step start, with v = ∂H/∂p).
2. Gravity flow: generalized-leapfrog step of the GN Hamiltonian over dt (Phase 1's `glf_step`).
3. Recompute density/forces for active particles at the new time; hydro half-kick: p ← p + (dt/2) a_hydro. u is updated with the same kick structure (du/dt at start and end).

**Why hydro kicks go into p:** dH/dt = v · a_hydro then exactly matches the work done by pressure and viscosity, and the SPH energy equation removes exactly that from u, so Σ m (H + u) + E_radiated + E_accreted is conserved by construction. Angular momentum Σ m (x × p) is conserved because SPH pair forces are central and antisymmetric. (Adding hydro forces to v instead breaks the energy ledger by about 2/r: ~4% at r = 50.) **Demonstrate both in a test; the wrong one is a nice figure for the report.**

**Inactive neighbours.** When particle i is active, its neighbours may be mid-step. Predict their x, v, u to the current time: positions with a gravity-only GLF substep from their last full state (accurate and cheap), u with du/dt. Prediction is for force calculation only; the particle's real state updates at its own step end.

**Time-step criteria (per particle; the smallest wins):**
```
orbit:        dt ≤ η_orb r^{3/2}                         (η from Phase 1, e.g. 0.005)
Courant:      dt ≤ C_cour h / v_sig,max                  (C_cour = 0.3)
force:        dt ≤ C_force √(h / |a_hydro|)              (C_force = 0.25)
viscosity:    dt ≤ C_visc h² / ν                          (Phase 7 physical viscosity)
cooling:      none (cooling is applied exactly)          (Phase 6)
```
**Level rules:** a particle may move to a *shorter* step at any of its sync points; to a *longer* step only when n is a multiple of the new step (otherwise it falls out of sync).

**Limiter (essential).** Neighbouring particles' levels may differ by at most 2 (factor 4) (Saitoh & Makino 2009). And when a shock arrives at a particle sitting on a long step, it must be **woken up**: its step is cut short at the current time, with a correctly sized closing kick (Durier & Dalla Vecchia 2012). Without this, strong shocks pass through "sleeping" particles. This is the classic failure that the Sedov test exposes.

**Snapshots at arbitrary output times.** For rendering, outputs are needed at times that don't fall on everyone's step boundary. For each particle, take a copy of its last full state and advance it to the output time with gravity-only GLF substeps (hydro over a fraction of one step is negligible). The real state is untouched.

---

## 4.2 Tests

1. **Phase 1 orbit tests** under the block scheme with hydro off: identical requirements (energy, L, pericentre times, precession).
2. **Phase 3 hydro tests** with particles forced onto mixed levels (e.g. half the particles randomly assigned 2 levels shorter): Sod and sound wave still pass.
3. **Sedov blast with and without the limiter/wake-up:** without it, the blast front is visibly wrong (cold particles on long steps let it through); with it, the radius vs time matches the self-similar law. This test *must* show the difference, otherwise it isn't testing anything.
4. **Energy ledger, hot blob around the hole:** a warm gas blob (500 particles, radius 3, ideal gas γ = 5/3, u = 4 × 10⁻⁶, i.e. sound speed ≈ 2 × 10⁻³: enough pressure to matter, but not so much that the blob blows apart during the orbit; no cooling, no accretion) with its centre on the orbit r_a = 500 → r_p = 25 (start at apocentre with x = (500, 0, 0), p = (0, ℓ/500, 0), ℓ = 7.1951756691; radial period ≈ 27,100), tidally squeezed and shocked at pericentre, which heats it strongly (that heating is the hydro work the ledger must account for). Measure the **maximum** over one radial orbit of |Δ(Σ m (H + u))| / (Σ m |H|).
   - **Why the settings matter:** the ledger is exact in continuous time, but the time stepping still has its own second-order error. Gravity alone on this orbit gives a maximum |ΔH/H| of 6.9 × 10⁻⁵ at η_orb = 0.005, 1.7 × 10⁻⁵ at 0.0025 and 4.3 × 10⁻⁶ at 0.00125 (plan-time check, expected-values §2). So this test runs with **η_orb = 0.00125 and C_cour = 0.15**, not the production values.
   - **Pass:** (a) maximum ledger error < 10⁻⁵; (b) it falls by a factor 3–5 from (η_orb, C_cour) = (0.0025, 0.3) to (0.00125, 0.15), which shows the residual is time-step error and not a coupling bug; (c) angular momentum to < 10⁻¹⁰.
   - **Wrong coupling:** repeat with hydro kicks applied to v. Its ledger error comes from the equations themselves (it is about (2/r) × the hydro work), so it must **not** shrink when the steps are halved. Plot both on one figure: one line falls by 4 per halving, the other stays flat. This is the figure for the report.
5. **Determinism:** same run twice → identical; 1 vs 4 Numba threads → equal to ~1e-12.
6. **Synchronization:** after any number of steps, every particle's time equals t₀ + (its n) · Δt_min exactly; no particle ever skips a sync.
7. **Speed-up:** the Phase 5 cloud setup (1k particles) to pericentre with block steps vs a single global step. Expect a large factor (the global step would be set by the pericentre region all the way). Record it.

---

## 4.3 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Float time accumulation | particles out of sync after long runs | integer timeline only |
| Level increase at the wrong moment | particle time ≠ grid time | only lengthen when n is divisible by the new step |
| No wake-up | shocks penetrate cold gas; Sedov fails | implement Durier & Dalla Vecchia wake-up; test 3 |
| Prediction errors for inactive neighbours | small forces noise; energy drift | predict positions with gravity-only GLF, not straight lines (orbits curve a lot near the hole) |
| Too many levels | int overflow, tiny dt | K_max ≤ 40; assert; investigate any particle that reaches the deepest level (usually a bug or a particle deep inside r_acc) |
| Energy ledger fails with accretion | sudden jumps | ledger must record H, u, m and angular momentum of each removed particle at removal |
| Time-step collapse in the nozzle shock | run slows to a crawl at pericentre | expected to some extent; measure; if extreme, check Courant uses v_sig correctly, and that h isn't collapsing because of a pairing problem |
| Hidden global sync | everything moves at the shortest step | profile active-particle counts per system step; plot the level histogram over time |

---

## 4.4 Diagnostics added in this phase

- Histogram of time-step levels per output (saved in diagnostics).
- Active-particle count per system step; total particle-steps; wall time per stage.
- The conservation ledger printed every N steps: mass (present + accreted), energy (Σ m(H + u) + radiated + accreted), angular momentum vector (present + accreted).

---

## Done when

- [ ] All tests pass, including the Sedov limiter demonstration and the ledger test with the coupling comparison.
- [ ] Speed-up measured; level histogram plot reviewed.
- [ ] Hiten has seen the ledger plot (flat line) and the wrong-coupling comparison; committed.
