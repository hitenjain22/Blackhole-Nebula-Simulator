# Phase 6: Stream crash, cooling and the ring (in-depth)

**Goal:** follow stages 4 and 5. The stretched cloud keeps orbiting; its relativistic precession makes the outgoing stream crash into the incoming one; the crash shocks and heats the gas; the gas loses orbital energy, circularizes and settles into a hot torus near r ≈ 51; radiative cooling decides whether that torus is thick and puffy or a thin ring.

**Updated 2026-10-15** (Hiten's decisions after the plunge review, `docs/DECISIONS.md`): all science runs use 10k (1k is a smoke test only, D1); the viscosity uses velocity reconstruction (D2); swallowing at r_acc = 4 is built (D6). **Updated 2026-10-16:** Hiten chose **cloud radius 60** for Phase 6 and later (DECISIONS, D4 update; review §7): the smallest cloud whose first stream crash falls inside the planned run, with a known numerical loss of ~0.13% of the gas at 10k. The radius-dependent numbers below are for radius 60 (expected-values §2c); the radius-100 values are kept for comparison.

**Ends with:** 10k runs from the start to a formed torus, with the gas + radiation equation of state and cooling on; a four-panel thin-vs-thick comparison (β scan); a Newtonian control run showing why relativity matters; the first light curve; flat ledgers throughout.

**Primary sources:** Bonnerot et al. 2016 (MNRAS 455, 2253: disc formation from tidal disruptions on eccentric orbits; efficient vs inefficient cooling), Hayasaki, Stone & Loeb 2013 (circularization driven by relativistic precession), Bonnerot & Stone 2021 (review: self-intersection, nozzle shock, circularization), Gammie 2001 (β-cooling), Lombardi et al. 2015 and Stamatellos et al. 2007 (approximate radiative cooling for SPH), Price 2008 (artificial conductivity at contact discontinuities). See `reference/references.md` for which were checked online.

---

## 6.1 The physical picture, with numbers to expect

All numbers here are in `reference/expected-values.md` §2b and §9 with how they were computed. They are **predictions to compare against**, not test tolerances: when the run disagrees, find out why and write it up (it may be the hydrodynamics doing something the test-particle estimate can't).

**Where the streams cross.** Gas that has just passed pericentre heads back out on an orbit rotated by 24.04°. The gas still falling in is on the un-rotated orbit. The two branches cross where both are 180° from pericentre on the exact orbit shape:
- self-intersection radius **r_SI ≈ 1,664** (the common "Kepler ellipse rotated at pericentre" formula gives 1,572; use the exact one);
- at polar angle 12.02° from the start direction (the +x axis);
- time from pericentre to r_SI: **36,970** (9.1 days);
- the streams meet almost **head-on**: both have speed 0.0283, radial components ∓0.02798, tangential 0.0044, so their velocities are 162° apart and the relative speed is **0.056 c ≈ 16,800 km/s**.

**When the crash starts.** Crossing needs gas on both branches at once: the incoming gas must still be passing r_SI when the outgoing gas arrives, so the spread of pericentre times must exceed about 2 × 36,970 ≈ 74,000. A test-particle model of the radius-60 cloud (300 particles, pressure ignored; expected-values §2c) gives:

| Pericentre passage | Mean time | Spread of pericentre times | Streams cross? |
|---|---|---|---|
| 1 | 396,000 (97 days) | 13,100 | no |
| 2 | 1,188,000 (291 days) | 39,100 | no |
| 3 | 1,980,100 (485 days) | 65,200 | no, but **only just**: the first outgoing gas reaches r_SI 8,700 after the last incoming gas has passed |
| 4 | 2,772,100 (679 days) | 91,400 | **yes**: first contact at **t = 2,765,139** (1.86 years), with 10% of the stream still incoming |

(For the old radius-100 cloud, expected-values §2b: spreads 21,800, 65,100, 108,700; crossing at passage 3, first contact t = 1,964,230, 25% incoming.)

The spread grows each orbit because the energy spread (±1.1% at radius 60) makes the return times differ by ±1.7% per orbit. Pressure and the nozzle shock at each pericentre add energy spread (Phase 5 measured ±14% at 50k and ±33% at 10k for the middle half of the gas after the first pass, radius 25, much of it numerical), so **the real crash probably starts earlier, at passage 3 (t ≈ 2.0 million) or even 2**. When it starts is one of this phase's results.

**How hot.** Turning the radial kinetic energy ½v_r² = 3.9 × 10⁻⁴ c² per unit mass into heat would make gas alone 1.7 × 10⁹ K. With radiation in equilibrium the temperature is far lower: 1.5 × 10⁵ K at ρ = 10⁻⁸ kg/m³ and 4.6 × 10⁵ K at 10⁻⁶ kg/m³, radiation-pressure dominated (Γ₁ → 4/3). This is why the radiation equation of state is essential.

**What happens after a crash.** The colliding gas loses most of its radial velocity but keeps its angular momentum (ℓ ≈ 7.35). It falls back on a shorter, still eccentric orbit (apocentre ~1,600, period ~1.4 × 10⁵), precesses again, and crosses again closer in. Repeated crashes walk the gas inward until it is roughly circular near the circularization radius **r_circ = 50.89** (orbital period 2,303 ≈ 13.6 hours).

**Energy budget.** Going from the cloud's orbit (ε = −1.989 × 10⁻⁴) to a circular orbit at r_circ (ε = −9.620 × 10⁻³) releases 9.42 × 10⁻³ c² per unit mass: 1.7 × 10⁴⁴ J (1.7 × 10⁵¹ erg) for 0.1 M☉, comparable to a supernova, or 36 days at the Eddington luminosity (5.41 × 10³⁷ W, using the standard pure-hydrogen convention; expected-values §9 explains the choice). Where that energy goes (radiated, stored as heat, used to unbind gas) is the energy-budget figure of the report.

**Thick or thin.** If the gas radiates the heat quickly (efficient cooling) it settles into a thin ring at r_circ; if it can't (inefficient cooling), it puffs up into a thick torus spread over a range of radii (Bonnerot et al. 2016 show both). For our 0.1 M☉ scenario the photon-diffusion estimate gives a cooling time of about 230/Ω, roughly 37 orbits, so **a thick torus is the expected default**.

---

## 6.2 New physics (all in `src/nebula/physics/`, all tested)

### Equation of state: gas + radiation (`eos.py`)

```
u = 3 k T / (2 μ m_p) + a T⁴ / ρ          (specific internal energy)
P = ρ k T / (μ m_p) + a T⁴ / 3
β_gas = P_gas / P
Γ₁ = β_gas + (4 − 3β_gas)² (γ − 1) / (β_gas + 12 (γ − 1)(1 − β_gas)),   γ = 5/3
c_s = √(Γ₁ P / ρ)
```
- Solve for T from (ρ, u) with Newton's method on **log T**, with bisection fallback on [1 K, 10¹³ K]; tolerance 10⁻¹² relative. Never solve the quartic in closed form (cancellation).
- Work in physical units inside the function; `units.py` converts code ρ (via the cloud mass scale) and code u (× c²). This is the only place the cloud mass enters the dynamics.
- The cloud itself at 10⁴ K and ρ ≈ 1.9 × 10⁻⁷ kg/m³ already has P_rad/P_gas ≈ 0.1, so **production runs use this EOS from t = 0**. (These numbers are for radius 100. The radius-60 cloud is 4.6× denser, ρ ≈ 8.6 × 10⁻⁷ kg/m³, so P_rad/P_gas ≈ 0.02 at 10⁴ K; radius 25: ~0.002. Using the full EOS from t = 0 is right either way.) (Phase 5 runs with ideal gas remain valid as development runs.)
- **Assumption to state in the report: local thermodynamic equilibrium (LTE)**, i.e. gas and radiation share one temperature. What makes radiation thermalize is **absorption**, not electron scattering (scattering only bounces photons around; it doesn't create or destroy them). The right measure is the effective optical depth τ_eff = κ_eff ρ L with κ_eff = √(κ_abs(κ_abs + κ_es)) (Phase 9 uses the same κ_eff). Rough plan-time estimates with a free–free (Kramers-type) κ_abs, coefficient from memory (check it in Frank, King & Raine before quoting): the 10⁴ K cloud has τ_eff ~ 10³ (thick: LTE fine; at 10⁴ K hydrogen's bound–free absorption makes it even thicker), but the 10⁶ K torus has τ_eff of only **a few to ~15** (free–free absorption falls as T^(−7/2); metal bound–free absorption adds up to ~20×). So **LTE in the hot torus is an approximation, not a given**: the gas could be hotter than the radiation in its thin outer layers. Say so in the report.
- Diagnostics (per particle, physical units, every snapshot): τ_es,h = κ_es ρ h and τ_eff,h = κ_eff ρ h, with κ_abs and κ_eff from `physics/opacity.py` (written here, with tests against hand-computed values from the textbook coefficient, and reused by Phase 9). Report the mass fractions with τ_eff,h < 1 and with τ_eff over the torus thickness < 1. If most of the torus mass has τ_eff < 1, stop and discuss with Hiten (cut-list option: the diffusion-cooling scheme below).

### Temperature floor
T_floor = 10⁴ K (gas kept warm by photo-ionization, as observed for G2). Floor energy u_floor(ρ) from the EOS at T_floor. Cooling never goes below it; nothing else may clip u. Any other clipping is a bug, logged.

### Cooling (`cooling.py`)

**Default: β-cooling** (Gammie 2001 form), relaxing toward the floor:
```
t_cool = β / Ω_K(r),   Ω_K = r^(−3/2)
u_new = u_floor + (u − u_floor) · exp(−Δt / t_cool)        (exact; no time-step limit)
E_radiated += m (u − u_new)
```
- Applied as an operator-split step in the middle of each particle's kick–drift–kick (Phase 4's Strang structure); u is updated exactly, and the radiated energy goes into the ledger and into a per-particle luminosity array (used for light curves and the renderer's emission).
- β is a config parameter. Default β = 230 (scenario estimate, expected-values §9). Scan β = 1, 10, 100, ∞ (∞ = no cooling).

**Option (cut-list item 4): approximate radiative diffusion cooling** (Lombardi et al. 2015 "pseudo-column density", or Stamatellos et al. 2007): estimates each particle's optical depth to the surface from local quantities and cools at the corresponding diffusion rate. **Read the papers before implementing;** the formulas are not reproduced here because they were not re-derived for this plan. It must reduce to the same thick-torus outcome at the scenario's density (cross-check with β = 230) and to fast cooling for thin gas.

---

## 6.3 Tests (before any long run)

| # | Test | Requirement |
|---|---|---|
| 1 | EOS test points (expected-values §9, five (ρ, T) points) | u, P, β_gas, Γ₁ to rtol 10⁻⁸; T recovered from (ρ, u) to 10⁻¹⁰ |
| 2 | EOS limits | Γ₁ → 5/3 as β_gas → 1 and → 4/3 as β_gas → 0: within 10⁻⁶ at the two extreme points of expected-values §9 (ρ = 10³, T = 10³ and ρ = 10⁻¹⁵, T = 10⁸), whose u and P also match to rtol 10⁻⁸ |
| 3 | EOS round trip | T → u → T on a 200 × 200 grid of ρ ∈ [10⁻¹⁵, 10³] kg/m³, T ∈ [10³, 10¹⁰] K: max error 10⁻¹⁰ |
| 4 | Radiation-dominated shock tube | Sod initial state, EOS forced into the radiation limit (P = ρu/3, an exact γ = 4/3 gas): match the γ = 4/3 exact solution: p* 0.306011, u* 0.953205, ρ*L 0.411437, ρ*R 0.278585, shock speed 1.728999 (same requirements as Phase 3's Sod) |
| 5 | Colliding streams | two cold uniform slabs colliding head-on at ±v (strong shock): post-shock gas at rest with u = v²/2 within 5%, compression 4 (γ = 5/3) or 7 (γ = 4/3) within 10%, shock speed v/3 or v/6 within 5%; total energy conserved to 10⁻⁶. This is the self-intersection in miniature, and the key check that real shocks are fully captured with velocity reconstruction (D2) on: at a sharp step reconstruction lowers the interface heating (~40% of plain in a unit test), so the jump conditions here must still hold. |
| 6 | β-cooling, static particle | u(t) matches u_floor + (u₀ − u_floor)e^(−t/t_cool) to 10⁻¹²; ledger: Σ m Δu = E_radiated to round-off |
| 7 | Cooling under block steps | the same particle cooled with 1 step or 1,024 steps over the same interval: identical u to 10⁻¹² |
| 8 | Hot blob ledger with cooling | Phase 4 test 4 as it now runs (D8: blob u = 4 × 10⁻⁸, from apocentre through pericentre to t = 18,000, not a full orbit; same orbit, η_orb = 0.00125, C_cour = 0.15) with β = 10 and the full EOS: maximum \|Δ(Σ m(H + u) + E_rad)\| / Σ m\|H\| < 10⁻⁵ over that run, and falling 3–5× when η_orb and C_cour are halved |
| 9 | Floor | a particle cooled for 100 t_cool sits at u_floor exactly and never below |

---

## 6.4 Runs

All runs: r_acc = 4 (removed particles go into the ledger with their m, H, u, x × p; built, D6); escape radius 50,000 (removed as "escaped", ledgered; **to build**: the same removed-particle ledger with a second reason); gas + radiation EOS; T_floor = 10⁴ K; the Phase 5 initial conditions with **radius 60** (Hiten, 2026-10-16); the hole-resolution check (D5) in every snapshot.

| Run | Particles | Gravity | β | End | Purpose |
|---|---|---|---|---|---|
| R6.0 | 1k | GN | 230 | short | smoke test only (D1): does it run, are the ledgers flat |
| R6.1 | **10k** | GN | 230 | t = **3.8 × 10⁶** (~2.6 years; was 3.0 × 10⁶ for radius 100: Hiten, 2026-10-16, DECISIONS) or torus formed + 10 ring orbits | first full circularization |
| R6.2 | 10k | GN | 230 | same | the everyday-resolution result (with R6.1 now also 10k: merge or use for a second configuration) |
| R6.3a–d | **10k** | GN | 1, 10, 100, ∞ | same | thin ring vs thick torus |
| R6.4 | **10k** | **Newtonian** point mass | 230 | same | control: no precession, so no early crossing |
| R6.5 | **10k** | GN | 230, seeds 1–3 | same | run-to-run scatter (random glass carving: **to build**, a random rotation/offset of the glass before carving; the current carving is deterministic) |

All runs 10k (Hiten, 2026-10-15; D1: 1k is never used near the hole).

**Run length budget.** Most of the 3.8 × 10⁶ is cheap (gas far out takes huge steps). Cost concentrates at pericentre passages, crashes and in the torus. The old estimate (R6.1 in tens of minutes, R6.2 in a few hours) assumed 1k runs and predates the measured speed (~30,000 particle-updates/s, Step 3.8), D2's extra gradient pass (~20–30%) and 10k for every run: **remeasure** on the first 10k run (Phase 5) and extrapolate before launching the Phase 6 set. Measured in Phase 5 (radius 25, D2): 10k to t = 400,000 in 1.7 h, 50k in 13.6 h; the pericentre passage dominates. Checkpoint every 15 minutes (simple version here; Phase 10 makes it robust).

**Stop condition for "torus formed".** Define with the diagnostics below: ≥ 50% of the remaining bound mass has orbital eccentricity < 0.3 and r < 200. Record that time as t_circ.

**Snapshots.** Use the schedule defined in Phase 10.2a, and implement it here (these are the first long runs): Δt_snap = clamp(T_orb(r_med)/40, 10, 10⁴), with T_orb(r) = 2π r^(3/2) and r_med the median radius of the gas inside r = 2,000 (all gas if none is inside yet), plus event snapshots at each pericentre passage, the first crash and t_circ. Expect ~4,000–5,000 snapshots per run (1–3 GB at 10k particles). The render director (Phase 8.5) never changes what the simulation writes; for exact frame times it uses `resim` (Phase 8.2).

---

## 6.5 Diagnostics (every snapshot, into `diagnostics.csv`)

- Ledgers: mass (present + accreted + escaped), energy (Σ m(H + u) + E_rad + E_acc + E_esc), angular momentum vector.
- **Per-particle orbit elements** from (H, ℓ) via the exact Schwarzschild turning points (E = √(1 + 2H), ℓ = |x × p|; solve E² = (1 − 2/r)(1 + ℓ²/r²) for r_peri, r_apo): eccentricity e = (r_apo − r_peri)/(r_apo + r_peri). Flag unbound (H ≥ 0) and plunging (no turning point above r = 4) particles.
- Eccentricity histogram (mass-weighted) over time: the circularization movie in one plot.
- Ring/torus mass (e < 0.3, r < 200); its mass-weighted mean radius; thickness H/r = rms(z)/r; radial extent (10th–90th percentile r).
- Mass in shocks and its location (marks crash sites): the marker is the **viscous heating rate after reconstruction** (Hiten, 2026-10-15), not α_AV > 0.5 (the switch's α sits at ~1 for all the cold gas and no longer marks shocks).
- Luminosity L(t) = Σ m (du/dt)_cool in watts and in units of L_Edd = 5.41 × 10³⁷ W; cumulative E_rad.
- Accretion rate Ṁ(t) at r_acc (binned; in M☉/yr and Eddington units, Ṁ_Edd = 0.0955 M☉/yr for 10% efficiency).
- Temperature distribution (mass-weighted histogram of log T); mass fractions with τ_eff,h < 1 and τ_es,h < 1.
- Unbound mass fraction.
- **Lost gas (D4 update):** particles below the capture line L < 4 plus those swallowed, as a share of the cloud; logged with every diagnostics line and flagged above 1%. Known numerical effect at 10k, radius 60: ~0.13% after the first pass (review §7).
- Time-step level histogram; wall time per 10⁵ time units.

---

## 6.6 Pictures and previews for Hiten (`outputs/checks/`)

1. The first crash: zoom on r_SI ≈ 1,664 at 12°, particles coloured by temperature, three moments (approach, impact, splash).
2. Circularization: face-on panels at passage 1, 2, 3 (or whenever the crash starts), torus forming, settled torus.
3. Edge-on torus at the end (shows thickness).
4. β comparison: 2 × 2 grid, face-on and edge-on insets, same moment after formation.
5. Newtonian vs GN at the same time.
6. Eccentricity histogram vs time (heat map), light curve, Ṁ(t), energy budget bar chart.
7. A particle-preview MP4 of R6.2 with a simple time warp.

---

## 6.7 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Stream thinner than h at the crash | crash looks smeared; dissipation depends strongly on particle number | report particles per stream cross-section at r_SI; compare 1k vs 10k (Phase 12 does 1k → 50k); it's a known resolution limit, report it |
| **Numerical circularization** (too much artificial viscosity bleeding orbital energy without a real crash) | gas circularizes before any crash; the Newtonian control circularizes almost as fast as GN | check R6.4: if it circularizes comparably, the effect is numerical; check the Cullen–Dehnen switch decays away from shocks; compare α_max = 1 vs 2 **and reconstruction (D2) on vs off**; check the hole-resolution numbers (D5) for the torus at r ≈ 51 (kernels must stay small compared with r there); never claim a circularization time without this check |
| No crash ever happens | stream keeps orbiting through passage 3+ | compare with the 6.1 table: crash at passage 3 is expected even without hydro; if not, check the stream's energy spread (Phase 5 histogram) and the precession (Phase 1 tests under hydro) |
| EOS solver fails for extreme u/ρ | NaN temperatures in hot, thin gas | log-space Newton + bisection on a wide bracket; test 3 covers the range |
| u clipped somewhere | energy ledger drifts upward | only cooling touches the floor; grep for any `max(u, …)`; test 9 |
| Huge time-step collapse during crashes | run slows sharply after the first crash | expected near crash sites; check the level histogram; if particles far from shocks are on tiny steps, the limiter is waking too many |
| Hot gas becomes unbound and flies off | particles at r > 50,000, neighbour searches slow | escape radius removes them (ledgered); report the unbound fraction (a real effect, from crash-driven outflows) |
| Optically thin gas treated as thick | unphysical radiation pressure in thin edges; LTE temperature too low in thin layers | report the τ_eff,h < 1 mass fraction; mention in the report; diffusion cooling (option) helps |
| Light curve noise | spiky L(t) | bin in time; report binned values; never smooth before computing totals |
| Results differ wildly between seeds | R6.5 scatter large | it's real stochasticity at low resolution; report mean and spread |
| Circularization slower than budget | t_circ beyond 3.8 × 10⁶ | extend in steps with checkpoints; don't raise η or C_cour to "speed up" (CLAUDE.md rule) |

**Scenario knob (only with Hiten's agreement, logged in DECISIONS.md):** if the crash starts too late for a good story, a larger cloud makes it start earlier (radius 100: passage 3; review §7). But at 10k a larger cloud loses more gas to the hole numerically (radius 75: 46+ of 10,000 against ~13 at radius 60), so it would need more particles (D3). Don't change the scenario quietly.

---

## 6.8 Physics questions this phase answers (for the report)

1. At which passage does the first crash happen, and how does that compare with the test-particle prediction (radius 60: passage 4, first contact t ≈ 2.77 million; passage 3 misses by only 8,700)? What made it earlier, if it was?
2. How much orbital energy is dissipated per crash, and where does it go (radiated, heat, unbinding gas)?
3. What is t_circ, in days and in ring orbits?
4. Thin ring or thick torus at β = 230? How does the outcome change with β (Bonnerot et al. 2016 comparison)?
5. How much slower is circularization without relativistic precession (Newtonian control)?
6. Peak luminosity in Eddington units.

---

## Done when

- [ ] Tests 1–9 pass with expected/actual printed.
- [ ] R6.1, R6.2, R6.3a–d, R6.4 complete with flat ledgers (|ΔE|/Σm|H| < 10⁻⁴ over the whole run including crashes). *(Hiten, 2026-10-21, DECISIONS: once the stream crash has started, the energy error is also judged against the energy the shocks process, Σ m (viscous heat), limit 10⁻³; the 10⁻⁴ limit on Σm|H| still applies before the crash. Reason: the crash turns ~4× the cloud's binding energy into heat.)*
- [ ] The numerical-circularization check (6.7) is done and written up.
- [ ] Resolution check of the lost gas (D4 update): a 50k radius-60 run through the first pass loses a smaller share than R6.1 (10k), both at the peak and after the pass (`scripts/loss_check.py R6.1_RUN 50K_RUN t=...`).
- [ ] Pictures and preview reviewed by Hiten.
- [ ] `docs/results-log.md` has dated entries answering 6.8's questions; committed.
