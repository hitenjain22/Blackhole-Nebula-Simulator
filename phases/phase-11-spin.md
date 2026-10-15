# Phase 11: The spinning black hole (in-depth)

**Goal:** add spin (a = J/M, from 0 to ~0.99). Spin shows up in three places: (1) **light**: the shadow becomes D-shaped and shifted, and the disk image becomes lopsided; (2) **gas orbits**: frame dragging makes tilted orbits precess around the spin axis (Lense–Thirring), which can make streams miss each other and can warp a tilted torus; (3) **the inner edge**: the innermost stable orbit moves inward for gas orbiting with the spin (r = 2.32 at a = 0.9).

This plan captures (1) exactly, (2) to about 1% for the precession rate, and does **not** capture (3) in the gas dynamics (explained below). That's an honest, defensible position; the report says it.

**Ends with:** Kerr ray tracing matching exact shadow shapes; a spinning-hole render of the torus; an aligned-spin run; a tilted-orbit run (30° or 60°) showing stream offsets and a precessing, possibly warped torus.

**Order and cut-list:** aligned first, tilted second. Both are cut-list items (PLAN §12). Don't start this phase before Phases 0–10 are done.

**Primary sources:** Bardeen, Press & Teukolsky 1972 (circular orbits and the ISCO in Kerr), Bardeen 1973 (Kerr shadow), Lense & Thirring 1918 (frame dragging, weak field), Chandrasekhar 1983 (*The Mathematical Theory of Black Holes*; reference for geodesics), James et al. 2015 (Kerr ray tracing for Interstellar), Guillochon & Ramirez-Ruiz 2015 (spin-induced stream misses delaying circularization), Nealon, Price & Nixon 2015 (warped disks around spinning holes in SPH), Bardeen & Petterson 1975 (alignment of inner tilted disks), Hayasaki, Stone & Loeb 2016 (circularization around spinning holes).

---

## 11.1 Numbers to know (expected-values §8 and §10)

| a | Horizon | ISCO prograde | ISCO retrograde | Photon orbit pro / retro | ISCO efficiency |
|---|---|---|---|---|---|
| 0 | 2 | 6 | 6 | 3 / 3 | 5.72% |
| 0.5 | 1.86603 | 4.23300 | 7.55458 | 2.34730 / 3.53209 | 8.21% |
| 0.9 | 1.43589 | 2.32088 | 8.71735 | 1.55785 / 3.91027 | 15.58% |
| 0.99 | 1.14107 | 1.45450 | 8.97186 | 1.16764 / 3.99110 | 26.40% |

Prograde circular orbit angular velocity: Ω_φ = 1/(r^(3/2) + a).

Nodal precession of a slightly tilted circular orbit (exact Kerr): Ω_nod = Ω_φ [1 − √(1 − 4a r^(−3/2) + 3a²/r²)]. For a = 0.9: 9.9560 × 10⁻⁵ (r = 25), 1.302237 × 10⁻⁵ (r = 50), 1.678400 × 10⁻⁶ (r = 100), 1.396527 × 10⁻⁸ (r = 500). The weak-field Lense–Thirring value 2a/r³ is 3% too high at r = 500 and 11% too high at r = 50, because spin also makes the hole oblate (a mass quadrupole).

---

## 11.2 Kerr light rays (renderer)

**Recommended formulation:** Hamiltonian ray integration in **Cartesian Kerr–Schild coordinates**: no coordinate singularity at the poles (Boyer–Lindquist has one), smooth through the horizon, and the inverse metric is simple:
```
g^μν = η^μν − f l^μ l^ν,     f = 2 r³ / (r⁴ + a² z²)
l_μ = (1, (r x + a y)/(r² + a²), (r y − a x)/(r² + a²), z / r)
r defined by  r⁴ − (x² + y² + z² − a²) r² − a² z² = 0
H = ½ g^μν p_μ p_ν = 0 for light;  dx^μ/dλ = ∂H/∂p_μ,  dp_μ/dλ = −∂H/∂x^μ
```
(Check every formula against a reference before coding; derive ∂H/∂x analytically and **test it against finite differences**. Signs of a and the l-vector depend on conventions; fix them by the tests below.)

- **Step control:** adaptive RK (float64 reference: DOP853 or RK45 with tight tolerance; WGSL: RK4 with step ∝ distance to the horizon and to the photon region). Terminate at r < r_h + 10⁻³ (captured) or r > r_escape (escaped). The second-order (Hamiltonian) form handles turning points naturally (Phase 2's lesson).
- **Camera:** a ZAMO (zero-angular-momentum observer) with an orthonormal tetrad built from the metric. At a = 0 it's the static observer of Phase 2.
- **Float64 CPU reference** in `physics/geodesics.py`; **WGSL** in `render/shaders/kerr_ray.wgsl`. The Phase 8–9 march loop takes positions along the ray from whichever integrator is active.

**Tests**

| # | Test | Requirement |
|---|---|---|
| 1 | a = 0 limit | escape angles and shadow sizes reproduce Phase 2's table (D = 20, 50) to the same tolerances |
| 2 | Hamiltonian | H = 0 conserved along rays to 10⁻¹⁰ (float64) |
| 3 | ∂H/∂x | analytic vs finite difference at 100 random points: 10⁻⁷ |
| 4 | **Shadow shape**, camera at D = 10⁴ in the equatorial plane, a = 0.9 | in impact-parameter coordinates (α horizontal, β vertical, spin axis up): shadow edge at α = −2.844421 and +6.832319 (β = 0); top at β = ±5.196152 (= 3√3) where α = 1.8 (= 2a). Rays 0.2% inside the curve are captured, 0.2% outside escape. Also a = 0.5 (α from −4.096267 to 6.138156) and a = 0.99 (−2.251724 to 6.983323). |
| 5 | **Which side is flat** | spin +z, camera on the +x axis looking at the hole, up = +z: the flattened edge is on the side where the hole's rotation moves toward the camera (prograde photons pass closer). Assert it explicitly in screen coordinates. |
| 6 | GPU vs CPU | as Phase 2 (capture classification identical outside ±10⁻⁴ of the critical curve; escape directions within the Phase 2 tolerances) |
| 7 | Kerr redshift | for gas on a prograde circular equatorial orbit (Kerr Ω_φ), g from the renderer matches the exact formula g = 1/(u^t (1 − Ω b)) with u^t from the Kerr circular-orbit expressions (Bardeen et al. 1972), to 10⁻⁶ |

**Gas 4-velocity in the spinning case:** the gas dynamics isn't exact Kerr (11.3), so its 4-velocity in Kerr is approximate: use the Phase 9 static-frame velocity (γ, β) as the velocity relative to the local ZAMO. Label it as an approximation in the report; it's accurate to the order of the spin corrections in the dynamics.

---

## 11.3 Spin in the gas dynamics

**Approach:** keep GN gravity (exact Schwarzschild orbit shapes) and add two spin accelerations: frame dragging (gravitomagnetic, Lense–Thirring) and the hole's spin-induced oblateness (mass quadrupole Q = −a²):
```
a_LT = (2 / r³) [ 3 (J · r̂)(r̂ × v) − J × v ]            J = a ĵ (spin vector)
a_Q  = −(a² / 2) [ (3 − 15 ζ²) x / r⁵ + 6 ζ ĵ / r⁴ ],     ζ = (ĵ · x) / r
```
(a_Q is minus the gradient of Φ_Q = (a²/(2r³))(3ζ² − 1).)

**How to couple it to the canonical (x, p) integrator:** convert the acceleration to a generalized force F = M(x) a_spin, where M(x) = (A I + C x xᵀ)⁻¹ is the inverse of Phase 1's `gn_mass_matrix_inv`. Then the equations of motion are exactly ẍ = a_GN + a_spin ("acceleration coupling"). Put F inside both implicit momentum stages of the generalized leapfrog (velocity v = ∂H/∂p at that stage), or as Strang half-kicks around it; choose by the tests.

**Why this choice (plan-time numerical experiment, expected-values §10):** inclined circular orbit (10°), a = 0.9:

| Coupling | r = 500 (rate / exact Kerr) | r = 50 (rate / exact Kerr) |
|---|---|---|
| acceleration coupling, LT + quadrupole | **1.0001** | **0.9879** |
| acceleration coupling, LT only | 1.0308 | 1.0900 |
| added directly to p, LT + quadrupole | 0.9961 | 0.9496 |
| added directly to p, LT only | 1.0266 | 1.0479 |

**Energy ledger:** these forces don't come from the GN Hamiltonian, so track their work W_spin = ∫ Σ m v · F dt and include it: Σm(H + u) + E_rad + E_acc + E_esc − W_spin = constant. (a_LT is perpendicular to v but F = M a_LT isn't, so W_spin ≠ 0.)

**Tests**

| # | Test | Requirement |
|---|---|---|
| 1 | a = 0 | identical to the non-spinning code (to 10⁻¹⁵) |
| 2 | Nodal rate, r = 500, i = 10°, a = 0.9, LT + Q | 1.3967 × 10⁻⁸ vs exact Kerr 1.396527 × 10⁻⁸: within 0.2%. LT only vs 2a/r³ = 1.44 × 10⁻⁸: within 0.2% |
| 3 | Nodal rate, r = 50, LT + Q | 0.988 ± 0.01 of exact Kerr (1.302237 × 10⁻⁵); record the measured value |
| 4 | Direction | for J ∥ +z and an orbit with L_z > 0, the ascending node advances in the direction of rotation |
| 5 | F conversion | ẍ computed from the canonical form equals a_GN + a_spin to 10⁻¹² at random states |
| 6 | Ledger | Phase 4 test 4's blob and settings (η_orb = 0.00125, C_cour = 0.15), on a 30°-tilted orbit: maximum ledger error (with W_spin) < 10⁻⁵ over one radial orbit, falling 3–5× when the steps are halved |

**What isn't captured: the ISCO shift.** GN's innermost stable orbit stays at r = 6 for any spin; at a = 0.9 the true prograde ISCO is 2.32 and the efficiency 15.6% instead of 5.7%. Options:
- (Default) accept and state it: spin runs show frame dragging and oblateness effects on the orbits, streams and torus, but the inner edge and efficiency are those of a non-spinning hole. The renderer still shows the true Kerr shadow.
- (Optional, only after research) a published pseudo-Newtonian or generalized-Newtonian description of Kerr that reproduces the ISCO and precession (candidates: Ghosh, Sarkar & Bhadra 2014, arXiv:1410.7900, a velocity-dependent Kerr analogue whose own abstract says the ISCO is reasonable only for a < 0.7; arXiv:1601.01034 on pseudo-Newtonian equations for stationary spacetimes; older Kerr pseudo-potentials such as Artemova et al. 1996 and Mukhopadhyay 2002). Use it only after reading the paper, reproducing its own test values, and logging the decision. **Don't invent one.**

---

## 11.4 Runs

| Run | Setup | What to look for |
|---|---|---|
| R11.1 | a = 0.9, spin ∥ orbit normal (aligned), 10k | small changes: in-plane periapsis precession is slightly reduced for a prograde orbit; crash location shifts slightly. Compare with R6.2/R7.1. |
| R11.2 | a = 0.9, spin tilted 30° to the orbit normal, 10k | the stream's orbital plane twists by roughly 1.6–1.8° per pericentre passage (weak-field Lense–Thirring estimate 1.85°; the quadrupole lowers it; measure). At the crash point r_SI ≈ 1,664 that's a vertical offset of ~27 per passage (30°) or ~46 (60°): **if the stream is thinner than that, the streams miss and the crash is delayed** (Guillochon & Ramirez-Ruiz 2015). Measure stream thickness at r_SI vs offset. |
| R11.3 | as R11.2 with 60° | stronger offsets; possibly a much later crash |
| R11.4 | render | Kerr shadow + lopsided torus; a = 0 vs a = 0.9 side by side, same camera |

**Torus precession:** a torus at r ≈ 51 precesses (exact Kerr rate for a = 0.9: 1.24 × 10⁻⁵, period ≈ 5.1 × 10⁵ ≈ 125 days). Our torus drains in about a week (Phase 7), so it **won't precess much** before draining. Rigid precession and inner alignment (Bardeen–Petterson) need longer-lived disks; mention them, compare with Nealon, Price & Nixon 2015 qualitatively, and measure tilt vs radius over time (a warp diagnostic: tilt and twist angles of the angular momentum of annuli).

---

## 11.5 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Sign convention of a or the l-vector | shadow flattened on the wrong side; nodes regress | tests 4–5 (rays) and test 4 (orbits) fix the conventions |
| Steps shrink to nothing near the horizon | ray integration stalls | terminate at r_h + 10⁻³; step ∝ (r − r_h) |
| Boyer–Lindquist pole singularity | streaks along the spin axis | that's why Kerr–Schild is recommended |
| float32 errors in WGSL near the photon region | noisy shadow edge | test 6; integrate the second-order form; tighten steps there |
| ZAMO tetrad wrong | everything slightly distorted; a = 0 limit fails | test 1 |
| Spin forces break the ledger | drift in tilted runs | include W_spin; test 6 |
| "Spin barely changes anything" | aligned run looks like a = 0 | expected for dynamics far from the hole; the visible differences are the shadow and the inner image; say so |
| Claiming ISCO effects | report says the disk extends to 2.32 | it doesn't in this model; the renderer may show gas to r = 6 only |

---

## 11.6 Physics questions this phase answers

1. How different does the spinning hole look (shadow shape, image asymmetry) for a = 0.5, 0.9, 0.99?
2. Does spin-tilt delay the crash, and by how many passages?
3. Does the tilted torus warp or precess measurably before it drains?

---

## Done when

- [ ] Ray tests 1–7 and dynamics tests 1–6 pass.
- [ ] R11.1 and R11.4 done; R11.2/R11.3 done or explicitly cut (cut list).
- [ ] The ISCO limitation is written into the report; committed.
