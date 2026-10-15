# Phase 1: Units and orbits around the hole (detailed)

**Goal:** the gravity of the black hole, exactly right for orbits, plus the integrator that will later carry the gas. Everything here is checked against independently computed numbers in `reference/expected-values.md` §1–2.

**Ends with:** a plot of the cloud's centre on its orbit, rotating 24.04° per pass, overlaid on the exact relativistic orbit and on Newton's (which doesn't rotate); a plunge demo at the innermost stable orbit; all tests passing.

---

## Background: the gravity model (read before coding)

We use the **generalized Newtonian (GN)** description of Tejeda & Rosswog 2013. Its Lagrangian per unit mass (M = 1, f(r) = r/(r − 2)):

```
L = ½ f |v|² + ½ (f² − f) ṙ² + 1/r,      ṙ = (x·v)/r
```

Its orbits have **exactly** the Schwarzschild shape r(φ): same precession, same innermost stable circular orbit (ISCO) at r = 6, same marginally bound orbit at r = 4. The conserved angular momentum equals Schwarzschild's L, and the conserved energy is ε = (E² − 1)/2 where E is Schwarzschild's energy. Only the *timing* differs slightly near the hole (circular period at r = 6 is 97.9 vs 92.3 in Schwarzschild coordinate time).

**Acceleration form** (derived from the Lagrangian and checked numerically for this plan):

```
a = −(1/r³)(1 − 2/r)² x  +  [2 ṙ / (r (r − 2))] v  −  [3 |x × v|² / r⁵] x
```

**Hamiltonian (canonical) form, which we integrate.** Canonical momentum p = f v + g ṙ r̂ with g = 2r/(r − 2)². Then

```
H(x, p) = ½ A(r) |p|² + ½ C(r) (p·x)² − 1/r
A(r) = (r − 2)/r,      C(r) = −2 (r − 2)/r⁴
v = ∂H/∂p = A p + C (p·x) x
∂H/∂x = [ ½ A'(r) |p|² + ½ C'(r) (p·x)² ] x/r + C (p·x) p + x/r³
A'(r) = 2/r²,          C'(r) = (6r − 16)/r⁵
```

H equals the GN energy ε, and the angular momentum is |x × p|, both exactly conserved.

**Why canonical variables:** with the **generalized leapfrog** (a symplectic, time-reversible, second-order method for Hamiltonians like this), angular momentum is conserved to machine precision and energy errors stay bounded. In a test for this plan, a circular orbit at r = 10 kept |ΔH/H| ≈ 6 × 10⁻⁹ and |ΔL/L| ≈ 10⁻¹⁴ over 200 orbits (step 0.02 r^{3/2}), while an explicit velocity-Verlet with a predictor drifted to 10⁻⁴ in energy. A naive "implicit kick" variant blew up entirely. So: integrate (x, p), not (x, v).

**Why this also matters for the gas (Phase 4):** pressure and viscosity forces will be added as kicks to **p** (as generalized forces). Then the total energy Σ m (H + u) and angular momentum Σ m (x × p) are conserved exactly by construction (dH/dt = v·F_hydro, and SPH pair forces are central and antisymmetric), which gives a clean conservation ledger. Adding hydro forces to the velocity instead would break the energy ledger by a fraction of order 2/r near the hole.

---

## Step 1.1: Units module

**Do:** `src/nebula/units.py`:
- CODATA 2018 constants: G, c, M☉, AU, σ_SB, a_rad, k_B, m_p, and conventional values (μ = 0.6 for ionized gas, κ_es = 0.034 m²/kg).
- `BlackHoleUnits(mass_msun)` → length unit GM/c² (m), time unit GM/c³ (s), velocity unit c, specific energy unit c².
- Gas conversions given a cloud mass scale: code density → kg/m³ (needs total cloud mass and the code-unit cloud mass), specific internal energy → temperature (gas only, and gas + radiation in Phase 6).
- Helpers to print times in hours/days/years and lengths in km/AU.

**Tests** (expected-values §1): for M = 4.3 × 10⁶ M☉, time unit 21.1803 s (rtol 1e-5), length unit 6.34968 × 10⁹ m (rtol 1e-5). 1 code time unit × 396,223.6 = 97.1 days.

**Hiten looks at:** a printed table: "1 M of time = 21.18 s; pericentre after 97 days; ring orbit 13.55 h" etc.

---

## Step 1.2: GN gravity functions

**Do:** `src/nebula/physics/gravity.py` (pure functions, float64, scalar and vectorized forms):
- `gn_accel(x, v)` (Lagrangian form above; used for cross-checks and plots).
- `gn_p_from_v(x, v)`, `gn_v_from_p(x, p)`, `gn_hamiltonian(x, p)`, `gn_dH_dx(x, p)`, `gn_dH_dp(x, p)`, `gn_angmom(x, p)`.
- `gn_mass_matrix_inv(x)` (A I + C x xᵀ, the matrix that turns p into v), used in Phase 4 consistency checks and in Phase 11, where its inverse turns the spin accelerations into forces on p.

**Tests:**
1. Round trip v → p → v to 1e-14 relative at random points (r between 4 and 10⁴).
2. **Independent derivation check:** compute the Euler–Lagrange acceleration by finite-differencing the Lagrangian numerically (central differences on ∂L/∂x, ∂²L/∂v∂v, ∂²L/∂v∂x, solve the 3×3 system) at 20 random points; must match `gn_accel` to 1e-6 relative. This catches any typo in the closed form.
3. Hamilton's equations from `gn_dH_dx`, `gn_dH_dp` reproduce `gn_accel` (differentiate v along a short numerical trajectory): 1e-6.
4. Circular orbits: for r = 6, 10, 50, 1000, with tangential speed v = (r − 2)/(r √(r − 3)):
   - acceleration is purely radial and equals v²/r inward (centripetal) to 1e-12;
   - H = (4 − r)/(2 r (r − 3)) to 1e-13;
   - |x × p|² = r²/(r − 3) to 1e-13.
5. ISCO: the function L²(r) = r²/(r − 3) has its minimum at r = 6 (check numerically to 1e-8).
6. Far field: at r = 10⁶ the acceleration matches Newton to 1e-5 relative.

---

## Step 1.3: Orbit utilities

**Do:** `src/nebula/physics/orbits.py`:
- `schwarzschild_EL_from_turning_points(r_peri, r_apo)` → (E, L): solve E² = (1 − 2/r)(1 + L²/r²) at both radii.
- `gn_energy_from_E(E)` = (E² − 1)/2.
- `apocentre_state(r_peri, r_apo)` → (x0, v0, p0) at apocentre on +x moving +y: v_t = L / (f(r_a) r_a).
- `precession_exact(r_peri, r_apo)`: Δφ − 2π with Δφ = 2∫ (L/r²) / √(E² − (1−2/r)(1+L²/r²)) dr, computed in u = 1/r with the substitution u = mid + half·cos θ to remove endpoint singularities.
- `circularization_radius(L)`: circular orbit with the same L: r = (L² + √(L⁴ − 12 L²))/2.
- `isco_radius(a=0)` (Kerr formula; a = 0 → 6, used in Phase 11).

**Tests** (expected-values §2), for r_p = 25, r_a = 5000:

| Quantity | Expected | Tolerance |
|---|---|---|
| L | 7.35374434336 | rtol 1e-10 |
| E | 0.99980106133 | rtol 1e-11 |
| ε_GN = H | −1.98918881503 × 10⁻⁴ | rtol 1e-9 |
| apocentre tangential speed (GN) | 1.47016057 × 10⁻³ | rtol 1e-8 |
| precession per radial orbit | 0.419573409 rad = 24.03979° | rtol 1e-8 |
| circularization radius | 50.8896237 | rtol 1e-8 |
| weak-field estimate 6π/(a(1−e²)) | 21.708° (sanity: exact must be larger) | — |

---

## Step 1.4: The integrator for test particles

**Do:** `src/nebula/sim/integrator.py`:
- `glf_step(x, p, dt)`: generalized leapfrog
  1. p½ = p − (dt/2) ∂H/∂x(x, p½)  (fixed-point iteration)
  2. x₁ = x + (dt/2) [∂H/∂p(x, p½) + ∂H/∂p(x₁, p½)]  (fixed-point iteration)
  3. p₁ = p½ − (dt/2) ∂H/∂x(x₁, p½)
  Iterate each implicit stage until the change is below 1e-15 relative or 10 iterations; count iterations and **raise** if not converged (never silently continue).
- `orbit_dt(r, eta)` = η r^{3/2}, quantized down to a power-of-two fraction of a maximum step (this is the seed of Phase 4's block steps).
- A reference integrator for tests only: SciPy `solve_ivp` (DOP853, rtol 1e-12) on `gn_accel`.
- Vectorized/Numba version for many particles (`glf_step_many`), tested against the scalar one.

**Tests** (these are *requirements*; choose η to meet them and record the η chosen, never the other way round):
1. Circular orbit r = 10, fixed step: over 1,000 orbits |ΔH/H| < 10⁻⁸ and |ΔL/L| < 10⁻¹². (For reference: η = 0.02 gave 6 × 10⁻⁹ over 200 orbits in the plan's own test.)
2. Second-order convergence: on an eccentric orbit (r_p = 10, r_a = 100, started at apocentre) with fixed steps dt = 2, 1, 0.5, the **maximum** |ΔH/H| over 4 radial orbits drops by 4 ± 0.2 per halving (plan-time values for reference: 1.5698 × 10⁻³, 3.9235 × 10⁻⁴, 9.8087 × 10⁻⁵; expect yours within ~5%). Don't use a circular orbit for this test: there the leading error term cancels and the error drops ~16× per halving (plan's check at r = 10: 5.6 × 10⁻⁹ → 3.5 × 10⁻¹⁰), which would look like a failure.
3. Time reversibility: integrate 50 orbits forward, flip p, integrate back: return to the start within 10⁻⁸ relative.
4. The cloud orbit (r_p = 25, r_a = 5000) with quantized adaptive steps, over 3 radial periods:
   - pericentre times match the DOP853 reference: first at t = 396,223.627 (rtol 1e-5), radial period 792,447.254 (rtol 1e-5);
   - precession per orbit 24.0398° (abs 0.01°);
   - |ΔH/H| < 10⁻⁴; |ΔL/L| < 10⁻¹².
   (Plan's own test: η = 0.005 gave |ΔH/H| = 4 × 10⁻⁵ with ~3,400 steps per radial orbit; η = 0.02 gave 2.7 × 10⁻³, too large.)
5. `glf_step_many` equals `glf_step` bit-for-bit (or to 1e-15) on 100 random particles.

**What can go wrong:**
- *Energy drift with adaptive steps.* Changing the step size breaks exact symplecticity; the error shows up mostly at pericentre. Fixes, in order: smaller η; choose the step from the *minimum* r over the step (predict r at mid-step); time-symmetric step selection (Hut, Makino & McMillan 1995). Don't paper over it.
- *Fixed-point iteration not converging* at very small r (< 4) or large dt: that's a dt problem; the step criterion must prevent it. Particles inside r_acc = 4 are removed in Phase 5 anyway.
- *Catastrophic cancellation* in A(r) = (r − 2)/r and C(r) near r = 2: irrelevant above r_acc = 4, but assert r > 2.5 in debug mode.
- *The "explicit with predictor" shortcut:* it looks fine for a few orbits and drifts over hundreds. Don't use it.

---

## Step 1.5: Orbit pictures for Hiten (and the report)

**Do:** `notebooks/01_orbits.ipynb` (or `scripts/plot_orbits.py`) producing in `outputs/checks/` (and copies of the best ones in `docs/figures/`):
1. `orbit_rosette.png`: cloud-centre orbit over 5 radial periods in the x–y plane: GN (solid), exact Schwarzschild from the precession integral / a proper-time geodesic reference (dashed, should overlap), Newton (dotted, closed ellipse). Hole drawn at the origin with radius 2; ISCO circle at 6.
2. `orbit_zoom_pericentre.png`: zoom on the pericentre passages, showing the 24.04° rotation with an annotated angle.
3. `isco_demo.png`: radius vs time for particles started on circular orbits at r = 6.2 and r = 5.8 with a 0.1% velocity nudge: the first oscillates and stays, the second plunges.
4. `timeline.txt`: a table of key times in code units and physical units (days/hours).

**Hiten looks at:** the rosette and the ISCO plot. Claude explains in two sentences what each shows.

---

## Done when

- [ ] All tests in Steps 1.1–1.4 pass, printed with expected/actual.
- [ ] η for eccentric orbits chosen and recorded in DECISIONS.md with the test evidence.
- [ ] Plots reviewed by Hiten; committed.
