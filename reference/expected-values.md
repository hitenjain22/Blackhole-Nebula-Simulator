# Expected values for tests (and how each was computed)

Every number here was computed independently of the project code, at planning time, with SciPy/NumPy scripts in `reference/scripts/` (listed in §11). Units: G = c = M = 1 unless stated. "Test" values are requirements (CLAUDE.md: never change them or loosen tolerances; if one looks wrong, stop and report). "Estimate" values are predictions to compare against, not pass/fail thresholds.

Physical constants used (CODATA 2018 / SI exact): G = 6.67430 × 10⁻¹¹, c = 299,792,458 m/s, M☉ = 1.98847 × 10³⁰ kg, AU = 1.495978707 × 10¹¹ m, k_B = 1.380649 × 10⁻²³, m_p = 1.67262192369 × 10⁻²⁷ kg, h = 6.62607015 × 10⁻³⁴, σ_SB = 5.670374419 × 10⁻⁸, a_rad = 4σ/c = 7.565733 × 10⁻¹⁶ J m⁻³ K⁻⁴. Conventions: μ = 0.6, κ_es = 0.034 m²/kg (solar composition).

---

## §1 Units (M = 4.3 × 10⁶ M☉)

| Quantity | Value | Type |
|---|---|---|
| Time unit GM/c³ | 21.180252 s | test, rtol 1e-5 |
| Length unit GM/c² | 6.349680 × 10⁹ m = 6.349680 × 10⁶ km = 0.042445 AU | test, rtol 1e-5 |
| First pericentre t = 396,223.627 | 97.1310 days | test |
| Radial period 792,447.254 | 194.2619 days | test |
| Circular orbit at r_circ = 50.8896: GN period 2,303.257 | 13.551 hours (0.5646 days) | test |

---

## §2 Orbits (Phase 1)

**Cloud orbit r_p = 25, r_a = 5,000** (exact Schwarzschild turning points; `01`, `10`):

| Quantity | Value | Tolerance |
|---|---|---|
| L (= GN ℓ) | 7.35374434336 | rtol 1e-10 |
| E | 0.999801061330200 | rtol 1e-11 |
| ε_GN = H = (E² − 1)/2 | −1.98918881503 × 10⁻⁴ | rtol 1e-9 |
| Apocentre tangential speed (GN), v = L / (f(r_a) r_a) with f(r) = r/(r − 2) | 1.4701605691 × 10⁻³ | rtol 1e-8 |
| Precession per radial orbit | 0.419573409 rad = 24.03979° | rtol 1e-8 |
| Angle pericentre → apocentre | 192.019893° | rtol 1e-8 |
| Circularization radius r = (L² + √(L⁴ − 12L²))/2 | 50.88962372 | rtol 1e-8 |
| Weak-field estimate 6π/(a(1 − e²)) | 21.708° (exact must be larger) | sanity |
| First pericentre (DOP853 on the GN acceleration, rtol 1e-12) | t = 396,223.627 | rtol 1e-5 |
| Radial period | 792,447.254 | rtol 1e-5 |

**Circular orbits** (v = (r − 2)/(r√(r − 3)), H = (4 − r)/(2r(r − 3)), ℓ² = r²/(r − 3); GN period 2πr/v; Schwarzschild coordinate period 2πr^(3/2)):

| r | v | H | ℓ² | GN period | Schw. period |
|---|---|---|---|---|---|
| 6 | 0.38490017946 | −0.0555555555556 | 12 | 97.9452 | 92.3436 |
| 10 | 0.30237157841 | −0.0428571428571 | 14.2857142857 | 207.7968 | 198.6918 |
| 50 | 0.14003039184 | −0.00978723404255 | 53.1914893617 | 2,243.5077 | 2,221.4415 |
| 50.8896237 | 0.13882469970 | −0.00962002249550 | 54.0775558675 | 2,303.2568 | 2,280.9918 |
| 1000 | 0.031606977125 | −4.99498495487 × 10⁻⁴ | 1,003.00902708 | 198,791.086 | 198,691.765 |

**Integrator reference behaviour** (generalized leapfrog on (x, p); `02`, `03`, `04`):
- Circular r = 10, fixed step 0.02 r^(3/2), 200 orbits: |ΔH/H| = 5.6 × 10⁻⁹, |ΔL/L| = 1.6 × 10⁻¹⁴. Step 0.01 r^(3/2): 3.5 × 10⁻¹⁰ (error falls ~16× per halving on circular orbits: leading term cancels).
- Eccentric r_p = 10, r_a = 100, fixed dt = 2, 1, 0.5, max |ΔH/H| over 4 orbits: 1.5698 × 10⁻³, 3.9235 × 10⁻⁴, 9.8087 × 10⁻⁵ (ratio 4.00: second order).
- Cloud orbit, quantized adaptive steps η r^(3/2), 3 radial periods: η = 0.02: 2,561 steps, |ΔH/H| = 2.7 × 10⁻³; η = 0.005: 10,278 steps (~3,400 per orbit), |ΔH/H| = 4.2 × 10⁻⁵; |ΔL/L| ~ 10⁻¹⁴ in both.
- Hot-blob ledger orbit (Phase 4 test 4; `21`): r_a = 500, r_p = 25, ℓ = 7.1951756691 (from the exact turning-point condition), radial period ≈ 27,100. Gravity only, quantized adaptive steps from apocentre, **maximum** |ΔH/H| over one radial orbit: η = 0.005: 6.9 × 10⁻⁵ (2,574 steps); η = 0.0025: 1.7 × 10⁻⁵; η = 0.00125: 4.3 × 10⁻⁶ (10,296 steps). The error at the end of the orbit is far smaller (7 × 10⁻⁷ at η = 0.005) because the leapfrog's energy error oscillates and returns: always test the maximum, not the end value.
- Rejected: explicit kick-drift-kick with a velocity predictor drifts to ~10⁻⁴ in energy on the circular test; an implicit-kick (x, v) scheme blew up (NaN). Don't use either.

**Pericentre variants** (r_a = 5,000; `19`):

| r_p | L | ε | Precession | Crossing radius r_SI | r_circ |
|---|---|---|---|---|---|
| 15 | 5.87468269 | −1.993100 × 10⁻⁴ | 43.16715° | 450.90 | 31.19266 |
| 25 | 7.35374434 | −1.989189 × 10⁻⁴ | 24.03979° | 1,663.76 | 50.88962 |
| 40 | 9.14015689 | −1.983298 × 10⁻⁴ | 14.48218° | 3,402.84 | 80.42623 |

---

## §2b The cloud and the stream (Phases 5–6; estimates unless marked)

Test-particle model: uniform sphere R = 100 at (5,000, 0, 0), all with the apocentre velocity, GN acceleration, DOP853 (`11`, `12`). Pressure ignored. Seed-dependent at the ±few % level.

| Quantity | Value |
|---|---|
| Orbital energy spread (half-range) | 3.71 × 10⁻⁶ = ±1.87% of \|ε\| (estimate R/r_a² = 4.0 × 10⁻⁶, ±2%) |
| Passage 1 pericentre times | mean 395,900; full spread 21,800; std 5,500; set by x offset (correlation 0.998) |
| Passage 2 | mean 1,187,700; spread 65,100; std 16,400 |
| Passage 3 | mean 1,979,500; spread 108,700; std 27,400 |
| Self-intersection radius (exact orbit shape, 180° from pericentre) | **1,663.76** at polar angle 12.02° (Kepler-rotated formula: 1,571.6) |
| Time pericentre → r_SI | 36,970 (9.1 days) |
| Velocities at r_SI | \|v\| = 0.0283258, v_r = ∓0.0279797, v_t = 0.0044147; 162° apart; relative speed 0.05596 (16,800 km/s) |
| Streams cross? (need spread > ~74,000) | passage 1: no; passage 2: no; passage 3: yes, ~25% of the stream still incoming at first contact |
| Cloud density at start | 1.854 × 10⁻⁷ kg/m³; electron-scattering optical depth across R: 4,003 |
| Cloud tidal radius R (M/M_cloud)^(1/3) | ≈ 35,000 |
| Jeans mass at 10⁴ K and that density | ≈ 19 M☉ |

## §2c The cloud and the stream for the radii in use (added 2026-10-16; estimates)

Hiten's decision (DECISIONS, D4 update 2026-10-16): **Phase 6 and later use cloud radius 60**; Phase 5's check runs use radius 25. Same test-particle model as §2b (uniform sphere at (5,000, 0, 0), apocentre velocity, GN, DOP853, pressure ignored); §2b's radius-100 values are unchanged. Scripts: `23_cloud_spread_by_radius R` (script 11 with R as an argument; 400 particles, seed 1; R = 100 reproduces script 11's output exactly) and `24_crash_by_radius R K N` (the review's generalisation of script 12; seed 2; R = 100, N = 300 reproduces §2b: 395,891 / 21,792, 1,187,690 / 65,142, 1,979,490 / 108,731, cross at passage 3 with 25% incoming).

| Quantity | Radius 25 (Phase 5) | Radius 60 (Phase 6) |
|---|---|---|
| Orbital energy spread, half-range (`23`, 400 particles) | 9.28 × 10⁻⁷ = ±0.47% of \|ε\| (estimate R/r_a² = 1.0 × 10⁻⁶) | 2.23 × 10⁻⁶ = ±1.12% (estimate 2.4 × 10⁻⁶) |
| Passage 1 pericentre times (`23`) | mean 396,150; full spread 5,631; std 1,359 | mean 396,060; full spread 13,517; std 3,261 |
| Passage 1 (`24`, N = 300) | | mean 396,009; spread 13,078 |
| Passage 2 (`24`) | | mean 1,188,039; spread 39,095 |
| Passage 3 (`24`) | crash only at passage 8, t ≈ 5.9 million (review §7) | mean 1,980,068; spread 65,245; streams do not cross |
| Passage 4 (`24`) | | mean **2,772,097**; spread 91,407; **streams cross: 10% of the stream still incoming at first contact** |
| First outgoing gas at r_SI vs last incoming gas there (`24`) | | passage 3: 1,985,703 vs 1,977,007 (**misses by only 8,700**); passage 4: 2,765,139 vs 2,782,604 (cross) |
| **First contact** (first outgoing gas reaches r_SI on the crossing passage) | | **2,765,139** (≈ 1.86 years); radius 100 for comparison: 1,964,230 (passage 3) |

Notes. (1) N = 200 gives the same passage 4 (2,771,718; spread 91,407; 10%), so the result does not depend on the sample. (2) The half-range from 400 random points is ~7% below R/r_a² because the sample rarely reaches the sphere's edge; a dense particle cloud gives the estimate almost exactly (Phase 5 10k and 50k radius-25 runs: 9.93 and 9.97 × 10⁻⁷). (3) Real pressure and nozzle-shock heating add energy spread (Phase 5: ±14% at 50k, ±33% at 10k, middle 50%), which may bring the crash earlier than this test-particle estimate: at radius 60 passage 3 misses by only 8,700, so a little extra spread moves the first crash to passage 3 (t ≈ 1.98 million).

---

## §3 Light rays, Schwarzschild (Phase 2, 8)

Deflection for a ray from infinity with impact parameter b (`05`, `06`): b = 1000: 0.00401182381 rad; 50: 0.085083450; 10: 0.590395788; 6: 1.719388310; 5.3: 3.557938042. Capture: b = 5.19 captured, 5.20 escapes (critical 3√3 = 5.196152).

Shadow angular radius, static observer at D (sin ψ = (3√3/D)√(1 − 2/D)): D = 20: 14.2690273°; 50: 5.8441783°; 100: 2.9485552°; 1000: 0.29742107°. (Printed to enough digits for the rtol 1e-7 test. Rule everywhere in this file: the rounding of a printed value must be at least 10 times smaller than the tolerance of the test that uses it; if a test needs more digits, rerun the script in `scripts/`.)

Escape angle φ∞ (camera at D, ψ = angle from the direction to the hole; `07`): D = 50: ψ = 10° → 210.86090°, 20° → 175.52870°, 45° → 140.94777°, 80° → 102.85233°. D = 20: 20° → 219.43474°, 45° → 151.84667°, 80° → 107.64767°; 10° captured.

Einstein rings, source at infinity exactly behind the hole: D = 50: primary 17.872702°, secondary 5.850814°. D = 20: 29.828318°, 14.283091°.

Einstein rings, **point source at finite radius** r_s on the axis behind the hole (`08`): camera D = 50, r_s = 30: 11.701931°, 5.849751°; D = 50, r_s = 100: 14.958880°, 5.850477°; D = 30, r_s = 15: 15.114892°, 9.639931° (shadow at D = 30: 9.632732°).

---

## §4 SPH and shocks (Phases 3, 6)

**Kernel (cubic spline, 3D, support 2h):** W(0) = 1/(πh³) = 0.3183098862/h³; W(h) = 0.0795774715/h³; ∫W dV = 1.000000000000. Neighbours for η = 1.2: (4π/3)(2η)³ = 57.9058.

**Sod, γ = 1.4** (ρ, P) = (1, 1) | (0.125, 0.1), exact Riemann solver (`13`):
p* = 0.303130, u* = 0.927453, ρ*L = 0.426319, ρ*R = 0.265574, shock speed 1.752156, rarefaction head −1.183216, tail −0.070273. At t = 0.2 with the diaphragm at x = 0: head −0.23664, tail −0.01405, contact 0.18549, shock 0.35043. Specific internal energy: 1.777600 (left of contact), 2.853541 (right).

**Same, γ = 4/3** (radiation-dominated limit, `14`): p* 0.306011, u* 0.953205, ρ*L 0.411437, ρ*R 0.278585, shock 1.728999, head −1.154701, tail −0.042628.
**Same, γ = 5/3:** p* 0.293945, u* 0.841195, ρ*L 0.479689, ρ*R 0.229806, shock 1.844473, head −1.290994, tail −0.169401.

**Head-on collision of two cold streams at ±v** (strong-shock jump conditions): post-shock gas at rest, u = v²/2; compression (γ+1)/(γ−1) = 4 (γ = 5/3) or 7 (γ = 4/3); shock speed in the lab frame v/3 or v/6.

**Sedov blast** (γ = 5/3): R = ξ₀ (E t²/ρ)^(1/5), ξ₀ ≈ 1.1517. *From the literature (e.g., Kamm & Timmes 2007), not recomputed here; check before using as a tight tolerance.* **Verified 2026-10-15** (Phase 3, Step 3.7) by `22`, which integrates the self-similar equations from the shock inward: ξ₀ = 1.151666 (γ = 5/3); cross-check γ = 1.4 gives 1.032777 (classic value for air, 1.033).

**Gresho–Chan vortex** (ρ = 1, γ = 5/3; Gresho & Chan 1990 as used by Springel 2010): v_φ = 5r (r < 0.2); 2 − 5r (0.2 ≤ r < 0.4); 0 (r ≥ 0.4). P = 5 + 12.5r² (r < 0.2); 9 + 12.5r² − 20r + 4 ln(5r) (0.2–0.4); 3 + 4 ln 2 (r ≥ 0.4). Steady: any change is numerical error.

---

## §5 1D viscous disk (Phase 7)

**Lynden-Bell & Pringle ring, ν constant** (τ = 12νt/R₀², x = r/R₀; Σ π R₀²/m = τ⁻¹ x^(−1/4) e^(−(1−x)²/τ) ive(¼, 2x/τ); `13`):

| τ | Mass remaining | Σ(x = 0.5) | Σ(x = 1) | Σ(x = 1.5) |
|---|---|---|---|---|
| 0.016 | 1.000000000 | 1.957777 × 10⁻⁷ | 0.7104152 | 8.579919 × 10⁻⁸ |
| 0.032 | 1.000000000 | 3.426427 × 10⁻⁴ | 0.5027209 | 1.500074 × 10⁻⁴ |
| 0.064 | 0.999999995 | 0.01208259 | 0.3560243 | 0.005278396 |
| 0.128 | 0.999978021 | 0.06064328 | 0.2525431 | 0.02637038 |
| 0.256 | 0.998280186 | 0.1156337 | 0.1797739 | 0.04972546 |
| 0.512 | 0.981639256 | 0.1370121 | 0.1290960 | 0.05783287 |
| 1.0 | 0.932078868 | 0.1218845 | 0.09491733 | 0.05361829 |
| 2.0 | 0.846486404 | 0.08303970 | 0.06580134 | 0.04389108 |

The Σ columns are Σ R₀²/m, i.e. τ⁻¹ x^(−¼) e^(−(1−x)²/τ) ive(¼, 2x/τ) / π (shown to 7 significant figures; the formula itself is the 10⁻⁹ test). Angular momentum ∫2πxΣx^(1/2)dx = 1.000000000 at all τ. Half the mass accreted at τ = 22.897.

**Steady state** with zero torque at r_in: ν Σ = (Ṁ/3π)(1 − √(r_in/r)).

**Self-similar drain** (estimate, independent plan-time solver `15`; tolerance 3% in Phase 7 test 5): ν = α (H/r)² r^(1/2), Gaussian ring width 3% at r₀ = 50.8896, zero torque at r = 6, outer edge 5,000: half-mass at 0.659 t_ν; peak Ṁ at 0.07 t_ν; peak Ṁ × t_ν = 1.945 (ring mass = 1). t_ν = r₀²/ν(r₀):

| α | H/r | t_ν | t_half |
|---|---|---|---|
| 0.1 | 0.3 | 40,337 (9.9 d, 17.7 orbits) | 26,567 (6.5 d, 11.6 orbits) |
| 0.03 | 0.3 | 134,456 (33.0 d) | 88,557 (21.7 d) |
| 0.3 | 0.3 | 13,446 (3.3 d) | 8,856 (2.2 d) |
| 0.1 | 0.1 | 363,031 (89.0 d) | 239,103 (58.6 d) |

---

## §6 Redshift and gas 4-velocity (Phase 9)

**GN → Schwarzschild identity** (`16`): 1 + 2H = (1 − 2/r)(1 + ℓ²/r²) + (1 − 2/r)² p_r², max relative violation 7.1 × 10⁻¹⁶ over 1,000 random states (r ∈ [2.5, 10⁴]). Static-frame: γ = E/√(1 − 2/r), γβ_r = √(1 − 2/r) p_r, γβ_⊥ = p_⊥.

**g = [√(1 − 2/r)/√(1 − 2/r_cam)] / [γ(1 − β·n̂)]**, camera at infinity:

| Emitter | local β | γ | approaching | receding | transverse | static |
|---|---|---|---|---|---|---|
| circular r = 6 | 0.50000000 | 1.15470054 | 1.41421356 | 0.47140452 | 0.70710678 | 0.81649658 |
| circular r = 10 | 0.35355339 | 1.06904497 | 1.29424459 | 0.61812119 | 0.83666003 | 0.89442719 |
| circular r = 50.8896237 | 0.14301831 | 1.01038673 | 1.13196905 | 0.84869746 | 0.97007674 | 0.98015267 |

Cross-check form for r = 10: 1/(u^t(1 − Ωb)), b = ±r/√(1 − 2/r) = ±11.1803: 1.2942445894, 0.6181211856; b = 0: 0.8366600265. Camera at r_cam = 50, static emitter at r = 10: g = 0.91287093.

---

## §7 Colour (Phase 9)

Official CIE 1931 2° colour-matching functions, 1 nm, 360–830 nm (471 samples; obtained via the `colour-science` package), Planck with exact constants, summation at 1 nm (`17`). Linear sRGB normalized to the largest channel.

| T (K) | x | y | linear sRGB (max = 1) | Luminous efficacy (lm/W) |
|---|---|---|---|---|
| 1,000 | 0.65275 | 0.34446 | (1, 0.0087, −0.0195) out of gamut | 0.00 |
| 1,500 | 0.58572 | 0.39312 | (1, 0.1331, −0.0196) out of gamut | 0.08 |
| 2,000 | 0.52668 | 0.41330 | (1, 0.2568, 0.0081) | 1.61 |
| 3,000 | 0.43693 | 0.40407 | (1, 0.4769, 0.1537) | 20.67 |
| 4,000 | 0.38044 | 0.37675 | (1, 0.6537, 0.3773) | 54.71 |
| 5,000 | 0.34510 | 0.35161 | (1, 0.7917, 0.6284) | 81.39 |
| 5,778 | 0.32643 | 0.33572 | (1, 0.8773, 0.8218) | 92.08 |
| 6,500 | 0.31353 | 0.32363 | (1, 0.9429, 0.9922) | 95.36 |
| 10,000 | 0.28063 | 0.28829 | (0.6093, 0.6952, 1) | 73.47 |
| 20,000 | 0.25646 | 0.25763 | (0.4055, 0.5364, 1) | 21.66 |
| 30,000 | 0.25007 | 0.24879 | (0.3613, 0.4975, 1) | 8.31 |
| 10⁵ | 0.24258 | 0.23803 | (0.3137, 0.4534, 1) | 0.31 |
| 10⁶ | 0.24013 | 0.23442 | (0.2992, 0.4393, 1) | 0.00 |
| → ∞ (Rayleigh–Jeans) | 0.23988 | 0.23404 | (0.2976, 0.4379, 1) | — |

Cross-checked against `colour-science`'s own `sd_blackbody` + `sd_to_XYZ` at 2,000 and 6,500 K: identical to 5 decimals. (An analytic multi-lobe fit to the CMFs, Wyman et al. 2013, gives 2,000 K as (0.5242, 0.4157): off by 0.0025; don't use fits for tests.)

XYZ → linear sRGB (IEC 61966-2-1): [[3.2406, −1.5372, −0.4986], [−0.9689, 1.8758, 0.0415], [0.0557, −0.2040, 1.0570]].
Wien: λ_peak T = 2.897771955 × 10⁻³ m K (10⁴ K → 289.777 nm; 10⁵ K → 28.978 nm; 10⁶ K → 2.898 nm).

**LUT interpolation** (`21`; chroma(T) = XYZ/(σT⁴/π) on a log-T grid from 10³ to 10⁸ K): interpolating **log chroma linearly in log T**, maximum relative error 3.81 × 10⁻⁴ (1,024 entries), 9.54 × 10⁻⁵ (2,048), **2.38 × 10⁻⁵ (4,096; 1.41 × 10⁻⁵ for T ≥ 2,000 K)**. Interpolating chroma itself linearly: 9.31 × 10⁻³ (1,024), 5.86 × 10⁻⁴ (4,096). Phase 9 uses 4,096 entries, log–log.

**Band fractions** (fraction of σT⁴ emitted in each band; exact Planck integral, `21`; band edges 0.1 keV = 12.39842 nm and 2 keV = 0.619921 nm):

| T (K) | IR 1–5 μm | optical 380–780 nm | UV 100–300 nm | EUV 12.4–100 nm | soft X-ray 0.1–2 keV |
|---|---|---|---|---|---|
| 10⁴ | 8.474691 × 10⁻² | 0.4045955 | 0.2729085 | 3.207698 × 10⁻⁴ | 1.012644 × 10⁻⁴⁵ |
| 10⁵ | 1.435799 × 10⁻⁴ | 2.110052 × 10⁻³ | 8.113420 × 10⁻² | 0.9112872 | 2.869748 × 10⁻³ |
| 10⁶ | 1.508351 × 10⁻⁷ | 2.426829 × 10⁻⁶ | 1.392287 × 10⁻⁴ | 5.047924 × 10⁻² | 0.9493758 |

(The optical column is the plain wavelength band, for orientation; the true-colour pipeline uses the CIE functions instead.)

**Doppler test ring** (Phase 9 test 12): circular orbit at r = 1,000 has static-frame speed β = 1/√(r − 2) = 0.03165445; approaching/receding bolometric ratio ((1 + β)/(1 − β))⁴ = 1.288296. At r = 10 the same formula gives 19.22, but lensing changes what is seen there, so only the sign is tested.

---

## §8 Kerr (Phase 11)

Bardeen–Press–Teukolsky ISCO; photon orbits r = 2[1 + cos(⅔ arccos(∓a))]; efficiency 1 − E_ISCO with E = (r^(3/2) − 2r^(1/2) + a)/(r^(3/4)√(r^(3/2) − 3r^(1/2) + 2a)) (`05`, `18`, `20`):

| a | Horizon | ISCO pro | ISCO retro | Photon pro | Photon retro | Efficiency |
|---|---|---|---|---|---|---|
| 0 | 2 | 6 | 6 | 3 | 3 | 5.7191% |
| 0.5 | 1.86602540 | 4.23300253 | 7.55458471 | 2.34729636 | 3.53208889 | 8.2118% |
| 0.9 | 1.43588989 | 2.32088304 | 8.71735228 | 1.55785463 | 3.91026794 | 15.5753% |
| 0.99 | 1.14106736 | 1.45449794 | 8.97186134 | 1.16764185 | 3.99110286 | 26.4030% |

**Shadow, observer at infinity, edge-on** (Bardeen 1973: ξ = (−r³ + 3r² − a²(r + 1))/(a(r − 1)), η = r³(4a² − r(r − 3)²)/(a²(r − 1)²), α = −ξ, β = ±√η):
a = 0.5: α ∈ [−4.096267, 6.138156]; a = 0.9: [−2.844421, 6.832319]; a = 0.99: [−2.251724, 6.983323]. Max β = 3√3 = 5.196152 at α = 2a (photon orbit r = 3).

Prograde circular Ω_φ = 1/(r^(3/2) + a), a = 0.9: r = 6: 0.064115147; r = 10: 0.030747682; r = 50: 0.0028212454.

---

## §9 Scenario, radiation and equation of state (PLAN §3, Phases 6–7, 9)

**Eddington convention.** L_Edd = 4πGMm_p c/σ_T (pure hydrogen, the standard convention for quoting Eddington ratios) = 1.257 × 10³¹ W per M☉ → **5.41 × 10³⁷ W** for 4.3 × 10⁶ M☉. Ṁ_Edd = L_Edd/(0.1 c²) = **0.0955 M☉/yr**. (With the solar-composition κ_es = 0.034 used for opacity, 4πGMc/κ_es = 6.32 × 10³⁷ W; don't mix the two: reports use 5.41 × 10³⁷ W.)

**Feeding-rate estimate.** 0.1 M☉ with half accreted over ~22 days gives ~8.5 Ṁ_Edd; the 1D drain estimate (half in 6.5 days) would give ~30 Ṁ_Edd; the stream's arrival spread (weeks) likely sets the pace. Either way strongly super-Eddington.

**Torus estimate** (half the cloud, 0.05 M☉, in a torus at r = 50.89 with H/r = 0.3): ρ ≈ 1.66 × 10⁻⁶ kg/m³; specific energy scale u ≈ 4.42 × 10¹⁴ J/kg; radiation-dominated T ≈ 9.92 × 10⁵ K; τ_es ≈ 5,500; diffusion time ≈ 1.77 × 10⁶ s; orbital period 4.83 × 10⁴ s; **β = t_cool Ω ≈ 230** (t_cool ≈ 230/Ω ≈ 37 orbits). Surface area ≈ 1.24 × 10²⁴ m²; effective temperature at L = L_Edd ≈ 1.7 × 10⁵ K.

**Energy budget:** from ε = −1.98919 × 10⁻⁴ to circular at r_circ (ε = −9.62002 × 10⁻³): 9.4211 × 10⁻³ c² per unit mass; 1.6837 × 10⁴⁴ J for 0.1 M☉ (36 days at 5.41 × 10³⁷ W).

**Gas sound speed** at 10⁴ K, μ = 0.6, γ = 5/3: 15.14 km/s = 5.0509 × 10⁻⁵ c; u = 2.2961 × 10⁻⁹ c².

**EOS test points** (u = 3kT/(2μm_p) + aT⁴/ρ, P = ρkT/(μm_p) + aT⁴/3, `16`):

| ρ (kg/m³) | T (K) | u (J/kg) | P (Pa) | β_gas | Γ₁ |
|---|---|---|---|---|---|
| 10⁻³ | 10⁴ | 2.0636755965 × 10⁸ | 1.3757585119 × 10⁵ | 9.9998166894 × 10⁻¹ | 1.6666361185 |
| 10⁻⁶ | 10⁶ | 7.5659396100 × 10¹⁴ | 2.5220486567 × 10⁸ | 5.4548245497 × 10⁻⁵ | 1.3333424248 |
| 10⁻⁹ | 10⁵ | 7.5659396100 × 10¹³ | 2.5220486567 × 10⁴ | 5.4548245497 × 10⁻⁵ | 1.3333424248 |
| 10⁻⁹ | 10⁷ | 7.5657332502 × 10²¹ | 2.5219110835 × 10¹² | 5.4551221167 × 10⁻¹¹ | 1.3333333333 |
| 1.6586853 × 10⁻⁶ | 9.9188303029 × 10⁵ (recovered from u = 4.415186 × 10¹⁴) | 4.415186 × 10¹⁴ (input) | 2.4412478743 × 10⁸ | 9.2714321654 × 10⁻⁵ | 1.3333487859 |

Extreme points for the limit test (Phase 6 test 2):

| ρ (kg/m³) | T (K) | u (J/kg) | P (Pa) | Γ₁ |
|---|---|---|---|---|
| 10³ | 10³ | 2.0635999392 × 10⁷ | 1.3757332928 × 10¹⁰ | 1.6666666667 (gas limit) |
| 10⁻¹⁵ | 10⁸ | 7.5657332500 × 10³¹ | 2.5219110833 × 10¹⁶ | 1.3333333333 (radiation limit) |

All values to 10 significant figures (`21`), so the rtol 10⁻⁸ test is meaningful. They use μ = 0.6, γ = 5/3 and a_rad = 4σ_SB/c computed in float64 from the exact SI constants above.

Limits: Γ₁ → 5/3 (β_gas → 1), → 4/3 (β_gas → 0).

**Crash heating** (½v_r² = 3.9143 × 10⁻⁴ c² thermalized): gas-only T = 1.705 × 10⁹ K; with radiation: 4.643 × 10⁵ K (ρ = 10⁻⁶), 1.468 × 10⁵ K (10⁻⁸), 4.644 × 10⁴ K (10⁻¹⁰).

---

## §10 Spin dynamics (Phase 11)

Exact Kerr nodal precession for a slightly tilted prograde circular orbit, Ω_nod = Ω_φ[1 − √(1 − 4a r^(−3/2) + 3a²/r²)], a = 0.9 (`18`):

| r | Exact Kerr | 2a/r³ (LT only) | LT + quadrupole, leading order | LT / Kerr |
|---|---|---|---|---|
| 25 | 9.955963 × 10⁻⁵ | 1.152 × 10⁻⁴ | 9.96480 × 10⁻⁵ | 1.157 |
| 50 | 1.302237 × 10⁻⁵ | 1.44 × 10⁻⁵ | 1.302538 × 10⁻⁵ | 1.106 |
| 100 | 1.678400 × 10⁻⁶ | 1.8 × 10⁻⁶ | 1.678500 × 10⁻⁶ | 1.072 |
| 500 | 1.396527 × 10⁻⁸ | 1.44 × 10⁻⁸ | 1.396531 × 10⁻⁸ | 1.031 |

**Numerical experiment** (GN + spin terms, circular orbit tilted 10°, 40 orbits, DOP853): measured nodal rates

| Coupling | r = 50 | / Kerr | r = 500 | / Kerr |
|---|---|---|---|---|
| acceleration, LT only | 1.419385 × 10⁻⁵ | 1.0900 | 1.439489 × 10⁻⁸ | 1.0308 |
| acceleration, LT + Q | 1.286469 × 10⁻⁵ | 0.9879 | 1.396669 × 10⁻⁸ | 1.0001 |
| generalized force on p, LT only | 1.364570 × 10⁻⁵ | 1.0479 | 1.433737 × 10⁻⁸ | 1.0266 |
| generalized force on p, LT + Q | 1.236613 × 10⁻⁵ | 0.9496 | 1.391088 × 10⁻⁸ | 0.9961 |

Torus at r_circ, a = 0.9: nodal rate 1.2363 × 10⁻⁵, period 5.08 × 10⁵ (125 days).
Weak-field Lense–Thirring node shift per radial orbit for r_p = 25, r_a = 5,000, a = 0.9: 4πa/p^(3/2), p = 49.75 → 0.03223 rad = 1.847° (estimate; quadrupole and strong-field effects lower it). Offset at r_SI per passage: 26.8 (tilt 30°), 46.4 (tilt 60°).

---

## §11 Scripts (`reference/scripts/`)

Plan-time checks, not project code (SciPy/NumPy; `17` needs `colour-science`). Run `10` first (it writes `orbit_consts.npy` used by `11` and `12`).

| Script | Computes |
|---|---|
| 01_orbit_constants | L, E, ε, precession, r_circ, pericentre times |
| 02_generalized_leapfrog_check | the canonical integrator on circular and eccentric orbits |
| 03_leapfrog_convergence | second-order check on an eccentric orbit |
| 04_rejected_integrators | the (x, v) schemes that failed |
| 05_shadow_deflection_kerr_sod | shadow sizes, deflections, Kerr radii and shadow, Sod |
| 06_deflection_crosscheck | deflections by direct integration |
| 07_escape_angles_einstein_rings | escape angles, rings for sources at infinity |
| 08_einstein_rings_finite_source | rings for sources at finite radius |
| 09_doppler_lbp_colour_fit | early Doppler, LBP mass checks, CMF-fit colours |
| 10_units_orbits_crossing | units, circular orbits, crossing radius and velocities |
| 11_cloud_energy_spread | cloud energy spread and first-passage arrival times |
| 12_stream_crossing_by_passage | arrival spreads and crossing for passages 1–3 |
| 13_sod_kernel_lbp | Sod (γ = 1.4), kernel, LBP table |
| 14_sod_gamma43_collision | Sod γ = 4/3 and 5/3, collision jump |
| 15_torus_drain_1d | 1D drain estimates |
| 16_eos_redshift | EOS points, crash temperatures, 4-velocity identity, g table |
| 17_blackbody_colour_cie | blackbody chromaticities from CIE tables |
| 18_kerr_and_spin_coupling | Kerr shadow, nodal rates, coupling experiment |
| 19_pericentre_variants | precession, r_SI, r_circ for r_p = 15, 25, 40 |
| 20_kerr_efficiency_scenario | ISCO efficiencies, torus precession, Eddington and energy numbers |
| 21_precision_and_review_checks | extra-digit EOS, ray and orbit values; hot-blob ledger orbit; colour-LUT interpolation error; band fractions; Doppler test ring |
| 22_sedov_constant | Sedov–Taylor constant ξ₀ from the self-similar ODEs (added in Phase 3) |
| 23_cloud_spread_by_radius | script 11 for any cloud radius (added 2026-10-16, §2c) |
| 24_crash_by_radius | stream crossing by passage for any radius (from the external review, added 2026-10-16, §2c) |
