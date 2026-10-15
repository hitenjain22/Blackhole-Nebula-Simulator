# Phase 9: Colour physics: true colour, Doppler and redshift (in-depth)

**Goal:** make every pixel's colour come from physics. Gas glows as a blackbody at its temperature; light climbing out of the hole's gravity is reddened; gas moving toward the camera is brightened and blued (Doppler beaming), gas moving away is dimmed and reddened. Then show the same physics three ways: true colour (default), a multi-band view from infrared to X-ray, and a clearly labelled cinematic grade.

**Ends with:** a colour pipeline tested against official colour-science data and exact relativistic formulas; true-colour frames of real snapshots with the lopsided, Doppler-beamed torus; multi-band frames; one cinematic-grade frame labelled as such.

**Primary sources:** CIE 1931 2° standard observer colour-matching functions (official CIE dataset); IEC 61966-2-1 (sRGB); Rybicki & Lightman 1979 (I_ν/ν³ is invariant along a ray); Misner, Thorne & Wheeler 1973 or any GR text (photon redshift g = (k·u)_obs / (k·u)_em); Luminet 1979 (Doppler-beamed disk image); James et al. 2015 (the Interstellar team turned Doppler beaming off for the film; we keep it on and say so).

---

## 9.1 Blackbody colour

**Data.** Store the official CIE 1931 2° colour-matching functions at 1 nm, 360–830 nm (471 rows) in `data/cie1931_2deg_1nm.csv`, with source and licence recorded in `data/README.md`. Get it from the CIE's published dataset or from the `colour-science` package (BSD licence; can be a dev dependency used once to export the table). Check and record the licence before committing.

**LUT.** For T on a log grid from 10³ to 10⁸ K (**4,096 entries**), compute
```
X(T) = Σ_λ B_λ(T) x̄(λ) Δλ     (same for Y, Z; Δλ = 1 nm; Planck with exact SI constants)
chroma(T) = (X, Y, Z) / (σ T⁴ / π)          (colour per unit bolometric intensity)
```
**Interpolate log(chroma) linearly in log T** (store the logs). Plan-time check (`21`): this gives a maximum relative error of 2.4 × 10⁻⁵ with 4,096 entries (1.4 × 10⁻⁵ for T ≥ 2,000 K). Interpolating chroma itself linearly is much worse (5.9 × 10⁻⁴ at 4,096 entries, 9.3 × 10⁻³ at 1,024), because chroma falls exponentially at low T.

In the shader, a blackbody at temperature T contributes XYZ = chroma(T) · σT⁴/π. Compute T⁴ as exp(4 ln(T/T_ref)) × T_ref⁴ in scaled units to avoid float32 overflow.

**To linear sRGB:** matrix (IEC 61966-2-1, as in expected-values §7):
```
 3.2406  −1.5372  −0.4986
−0.9689   1.8758   0.0415
 0.0557  −0.2040   1.0570
```
**Out-of-gamut colours** (cool blackbodies below ~1,900 K have negative blue): never clamp channels independently (that shifts hue); desaturate toward the same luminance Y until in gamut.

**What true colour will show** (worth explaining in the README): hot blackbodies all look nearly the same blue-white. Chromaticity at 10⁴ K is (0.2806, 0.2883), at 3 × 10⁴ K (0.2501, 0.2488), and it converges to (0.2399, 0.2340) as T → ∞. Our gas is ≥ 10⁴ K (the floor), so in true colour the torus is **blue-white**, not orange. Orange-red needs 2,000–4,000 K. The orange look Hiten imagined lives in the cinematic grade (9.6), labelled.

---

## 9.2 From gas state to 4-velocity (exact)

The renderer needs each gas element's 4-velocity in the Schwarzschild spacetime. For GN dynamics this map is **exact** (derived for this plan and checked numerically to 7 × 10⁻¹⁶ at 1,000 random states):

```
H   = GN Hamiltonian of the particle (Phase 1),   E = √(1 + 2H)      (energy at infinity per unit mass)
p_r = p · r̂,    p_⊥ = p − p_r r̂                    (|p_⊥| = ℓ / r)
Identity: 1 + 2H = (1 − 2/r)(1 + ℓ²/r²) + (1 − 2/r)² p_r²     (so E is always real for r > 2)
```
In the frame of a **static observer** at the gas's position (orthonormal r̂, angular directions):
```
γ = E / √(1 − 2/r)
γ β_r = √(1 − 2/r) · p_r
γ β_⊥ = p_⊥                       (vector)
check: γ² − (γβ_r)² − |γβ_⊥|² = 1 exactly
```
So the GN canonical momentum is the spatial part of the Schwarzschild 4-velocity (covariant components). For gas with pressure the map still gives a valid 4-velocity at each instant; the renderer uses it as the gas velocity. The grids (Phase 8) store mass-weighted γβ (not β: averaging γβ keeps it physical).

---

## 9.3 The redshift factor g

For gas at radius r emitting light that travels toward the camera (a static observer at r_cam), with n̂ the direction the light travels (in the static frame at the gas):
```
g = ν_received / ν_emitted = [ √(1 − 2/r) / √(1 − 2/r_cam) ] / [ γ (1 − β · n̂) ]
```
The first factor is gravitational, the second Doppler. n̂ is **minus** the direction of the backward-traced camera ray at that point. Static-frame ray direction from Phase 2's u(φ): the backward ray's orthonormal components are ∝ (dr/√(1 − 2/r), r dφ) in its plane, with dr = −du/u².

**Tests (expected-values §6, camera at infinity unless stated):**

| Emitter | g approaching (n̂ = +v̂) | g receding | g transverse (n̂ ⊥ v) | static emitter |
|---|---|---|---|---|
| circular orbit r = 6 (β = 0.5) | 1.41421356 | 0.47140452 | 0.70710678 | 0.81649658 |
| circular orbit r = 10 (β = 0.35355339) | 1.29424459 | 0.61812119 | 0.83666003 | 0.89442719 |
| circular orbit r = 50.8896 (β = 0.14301831) | 1.13196905 | 0.84869746 | 0.97007674 | 0.98015267 |

- Camera at r_cam = 50, static emitter at r = 10: g = 0.91287093.
- Cross-check with the impact-parameter form for circular orbits, g = 1/(u^t (1 − Ω b)) with u^t = 1/√(1 − 3/r), Ω = r^(−3/2), b = ±r/√(1 − 2/r): must agree to 10⁻¹².
- Circular-orbit inputs: E = (1 − 2/r)/√(1 − 3/r), ℓ = r/√(r − 3) (Schwarzschild; the GN circular orbit has exactly these E and ℓ).

---

## 9.4 Relativistic radiative transfer along the ray

Replace Phase 8's simple step with the exact version. With the gas emitting as a blackbody at temperature T (its own frame) with a grey absorption coefficient α_gas (its own frame), each march step contributes
```
Δs_static = √(dr² / (1 − 2/r) + r² dΩ²)                  (static-frame length of the step)
Δs_gas    = Δs_static · γ (1 − β · n̂)                     (length in the gas frame)
Δτ        = α_gas · Δs_gas
I_XYZ    += T_trans · (1 − e^(−Δτ)) · XYZ_blackbody(g T)
T_trans  *= e^(−Δτ)
```
**Why this is exact:** I_ν/ν³ is conserved along rays, and a blackbody obeys g³ B_(ν/g)(T) = B_ν(gT). So an observed blackbody is a blackbody at the shifted temperature gT, in both the opaque and the transparent limits. The famous g⁴ brightening is already inside σ(gT)⁴: **do not multiply by g⁴ again.**

**Opacity model (α_gas).** Electron scattering (κ_es = 0.034 m²/kg) makes the torus opaque but doesn't create or destroy photons, so it doesn't set the colour by itself. Options, in order:
1. Default: grey effective absorption κ_eff = √(κ_abs (κ_abs + κ_es)) with κ_abs a free–free (Kramers-type) mean opacity ∝ ρ T^(−7/2), from `physics/opacity.py` (already written in Phase 6 for the LTE diagnostic). **Look up the coefficient in a textbook** (e.g., Frank, King & Raine, *Accretion Power in Astrophysics*) before using it; it isn't reproduced here because it wasn't re-checked for this plan.
2. Simpler fallback: κ_abs a documented constant.

**Consistency check (not a test tolerance):** the visible torus surface should come out near the effective temperature implied by its luminosity: T_eff = (L / (σ A))^(1/4) ≈ 1.7 × 10⁵ K for L = L_Edd and the torus surface area ≈ 1.2 × 10²⁴ m² (expected-values §9). If the image's surface temperature is wildly different, the opacity model or the cooling luminosity is off.

---

## 9.5 Exposure and tone mapping

- Pipeline: HDR XYZ → linear sRGB → exposure → **tone curve on luminance only** (filmic/ACES-like curve applied to Y, then RGB scaled by Y_out/Y_in: preserves hue) → gamut handling (9.1) → sRGB transfer function (12.92x below 0.0031308, 1.055x^(1/2.4) − 0.055 above) → 8-bit with ordered dithering (or 16-bit PNG for masters).
- Exposure: keyframed per shot (final renders); automatic for previews (a high percentile of luminance, smoothed over ~1 s of video so it doesn't pump).
- **Stars:** physically, stars would be invisible next to a torus this bright. The star-field brightness is an artistic multiplier, labelled in the shot file and the video credits.

---

## 9.6 Three ways to show the same physics

1. **True colour (default).** Everything above.
2. **Multi-band mode.** LUTs of band-integrated Planck intensity per unit bolometric for five contiguous-where-it-matters bands: infrared (1–5 μm), optical (CIE), ultraviolet (100–300 nm), **extreme ultraviolet (EUV, 12.4–100 nm)** and soft X-ray (0.1–2 keV ≈ 0.62–12.4 nm). Each band shown in a fixed false colour, or a slider sweeping IR → X-ray. Where the gas emits (expected-values §7 band table): the 10⁴ K cloud mostly in optical and UV; 10⁵ K gas 91% in the EUV; 10⁶ K gas 95% in soft X-rays. The optical view of the hot torus shows a tiny fraction (10⁻³ to 10⁻⁶) of its light, still very bright because the total is huge. Caveats for the report: EUV light is absorbed by interstellar hydrogen, so no telescope would see that band from Earth; real systems also have non-thermal X-ray coronae, not modelled.
3. **Cinematic grade.** The same luminance, re-coloured with an artistic temperature palette (deep red → orange → yellow-white). File names end `_cinematic`; an on-screen label says "cinematic colour grade, not true colour". Doppler beaming stays on unless the shot says otherwise (and then that's labelled too).

---

## 9.7 Tests

| # | Test | Requirement |
|---|---|---|
| 1 | CMF table integrity | 471 rows, 360–830 nm; an equal-energy spectrum gives x = y = 1/3 to 10⁻⁴ |
| 2 | Planck chromaticity | (x, y) at 2,000 K (0.52668, 0.41330), 6,500 K (0.31353, 0.32363), T → ∞ (0.23988, 0.23404) to 2 × 10⁻⁴; the full table in expected-values §7 to 5 × 10⁻⁴ |
| 3 | Normalization | luminous efficacy 683 · Y / (σT⁴/π) = 95.36 lm/W at 6,500 K (±0.1); peak efficacy between 6,000 and 7,000 K |
| 4 | Wien | peak of B_λ at λ T = 2.897771955 × 10⁻³ m K (10⁻⁶) |
| 5 | sRGB matrix | D65 white (x 0.3127, y 0.3290, Y = 1) → (1, 1, 1) within 10⁻³ |
| 6 | sRGB transfer | round trip to 10⁻⁷; continuous at the 0.0031308 knee |
| 7 | Hue preservation | tone mapping leaves in-gamut chromaticities unchanged (10⁻³) |
| 8 | LUT accuracy | LUT (log-log interpolation, 4,096 entries) vs direct integration at 50 random T between nodes: < 5 × 10⁻⁵ relative |
| 9 | 4-velocity identity | γ² − \|γβ\|² = 1 to 10⁻¹² at 1,000 random GN states (r from 2.5 to 10⁴) |
| 10 | g table (9.3) | to 10⁻⁸; cross-check form to 10⁻¹² |
| 11 | Opaque static shell, r = 10, T = 10⁴ K, viewed from far | rendered bolometric intensity σ(gT)⁴/π with g = 0.89442719 (0.5%); colour = blackbody at gT |
| 12 | **Which side is bright** | (a) Quantitative: a thin opaque ring at r = 1,000 on circular orbits (β = 1/√(r − 2) = 0.0316544 in the static frame), rotating counterclockwise seen from +z, viewed edge-on from D = 10⁵ on the +x axis. At the two ends of the ring (moving straight toward and away from the camera) the bolometric intensity ratio is ((1 + β)/(1 − β))⁴ = 1.28830, within 2% (light bending is negligible this far out). (b) Qualitative, at r = 10 near the hole: the approaching side is brighter and bluer (the exact ratio there depends on lensing, so only the sign is tested). Getting this backwards is the classic sign error. |
| 13 | GPU vs CPU colour pipeline | per-pixel difference < 10⁻³ on a 64 × 64 frame |
| 14 | Band LUTs | at LUT nodes: each band integral vs direct numerical integration to 10⁻⁶ relative (expected-values §7 band table at 10⁴, 10⁵, 10⁶ K); between nodes, the same < 5 × 10⁻⁵ as test 8 |

---

## 9.8 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| g⁴ applied on top of B(gT) | the beamed side absurdly bright | the shift is inside B(gT); test 11 |
| g applied to brightness but not temperature | Doppler side brighter but not bluer | colour from XYZ(gT); test 12 |
| Light direction sign flipped | the receding side is the bright one | n̂ = −(backward ray direction); test 12 |
| Per-channel clipping | hot gas turns cyan or flat white | luminance tone curve + hue-preserving gamut handling; test 7 |
| float32 overflow | NaN or infinite pixels near the hole | T⁴ in scaled units; clamp nothing silently, count NaNs per frame and fail the frame if any |
| Everything white | blown-out frame | exposure keyframes; the dynamic range is huge (g up to 1.4 at r = 6 means 3.9× from beaming alone) |
| Banding in smooth gradients | contour lines in the glow | dithering; 16-bit masters |
| Unit mix-ups in the LUT | wrong colours at all T | test 2 catches it; Δλ and B_λ units consistent |
| "Why isn't it orange?" | true-colour torus looks blue-white | that's correct physics; explain it; use the cinematic grade for the orange look, labelled |

---

## Done when

- [ ] Tests 1–14 pass.
- [ ] True-colour, multi-band and cinematic frames of the same snapshot made, side by side.
- [ ] The surface-temperature consistency check written up.
- [ ] Hiten has seen the three looks and picked the defaults for the final videos; committed.
