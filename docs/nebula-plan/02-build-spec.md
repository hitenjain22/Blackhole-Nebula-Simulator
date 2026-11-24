# Build spec for Claude Code

A new phase, "Phase 14: the nebula", in ten steps (N0–N9). The usual CLAUDE.md rules apply: one step at a time, tests with expected values that are never loosened, stop for Hiten's check, then commit. Numbers marked *start* are first guesses to tune with Hiten; everything else is a requirement.

## Hiten's decisions (Nov 23)

1. **Story:** the whole nebula must visibly get ripped apart, swing around the hole, show the spinning lensing with the camera orbiting, then the disc. That means the two-clock version (Option C).
2. **Length:** no limit; it must not feel rushed. Don't trim the current story. Give the time-lapse and the dive as long as they need (*start:* 45–60 s for the time-lapse). Each extra 10 s at 4K costs roughly 6–8 more Mac hours (my estimate).
3. **Teal:** a clear accent against the pink, not half and half (*start:* teal in ~15–25% of the glowing area, placed near hot stars and on the inner streamer faces). Tune by eye.
4. **Size:** a huge nebula, far bigger than the hole. Use real parsec scale (Mapelli & Trani's 15 pc cloud). Physical size doesn't change simulation or render cost; particle count and grid detail do, so those are the dials if renders get too slow.
5. **Self-gravity in Act 1:** off (Hiten's choice). Softer, wispier streamers, cheaper, and no risk of collapsing knots stalling the run. The lumpy starting density still gives clumps and contrast; they stretch instead of collapsing. Optional: one 50k run with it on, only for comparison.

**All five approved by Hiten on Nov 23, along with everything else in this doc. N0 can start.**

## Summary

| Step | What | Gate (Hiten checks) | Target date |
| --- | --- | --- | --- |
| N0 | Phase file and DECISIONS entries | Approves the plan | Nov 26 |
| N1 | Emission-line colour model, CPU and GPU | Colour swatch sheet; one recoloured still | Dec 5 |
| N2 | Act 1 setup and 50k test run | 50k time-lapse next to Mapelli & Trani's figures; measured run time | Dec 12 |
| N3 | Act 1 production run and backdrop bake | Stills of the frozen nebula from four angles; 4-s orbit clip | Dec 31 (run done before Jan 16) |
| N4 | Act 2 lumpy cloud run | Day-97 still and a stream close-up | Jan 7 |
| N5 | Gravity-only streamer particles | Clip of the streamer peeling in around the cloud | Jan 21 |
| N6 | Rest-field wisps | 4-s close-up, before and after | Jan 14 |
| N7 | Exposure and look integration | Contact sheet of the whole story | Jan 28 |
| N8 | Time-lapse and dive camera | Low-res full preview | Jan 31: feature freeze |
| N9 | Render tests, final 4K, QA | Final video | Feb 21 (spare until Mar 7) |

## N0. Plan and decisions

- Write `phases/phase-14-nebula.md` from this tab and the main tab.
- DECISIONS entries, each needing Hiten's approval (rule 7): the two-clock story; emission-line colour; gravity-only particles and the 2,000-unit split rule; rest-field noise; the nebula exposure gain as a labelled cheat.
- Add a LICENSE file before more open-source code goes in (the repo is public and includes a Phantom GPL patch).

## N1. Emission-line colour model

**Build:** `physics/emission_lines.py` (pure functions, readable reference) and the same in the colour shader.

- **Lines (air wavelengths):** Hα 656.28, Hβ 486.13, Hγ 434.05, [O III] 500.68 and 495.89, [N II] 658.35 and 654.80, [S II] 671.65 and 673.08 nm.
- **Emissivities:** Hα from the Dong & Draine fit; Hβ and Hγ as case-B ratios; [O III], [N II], [S II] from PyNeb tables over T × n, computed offline into small lookup textures. Multiply by ion-fraction tags and abundances.
- **Ionized skin (Act 1 and backdrop):** ionized fraction = exp(−N / N₀), where N is the gas column from the cell toward the central star cluster, found by one march toward the centre during the bake. N₀ sets the skin thickness (*start:* a skin ~0.1 pc thick). Label it a model.
- **Dust:** opacity per wavelength from `dust_extinction` (*start:* R_V = 3.1) × a dust-to-gas ratio × density. No dust above a set temperature (*start:* 2 × 10⁴ K), since hot gas destroys grains. Label the cutoff a choice.
- **Tests (expected values from a script in `reference/scripts/`):**
    1. A single line's chromaticity equals the CIE table's chromaticity at that wavelength (to 1e-6).
    2. Hα emissivity equals the fit at 5,000, 10,000 and 20,000 K (relative 1e-6).
    3. Light through a uniform dust slab equals exp(−τ) at each line (1e-6).
    4. GPU equals CPU on a test grid (same limit as your existing colour test, 1e-3).
    5. No NaN or inf at zero density, the temperature floor and 10⁶ K.
    6. At 10⁵ K and above, the result is within a small colour difference of blackbody-only (lines have faded). Derive the tolerance from the line-to-continuum ratio at that temperature and say so.
- **Hiten checks:** a swatch sheet of colour vs temperature (5 × 10³ to 10⁶ K) and tags; one Act 2 still at day ~97 in the new colours.

## N2. Act 1 setup and 50k test

- **Phantom setup:** hole as an external point mass of 4.3 × 10⁶ M☉; a sink of radius 0.05 pc (*start*; Mapelli & Trani used 0.005 pc, larger is faster, and it is still well inside their 0.4 pc inner ring).
- **Cloud (*start*, from Mapelli & Trani):** 1.28 × 10⁵ M☉, radius 15 pc, starting 26 pc out, impact parameter 8–26 pc, speed 0.2–0.5 of escape speed. k⁻⁴ divergence-free turbulence via `set_velfield_from_cubes` or TurbGen; lumpy density from pyFC.
- **Self-gravity:** off (decision 5).
- **Thermal:** cold gas with a 100 K floor and simple cooling. Dense gas renders as dust; the glow comes from the ionized skin in N1.
- **Length:** ~0.5–1 Myr, covering the first pericentre and the start of the ring.
- **Tests:**
    1. A single test particle's orbit matches the exact Kepler orbit (tolerance from the integrator's order and step).
    2. Angular momentum, including the sink's share, conserved within 1% (Mapelli & Trani's figure).
    3. The measured velocity spectrum slope is −4 within sampling error (derive it from the number of modes), and the compressive share is near zero.
    4. Total mass and the start's turbulent energy match targets to 1e-10 and 1%.
- **Gate:** measure 50k run time, then choose the production particle count so the run fits in ≤ 5 days on 8 vCPUs.
- **Hiten checks:** a 50k time-lapse beside Mapelli & Trani's figures.

## N3. Act 1 production and backdrop bake

- Run on the remaining trial credit in December; it must finish before ~Jan 16. Fallback: chained 12-hour Kaggle CPU sessions.
- Dumps: ~400 evenly spaced in time for the time-lapse.
- **Bake** the final dump into sparse bricks: density, temperature, dust, ionized skin, tags. Use per-volume local coordinates.
- **Tests:** deposited mass equals particle mass (your existing ledger, 1e-12); the skin fraction decreases monotonically with column; two bakes are bit-identical.
- **Hiten checks:** stills of the frozen nebula from four angles; a 4-second orbit clip.

## N4. Act 2 lumpy cloud

- Settings as in the main tab's "Making the simulated cloud clumpy" table: Mach 1–2; σ of ln ρ 1.0–1.3; correlation length 10–25 units; clumps ≥ 20 units; prolate; tags; orbit plane aligned with the streamer feeding Act 1's sink.
- **Tests:** centre-of-mass position and velocity equal the old start (relative 1e-12); velocity spectrum slope and divergence as in N2; measured density spread within ±0.05 of target (finite-sampling tolerance, derived and stated).
- One 100k Phantom run with your video settings.
- **Hiten checks:** a day-97 still and a close-up of the stream.

## N5. Gravity-only streamer particles

- Generate 0.4–2 M particles on neighbouring orbits behind the cloud: a long tidal tail reaching outward toward Act 1's sink radius, the inflow that feeds the cloud.
- Integrate with your generalized-Newtonian orbit integrator (the one used for in-between frames).
- **Split rule:** any particle whose closest approach during the story is under 2,000 units is removed from this set (or added to the SPH run).
- Deposit them on a coarser schedule (every N frames, grids interpolated between) if per-frame deposition is too slow.
- **Tests:** per-orbit energy conserved to the integrator's tolerance; no gravity-only particle comes within 2,000 units; deposited mass equals particle mass.
- **Hiten checks:** a clip of the streamer peeling in around the cloud.

## N6. Rest-field wisps

- As in the main tab's "Sub-particle detail" section: deposit X₀, psrdnoise fBm at X₀, band-limit from the X₀ gradient, fade where streams overlap.
- **Tests:** X₀ deposition ledger; the noise's mean over a large volume is ~0 (density scale preserved within a stated tolerance); a flicker metric (frame-to-frame change on a slowly moving camera) no worse than without wisps.
- **Hiten checks:** a 4-second close-up of the stream, before and after.

## N7. Look integration

- Nebula gain as a separate, labelled exposure; check that under 0.5% of pixels are pure white (your existing saturation check).
- Teal share tuned to decision 3.
- One contact sheet of the whole story for Hiten.

## N8. Time-lapse and dive camera

- Time-lapse: a slow orbit around the Act 1 nebula over 45–60 s (*start*) while its dumps play.
- Dive: distance shrinks exponentially, r(t) = r₀ (r₁ / r₀)^(t/T), handing over to Act 2's first shot. The lensing becomes visible during the dive, as the nebula behind the hole bends into a ring around the shadow.
- On-screen clock switches from years to days at the hand-over.
- **Precision test:** the same scene rendered with the camera offset by 10⁷ units versus in local coordinates gives the same image within 1e-4.
- **Hiten checks:** a low-res full preview. **Feature freeze on Jan 31.**

## N9. Render tests, final 4K, QA

- Early January: time a 4K frame at 2 rays per pixel on the Mac; test Kaggle with NVIDIA Vulkan and with lavapipe; build the chunked-encode pipeline (see [Free compute](03-free-compute.md)).
- Feb 1–21: the final render, split between Mac overnights and Kaggle sessions, with the watchdog.
- **QA:** your frame-to-frame scan for spikes; joins; flare; NaN and inf check; pure-white share; spot-check stills at full resolution.
