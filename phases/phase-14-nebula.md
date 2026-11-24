# Phase 14: the nebula (in-depth)

**Goal:** replace the smooth blue-white blob with a giant, realistic-looking nebula that is visibly ripped apart by the black hole, in real nebula colours (pink-red hydrogen, teal accents, dark dust lanes), and that the camera can orbit in 3D. "It must look realistic, not be realistic": labelled cheats are allowed where they make the video better (Hiten 2026-11-23).

**Ends with:** a two-clock story: Act 1, a time-lapse of a whole nebula (real parsec scale) torn into streamers and a clumpy ring over ~10⁵ years; a nested-scale camera dive; Act 2, today's close-up story (one streamer tip falls in, crashes, builds the ring and disc) with a turbulent, lumpy cloud, gravity-only streamer gas around it, emission-line colours and rest-field wisps everywhere; the final 4K render.

**Source material:** `docs/nebula-plan/` (Hiten's planning doc, approved 2026-11-23): [00-plan.md](../docs/nebula-plan/00-plan.md) (plan, techniques, timeline), [02-build-spec.md](../docs/nebula-plan/02-build-spec.md) (steps N0-N9; this file follows it), [01-realism-check.md](../docs/nebula-plan/01-realism-check.md) (real vs labelled cheat), [03-free-compute.md](../docs/nebula-plan/03-free-compute.md), [04-open-source-tools.md](../docs/nebula-plan/04-open-source-tools.md), [05-reading-list.md](../docs/nebula-plan/05-reading-list.md), [06-look-references.md](../docs/nebula-plan/06-look-references.md).

**Primary sources:** Mapelli & Trani 2016 (A&A 585, A161: a 1.28 × 10⁵ M☉, 15 pc turbulent cloud captured by Sgr A*; Act 1's model), Bonnell & Rice 2008 (Science 321, 1060: k⁻⁴ turbulent cloud onto Sgr A*; clumps survive), Lucas et al. 2013 (prolate structured cloud keeps angular momentum), Guillochon et al. 2014 (clumpy streams; recombination glow ends past ~10⁵ K), Price, Federrath & Brunt 2011 (σ²_ln ρ = ln(1 + b²𝓜²)), Dong & Draine 2011 (Hα emissivity fit), Osterbrock & Ferland 2006 (case-B ratios), PyNeb (Luridiana et al. 2015; collisional line emissivities), Cardelli, Clayton & Mathis 1989 / `dust_extinction` (extinction curves), Bridson et al. 2007 (curl noise), Gustavson & McEwan (psrdnoise). Production references: AMNH/SDSC Orion (glow in skins, nested volumes), BUF *Cosmos* (quasi-static nebulae, exponential dive), DNEG *Interstellar* (filter detail against flicker).

**Hiten's decisions (2026-11-23, DECISIONS):** (1) two-clock story (option C); (2) no length limit, don't trim the current story (*start:* 45-60 s time-lapse); (3) teal a clear accent, not half and half (*start:* ~15-25 % of the glowing area, near hot stars and on inner streamer faces); (4) a huge nebula at real parsec scale (size costs nothing; particle count and grid detail are the dials); (5) self-gravity off in Act 1 (optional 50k comparison run with it on).

**Conventions:** numbers marked *start* are first guesses to tune with Hiten; everything else is a requirement. Expected values for tests come from scripts in `reference/scripts/` (CLAUDE.md, rule 3). Every new formula lives in `physics/` with a CPU reference and a GPU-vs-CPU test (rules 1-2). Units stay G = c = M = 1 (1 code length = 6.35 × 10⁶ km; 1 pc ≈ 4.86 × 10⁶): Act 1's Phantom units are converted on load. Each step ends with full-frame stills or a 4-second clip for Hiten (full frames side by side, not crops).

**How it fits the code today (checked 2026-11-23):**
- Colour: `render/march.py` `RelativisticBlackbody` (CPU) and the `// #if COLOUR` block of `shaders/gas_march.wgsl` (GPU) march X, Y, Z, bolometric with a blackbody chroma table (`physics/colour.py`, CIE table `data/cie1931_2deg_1nm.csv` via `render/colour_data.py`). Emission lines are a new source term in the same block; the blackbody stays.
- Per-cell fields: textures hold (ρ, ρ T⁴) plus ρ γβ (rgba32float, alpha unused). Tags (2-3 channels) and the rest field X₀ (3 channels + |X₀|²) need new deposited fields in `render/gasgrid.py` and `render/bricks.py` (Numba + reference twins, `N_FIELDS`) and new textures/bindings.
- Gas look and the late-story extras already in place: soft gas (`[gas] h_scale`), the render-time inner disc (`[disc]`, look C3), Kerr light (`spin`, trial orbit shot `configs/shots/full_story_orbit.toml`), pinpoint lensed stars.
- Simulations: Act 2 sources are Phantom runs (`phantom/inputs/`, the project's Phantom patch, GPL); `sim/cloud.py` `cloud_ic` builds today's uniform glass ball (radius 100, mass 1, 10⁴ K, orbit r_p 25 / r_a 5,000).
- Precision: camera and grids are float32 on the GPU; parsec-scale scenes (~10⁷ units) need camera-relative / per-volume local coordinates (N3, N8).

---

## N0. Plan and decisions (this file)

- This phase file; DECISIONS entries (two-clock story; emission-line colour; gravity-only particles and the 2,000-unit split rule; rest-field noise; the nebula exposure gain as a labelled cheat; Hiten's five decisions); a GPL-3.0 `LICENSE` (the repo holds a Phantom patch; DECISIONS 2026-10-26: the project may become GPL open source); HANDOFF §3-§4 brought up to date.
- **Gate:** Hiten approves this file before N1.

## N1. Emission-line colour model (target Dec 5)

**Build:** `physics/emission_lines.py` (pure functions with docstrings), its CPU use in `render/march.py` (a colour model option), and the same in the shader's colour block; small lookup textures for the collisional lines.

- **Lines (air wavelengths, nm):** Hα 656.28, Hβ 486.13, Hγ 434.05, [O III] 500.68 and 495.89, [N II] 658.35 and 654.80, [S II] 671.65 and 673.08.
- **Emissivities:** Hα from the Dong & Draine fit, j = 2.82 × 10⁻²⁶ T₄^(−0.942 − 0.031 ln T₄) n_e n_H⁺ erg cm⁻³ s⁻¹ sr⁻¹; Hβ and Hγ as case-B ratios; [O III], [N II], [S II] from PyNeb tables over T × n, computed offline (`reference/scripts/`) into small lookup textures. Multiplied by ion-fraction tags and abundances. Each line's colour: emissivity × the CIE x̄ȳz̄ at its wavelength.
- **Ionized skin (Act 1 and backdrop):** ionized fraction = exp(−N / N₀), N the gas column toward the central star cluster (one march during the bake). *start:* a skin ~0.1 pc thick. Labelled a model.
- **Dust:** opacity per line from `dust_extinction` (*start:* R_V = 3.1) × a dust-to-gas ratio × density; none above *start* 2 × 10⁴ K (grains destroyed; the cutoff is a labelled choice).
- **Blackbody stays on:** above ~2-3 × 10⁴ K the lines fade and it takes over: the pink → teal → white-hot arc comes from the physics.

| # | Test | Requirement |
|---|---|---|
| 1 | Single line's chromaticity | equals the CIE table's chromaticity at that wavelength, 1e-6 |
| 2 | Hα emissivity | equals the fit at 5,000, 10,000, 20,000 K, relative 1e-6 |
| 3 | Uniform dust slab | transmitted light equals exp(−τ) at each line, 1e-6 |
| 4 | GPU vs CPU | on a test grid, the existing colour test's limit (1e-3) |
| 5 | Edge values | no NaN / inf at zero density, the temperature floor and 10⁶ K |
| 6 | Hot limit | at ≥ 10⁵ K within a small colour difference of blackbody-only; tolerance derived from the line-to-continuum ratio there, and stated |

**Hiten checks:** a swatch sheet of colour vs temperature (5 × 10³ to 10⁶ K) and tags; one Act 2 still at day ~97 in the new colours.

## N2. Act 1 setup and 50k test (target Dec 12)

- **Phantom setup:** hole as an external point mass of 4.3 × 10⁶ M☉ (Newtonian: nothing in Act 1 comes near it); a sink of radius *start* 0.05 pc (Mapelli & Trani: 0.005 pc; larger is faster and still inside their 0.4 pc inner ring).
- **Cloud (*start*, Mapelli & Trani):** 1.28 × 10⁵ M☉, radius 15 pc, starting 26 pc out, impact parameter 8-26 pc, speed 0.2-0.5 of escape speed; k⁻⁴ divergence-free turbulence (`set_velfield_from_cubes` or TurbGen); lumpy density (pyFC lognormal cube).
- **Self-gravity off** (decision 5). **Thermal:** cold gas, 100 K floor, simple cooling; dense gas renders as dust, the glow comes from N1's ionized skin.
- **Length:** ~0.5-1 Myr: the first pericentre and the start of the ring.

| # | Test | Requirement |
|---|---|---|
| 1 | Single test particle | orbit matches the exact Kepler orbit (tolerance from the integrator's order and step, derived) |
| 2 | Angular momentum incl. the sink's share | conserved within 1 % (Mapelli & Trani's figure) |
| 3 | Velocity spectrum | slope −4 within the sampling error (derived from the mode count); compressive share ≈ 0 |
| 4 | Totals | mass 1e-10; starting turbulent energy 1 % |

- **Gate:** measure the 50k run time, then pick the production count so it fits ≤ 5 days on 8 vCPUs. **Hiten checks:** a 50k time-lapse beside Mapelli & Trani's figures.

## N3. Act 1 production run and backdrop bake (target Dec 31; run done before ~Jan 16, end of the trial credit)

- Remaining Google Cloud trial credit in December (cost checked with Hiten first); fallback: chained 12-hour Kaggle CPU sessions (03-free-compute.md recipe). ~400 dumps evenly spaced in time.
- **Bake** the final dump into sparse bricks (density, temperature, dust, ionized skin, tags) in per-volume local coordinates.
- **Tests:** deposited mass = particle mass (the existing ledger, 1e-12); skin fraction decreases monotonically with column; two bakes bit-identical.
- **Hiten checks:** stills of the frozen nebula from four angles; a 4-second orbit clip.

## N4. Act 2 lumpy cloud (target Jan 7)

| Setting | Value (*start* where marked) |
|---|---|
| Velocity field | divergence-free Gaussian, P(k) ∝ k⁻⁴, 128³ FFT grid interpolated to particles; mean velocity and spin removed (keeps the tested orbit) |
| Amplitude | Mach 1-2 (c_s = 5.05 × 10⁻⁵ = 15 km/s); the cloud puffs out ~σ_v × time to pericentre (~20-40 units) |
| Density lumps | lognormal, σ of ln ρ 1.0-1.3, correlation length 10-25 units; equal-mass particles placed through the cumulative mass of the field, h recomputed |
| Clumps | ≥ 20 units across (smaller ones wash out at the sound speed by pericentre) |
| Shape | prolate / teardrop, orbit plane along the streamer feeding Act 1's sink |
| Tags | per particle from noise at its start position: oxygen excitation, N/S strength, dust fraction; the backdrop's own noise, so the cloud matches the patch it is cut from |
| Particles | 100k (one Phantom run with the video settings) |

- **Tests:** centre-of-mass position and velocity equal the old start (relative 1e-12); spectrum slope and divergence as N2; measured density spread within ±0.05 of target (finite-sampling tolerance, derived and stated).
- **Hiten checks:** a day-97 still and a close-up of the stream.

## N5. Gravity-only streamer particles (target Jan 21)

- 0.4-2 M particles on neighbouring orbits behind the cloud (a tidal tail reaching toward Act 1's sink radius), integrated with the existing GN orbit integrator (the one used for in-between frames); deposited on a coarser schedule if per-frame deposition is too slow.
- **Split rule:** any particle whose closest approach during the story is under 2,000 units (outside the stream-crash radius ~1,664) is removed from this set (or added to the SPH run).
- **Tests:** per-orbit energy conserved to the integrator's tolerance; none within 2,000 units; deposited mass = particle mass.
- **Hiten checks:** a clip of the streamer peeling in around the cloud.

## N6. Rest-field wisps (target Jan 14)

- Deposit each particle's start position X₀ (density-weighted, same kernel) and |X₀|²; while marching, scale density by exp(a (noise(X₀) − mean)) with psrdnoise fBm; band-limit octaves from the X₀ gradient across a pixel (~2 px); fade where the X₀ spread is large (overlapping streams at the crash).
- **Tests:** X₀ deposition ledger; mean of the noise factor over a large volume preserves density within a stated tolerance; flicker metric (frame-to-frame change on a slow camera) no worse than without wisps; GPU vs CPU for the noise function.
- **Hiten checks:** a 4-second close-up of the stream, before and after.

## N7. Look integration (target Jan 28)

- Nebula gain as a separate, labelled exposure (a dim nebula sits many stops below the hot stream); under 0.5 % of pixels pure white (existing saturation check); teal share tuned to decision 3.
- **Hiten checks:** one contact sheet of the whole story.

## N8. Time-lapse and dive camera (target Jan 31: feature freeze)

- Time-lapse: a slow orbit around the Act 1 nebula over *start* 45-60 s while its dumps play. Dive: r(t) = r₀ (r₁ / r₀)^(t/T), handing over to Act 2's first shot; the lensing appears during the dive. On-screen clock switches from years to days at the hand-over. The close-up orbit around the spinning hole (trial shot) joins here.
- **Precision test:** the same scene rendered with the camera offset by 10⁷ units vs in local coordinates gives the same image within 1e-4.
- **Hiten checks:** a low-res full preview. **No new features after Jan 31.**

## N9. Render tests, final 4K, QA (target Feb 21; spare until Mar 7)

- Early January: time a 4K frame at 2 rays per pixel on the Mac; test Kaggle with NVIDIA Vulkan and with lavapipe; chunked encode pipeline (03-free-compute.md).
- Feb 1-21: final render split between Mac overnights and Kaggle sessions, with the watchdog.
- **QA:** frame-to-frame spike scan; joins; flare; NaN / inf; pure-white share; full-resolution spot checks.

---

## What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Looks painted / procedural | plain fBm everywhere | shape from flow (Act 1's real run), long thin filaments, few dominant forms; compare with real photos |
| Seam between backdrop and simulated gas | different colour, brightness or texture | one emission model, carve the cloud from a streamer (feathered), same noise and tags |
| Flicker / shimmer | noise finer than a pixel; per-frame jitter | band-limited octaves; fixed jitter per pixel; extra rays only where needed |
| Nebula goes black | auto exposure follows the stream (~20 stops brighter) | separate labelled nebula gain (N7) |
| float32 at parsec scale | blocky or jittering detail far from the origin | camera-relative / per-volume local coordinates; N8 precision test |
| Iris GPU memory | dense 1024³ × 4 channels ≈ 8.6 GB | sparse bricks or nested 256³ grids plus noise octaves while marching |
| Act 1 run slower than planned | 50k timing extrapolates past 5 days | fewer particles (the dial), Kaggle chaining, or start earlier |
| Render time balloons | long rays through huge volumes | skip empty space with bricks; bigger steps far from the camera; 4-s clip first |

## Done when

- [ ] N1-N9 tests pass; each step approved by Hiten from stills or clips.
- [ ] Every labelled cheat (01-realism-check.md) recorded in DECISIONS and the report.
- [ ] Final 4K video encoded, QA'd and tagged.
