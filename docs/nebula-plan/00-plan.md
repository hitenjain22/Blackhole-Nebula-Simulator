# Building a giant nebula for the black hole video

2026-11-23 · Hiten Jain

## Bottom line

A giant, realistic nebula that gets ripped apart on screen is feasible for $0, with the 4K final done by the first week of March. Four facts make it work:

1. **Nobody simulates a whole nebula in full detail.** Films, planetariums and NASA freeze the huge parts and simulate what moves. Here freezing is also physically right: over the 320 days of the close-up, the real nebula moves 0.004% of its size.
2. **The big tear is cheap to simulate.** A whole nebula swinging past Sgr A\* over ~100,000 years never comes close to the hole, so its time steps stay long. 300k–500k particles is realistic on free CPUs.
3. **4K costs about as much as your 1080p final** at 1–2 rays per pixel, instead of 4×.
4. **Free compute is enough:** your Mac, Kaggle's free GPU and CPU sessions, your remaining Google Cloud trial credit (before ~Jan 16), and Mesa's lavapipe, which lets your renderer run on CPU-only machines.

**The plan in one line:** a time-lapse of the whole nebula being torn into streamers (Act 1) → a dive toward the hole → one streamer tip falls in, crashes and builds the disc in full detail (Act 2, your current story). One emission-line colour model and "rest-field" wisps across everything.

| Part | Effort (my estimate) | Money |
| --- | --- | --- |
| Emission-line colour model | 2–4 days | $0 |
| Act 1 setup and 50k test runs | ~1 week | $0 (Mac or Kaggle CPU) |
| Act 1 production run (300k–500k) | 3–6 days of run time | $0–50 (trial credit before Jan 16, or chained Kaggle CPU sessions) |
| Act 2 lumpy cloud and gravity-only streamer | ~1 week plus ~4 days of run time | $0–40 |
| Rest-field wisps | 2–4 days | $0 |
| Time-lapse and dive camera | ~1 week | $0 |
| Final 4K render | 60–90 hours on Mac plus Kaggle | $0 |

**Main risks:** Act 1's real run time is unknown until the 50k test; your Mac's GPU has crashed on long renders before; headless Vulkan on Kaggle is untested. The first two weeks of January are reserved for testing exactly these.

More detail: [Realism check](01-realism-check.md) · [Build spec](02-build-spec.md) · [Free compute](03-free-compute.md) · [Open-source tools](04-open-source-tools.md) · [Reading list](05-reading-list.md) · [Look references](06-look-references.md)

## The new plan: the nebula itself gets ripped apart

Tell it on two clocks. First a ~30-second time-lapse: a whole nebula of 300k–500k particles is torn into streamers and a clumpy ring over ~100,000 years. Then the camera dives in, and the last ~320 days play out in full detail as one streamer's tip falls in and builds the disc. This is what the real Galactic Centre is doing, and it fits free compute because the big run never gets close to the hole.

(Hiten's decisions on 2026-11-23: no length limit, so the time-lapse can run 45–60 s; see the build spec.)

**Why it's real**

- [Mapelli & Trani 2016](https://ar5iv.labs.arxiv.org/html/1510.05259) simulated exactly this with SPH: a 1.28 × 10⁵ M☉ cloud 15 pc in radius, a k⁻⁴ turbulent start, beginning 26 pc from the hole (run R1 had ~108k gas particles). First pericentre came at ~1–2.5 × 10⁵ years and lasted ~1.5 × 10⁵ years. The cloud was squeezed into a stream and formed a small inner ring and a clumpy ring ~2 pc across, like the real circumnuclear disc.
- The real Northern Arm of the minispiral is a broad sheet ["drawn out into a narrow stream in the vicinity of Sgr A\*"](https://ar5iv.arxiv.org/html/astro-ph/0405197). The nebula around Sgr A\* is being ripped apart right now, just slowly.

**The layers**

| Layer | What it shows | Particles | Method | Time covered | Cost (my estimates) |
| --- | --- | --- | --- | --- | --- |
| Act 1: the big tear | A whole nebula (10⁴–10⁵ M☉, 10–30 pc) swinging past Sgr A\*, stretched into streamers and a clumpy ring | 300k–500k SPH | Phantom, Newtonian point-mass hole, a sink ~0.05 pc wide, turbulent start, cooling with a floor | 10⁵–10⁶ years, shown in ~30 s | Days on 8 cores. Nothing gets near the hole, so time steps stay long. Test at 50k first. |
| Act 2: the close-up fall | One streamer tip: your current cloud (0.1 M☉), now turbulent and lumpy | 100k–150k SPH | Your current Phantom setup | ~320 days | ~4 days, as now |
| Act 2 surroundings | The rest of that streamer around the falling cloud, slowly stretching | 0.4M to several M gravity-only | Your existing orbit integrator, no gas pressure | ~320 days | Minutes to hours on the Mac |
| Backdrop | Act 1's final state, frozen | Deposited into the grid | Static bake plus noise detail | — | One bake on the Mac |

**The biggest win: Act 1 replaces the painted backdrop.** Its final frame becomes the giant nebula, so the backdrop's shape comes from real physics. Noise only adds fine detail, which removes most of the "looks painted" risk.

**How the pieces hand over**

- **Act 1 to Act 2:** Act 1's sink (0.05 pc ≈ 10,000 AU ≈ 240,000 code units) is far bigger than Act 2's starting distance (5,000 units). The whole close-up therefore happens inside Act 1's sink, and the camera dive bridges the scales. Point the Act 2 cloud's orbit along the streamer that is feeding the sink in Act 1's last frames.
- **Gravity-only to SPH:** integrate every gravity-only particle's orbit first. Any that comes within ~2,000 units of the hole during the story (outside the stream-crash radius of ~1,664) gets switched to SPH. Gravity-only motion is the "frozen-in" approximation from TDE theory. It gets debris energies [within ~20% for disrupted stars](https://arxiv.org/pdf/2008.04922) but [misses features shocks and self-gravity create](https://arxiv.org/pdf/2309.07216), so keep it away from the crash.
- **One look for everything:** the same emission-line colours, dust and rest-field detail on all layers.
- **An honest clock:** the on-screen timer switches from years to days at the dive, so the time jump is stated, not hidden.

**Three options, from cheapest to most ambitious**

| Option | Looks like | Extra simulation | Fits free by March? |
| --- | --- | --- | --- |
| A. Frozen painted nebula + your cloud | A still nebula; one clump falls | None | Yes, easily |
| B. Three kinds of gas within 320 days | The inner streamer visibly peels in around the falling cloud | Gravity-only particles only | Yes |
| **C. Two clocks (chosen by Hiten)** | The whole nebula torn apart in a time-lapse, then B up close | One Act 1 run of 300k–500k | Yes, if Act 1 runs in December on the remaining trial credit or chained Kaggle CPU sessions |

It is a method change, so it goes in DECISIONS (approved by Hiten on 2026-11-23) before Claude Code builds it.

## How films and planetariums build nebulae

No production simulates a whole nebula. Every one below builds a static or "quasi-static" volume from shapes plus noise, and simulates only what visibly moves.

| Production | How the nebula was built | What to borrow |
| --- | --- | --- |
| [AMNH + SDSC, Orion Nebula (2000)](https://www.cs.uaf.edu/~genetti/Research/Papers/EG00/Orion.html) | A surface model of the glowing layer, turned into a distance field and roughened with turbulence. Glow lives in a thin skin; dark dust cores are opaque. 86 separate volumes (main 652 × 326 × 652, small ones 64³), because one uniform grid would have needed ~2,791 TB. Rendered in 12 h on 900+ processors. | Glow on the skins of dark clumps (edge brightening). Many nested volumes instead of one huge grid. |
| [STScI + Caltech, Orion flythrough (2018)](https://science.nasa.gov/missions/hubble/nasa-space-telescopes-provide-a-3d-journey-through-the-orion-nebula) and [Carina "Mystic Mountain"](https://svs.gsfc.nasa.gov/30683) | Layers built from Hubble and Spitzer images: a visible-light surface over infrared structure. Custom code blends tens of millions of semi-transparent elements. Stars, disks, bow shocks and the front "veil" are rendered as separate layers, then recombined. | Render stars, veil and main gas as separate layers you can balance. |
| [BUF, Cosmos: A Spacetime Odyssey (2014)](https://history.siggraph.org/wp-content/uploads/2022/10/2015-Talks-Vidal_Visualizing-the-Cosmos_-a-Procedural-Approach.pdf) | "Quasi-static" nebulae: fractal trees of curves, each branch the path of a particle in a simple dynamic sim, sampled into particles with positional noise. Huge scales done as nested passes, one per order of magnitude, with an exponential camera move. ~5 min per frame on 8 cores. | Shape from motion paths: looks like a frozen simulation without running one. Nested-scale camera moves. |
| [DNEG, Interstellar (2014)](https://cerncourier.com/a/building-gargantua/) | Stars, dust clouds, nebulae and the disc all lensed by their renderer. Ray bundles plus spatial filtering gave IMAX smoothness without flicker; renders up to 100 h. | Filter fine detail so it can't sparkle or flicker. |
| [Adam Swaab, Houdini nebulae](https://80.lv/articles/how-to-create-gorgeous-space-nebulas-in-3d) | Pure render-time noise "felt very much like procedural noise" and was hard to direct. 30–40 M raw particles showed up as dots close in. Final method: ~7.5 M particles rasterized into a volume, advected for fine detail, a smoke solver with the pressure step removed (no mushroom look), and a fake attraction field. | Big shapes from flow, fine detail from noise. Never noise alone. |
| [Gaia Sky nebulae](https://tonisagrista.com/blog/2024/rendering-aurorae-nebulae/) | Real-time: march a bounding sphere, accumulate density and colour. Based on Duke's Shadertoy "Dusty Nebula 4" (CC BY-NC-SA). | Quick look-dev reference. Don't copy the code (non-commercial licence). |

**The noise toolkit, and what each piece is for**

- **fBm** (several octaves of noise summed): the billowy base. On its own it is what reads as "procedural noise".
- **Domain warping** ([Quilez](https://iquilezles.org/articles/warp)): sample noise at a point pushed by other noise. Gives swirls and stretched filaments.
- **Ridged noise** (1 − |noise|): sharp creases, which become filaments and sheet edges.
- **Worley (cellular) noise**: bubble walls and cavities, like wind-blown shells.
- **Curl noise** ([Bridson et al. 2007](https://www.dneg.com/wp-content/uploads/2022/11/2007.pdf)): a swirling velocity field with no sources or sinks. Moving particles through it makes natural filaments without bunching.
- **Advection**: push density through a velocity field for a few steps. Turns blobs into streaks with real "flow".
- **Pillars**: a dense column (a cylinder shape plus noise) with an opaque dusty core and a bright rim on the side facing the light.
- **Dark lanes**: a separate dust density that only absorbs, crossing in front of glowing gas.
- **Band-limiting** ([Quilez](https://iquilezles.org/articles/)): drop noise octaves finer than a pixel. This is what keeps detail from shimmering frame to frame.

## Physically based nebula colour

Model the glow as five emission lines plus dust absorption. Turn each line into colour with the CIE table the repo already ships (`data/cie1931_2deg_1nm.csv`), and keep the existing blackbody on top. The pink → teal → white-hot story arc then falls out of the physics, with no colour switch.

This is also what the real gas near Sgr A\* does. The minispiral streamers around the hole are [photoionized by UV from the young massive stars of the central parsec](https://arxiv.org/pdf/1006.0064), and G2 itself was tracked in a hydrogen recombination line ([Schartmann et al.](https://www.arxiv.org/pdf/1609.07308)). [Guillochon et al. 2014](https://arxiv.org/pdf/1401.2990) found that infalling clumps heated past ~10⁵ K at pericentre stop glowing in recombination lines. So "pink nebula turns white-hot as it falls" is the correct behaviour.

| Line | Wavelength | Colour on its own | What makes it | How it scales |
| --- | --- | --- | --- | --- |
| Hα | 656.3 nm | deep red | recombination | n² × T₄^(−0.942 − 0.031 ln T₄): drops ~10× from 10⁴ K to 10⁵ K |
| Hβ (+ Hγ) | 486.1 nm (434.0) | blue-green (blue) | recombination | same shape as Hα; Hβ ≈ Hα / 2.86 in standard case B |
| [O III] | 500.7 nm (+ 495.9 at ~1/3) | teal / cyan-green | collisions with O⁺⁺ | n² × x(O⁺⁺) × T^−0.5 e^(−28,764 / T) |
| [N II] | 658.4, 654.8 nm | red | collisions, low-ionization gas | strongest at clump edges; deepens Hα red |
| [S II] | 671.6, 673.1 nm | deep red | collisions at ionization fronts | thin bright rims on dense clumps |

Sources for the scalings: Hα fit from [Dong & Draine 2011 as used in SILCC](https://arxiv.org/pdf/1610.06569) (coefficient 2.82 × 10⁻²⁶ erg cm⁻³ s⁻¹ sr⁻¹); [O III] / Hβ ∝ T^0.5 e^(−28,764/T) from [Stasińska](https://arxiv.org/pdf/astro-ph/0207500). The 2.86 and 1/3 ratios are standard textbook values, not taken from these pages.

```latex
j_{\mathrm{H}\alpha} = 2.82\times10^{-26}\,T_4^{-0.942-0.031\ln T_4}\,n_e n_{\mathrm{H}^+}\quad[\mathrm{erg\,cm^{-3}\,s^{-1}\,sr^{-1}}]
```

**Why real nebulae look pink, teal and red-brown**

- **Pink** is Hα red mixed with Hβ and Hγ blue. In M17 it [turns redder toward the edges, because dust removes the blue lines first](https://clarkvision.com/galleries/gallery.astrophoto-1/web/m17-500mm-09-04-2021-acr-IMG_0993-003-av10-c2-1x-rs2,2-e.67s.html).
- **Teal** is [O III]. The Orion Trapezium region is [teal in true colour](https://clarkvision.com/articles/astrophotography.m42-trapezium.true.color): [O III] dominates green, and the eye barely responds to Hα.
- **Red-brown and black** is dust: thin dust reddens what is behind it; thick dust is a black lane.

**The recipe for your renderer**

1. Per grid cell, compute the five line emissivities from density and temperature. Two "tag" multipliers set the region-to-region variety: oxygen excitation (how much O⁺⁺, i.e. how close to hot stars) and nitrogen/sulfur strength.
2. Turn each line into colour: emissivity × the CIE x̄ȳz̄ value at that wavelength. One table lookup per line, no spectrum integral.
3. Dust: a separate density that absorbs more blue than red (roughly 1/λ in the optical, the [Cardelli–Clayton–Mathis](https://iancze.github.io/courses/astro542/lectures/18-dust) shape). Evaluate it at each line's wavelength: a few multiplies per step.
4. Keep the blackbody on: above ~2–3 × 10⁴ K the lines fade and it takes over by itself.
5. Give the nebula its own exposure gain, labelled as a cheat. A dim emission nebula sits many stops below the hot stream, so automatic exposure will turn it black once the stream brightens.

**Open-source renderers:** [yt's volume renderer](https://yt-project.org/doc/visualizing/volume_rendering.html) does emission and absorption transfer functions on simulation grids but runs on the CPU and doesn't bend light. None of the tools found here does curved light plus line emission. Your own GPU marcher is the right place for this: the change is a new colour function, not a new renderer.

## Making the simulated cloud clumpy, cheaply

Start the cloud with a turbulent velocity field and a lumpy (lognormal) density at Mach ~1–2, with clumps at least ~20 code units across. That is about a day of setup and a normal-sized Phantom run. Published simulations show the structure survives the tidal stretching.

**What the literature says**

- [Bonnell & Rice 2008](https://arxiv.org/pdf/0810.2723): 4.5 M SPH particles; a divergence-free Gaussian velocity field with P(k) ∝ k⁻⁴, scaled so kinetic energy equals the cloud's gravitational binding energy. The black hole's tides "do not destroy the small-scale structures": shearing made coherent spirals and kept the captured gas clumpy.
- [Lucas et al. 2013](https://ar5iv.arxiv.org/html/1305.0012): a prolate (stretched) cloud falling onto a 4 × 10⁶ M☉ hole; "a turbulent or otherwise highly structured cloud is necessary" to keep enough angular momentum through the pericentre shocks.
- [Guillochon et al. 2014](https://arxiv.org/pdf/1401.2990): with cooling that grows as density squared, small overdensities in a stream grow until it breaks into clumps. The clumps later stretch and heat past ~10⁵ K at pericentre.
- [Price, Federrath & Brunt 2011](https://arxiv.org/abs/1010.3754v1): in turbulent gas the spread of log-density is σ² = ln(1 + b²𝓜²) with b = 1/3. That tells you how lumpy a given Mach number makes the gas.

**Settings for your cloud** (my estimates in code units, from c_s = 5.05 × 10⁻⁵, i.e. 15 km/s, and 396,224 time units to pericentre)

| Setting | Value | Why |
| --- | --- | --- |
| Velocity field | Divergence-free, Gaussian, P(k) ∝ k⁻⁴, made on a 128³ FFT grid and interpolated to particles. Subtract the mean velocity and spin. | Same recipe as Bonnell & Rice. Removing bulk motion keeps the orbit you've already tested. |
| Amplitude | Mach 1–2 (15–30 km/s rms) | The virial scaling doesn't apply (no self-gravity). The cloud puffs out by about σ_v × time to pericentre: ~20 units at Mach 1, ~40 at Mach 2, ~60 at Mach 3 (doubling a radius-60 cloud). |
| Pre-decay run | Not needed | One eddy turnover (60 / σ_v ≈ 1.2 × 10⁶ at Mach 1) is longer than the fall to pericentre. A short settling run is optional. |
| Density lumps | Lognormal, σ of ln ρ ≈ 1.0–1.3 (clumps 3–4× the mean), correlation length 10–25 units | Mach 2 alone gives σ ≈ 0.6 (clumps only ~1.8×). Imposing density directly gives visible clumps from frame one. |
| Clump size | ≥ 20 units across | Equal-temperature clumps are overpressured and spread at the sound speed: ~20 units by pericentre. Smaller clumps wash out. |
| Placing particles | Map your glass through the cumulative mass of the density field (equal-mass particles, uneven spacing), then recompute h | Keeps SPH particles equal-mass, which Phantom expects. |
| Shape | Teardrop or prolate, not a sphere | Reads as a piece torn off a larger structure. Lucas et al. used a prolate cloud. |
| Tags | One per particle, from noise at its start position: oxygen excitation, N/S strength, dust fraction | Drives the colour variety above and travels with the gas for free. |
| Particles | 100k–200k | A clump with 1% of the mass gets ~1,000 particles at 100k: enough to hold its shape. |

**Optional: clumps that grow by themselves.** Your β-cooling doesn't depend on density, so it can't make clumps. Guillochon's cooling (∝ ρ², peaking near 10⁵ K) would let the stream fragment on its own. This is a physics change, so it needs your approval (CLAUDE.md rule 7) and its own tests. Treat it as a later experiment, not part of the first nebula pass.

**Cost:** your 100k Phantom video runs take about 4 days on the c4d-highcpu-8 VM (~$0.30–0.40/h), so ~$30–40 per run. 200k is roughly 2–2.5× that (my estimate). Free routes are in the [Free compute](03-free-compute.md) tab.

## Sub-particle detail at render time

Use each particle's starting position as a noise coordinate. VFX tools call this a "rest field". Deposit it into the grid beside density, then sample noise at it while marching. The noise stretches and bends exactly with the gas: a stream drawn out 1,000× turns into fine lengthwise filaments with no extra particles. Because the noise is pinned to the gas itself, it can't drift or swim between frames.

**How to build it**

1. Give every particle its start position X₀. Phantom keeps particle IDs, so X₀ can be looked up from the first dump.
2. Deposit X₀, density-weighted, as three extra grid channels with the same kernel as density. The renderer's in-between interpolation carries X₀ unchanged.
3. While marching, read X₀ at the sample point, evaluate fBm there, and scale density by exp(a × (noise − mean)). The mean brightness stays about the same; only the texture changes.
4. Band-limit: from how fast X₀ changes across a pixel (finite differences of the X₀ channels), drop octaves finer than ~2 pixels on screen. This one step stops shimmering in the most stretched parts of the stream.
5. Where unrelated streams overlap (the crash), X₀ is an average of gas from different places and the noise turns to mush. Also deposit |X₀|²; where its spread is large, fade the noise out.

**Compared with the other options**

| Method | How it works | Fit for this project |
| --- | --- | --- |
| Rest field (recommended) | Noise sampled at each gas parcel's starting position | Pinned to the gas, so it holds still through time warps and joins between runs. Costs 3 grid channels plus noise per sample. |
| Flow-aligned noise | Noise stretched along the local velocity | Simple, but not tied to the gas: the pattern slides as the time warp changes speed. |
| [Advected textures](https://diglib.eg.org/items/c4bf80cc-23f0-4107-958d-e03c0b9c8ac4/full) (Neyret 2003) | Noise carried by the flow, regenerated and cross-faded periodically | Built for steady sim time. Your time warps would make the regeneration pulses visible. |
| [Wavelet turbulence](https://www.cs.cornell.edu/~tedkim/WTURB) (Kim et al. 2008) | Adds missing small eddies to a coarse grid simulation | Reported 50³ → 400³ at ~30 s per frame on a 4-core CPU (2008). Needs a grid velocity history; better suited to baking the backdrop than to SPH streams. |

**GPU cost (my estimate; measure on a clip):** 3–4 octaves of 3D gradient noise per march step is tens of arithmetic operations. On the L4 that should be small next to the bent-ray stepping. On the Iris Plus, expect roughly 1.3–2× the current frame time. Memory: three more float16 channels per cell.

I found no paper using a rest field for SPH visualization specifically. It is a standard VFX trick applied to a new case, so test it on a 4-second close-up clip before wiring it into the story.

## Making it read as enormous

The real nebula around Sgr A\* is about 2,000–3,000× wider than the cloud's starting distance from the hole, and it barely moves during the story. So build it static and truly large, and use a nested-scale camera dive to get from orbiting the nebula down to the black hole.

| Structure | Real size | In code units (1 unit = 6.35 × 10⁶ km) |
| --- | --- | --- |
| Ring / torus | 2.2 AU | ~51 |
| Cloud's starting distance | 212 AU (0.001 pc) | 5,000 |
| [Sgr A West minispiral](https://arxiv.org/pdf/astro-ph/0102382) (ionized streamers) | 2.1 × 2.9 pc | ~10–14 million |
| [Circumnuclear disc](https://arxiv.org/pdf/astro-ph/0102382) (dusty molecular ring) | 1.7–7 pc from the centre | ~8–34 million |
| [Orion Nebula](https://www.cs.uaf.edu/~genetti/Research/Papers/EG00/Orion.html), for comparison | ~6 light-years | ~9 million |

**Static is physically right.** Gas in the circumnuclear disc moves at ~100 km/s. Over the 320 days the story covers, that is ~18 AU, or 0.004% of a 2 pc structure (my arithmetic). Freezing the nebula, as BUF did for Cosmos, costs no realism here.

**Two ways to get "revolve around it, then get sucked in"**

- **True scale plus an exponential dive (recommended).** Open orbiting the whole nebula, then dive about four orders of magnitude, with the camera's distance shrinking exponentially, like BUF's [Cosmos pull-out](https://history.siggraph.org/wp-content/uploads/2022/10/2015-Talks-Vidal_Visualizing-the-Cosmos_-a-Procedural-Approach.pdf) reversed. Lensing only appears once you're close, which makes the arrival dramatic. Render it as nested volumes (nebula, one streamer, simulation grids), each in its own local coordinates.
- **Shrunk nebula (art cheat).** Scale the nebula to ~10⁴–10⁵ units so it wraps around the action. Simpler, but label it in the report.

**Depth cues that sell the size**

- **Parallax layers:** thin foreground wisps sliding fast past the camera against slow-moving far structure. This is the STScI "veil" layer.
- **Occlusion:** dark dust lanes crossing in front of bright gas show what is nearer.
- **Stars at many depths:** a DNEG artist working on Men in Black found [the key difference between cosmic dust and ordinary dust is millions of stars inside it](https://www.sidefx.com/community/men-in-black-international/). The nuclear star cluster is real, so put stars inside the nebula, not just behind it.
- **Extinction with depth:** farther gas looks redder and dimmer through more dust, the space version of haze on distant mountains.
- **Detail at every zoom:** add finer noise octaves as the camera closes in (band-limiting run in reverse), so nothing goes soft or blocky.
- **Slow camera:** big things drift slowly across the screen. Fast motion reads as small.

## Free and cheap compute

The whole project can run free: your Mac, Kaggle's free GPU and CPU sessions, your remaining Google Cloud trial credit (spend it before ~Jan 16), and Mesa's lavapipe, which runs your renderer on CPU-only machines. NSF ACCESS and UC Davis Hive both need a faculty or grad-student lead, so they're out for now. The Open Science Pool is worth one application (see the tab). Every option, with its terms checked: [Free compute](03-free-compute.md)

**Free renderers:** your own GPU marcher is the only one that bends light, so final frames come from it. Free [Blender Cycles](https://www.blender.org), [OpenSpace](https://www.Github.com/OpenSpace/OpenSpace) or Houdini's free non-commercial edition with the [ytini tools](https://arxiv.org/html/1701.01730) are fine for sketching backdrop shapes. None of them can lens.

## Better ideas than the current plan, and pitfalls

The "movie-effects split" is the right core idea. These six changes make it better, and the biggest is basing the backdrop on the real Galactic Centre instead of generic noise.

1. **Model the backdrop on what is really there.** Sgr A\* sits inside the [minispiral: clumpy, photoionized streamers on orbits around the hole, some plunging to within a few arcseconds of it](https://arxiv.org/pdf/1006.0064), ringed by a dusty molecular circumnuclear disc. That gives you glowing pink-teal spiral arms, black dust lanes with lit inner rims, and a story: your cloud is a clump torn off a streamer like the Northern Arm. It is real, and it has a shape that generic noise can't fake.
2. **Build the backdrop's shape from flow, not raw noise.** With Option C, Act 1's final frame does this with real physics. For Options A and B, seed particles along a few orbit-like streamer curves, move them a few steps through curl noise plus the hole's gravity, rasterize into the grid, then add domain-warped detail. This is the step that separates "nebula" from "procedural noise" in [Swaab's work](https://80.lv/articles/how-to-create-gorgeous-space-nebulas-in-3d), and it is close to [BUF's frozen-simulation trees](https://history.siggraph.org/wp-content/uploads/2022/10/2015-Talks-Vidal_Visualizing-the-Cosmos_-a-Procedural-Approach.pdf).
3. **Put the glow in skins.** Make the emission strongest in a thin band at the surface of dense, dusty clumps (the side facing the centre's hot stars), with opaque dark cores, as in the [AMNH Orion model](https://www.cs.uaf.edu/~genetti/Research/Papers/EG00/Orion.html). Bright rims on dark shapes are the single most "real nebula" cue.
4. **Use the rest field instead of flow-aligned noise** (sub-particle detail section). It is stable under your time warps.
5. **Cut the cloud from the backdrop with the same noise.** If the cloud's rest-field noise is the backdrop's own noise function, sampled at the particles' starting positions, then at frame 0 the cloud looks exactly like the patch of nebula it was cut from. The seam disappears by construction.
6. **Freeze the nebula.** It moves 0.004% of its size during the story, so a static bake is both cheapest and correct.

| Pitfall | Why it happens | Fix |
| --- | --- | --- |
| Looks painted or "procedural" | Plain fBm, the same in every direction and at every scale | Shape from flow; long thin filaments; a few big dominant forms; check against real photos |
| Seam between painted and simulated gas | Different colour model, brightness scale or texture | One emission model for both; carve the cloud out of a streamer (remove backdrop density there, feathered); same noise and tags (idea 5) |
| Flicker and shimmer | Noise finer than a pixel; ray jitter that changes every frame | Band-limit octaves; a fixed jitter pattern per pixel; extra rays only where needed, like your photon-ring supersampling |
| Nebula goes black | Automatic exposure follows the stream, which brightens ~20 stops | A separate, labelled nebula gain, or tone-map the layers apart |
| Precision at parsec scale | Coordinates of ~10⁷ units in float32 (~7 digits) can't hold detail of ~1 unit | Camera-relative or per-volume local coordinates for the big grids |
| GPU memory on the Iris | A dense 1024³ grid with 4 float16 channels is ~8.6 GB | Your sparse bricks, or nested 256³ grids plus noise octaves evaluated while marching |
| Render time balloons | Long rays through a huge volume | Skip empty space with bricks; bigger steps far from the camera; always test a 4-second clip before a full render |

## Timeline to the 4K final

End of February for the final render, with the first week of March as spare time. Each step ends with a still or a 4-second clip for you to approve, as now. Full steps, settings and tests for Claude Code: [Build spec](02-build-spec.md)

| When | What |
| --- | --- |
| Nov 24 – Dec 7 | Emission-line colour model. Act 1 setup and a 50k test run (Mac or Kaggle CPU). Go lighter during finals. |
| Dec 8 – Dec 31 | Act 1 production run on the trial credit (must finish before ~Jan 16). Bake the backdrop from it and do look-dev. Act 2 lumpy cloud run. |
| Jan 1 – 14 | Rendering tests: 4K at 2 rays per pixel on the Mac; Kaggle GPU (Vulkan) or CPU (lavapipe); the 20 GB storage workaround. Rest-field wisps. |
| Jan 15 – 31 | Gravity-only streamer around the cloud, time-lapse and dive camera, short clips. |
| **Jan 31** | **Stop adding features.** Only fixes after this. |
| Feb 1 – 21 | Final 4K render, split between Mac overnights and Kaggle sessions. |
| Feb 22 – Mar 7 | Spare time: re-render bad frames, encode, handle surprises. |
