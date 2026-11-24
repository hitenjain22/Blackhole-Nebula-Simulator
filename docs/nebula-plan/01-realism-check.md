# Realism check

Most of what's on screen can be real physics. The rest is a handful of labelled cheats that a scientist would accept in a film, and each one gets a line in the report.

## Real vs cheat

| Element | Status | Why |
| --- | --- | --- |
| Lensing, Kerr shadow, photon ring | Real | GR ray tracing, tested against float64 references |
| Close-up cloud: stretching, 24° precession, crash, ring | Real | SPH with relativistic-accurate gravity; your test suite |
| Act 1 nebula torn into streamers and a ring | Real physics, lower resolution | Newtonian SPH, as in [Mapelli & Trani 2016](https://ar5iv.labs.arxiv.org/html/1510.05259) |
| Gravity-only streamer particles | Approximation | Valid away from the hole and the crash; switched to SPH where they get close |
| Frozen backdrop | Real approximation | It moves ~0.004% of its size during the close-up |
| Pink and red emission-line colours | Real physics, simplified | Line formulas are real; which gas is ionized is set by tags, not computed |
| Amount of teal [O III] | Mild artistic choice | The UV light ionizing gas near Sgr A\* is relatively soft (see below), so real teal would be limited |
| Dust lanes and reddening | Real, simplified | Standard extinction curve, one dust-to-gas ratio |
| Rest-field wisps | Labelled cheat | Detail finer than the simulation resolves; plausible, not computed |
| Separate nebula exposure | Labelled cheat | No camera or eye can show a dim nebula and the white-hot stream at once |
| Time jump (years → days) | Honest editing | Stated by the on-screen clock |
| Inner accretion disc | Physically motivated model | Already in the project (steady thin-disc model) |
| Seeing the minispiral in visible light | Right for a nearby camera | From Earth it's hidden by dust, which is why it's [mapped only in infrared and radio](https://arxiv.org/pdf/astro-ph/0405197v1) |

## Timescales

| Process | Time | Source |
| --- | --- | --- |
| Close-up story | ~320 days | Project |
| Cloud's orbit | ~194 days | Project |
| Gas moving at 100 km/s over the close-up | ~18 AU | My arithmetic |
| Free fall to the hole from 1 pc | ~8,000 years | My arithmetic (point mass 4.3 × 10⁶ M☉) |
| Circular orbit at 1 pc | ~45,000 years | My arithmetic |
| A giant cloud's first pericentre passage | Starts at 1–2.5 × 10⁵ years, lasts ~1.5 × 10⁵ years | [Mapelli & Trani 2016](https://ar5iv.labs.arxiv.org/html/1510.05259) |

This gap of ~10⁵ between the nebula's clock and the close-up's clock is why the video needs two clocks to show both.

## What the real gas around Sgr A\* would look like up close

- **Glowing streamers:** the minispiral is [clumpy ionized streamers orbiting Sgr A\*, some reaching within a few arcseconds of it, photoionized by the massive young stars of the central parsec](https://arxiv.org/pdf/1006.0064). They glow mainly in hydrogen recombination light.
- **Less teal than Orion:** the ionizing UV there is [relatively soft, with an effective temperature of ~35,000 K](https://arxiv.org/pdf/1006.0064), and Galactic Centre surveys find [very low excitation ionized gas](https://www.arxiv.org/pdf/1803.02806). Expect pink and red to dominate, with teal only near the hottest stars. A bit more teal is a defensible style choice; a teal-dominated nebula is not.
- **A dark, clumpy ring:** the circumnuclear disc is [a clumpy, inhomogeneous ring of molecular gas](https://www.aanda.org/articles/aa/full_html/2016/01/aa27195-15/aa27195-15.html) with dust. In visible light it's dark, with lit inner rims facing the central stars.
- **Stringy filaments:** Webb found that ionized hydrogen near the Galactic Centre forms [long filaments threaded by strong magnetic fields](https://www.colorado.edu/today/2025/04/02/webb-telescope-captures-images-insight-one-milky-ways-most-extreme-environments), unlike nebulae elsewhere. Your SPH has no magnetic fields, so stretched rest-field wisps are the honest way to suggest this.
- **A very crowded star field:** Webb's 50-light-year view of nearby Sagittarius C holds [an estimated 500,000 stars](https://science.nasa.gov/asset/webb/sagittarius-c-nircam-image/). Stars embedded at every depth are both realistic and your best depth cue.
- **Invisible hot gas:** the central cavity is also filled with very hot, thin gas that glows in X-rays and would look empty to the eye.

## Physics notes for Act 1

- **Two gas phases:** dense gas is cold and molecular (Mapelli & Trani used a 100 K floor), rendered as dark dust. The ionized skin is added by the colour model, not by the simulation's temperature.
- **Self-gravity:** off (Hiten's choice, 2026-11-23). The lumpy starting density still gives clumps and contrast; they stretch instead of collapsing. Mapelli & Trani's high-resolution run with self-gravity is the comparison if a 50k "on" test is ever wanted.
- **Magnetic fields:** left out, as stated in PLAN.md's out-of-scope list. Mention the Webb filaments in the report as what real fields would add.
