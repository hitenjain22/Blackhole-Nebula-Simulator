# Colour fixes (round 7)

2026-11-24. For the round 5 sheets: `n5_00575` (stream around the hole) and `n4_02112` (torus). Answers Claude Code's five research questions; the mapping is at the end.

Three root causes, all fixable before N2:

1. **Neon:** every line is drawn at its pure single-wavelength colour, far outside what a screen can show, and today's display step keeps colours fully saturated at every brightness. Out-of-range colours are pinned to the edge of the screen's colour range, which is exactly what reads as neon.
2. **Black gaps:** thin gas is dimmed twice: its glow falls as density squared (real), and the tone curve's dark end crushes what's left (not real). The result is bright blobs on black.
3. **Patches:** colours come from a fixed pattern in space, and a line can appear on its own (green-only, violet-only gas). That never happens in a real nebula: hydrogen always glows wherever the other lines do.

## What the numbers say

| Line | Wavelength | Eye brightness per watt (CIE ȳ) | Colour on its own | Strength vs Hβ in the Orion Nebula |
| --- | --- | --- | --- | --- |
| Hα | 656.3 nm | 0.076 | deep red, far outside sRGB | 2.86 (standard case B) |
| Hβ | 486.1 nm | 0.177 | blue-cyan | 1 |
| Hγ | 434.0 nm | 0.016 | violet-blue | ~0.47 (case B) |
| [O III] | 500.7 nm | 0.334 | pure green, far outside sRGB (linear R = −1.9 at Y = 1) | [0.8–2.3](https://arxiv.org/pdf/1008.1002) |
| He I | 587.6 nm | 0.786 | gold | [0.04–0.13](https://arxiv.org/pdf/1008.1002) |
| [Ne III] | 386.9 nm | 0.0001 | violet the eye barely sees | [0.08–0.14](https://arxiv.org/pdf/1008.1002) |

ȳ values and colours: computed from the repo's CIE table (`data/cie1931_2deg_1nm.csv`) and `XYZ_TO_SRGB`. Orion ratios: O'Dell et al. 2010, reddening-corrected, central region.

What follows from the table:

- **Real teal is a mix, not [O III] alone.** With Orion's central ratios (all lines together), the mix lands at chromaticity (0.227, 0.308): a soft teal almost inside sRGB (linear RGB per unit Y = 0.10, 1.22, 1.44). [O III] alone lands at (0.010, 0.525): neon green. That's the green in the n4 and n5 sheets.
- **Violet should be invisible.** [Ne III]'s ȳ is ~0.0001, so a "neon violet" patch is unphysical. Drop the line from the look; at real strength it adds nothing you can see.
- **No gold patches.** He I is gold, but only ~0.1 Hβ: a faint warm tint in the mix, never its own region.
- **Pure hydrogen is magenta-pink** (chromaticity 0.351, 0.170). Photos look softer pink because of continuum light and dust reddening, which fixes 2 and 3 add.

## Fix 1: lock the line ratios to physics (kills neon green and violet)

Each bit of gas emits one spectral recipe, all lines together; a line is never on its own. Tags only move ratios within real ranges.

| Ratio | Range | *Start* | Source |
| --- | --- | --- | --- |
| Hα / Hβ | 2.86, fixed (dust then reddens it) | 2.86 | Case B recombination (textbook) |
| Hγ / Hβ | 0.47, fixed | 0.47 | Case B (textbook) |
| [O III] 5007 / Hβ | 0–3 | 0.5 base; 1.5–2.5 in teal regions | [Orion 0.8–2.3](https://arxiv.org/pdf/1008.1002); Galactic Centre gas is [low excitation](https://www.arxiv.org/pdf/1803.02806), so a low base |
| [O III] 4959 / 5007 | 1/3, fixed | 0.33 | Atomic physics (textbook) |
| [N II] 6584 / Hα | 0.1–0.5 | 0.3 | Typical H II regions (textbook) |
| [S II] / Hα | 0.05–0.3 | 0.15 | Typical H II regions (textbook) |
| He I 5876 / Hβ | ~0.1, fixed | 0.1 | [Orion 0.04–0.13](https://arxiv.org/pdf/1008.1002) |
| [Ne III] 3869 | off | 0 | Invisible (ȳ ≈ 0.0001) |

Hiten's teal accent (decision 3) comes only from raising [O III]/Hβ in some regions, so teal always arrives mixed, with pink at its edges.

## Fix 2: new display step: AgX plus soft gamut compression (kills neon, adds the path to white)

Today (`render/tone.py`): a tone curve on luminance only, then the smallest desaturation that fits the screen's colour range. Colours stay fully saturated at any brightness, and out-of-range lines sit exactly on the edge of the range. Real cameras and film do the opposite: [bright colours go towards white](https://wiki.blender.org/release_notes/4.0/color_management/).

| Option | Saturated colours | Faint gas | Verdict |
| --- | --- | --- | --- |
| Current: ACES fit on luminance only | Stay neon at every brightness | Crushed by the curve's toe (slope ~0.21 at 0) | Replace |
| ACES 1.x per channel | Known hue shifts (saturated blue drifts purple) | Fine | No |
| Blender's old Filmic | [Collapses all colours into six before attenuating to white](https://github.com/EaryChow/AgX) | Fine | No |
| **AgX** | [Better handling of saturated highlights, without ACES's hue shifts or Filmic's "Notorious Six"](https://www.blendernation.com/2023/09/05/filmics-successor-new-agx-view-transform-in-blender-4-0/); a ["graceful path to white"](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/agx/) | [16.5 stops of dynamic range](https://github.com/EaryChow/AgX), so faint gas survives | **Use** |

- **AgX is Blender's default since 4.0.** It's [less saturated than ACES Filmic out of the box](https://discourse.threejs.org/t/is-agx-tonemapping-implemented-correctly/60609); if it's too grey, add a mild "punchy" look (a little contrast and saturation after the curve), not more line gain.
- **Soft gamut compression before AgX**, in the style of the ACES Reference Gamut Compression: leave the inner part of the colour range alone and squeeze only the outer part with a power curve. A [threshold of 0.2 leaves the inner 80% unaffected](https://github.com/jedypod/gamut-compress/blob/master/GamutCompress.fuse). This replaces "least desaturation".
- **Implementation:** port three.js's compact AgX (MIT) to WGSL, with a Python reference and a GPU-vs-CPU test (rule 2). A few dozen operations per pixel.
- **Tests:** neutral grey stays neutral; output is monotonic in exposure; a very bright saturated input comes out with lower saturation than a dim one (the path to white); GPU equals CPU within 1e-3.

## Fix 3: fill the black gaps and make it look like gas

- **Scattered light (labelled model):** real nebulae also glow by dust scattering starlight, and in Orion [scattered light is strong enough to distort line measurements](https://arxiv.org/pdf/1008.1002). Add a faint bluish-white continuum proportional to density (not density²), so thin gas glows softly between the clumps. *Start:* 10–25% of the line brightness at mean density. It also takes the edge off neon, as astrophotographers do when they [keep a natural RGB base and blend the narrowband lines in](https://app.astrobin.com/forum/topic/259714?page=1&post=400321).
- **Dust that isn't pure black:** cap dust's optical depth in thin gas, and give dust an albedo (*start:* 0.5) so lanes look brown-grey and lit at their edges, not like holes.
- **The dark end:** AgX's log encoding replaces the toe that crushes faint gas. Astronomers solve the same problem with Lupton's asinh stretch, which [shows faint objects while preserving the colours of bright ones](https://arxiv.org/abs/astro-ph/0312483).
- **Bloom:** physically based, no brightness threshold, energy-conserving, in linear light before AgX. Down-sample and blur directly instead of thresholding, with [the blurred layer set to a low value (say 0.04)](https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom) (the Call of Duty 2014 down/up-sample chain). Cheap even on the Iris.

## Fix 4: hot gas goes to blue-white

- **Physics:** recombination lines fade as gas heats; [clumps heated past ~10⁵ K stop glowing in them](https://arxiv.org/pdf/1401.2990). Today's fade (t_fade 4 × 10⁴ K, power 8) is a near on/off switch.
- **Smoother fade:** line weight = 1 / (1 + (T / T_f)^p) with *start* T_f = 3 × 10⁴ K, p = 3. That gives 50% at 30,000 K, 11% at 60,000 K, and ~3% at 100,000 K.
- **Drama floor (labelled cheat):** keep 8–12% of the line colour in hot gas as a tint on the blackbody, so the torus keeps a hint of pink and teal at its edges. This is the "n4 top right, but less colour" look Hiten asked for.
- **AgX does the rest:** its path to white whitens the brightest torus core without any extra rule.

## Fix 5: colours that move with the gas (step N4, already planned)

- Per-particle tags ([O III]/Hβ, [N II]/Hα, dust), from smooth noise at each particle's start position, deposited density-weighted (tag × ρ, then divide by ρ). Where streams mix in the crash, the colours blend.
- A large-scale tag field (correlation length ≥ a third of the cloud radius) with smooth weights and no thresholds, so there are no hard patch edges.

## Bonus: the star field

Every star has the same brightness, and they fill the frame like confetti, competing with the gas. Real fields have a few bright stars and many near the limit of visibility. Dim the typical star by a few stops, keep a handful bright, and pass the stars through the same AgX step.

## Round 7 for Claude Code

Do these in order and show one comparison sheet per test frame (`n5_00575` and `n4_02112`):

| Variant | Contains |
| --- | --- |
| A | Round 5 soft_mix (today's best) |
| B | Fix 1 (locked ratios) + Fix 2 (AgX and gamut compression) |
| C | B + Fix 3 (scattered light, dust albedo, bloom) |
| D | C + Fix 4 (smooth temperature fade with a drama floor) |

New functions follow the usual rules: formulas in `physics/` (line ratios, temperature fade), display code in `render/tone.py` with its WGSL twin, tests with expected values from a `reference/scripts/` script. Record AgX, scattered light, dust albedo, bloom and the drama floor in DECISIONS as labelled look choices.

## Answers to Claude Code's five questions

| Question | Where |
| --- | --- |
| 1. Natural colour from narrowband data | "What the numbers say", Fix 1, Fix 3 (continuum) |
| 2. Tone mapping: filmic vs ACES vs AgX | Fix 2 (AgX) |
| 3. Bloom and soft scattering | Fix 3 |
| 4. Colours mixing inside the volume | Fix 5, plus Fix 1 (lines never alone) |
| 5. Smooth colour → blue-white with temperature | Fix 4 |

Background reading: [Rector et al. 2007](https://arxiv.org/pdf/astro-ph/0412138), how NASA/NOAO image makers build colour images, and [Rector et al. 2017](https://arxiv.org/pdf/1703.00490), on appealing colour composites that still convey the science.
