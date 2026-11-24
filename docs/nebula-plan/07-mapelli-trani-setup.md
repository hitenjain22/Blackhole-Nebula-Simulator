# Mapelli & Trani 2016: exact simulation setup (for Phase 14 N2, Act 1)

Source: Mapelli & Trani 2016, A&A, "Modelling the formation of the circumnuclear ring in the Galactic centre", [arXiv:1510.05259](https://ar5iv.labs.arxiv.org/html/1510.05259). Read 2026-11-23 (Sections 1-4.1 and the start of 4.2 verbatim; the rest from a summary, marked). Values to reproduce or compare against in N2.

## Code and black hole
- gasoline N-body/SPH (Wadsley et al. 2004) with the Read et al. 2010 OSPH changes.
- SMBH: sink particle, M = 3.5 × 10⁶ M☉ (Ghez et al. 2003), **fixed in place** (no recoil, "to prevent spurious kicks"). Sink radius 5 × 10⁻³ pc; softening 10⁻³ pc.
- Gas softening 10⁻³ pc (R1-R5), 4.6 × 10⁻⁴ pc (R6). Self-gravity: not stated as a switch, but self-gravitating clumps collapse in the outer ring, so **on** (inferred). Our decision 5: off.

## External potential (important)
- Stellar cusp, rigid: ρ(r) = 2.8 × 10⁶ M☉ pc⁻³ (r / 0.22 pc)^(−γ), γ = 1.2 inside 0.22 pc, 1.75 outside.
- Bulge: Hernquist, ρ(r) = M_b a / (2π r (r + a)³), M_b = 2.9 × 10¹⁰ M☉, a = 0.7 kpc.
- Stellar mass inside 26 pc ≈ 1.3 × 10⁸ M☉; stars equal the SMBH's mass at ~1.6 pc. **At the cloud's distance the stars dominate gravity**: a point-mass-only hole gives different orbits and timescales.

## Cloud and runs
- Uniform-density sphere, radius 15 pc, centre 26 pc from the SMBH; v_esc(26 pc) ≈ 34 km/s.
- Turbulence: divergence-free (solenoidal) Gaussian random field on a grid, P(k) ∝ k⁻⁴; σ(l) ∝ l^(1/2) (Larson 1981); "supersonic, marginally self-bound" (Hayfield et al. 2011). No Mach number or virial ratio given; seed not given.

| Run | M (10⁴ M☉) | v_in / v_esc | b (pc) | m_gas (M☉) |
|---|---|---|---|---|
| R1 | 12.7 | 0.208 | 26 | 1.2 |
| R2 | 4.27 | 0.208 | 26 | 0.4 |
| R3 | 12.7 | 0.375 | 26 | 1.2 |
| R4 | 12.7 | 0.52 | 8 | 1.2 |
| R5 | 12.7 | 0.52 | 26 | 1.2 |
| R6 | 12.7 | 0.52 | 26 | 0.12 (10× resolution test) |

- R1 gas particle count: 107,783 (Fig. 3 caption; does not divide exactly into the stated mass at 1.2 M☉: treat with care). Others not stated.
- v_in ≈ 7.1 / 12.8 / 17.7 km/s for 0.208 / 0.375 / 0.52 (derived from v_esc 34 km/s, not stated).

## Thermodynamics
- Cooling: Boley 2009 / Boley et al. 2010 radiative scheme, ∇·F = −(36π)^(1/3) s⁻¹ σ (T⁴ − T_irr⁴)(Δτ + 1/Δτ)⁻¹, s = (m/ρ)^(1/3), Δτ = s κ ρ; D'Alessio et al. 2001 opacities (1 μm grains); T_irr = 100 K; **floor 100 K**. SMBH feedback: compressional heating only. EOS and initial temperature not stated.

## Timeline and results (R1 unless noted)
- First pericentre at 1.0-2.5 × 10⁵ yr, lasting ~1.5 × 10⁵ yr; cloud → stream in ≲ 2.5 × 10⁵ yr.
- Inner ring forms in the first passage: outer radius ≲ 0.3-0.4 pc, inner edge at the sink radius.
- Outer ring (the circumnuclear-ring analogue) from ≳ 3.5 × 10⁵ yr (second passage), radius ~2 pc (2-4 pc across runs); r_CNR ∝ v_in^0.8 b^0.5.
- Angular momentum conserved within ≲ 1% over 2 Myr (0.7% R1 … 0.1% R6).
- At 2 Myr (Table 2): R1 M_CNR 1.15 × 10⁵ M☉, r_CNR 2.1 pc, width 1.2 pc, v_circ 125 km/s, inner ring 1.25 × 10⁴ M☉ at 0.35 pc. Inner/outer rings misaligned ~24°. Gas mostly 100-160 K, max ~1,000 K; ~500 K inside ~0.02 pc.
- Run length 2 Myr; figures at 1.5, 3.5, 5.5, 7.5, 9.5, 10.5 × 10⁵ yr (Figs. 1-2, R1), 1.5 Myr (Fig. 4, all runs), 2 Myr (Fig. 6, face-on and edge-on).

## Differences from the build spec to decide in N2
1. SMBH mass: paper 3.5 × 10⁶, project 4.3 × 10⁶ M☉ (keep ours; note it).
2. **External stellar potential**: the build spec has the hole only; the paper's cusp + bulge dominate gravity at 26 pc. Recommendation: include the rigid cusp + bulge potential (a few lines in Phantom's external forces), else the orbits and timings won't match the paper's figures (the N2 gate compares with them).
3. Self-gravity off (Hiten's decision 5) vs on in the paper: expected softer streamers, fewer collapsed clumps; the comparison with Figs. 1-2 must allow for it.
4. Cooling: paper uses the Boley radiative scheme with a 100 K floor; spec says "simple cooling, 100 K floor".
5. Sink 0.05 pc (spec) vs 5 × 10⁻³ pc (paper): larger is faster; the inner ring (0.3-0.4 pc) stays outside both.
