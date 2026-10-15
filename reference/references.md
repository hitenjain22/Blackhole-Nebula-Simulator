# References: what each is for, and how sure we are of it

**Status key**
- ✅ **Checked online** while writing this plan (title and arXiv id or journal confirmed by search on 2026-10-14/15). Volume and page numbers not shown in the search results are still from memory: confirm them before citing in the report.
- 📝 **From memory**: a standard reference, very likely right, but not re-checked. Look it up before relying on a specific equation, number or citation detail.

Rule for Claude Code: before implementing an equation "from paper X", open paper X (arXiv is free) and check the equation. Don't implement from this list's one-line summaries.

---

## Gravity and orbits

| Reference | Used for | Status |
|---|---|---|
| Tejeda & Rosswog 2013, MNRAS (doi 10.1093/mnras/stt853), arXiv:1303.4068, "An accurate Newtonian description of particle motion around a Schwarzschild black hole" | the GN acceleration and Lagrangian (Phase 1) | ✅ |
| Tejeda & Rosswog 2014, arXiv:1402.1171, "Generalized Newtonian description of particle motion in spherically symmetric spacetimes" | background, extensions | ✅ |
| Ghosh, Sarkar & Bhadra 2014, MNRAS (doi 10.1093/mnras/stu2046), arXiv:1410.7900, "Newtonian analogue of corresponding spacetime dynamics of rotating black holes" | candidate Kerr extension (Phase 11 option). Its own abstract says the ISCO is reproduced reasonably only for a < 0.7 | ✅ |
| arXiv:1601.01034, "Pseudo-Newtonian equations for evolution of particles and fluids in stationary space-times" | another candidate for spin (Phase 11 option); read before use | ✅ title; authors not checked |
| Artemova, Björnsson & Novikov 1996, ApJ 461, 565 | older Kerr pseudo-potential (Phase 11 option, background) | 📝 |
| Mukhopadhyay 2002, ApJ 581, 427 | Kerr pseudo-Newtonian potential (Phase 11 option, background) | 📝 |
| Bardeen, Press & Teukolsky 1972, ApJ 178, 347 | Kerr circular orbits, ISCO formula, photon orbits | 📝 |
| Lense & Thirring 1918, Phys. Z. 19, 156 | frame dragging (weak field) | 📝 |
| Hairer, Lubich & Wanner, *Geometric Numerical Integration* (Springer, 2nd ed. 2006) | generalized leapfrog for non-separable Hamiltonians | 📝 |
| Hut, Makino & McMillan 1995, ApJ 443, L93 | time-symmetric adaptive steps (Phase 1 fix option) | 📝 |

## Smoothed-particle hydrodynamics

| Reference | Used for | Status |
|---|---|---|
| Price 2012, J. Comput. Phys. 231, 759, arXiv:1012.1885, "Smoothed particle hydrodynamics and magnetohydrodynamics" | the SPH equations (grad-h formulation) (Phase 3) | ✅ |
| Price et al. 2018, PASA 35, e031, arXiv:1702.03930, "Phantom: a smoothed particle hydrodynamics and magnetohydrodynamics code for astrophysics" | reference implementation choices: viscosity, switches, disc viscosity, accretion | ✅ |
| Cullen & Dehnen 2010, MNRAS 408, 669, "Inviscid smoothed particle hydrodynamics" | viscosity switch (Phase 3) | ✅ title |
| Price 2008, J. Comput. Phys. 227, 10040, arXiv:0709.2772, "Modelling discontinuities and Kelvin–Helmholtz instabilities in SPH" | artificial conductivity (Phases 3, 6) | ✅ |
| Monaghan 1997, J. Comput. Phys. 136, 298 | signal-velocity artificial viscosity | 📝 |
| Monaghan 1992, ARA&A 30, 543 | SPH review | 📝 |
| Springel 2010, ARA&A 48, 391 | SPH review; Gresho test setup | 📝 |
| Balsara 1995, J. Comput. Phys. 121, 357 | shear limiter for viscosity | 📝 |
| Dehnen & Aly 2012, MNRAS 425, 1068 | Wendland kernels, pairing instability | 📝 |
| Gresho & Chan 1990, Int. J. Numer. Methods Fluids 11, 621 | Gresho vortex test | 📝 |
| Kamm & Timmes 2007, LA-UR-07-2849, "On efficient generation of numerically robust Sedov solutions" | Sedov test values | ✅ |
| Liptai & Price 2019, MNRAS 485, 819, arXiv:1901.08064, "General relativistic smoothed particle hydrodynamics" | context: what full GR SPH does (we don't) | ✅ |
| Price 2007, PASA 24, 159 (SPLASH) | interpolating particles to grids for rendering (Phase 8) | 📝 |

## Time stepping

| Reference | Used for | Status |
|---|---|---|
| Springel 2005, MNRAS 364, 1105 (GADGET-2) | block time steps, kick-drift-kick | 📝 |
| Quinn et al. 1997, arXiv:astro-ph/9710043 | time-stepping for N-body | 📝 |
| Saitoh & Makino 2009, ApJ 697, L99, arXiv:0808.0773, "A necessary condition for individual time steps in SPH simulations" | neighbour time-step limiter (Phase 4) | ✅ |
| Durier & Dalla Vecchia 2012, MNRAS 419, 465, arXiv:1105.3729, "Implementation of feedback in SPH: towards concordance of methods" | waking inactive particles (Phase 4) | ✅ |

## The cloud, tidal disruption and circularization

| Reference | Used for | Status |
|---|---|---|
| Gillessen et al. 2012, Nature 481, 51, arXiv:1112.3264, "A gas cloud on its way towards the supermassive black hole at the Galactic Centre" | G2, the real cloud our scenario is modelled on | ✅ |
| Plewa et al. 2017, ApJ, arXiv:1704.05351, "The post-pericenter evolution of the Galactic Center source G2" | G2's measured pericentre: 1,560 ± 70 R_S (18.4 light hours) in 2014.18 (Phase 12) | ✅ |
| Burkert et al. 2012, ApJ 750, 58 | G2 physics and simulations | 📝 |
| Schartmann et al. 2012, ApJ 755, 155 | G2 simulations | 📝 |
| Stone, Sari & Loeb 2013, MNRAS 435, 1809, arXiv:1210.3374, "Consequences of strong compression in tidal disruption events" | energy spread, compression at pericentre (Phase 5) | ✅ |
| Hayasaki, Stone & Loeb 2013, MNRAS 434, 909, arXiv:1210.1333, "Finite, intense accretion bursts from tidal disruption of stars on bound orbits" | circularization of bound debris by precession (Phases 6, 12) | ✅ |
| Hayasaki, Stone & Loeb 2016, arXiv:1501.05207, "Circularization of tidally disrupted stars around spinning supermassive black holes" | spin and circularization (Phase 11) | ✅ |
| Bonnerot, Rossi, Lodato & Price 2016, MNRAS 455, 2253, arXiv:1501.04635, "Disc formation from tidal disruptions of stars on eccentric orbits by Schwarzschild black holes" | thin ring vs thick torus with cooling (Phases 6, 12); closest published analogue | ✅ |
| Bonnerot & Stone 2021, Space Sci. Rev. 217, 16, arXiv:2008.11731, "Formation of an accretion flow" | review: nozzle shock, self-intersection, circularization | ✅ |
| Guillochon & Ramirez-Ruiz 2015, ApJ, arXiv:1501.05306, "A dark year for tidal disruption events" | spin-induced stream misses (Phase 11) | ✅ |
| Steinberg & Stone 2024, Nature 625, 463, "Stream–disk shocks as the origins of peak light in tidal disruption events" | radiation-hydrodynamics comparison (Phase 12) | ✅ (volume from the search listing; page from memory) |
| Price et al. 2024, ApJL 971, L46, arXiv:2404.09381, "Eddington envelopes: the fate of stars on parabolic orbits tidally disrupted by supermassive black holes" | state-of-the-art GR SPH comparison (Phase 12) | ✅ |
| arXiv:2608.16986 (2026), "General relativistic hydrodynamics of stellar tidal disruptions in Kerr spacetime: methods, validation, and first applications" | very recent; worth reading for Phases 11–12 | ✅ title; authors not checked |
| Rees 1988, Nature 333, 523 | classic tidal-disruption picture (background) | 📝 |

## Disks, viscosity and cooling

| Reference | Used for | Status |
|---|---|---|
| Shakura & Sunyaev 1973, A&A 24, 337 | α viscosity | 📝 |
| Lynden-Bell & Pringle 1974, MNRAS 168, 603 | spreading-ring solution (Phase 7 tests) | 📝 |
| Pringle 1981, ARA&A 19, 137 | accretion-disc review, diffusion equation | 📝 |
| King, Pringle & Livio 2007, MNRAS 376, 1740 | observed α values | 📝 |
| Lodato & Price 2010, MNRAS 405, 1212, arXiv:1002.2973, "On the diffusive propagation of warps in thin accretion discs" | α_SS ≈ (α_AV/10)⟨h⟩/H calibration of SPH disc viscosity (Phase 7) | ✅ |
| Flebbe et al. 1994, ApJ 431, 754 | Navier–Stokes physical viscosity in SPH (Phase 7 Option A) | 📝 (search didn't find it; verify) |
| Murray 1996, MNRAS 279, 402 | SPH accretion discs, viscosity | 📝 |
| Nealon, Price & Nixon 2015, MNRAS 448, 1526, arXiv:1501.01687, "On the Bardeen–Petterson effect in black hole accretion discs" | warped discs around spinning holes (Phase 11) | ✅ |
| Bardeen & Petterson 1975, ApJ 195, L65 | inner disk alignment | 📝 |
| Balbus & Hawley 1998, Rev. Mod. Phys. 70, 1 | why discs are turbulent (background) | 📝 |
| Gammie 2001, ApJ 553, 174 | β-cooling (Phase 6) | 📝 |
| Lombardi, McInally & Faber 2015, MNRAS 447, 25, arXiv:1411.3678, "An efficient radiative cooling approximation for use in hydrodynamic simulations" | diffusion cooling option (Phase 6) | ✅ |
| Stamatellos et al. 2007, A&A 475, 37 | radiative cooling approximation for SPH (Phase 6 option) | 📝 |
| Frank, King & Raine 2002, *Accretion Power in Astrophysics* (3rd ed., CUP) | textbook: discs, opacities (Kramers coefficient for Phase 9) | 📝 |

## Light, rendering and colour

| Reference | Used for | Status |
|---|---|---|
| Synge 1966, MNRAS 131, 463 | shadow size for a static observer (Phase 2) | 📝 |
| Bardeen 1973, in *Black Holes* (Les Houches, eds. DeWitt & DeWitt) | Kerr shadow curve (Phase 11) | 📝 |
| Luminet 1979, A&A 75, 228 | first image of a disk around a black hole (visual reference) | 📝 |
| James, von Tunzelmann, Franklin & Thorne 2015, Class. Quantum Grav. 32, 065001 | Interstellar rendering: ray bundles, anti-flicker, Kerr | 📝 |
| Rybicki & Lightman 1979, *Radiative Processes in Astrophysics* (Wiley) | radiative transfer, I_ν/ν³ invariance (Phases 8–9) | 📝 |
| Misner, Thorne & Wheeler 1973, *Gravitation*; Chandrasekhar 1983, *The Mathematical Theory of Black Holes* | geodesics, redshift | 📝 |
| CIE 1931 2° standard observer colour-matching functions (CIE dataset; also in the `colour-science` package) | blackbody colour (Phase 9) | values ✅ (computed from the package's table) |
| IEC 61966-2-1 (sRGB) | XYZ → sRGB, transfer function | 📝 (matrix matches the `colour-science` package) |
| Wyman, Sloan & Shirley 2013, JCGT 2(2), 1 | analytic CMF fits (not used for tests; shown to be 0.0025 off at 2,000 K) | 📝 |

## Tools and platform facts

| Fact | Used for | Status |
|---|---|---|
| PyTorch publishes no macOS Intel (x86_64) wheels after 2.2.2 (PyTorch dev-discuss, "PyTorch macOS x86 builds deprecation starting January 2024") | Phase 13 framework choice | ✅ |
| JAX on Intel Macs | Phase 13 | not confirmed either way: check with uv before relying on it |
| Homebrew doesn't support this Intel Mac | setup (Phase 0) | confirmed by Hiten's own installer message |
