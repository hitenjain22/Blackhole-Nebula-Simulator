# Nebula–Black Hole: master plan

A gas cloud falls toward a supermassive black hole, is stretched into a stream, whips around the hole, crashes into itself, forms a hot glowing torus, and drains into the hole. The project produces (a) a physically defensible simulation with measured results and (b) cinematic, ray-traced videos of it, on a 2020 Intel MacBook Air, using overnight runs.

Read with: `CLAUDE.md` (rules), `phases/` (per-phase specs), `reference/expected-values.md` (test numbers), `reference/references.md` (sources).

---

## 1. Deliverables

1. **Simulator** (`src/nebula/sim`): 3D smoothed-particle hydrodynamics (SPH) with relativistic-accurate gravity from the hole, shock heating, radiative cooling, disk viscosity and accretion. Writes snapshots and checkpoints.
2. **Renderer** (`src/nebula/render`): GPU ray tracer that bends light around the hole, renders the gas as a glowing volume with true-colour blackbody emission, Doppler and gravitational shifts, and a lensed star field. Writes frames; ffmpeg assembles videos.
3. **1D disk model** (`src/nebula/physics/disk1d.py`): the viscous-disk equation, used to validate the SPH drain and to run slow-drain variants.
4. **Results**: accretion rate over time, circularization time, energy and angular-momentum budgets, resolution study (1k → 50k particles), thin-vs-thick cooling comparison, comparison with published simulations.
5. **Videos**: the full event (about 2 minutes), stage close-ups, and a multi-band (infrared → X-ray) sequence.
6. **Write-up**: README, technical report with figures, `AI_LOG.md` showing how AI-assisted development was done and checked.

---

## 2. Decisions already made

| Topic | Decision | Why |
|---|---|---|
| Machine | MacBook Air 2020, Intel i3 2-core, Iris Plus GPU, 8 GB | Hiten's computer |
| Mode | Offline simulation + offline rendering, replayed as video | Lets a modest laptop produce high-quality results overnight |
| Language | Python; Numba for hot loops; WGSL shaders via wgpu-py for the GPU | Readable, standard in research and AI; fast enough |
| Environment | uv (Homebrew does not support this Mac) | Pins Python and libraries reproducibly |
| Gas method | SPH | Follows gas over huge ranges of distance; standard for this problem (Phantom) |
| Gravity on gas | Tejeda & Rosswog 2013 "generalized Newtonian" (GN) acceleration | Exact Schwarzschild orbit shapes, ISCO and precession, in a form that mixes with SPH |
| Time stepping | Generalized (implicit) leapfrog on positions and canonical momenta, hydro forces as kicks on the momenta, individual power-of-two (block) time steps | Long-term stability, an exact energy ledger, and efficiency over a huge range of orbital speeds |
| Black hole | Non-spinning first; spin added in Phase 11 | Get everything right once before adding complexity |
| Colour | True-colour blackbody (default), multi-band mode, labelled cinematic grade | Science and looks from one physics model |
| Particles | 1k development, 10k everyday, 50k overnight showcase | Estimated feasible in the earlier planning notes; Phase 0 measures the real speed |

---

## 3. The scenario

| Quantity | Value (code units, G = c = M = 1) | Physical (M = 4.3 × 10⁶ M☉) |
|---|---|---|
| Hole mass | 1 | 4.3 million solar masses (like Sagittarius A*) |
| Length unit GM/c² | 1 | 6.35 × 10⁶ km = 0.0424 AU |
| Time unit GM/c³ | 1 | 21.18 s |
| Cloud start (apocentre) | r = 5,000 | 212 AU |
| Closest approach (pericentre) | r = 25 | 1.06 AU |
| Cloud radius | 100 | 4.2 AU |
| Cloud mass | (enters only via radiation and units) | **0.1 M☉** |
| Cloud temperature | 10⁴ K (warm ionized gas, like G2) | sound speed 5.05 × 10⁻⁵ c |
| Ring forms at | r ≈ 50.9 (from the orbit's angular momentum) | 2.2 AU |
| Precession per orbit | 24.04° | |
| Time from start to pericentre | 396,224 | 97 days |
| Ring orbit period | ≈ 2,303 | 13.55 hours |
| First stream crash (estimate) | at the 2nd or 3rd pericentre passage, r ≈ 1,664 | 0.8–1.3 years after the start |
| Torus drain, half the mass (estimate, α = 0.1, H/r = 0.3) | ≈ 26,600 | ≈ 6.5 days |

**Why 0.1 M☉.** With no self-gravity, the cloud's mass enters the dynamics only through the equation of state (how much radiation pressure the gas has; Phase 6). Mostly it sets how bright and how opaque the gas is. At 0.1 M☉ the hole is fed at about 8 to 30 times its Eddington rate (depending on whether the stream's arrival or the torus drain sets the pace; the Eddington rate is where radiation pressure balances gravity). That makes the torus optically thick (electron-scattering optical depth of thousands; true absorption is much weaker in the hot torus, see Phase 6 on LTE), radiation-pressure supported (γ ≈ 4/3), slowly cooling (cooling time ≈ 230/Ω, about 37 orbits) and glowing as a blackbody at 10⁴–10⁶ K: blue-white. This is consistent with observed tidal-disruption flares. A G2-mass cloud (about 3 Earth masses) would instead feed the hole at 0.1% of Eddington, and the gas would form a dim, hot, transparent flow that doesn't glow like a blackbody. The cloud's own gravity is negligible at 0.1 M☉ (its tidal radius is ~35,000 M, far outside the start point) and its Jeans mass is about 19 M☉, so it is not collapsing.

Numbers are in `reference/expected-values.md` with how they were computed (scripts in `reference/scripts/`).

**What this means for the story.** Most of the event's clock time is quiet orbiting: the cloud swings past the hole once or twice (one orbit takes ~194 days) before, on its second or third pass, its stream is spread out enough to crash into itself. Then things happen fast: crashes, heating, a thick glowing torus, and a drain of days to weeks. The video's time warp (Phase 8) compresses the quiet orbits and slows down the crash and torus. The years-long "slow drain" of thin, cool disks is shown with the 1D disk model (Phase 7), since this torus is thick and drains quickly.

---

## 4. Stages and how each is modelled

| Stage | Physics | Method |
|---|---|---|
| 1. Approach | Free fall on a bound orbit; gas pressure negligible | SPH + GN gravity, large time steps far out |
| 2. Tidal stretching | Tidal shear into a stream; vertical compression | Same; resolution along the stream is the limit |
| 3. Swing-by | Nozzle shock at pericentre; relativistic precession | Small time steps; artificial viscosity captures the shock |
| 4. Return and crash | Stream self-intersection shocks; circularization | SPH shocks, heating; energy and angular-momentum ledgers |
| 5. Cooling and settling | Radiative cooling vs shock heating; thick torus or thin ring | Gas + radiation equation of state; β-cooling or diffusion cooling |
| 6. Drain | Viscous angular-momentum transport | Physical α viscosity in SPH, validated against the 1D disk model |
| 7. Plunge | No stable orbits inside r = 6 | GN gravity handles it; particles removed at r_acc = 4 with full ledger |

---

## 5. Architecture

```
configs/runs/<run>.toml ──► simulator ──► runs/<run-id>/
                                     ├── manifest.json
                                     ├── checkpoints/ (float64, full state)
                                     ├── snapshots/  (float32, render fields)
                                     └── diagnostics.csv (ledgers, accretion rate, timings)
                                                │
configs/shots/<shot>.toml (camera, time warp, look) ───►│
                                                ▼
                                renderer ──► frames/<shot-id>/frame_00001.png …
                                                │
                                                ▼
                                ffmpeg ──► videos/<shot-id>.mp4
analysis notebooks ◄── diagnostics.csv, snapshots ──► docs/figures/
```

- **Simulator and renderer are separate programs.** They share only the snapshot format and `src/nebula/physics`.
- **Snapshot times follow a physics-based schedule** (about 40 per orbit of the gas in view; Phase 10.2a). The renderer interpolates between snapshots (Phase 8.2). For hero shots, `nebula resim` re-runs a window from a checkpoint and writes snapshots exactly at the director's frame times (Phase 8.2, 8.5).
- **The 1D disk model** is a separate tool: `analysis/disk_profile.py` reads a snapshot and measures Σ(r) and ν(r); `physics/disk1d.py` evolves them (pure arrays in, arrays out).

---

## 6. Repository layout

```
nebula-blackhole/
  CLAUDE.md  PLAN.md  README.md  AI_LOG.md
  phases/  reference/  docs/ (DECISIONS.md, report, figures)
  pyproject.toml  uv.lock  .gitignore
  src/nebula/
    units.py                 # constants and conversions (only place with physical units)
    physics/                 # pure formulas, each tested
      gravity.py             # GN acceleration, exact Schwarzschild references, Lense–Thirring (Phase 11)
      orbits.py              # E, L from turning points, precession, circular orbits
      kernels.py  eos.py  opacity.py  viscosity.py  cooling.py
      geodesics.py           # light rays: Schwarzschild and Kerr reference integrators
      redshift.py  blackbody.py  colour.py
      disk1d.py
    sim/
      state.py  neighbours.py  sph.py  integrator.py  timesteps.py
      setup_cloud.py  accretion.py  diagnostics.py  io.py  run.py
    render/
      device.py  camera.py  starfield.py  gasgrid.py  director.py
      shaders/*.wgsl  frames.py  video.py
    analysis/                # metrics.py, disk_profile.py (snapshot → Σ(r), ν(r)), light curves, figures
  tests/       # mirrors src; physics tests use reference/expected-values.md
  configs/
    runs/      # one TOML per simulation run
    shots/     # one TOML per rendered shot (camera path, time warp, look)
  scripts/     # overnight.sh and other small shell helpers (Phase 10)
  data/        # small committed data files: CIE tables, star catalogue (with licences)
  notebooks/   # exploration and figures
  runs/ frames/ videos/ outputs/   # git-ignored
```

---

## 7. Conventions

- **Coordinates:** hole at the origin. Initial orbit in the x–y plane, cloud starts on the +x axis at apocentre, moving in +y. Spin (Phase 11) along +z by default.
- **Time:** the GN coordinate time t. It matches Schwarzschild coordinate time far from the hole; near the hole it differs by a few percent (orbit *shapes* are exact; *timing* is approximate). Physical times on screen use t × 21.18 s.
- **Particle data:** structure of arrays (`pos[N,3]`, `mom[N,3]` (the canonical momentum p that the integrator evolves; the velocity v = ∂H/∂p is derived from it), `mass[N]`, `u[N]`, `h[N]`, `rho[N]`, …), contiguous float64.
- **Configs:** TOML; every parameter has a default in code and is echoed into the run manifest.
- **Random numbers:** one seeded `numpy.random.Generator` per run, seed in the config.
- **Snapshot format:** one `.npz` per snapshot (float32 fields + float64 time), plus a JSON index. Checkpoints: `.npz` float64 with integrator state.
- **Names:** `r_acc`, `r_isco`, `alpha_ss`, `beta_cool` etc. match the symbols in the phase docs.

---

## 8. Phases

| # | Phase | Ends with (what Hiten sees) | Detail |
|---|---|---|---|
| 0 | Setup and platform check | Environment works; a test GPU image and a Numba speed number | Detailed |
| 1 | Units and orbits | Plot of the cloud-centre orbit precessing by 24.04°; all orbit tests pass | Detailed |
| 2 | First black-hole render | Lensed star-field image with the correct shadow size | Detailed |
| 3 | SPH core | Shock-tube plot matching the exact solution | In-depth |
| 4 | Block time stepping | Orbits and SPH tests unchanged with block steps; speed-up measured | In-depth |
| 5 | The cloud: approach and swing-by | 1k-particle run to first pericentre; plots of the stream | In-depth |
| 6 | Shocks, cooling and the ring | Torus forms; thin-vs-thick comparison; 10k run | In-depth |
| 7 | Viscosity and the drain | Ring-spreading test vs 1D model; accretion-rate curve; 50k overnight run | In-depth |
| 8 | Rendering the gas | Lensed, glowing gas frames from real snapshots | In-depth |
| 9 | Colour physics | True-colour, Doppler-shifted frames; multi-band mode | In-depth |
| 10 | Production pipeline | Overnight run → overnight render → video, unattended | In-depth |
| 11 | Spinning black hole | Kerr shadow; aligned then tilted-orbit runs | In-depth |
| 12 | Science results | Resolution study, figures, technical report | In-depth |
| 13 | AI/ML extensions and portfolio | Differentiable fit of α; README, videos, AI log | In-depth |

Phases 2 and 3 can swap order. Phases 8–9 can start as soon as Phase 5 produces snapshots.

---

## 9. How correctness is established

Five layers. A result is trusted only when all layers relevant to it pass.

1. **Unit tests with known answers.** Every physics function, against values in `reference/expected-values.md` (exact solutions, published numbers, or independent high-precision computations).
2. **Reference vs fast.** Every Numba or GPU kernel against its readable Python reference.
3. **Standard hydro tests.** Sod shock tube, sound wave, Sedov blast, Gresho vortex, Keplerian ring spreading: the tests every SPH code paper shows.
4. **Conservation ledgers.** Mass, energy, angular momentum, with accreted and radiated parts, on every run.
5. **Convergence and comparison.** The same run at 1k, 5k, 10k, 20k, 50k particles; results that change with resolution are reported as such. Qualitative comparison with Bonnerot et al. 2016 and Price et al. 2024.

---

## 10. Project-wide risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A library (Numba, wgpu-py) no longer supports Intel Macs | Medium | High | Phase 0 checks before anything else; fallbacks listed there |
| SPH bugs that produce plausible-looking but wrong gas | High | High | Standard tests (layer 3) before any science run; ledgers on every run |
| Coupling GN gravity with SPH breaks energy conservation | Medium | Medium | Use the generalized-force coupling of Phase 4/5 (makes the ledger exact) and test it |
| Too slow on this laptop | Medium | Medium | Profile early (Phase 3–4); KD-tree and Numba; reduce particle count before reducing accuracy |
| Resolution too low in the thin stream | High | Medium | Report it honestly; convergence study; particle count where it matters most |
| Renderer memory/time blows up | Medium | Medium | Tiled GPU dispatch, nested grids, measured per-frame budget |
| Physical story inconsistent (e.g. thin disk with fast drain) | Medium | High (credibility) | Choices tied to the 0.1 M☉ scenario; every approximation labelled in the report |
| Scope creep | High | Medium | Phases end with a demo; cut list below |

---

## 11. Where to spend the most care

1. **SPH correctness (Phase 3).** Everything depends on it. Do not move on until all standard tests pass.
2. **Time stepping (Phase 4).** The most bug-prone code. Silent errors here look like physics.
3. **The energy and angular-momentum ledgers (Phases 4–7).** They are the evidence for every claim.
4. **The light-ray integrator and redshift (Phases 2, 9).** The visual centrepiece and easy to get subtly wrong.
5. **Reproducible overnight runs (Phase 10).** A crashed overnight run costs a day.

---

## 12. Cut list (if time runs short, cut from the top)

1. Phase 13 ML extension (keep the AI log and portfolio).
2. Tilted spinning-hole run (keep the aligned one).
3. Spinning hole entirely (Phase 11).
4. Diffusion cooling (keep β-cooling).
5. Multi-band mode (keep true colour).
6. 50k showcase run (keep 10k with the convergence study up to 20k).

Never cut: the standard SPH tests, the ledgers, lensing.

---

## 13. Out of scope (state this in the report)

Full radiation transport (replaced by an equation of state with radiation pressure, and a cooling law), magnetic fields (replaced by α viscosity), relativistic hydrodynamics (gravity is relativistic-accurate, gas dynamics is Newtonian), outflows driven by radiation pressure, research-level resolution (10⁵–10⁶ particles).
