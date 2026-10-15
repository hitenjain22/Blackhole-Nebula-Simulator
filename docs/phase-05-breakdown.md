# Phase 5 step breakdown (written 2026-10-15; path approved by Hiten)

Phase 5 follows the cloud from apocentre through its first pericentre to t = 500,000 (phases/phase-05-cloud-infall.md). Phase 4 found a problem that must be understood first: in the 1k cloud, one particle of the leading edge was heated to u ≈ 0.025 (~10¹¹ K) as the stream swung through pericentre, lost its angular momentum and fell into the hole at t = 386,646 — identically with block and shared time steps (DECISIONS, 2026-10-15). The phase file itself says particles reaching r_acc on the first pass mean something is wrong: "stop and investigate". Hiten: take whatever time is needed.

## Checked against what exists

- Gravity, coupling, block steps, limiter: Phase 4 (`sim/coupled.py`, `sim/blocks.py`), all tests passing.
- A minimal cloud set-up exists (`sim/cloud.py`: glass ball, radius 100, 10⁴ K, apocentre state).
- Missing: accretion and escape removal with ledger entries, run folders with manifests (rule 8), checkpoints (every ≤ 15 minutes, resumable), snapshots, the diagnostics file, `docs/results-log.md`.
- Speed: block steps gave ×3.0 fewer particle-steps but only ×1.5 in wall time on the 1k approach; the 10k run needs better.

## Steps

| Step | Builds / does | Tests / evidence |
|---|---|---|
| **5.0** | **Investigation of the pericentre heating and plunge** (below). Fix whatever is wrong; nothing else is built until this is understood. | A written finding in DECISIONS/AI_LOG with the evidence; regression test(s) for any bug found. |
| 5.1 | Accretion (r_acc = 4) and escape (r = 50,000) removal, each removed particle's m, H, u and x × p in the ledger; works with block steps | Ledger exact across removals (unit test); the plan's original hot blob (u = 4 × 10⁻⁶) for one full orbit with gas falling in: ledger < 10⁻⁵ including the accreted part |
| 5.2 | Run infrastructure: TOML config, run folder with manifest (git commit, config, versions, seed, machine, times), float64 checkpoints every ≤ 15 min, clean resume, float32 snapshots on a schedule | Resume gives the same result as an uninterrupted run (bit for bit); manifest contents; never overwrites a run folder |
| 5.3 | Diagnostics per snapshot (`diagnostics.csv`): ledgers, centre-of-mass orbit vs the point-mass orbit, stream length/width/thickness, max density and temperature, mass above 10⁵ / 10⁶ K, dM/dε, level histogram; `docs/results-log.md` | Each diagnostic checked on a hand-made particle set with a known answer |
| 5.4 | Speed work before the long runs (profile first; candidates: lighter neighbour-density pass, fewer list rebuilds, prediction of only the particles that interact) | Results unchanged (bit-identical where possible); measured speed-up |
| 5.5 | The 1k run to t = 500,000; plots (six panels, edge-on view), preview animation | Expected-value checks: tidal energy spread ±R/r_a² (within ~30%), pericentre arrival spread ~21,800, centre at pericentre near 396,224, expansion ~20% before pericentre, flat ledgers |
| 5.6 | The 10k run (overnight); resolution comparison; the three physics questions of phase-05 §5.4 answered in the results log | Same checks at 10k; differences between 1k and 10k explained |

## Step 5.0 in detail (cheapest and most telling first)

1. **Do the orbits cross?** Gravity only (no gas forces), the same 1k cloud: track the leading-edge particles through pericentre. If their paths do not intersect near the hole, the collision seen with gas is not real, and the gas code is creating it.
2. **Where does the heat come from?** Split du/dt into compression (P dV), viscous heating and conduction, per particle, through the heating episode; repeat with conduction off, with each conduction signal, and with the viscosity switch off/on.
3. **Resolution:** the same run at 10k particles. A real effect stays roughly the same; an artefact usually changes a lot. Check whether the stream becomes thinner than h (particles per cross-section).
4. **The slow angular-momentum drift on the way in** (leading edge 7.25 → 6.3 before pericentre): is pressure really doing this? Compare with pressure off, and with the expected tidal spread.
5. **Cross-check with published and open-source work:** how Phantom, GADGET-2/4 and SWIFT handle the same situation (their viscosity/conduction limiters, time-step criteria near compact objects), and published tidal-stream simulations at pericentre (e.g. Bonnerot & Stone 2021 review) for the expected compression and heating. Formulas are re-derived from the papers; code is compared, and copied only where the license allows.

Each sub-step is a short experiment with a plain-language result for Hiten.
