# Results log

One dated entry per meaningful result (phases/phase-05-cloud-infall.md, "Done when"). Numbers are as measured on this Mac; details and decisions are in `docs/DECISIONS.md`.

---

## 2026-10-15: Why gas fell into the hole too early (Phase 4 → 5)

- The 1k cloud lost a particle into the hole at t = 386,646 (before its centre reached pericentre), heated to ~10¹¹ K. An external review (`docs/review-2026-10-15-plunge.md`), confirmed here run for run, showed this is **numerical**: (1) Phantom's artificial viscosity mistakes the cold stream's smooth supersonic squeeze for a shock, heating it 25–35× and stirring its angular momentum on the way in; (2) near the hole the 1k cloud is under-resolved, its SPH kernels reaching across the hole (90% → 35% of neighbour pairs), so gas moving in opposite directions is braked against itself.
- Gravity only, and with viscosity off, nothing falls in (lowest angular momentum 7.213 and 7.077; capture needs < 4).
- Literature agrees: converged simulations find the real pericentre ("nozzle") heating is tiny (~4 × 10⁻⁵ of the orbital energy); low-resolution SPH overestimates it.

## 2026-10-15: Velocity reconstruction in the viscosity (D2)

- In a smooth supersonic squeeze the viscous heating drops from 2.8 × 10⁻² to 1.6 × 10⁻¹⁹ (zero to round-off).
- All Phase 3 and 4 tests pass with unchanged tolerances. Shock tube plateaus within 1.0–1.4%, post-shock ripple 1.5% (limit 5%); Sedov blast radius within 0.5%; sound wave unchanged.
- Gresho vortex error with the switch improves 12–13% (0.143 → 0.124 low, 0.091 → 0.080 high resolution).
- Hot-blob energy ledger unchanged (4.7 × 10⁻⁶ shared step, 4.06 × 10⁻⁶ block steps, both falling 4.0× per halving); the block-step angular-momentum drift falls from 3.1 × 10⁻¹⁰ to 2.6 × 10⁻¹¹.
- The "wrong coupling" comparison error falls from 1.5 × 10⁻³ to 8.5 × 10⁻⁵ (still flat with step size and 18× the right coupling's): the viscosity no longer does work on the smooth approach, so there is less force to misapply.

## 2026-10-16: The cloud's first pass at 10k and 50k particles (radius 25, Phase 5)

Both runs: t = 0 → 400,000 (cloud starts at r = 5,000; centre of mass closest to the hole at t = 396,235, r = 134), velocity reconstruction on, no cooling.

| | 10k | 50k | review's gravity-only ceiling |
|---|---|---|---|
| particles swallowed | 0 | 0 | 0 |
| energy ledger drift (end) | 1.8 × 10⁻⁵ | 1.6 × 10⁻⁵ | |
| kernels reaching across the hole | 0% | 0% | |
| typical heat u near the hole, at pericentre | 1.3 × 10⁻⁶ | 6.4 × 10⁻⁷ | ~2 × 10⁻⁸ (the most heat possible if all the squeezing motion became heat; review §2d) |
| hottest 1% at pericentre | 6.9 × 10⁻⁶ | 2.6 × 10⁻⁶ | 1–3 × 10⁻⁶ |
| hottest particle at pericentre | 7.3 × 10⁻⁴ | 2.6 × 10⁻⁵ | ~5 × 10⁻⁶ |
| spread of orbital energy at the end (middle 50%) | ±33% | ±14% | |
| gas made unbound | 5.0% | 0.7% | |
| wall time | 1.7 h | 13.6 h | |

- With 5× more particles the pericentre heating halves, the hottest particle is 28× cooler, and the energy spread and unbound gas fall 2.3× and 7×: much of the 10k heating is numerical (resolution), as the review predicted. The 50k hottest 1% sits inside the review's ceiling; the hottest particle is still ~5× above it.
- Not yet converged, and still far too warm: typical gas at 50k is ~30× above the gravity-only ceiling (10k: ~65×), so the typical temperatures at closest approach are still numerical. (Corrected 2026-10-16: this line first compared with a literature figure for the radius-100 cloud, ~3 × 10⁻⁷; for radius 25 the ceiling is ~2 × 10⁻⁸, review §2d.) The heat is still far too small to change the orbits, but the energy spread it causes is not (next point).
- This matters for Phase 6: the energy spread after the first pass sets when the returning stream crashes into itself. At 10k it is mostly numerical; Phase 6 runs at 10k (Hiten's decision) should be read with that in mind.
- Pictures: `docs/figures/cloud50k_r25_D2_panels.png`, `_metrics.png`, `_edge_on.png`, `_preview.mp4`. (This run predates the viscous-heating snapshot field, so its "mass in shocks" is not available.)

## 2026-10-17: Radius 60, first pass at 10k vs 50k (resolution check before Phase 6)

Same settings (radius 60, gas + radiation EOS, β = 230, swallowing at r_acc = 4); R6.1 (10k) paused at t = 403,901, `R6.1_r60_50k` stopped at t = 404,962. Compared at the same times:

| | 10k at 396,284 | 50k at 396,262 | 10k at 402,955 | 50k at 402,929 |
|---|---|---|---|---|
| gas flung out (unbound) | 4.3% | 0.32% | **10.9%** | **0.75%** |
| orbital-energy spread, middle 50% | ±10% | ±2.8% | **±43%** | **±13%** |
| typical heat near the hole (u, r < 300) | 4.5 × 10⁻⁶ | 8.9 × 10⁻⁷ | 3.1 × 10⁻⁵ | 2.7 × 10⁻⁶ |
| lost (swallowed + below L < 4), peak during the pass | | | 0.13% | 0.016% |
| compute time to the end of the pass | | | 5.0 h | 29 h |

- **Resolution check of the lost gas (D4 update): PASS** (peak 0.016% < 0.13%; at t = 403,901: 0.004% < 0.080%; `scripts/loss_check.py`).
- Flung-out gas falls 15×, the energy spread 3×, the heating ~10×: at 10k most of the post-pass spread and outflow is numerical, so a 10k Phase 6 would mostly measure numerical error (the crash would come at passage 2 for that reason). The hot bands of the 10k run are gone at 50k (one small hot spot remains).
- 50k is not converged either: its spread grows from ±2.8% to ±13% after the pass (tidal estimate ±1.1%). Consistent with the literature: SPH over-heats the stream at pericentre at low resolution and the effect shrinks with resolution (Price et al. 2024; SPH-EXA, arXiv:2510.26663).
- Pictures: `docs/figures/R6.1_10k_vs_50k.png`, `docs/figures/loss_R6.1_r60_vs_R6.1_r60_50k.png`.
