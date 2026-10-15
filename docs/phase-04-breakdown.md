# Phase 4 step breakdown (written 2026-10-15, overnight; for Hiten's review)

CLAUDE.md workflow 5 asks for a breakdown and approval before a new phase. Hiten was asleep and asked for work to continue, so this breakdown was written down and then followed; review it in the morning and redirect anything you disagree with.

## Checked against what exists

- Phase 1 gives the gravity-only generalized leapfrog (`sim/integrator.py`, η_orb = 0.00125 chosen in Step 1.4) and the integer tick timeline idea (`integrate_orbit`).
- Phase 3 gives SPH density/forces/gradients with list reuse, the shock switch, and a global-step kick-drift-kick integrator (`sim/hydro.py`) with no black hole (p = v).
- Phase 4 joins them: the gas moves in the hole's gravity using canonical momenta p (hydro forces are kicks on p, with v = ∂H/∂p), on individual power-of-two time steps.

Things that no longer fit or need care:
1. **η_orb = 0.00125, not the plan's 0.005** (Step 1.4 decision): ~4× more orbit steps per particle than the plan's cost estimates assumed.
2. **The kernel is Wendland C4** (Step 3.6): ~2.4× the plan's neighbour count; measured speed ≈ 30,000 full-step updates/s (Step 3.8). Block steps are the main speed lever from here.
3. **Conductivity with gravity:** Phantom uses the velocity-based signal speed (|v_ab · r̂_ab|) when external forces act, because the pressure-based one leaks heat in gas held up by gravity. The gas around the hole is exactly that case. **Decision (overnight, recommended option):** use the velocity-based signal (`COND_VELOCITY`) for runs with the hole, keep the pressure-based one for the Phase 3 tests without gravity. Recorded in DECISIONS.md.
4. **Phase 4 test 7** (speed-up on the 1k cloud to pericentre) needs the Phase 5 cloud set-up; a minimal version of it (glass sphere on the cloud orbit) is built here.

## Steps

| Step | Builds | Tests |
|---|---|---|
| 4.1 | `sim/timesteps.py`: integer timeline, levels, sync and level-change rules, neighbour limiter (levels differ by ≤ 2) | sync exactness after many steps; lengthening only at divisible ticks; limiter on hand-made level patterns |
| 4.2 | Coupled gravity + hydro on one shared level: Strang split (hydro half-kick on p → GLF gravity flow → hydro half-kick), u kicks, ledger Σm(H + u); a "no hole" option (free particles) | Phase 3 problems unchanged without the hole; gravity-only orbits = Phase 1; **hot-blob energy ledger** (plan test 4) with right vs wrong coupling |
| 4.3 | Block time steps: active sets, per-particle criteria (orbit, Courant, force), predicted states for inactive neighbours (gravity-only GLF for x, p; du/dt for u) | Phase 1 orbit tests under block steps (test 1); Sod and sound wave with forced mixed levels (test 2); determinism (test 5); synchronization (test 6) |
| 4.4 | Limiter and wake-up (Saitoh & Makino 2009; Durier & Dalla Vecchia 2012; read first) | Sedov with vs without limiter/wake-up: the difference must show (test 3) |
| 4.5 | Diagnostics (level histogram, active counts, ledger print); speed-up measurement | hot-blob ledger under block steps (test 4 full); speed-up vs global step on a 1k cloud to pericentre (test 7) |
