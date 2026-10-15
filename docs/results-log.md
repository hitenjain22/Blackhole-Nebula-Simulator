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
