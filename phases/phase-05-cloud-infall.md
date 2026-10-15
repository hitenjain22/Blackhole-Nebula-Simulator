# Phase 5: The cloud: approach, stretching and swing-by (in-depth)

**Goal:** the first real run. Build the cloud, send it on its orbit, and follow stages 1–3: the approach, tidal stretching into a stream, and the swing-by with its nozzle shock and orbit rotation.

**Ends with:** a 10k-particle run from the start to well past the first pericentre (the 1k cloud is a smoke test only: D1), a 50k run for the gas temperatures at closest approach (D3), with plots of the stream at key moments, the energy distribution of the gas, the ledgers, and a quick preview animation.

**Updated 2026-10-15** after the plunge review (`docs/review-2026-10-15-plunge.md`) and Hiten's decisions D1–D11 (`docs/DECISIONS.md`): velocity reconstruction in the viscosity (D2), cloud radius 25 for Phase 5 (D4), the hole-resolution check (D5), swallowing at r_acc = 4 with ledger entries (D6), Phase 10.2a's snapshot rule. Items that depend on the cloud radius are marked *(pending: cloud size)* and wait for Hiten's choice of the Phase 6 cloud size (review §7).

**Primary sources:** Gillessen et al. 2012, Burkert et al. 2012, Schartmann et al. 2012 (G2 cloud: initial conditions and tidal stretching), Bonnerot & Stone 2021 (review: nozzle shock and stream physics), Stone, Sari & Loeb 2013 (energy spread of disrupted material).

---

## 5.1 Initial conditions

| Parameter | Value | Notes |
|---|---|---|
| Orbit | r_p = 25, r_a = 5000, GN apocentre state from Phase 1 | cloud centre at (5000, 0, 0), velocity (0, 1.47016 × 10⁻³, 0) |
| Cloud radius | **25** (D4; was 100) | uniform density (option: a mild central concentration). A 4× smaller cloud has ~4× smaller smoothing lengths near the hole at the same particle count; at radius 100 even 10k particles let kernels span the hole (review §2b). The Phase 6 size is still open (review §7). |
| Cloud mass | 1 code mass unit ↔ 0.1 M☉ physically | the dynamics doesn't depend on it; units.py maps it for temperature and radiation |
| Temperature | 10⁴ K → u from units (ideal gas only in this phase; radiation pressure is about 10% of gas pressure in the cloud, not negligible, and Phase 6 switches to the full gas + radiation EOS for production) | sound speed ≈ 5 × 10⁻⁵ |
| Internal velocities | zero (option: small seeded turbulence, ~c_s) | |
| Particle layout | sphere carved from a relaxed glass, then scaled | random points cause density noise and spurious pressure kicks |
| Accretion radius | r_acc = 4 | removal with full ledger entry (built: `sim/accretion.py`, D6); removals on a particle's first approach are flagged |
| Self-gravity | off (justified: cloud tidal radius R (M/M_cloud)^(1/3) ≥ 8,750 ≫ 5,000 for any radius from 25 to 100; Jeans mass ≥ 2.4 M☉ ≫ 0.1 M☉ (19 M☉ at radius 100, scaling as R^(3/2))) | write the justification into DECISIONS.md |
| Viscosity | Phantom's, with velocity reconstruction (D2) | without it, the cold gas squeezed on its way in is heated and stirred by the viscosity (review §2c) |

**Expected numbers to check against** (from expected-values §2 and §9). *(pending: cloud size)*: the first, second and fourth items below were computed for radius 100; they scale with the radius and are not changed here until Hiten picks the cloud size.
- Each particle's orbital energy is fixed from the start (no self-gravity, negligible pressure). The spread across the cloud is about ±R/r_a² = ±4 × 10⁻⁶ around ε = −1.989 × 10⁻⁴: ±2%. So the return times of the stream's ends will differ by about ±3% of the radial period (±24,000 time units ≈ ±6 days) on the next return. This is the "tidal energy spread" result; measure it.
- The cloud expands slightly from its own pressure before pericentre: by roughly c_s × t ≈ 5 × 10⁻⁵ × 4 × 10⁵ ≈ 20 (20% of its radius).
- Centre of mass reaches pericentre near t = 396,224 (the cloud's size shifts individual particles by a few percent).
- Pericentre arrival times across the cloud span about 21,800 (standard deviation ≈ 5,500; test-particle estimate in expected-values §2b). They are set almost entirely by each particle's starting x offset (its orbital energy), not by its y or z offset.
- **No stream crossing is expected in this phase:** the incoming gas has fully passed the crossing radius r_SI ≈ 1,664 before the outgoing gas reaches it (Phase 6 §6.1). A crash before t = 500,000 means a bug, or a pressure effect worth investigating.

---

## 5.2 Run and diagnostics

**Runs:** 1k as a smoke test (does it run, are the ledgers flat; never used near the hole, D1); 10k to t = 500,000 (past first pericentre); 50k overnight for the temperatures at closest approach (D3). Snapshots follow Phase 10.2a's rule from Phase 5 on (Hiten, 2026-10-15): Δt_snap = clamp(T_orb(r_med)/40, 10, 10⁴), T_orb(r) = 2π r^(3/2), r_med = median radius of the gas inside r = 2,000 (all gas if none), plus an event snapshot when the centre of mass passes pericentre (built: `scripts/cloud_run.py`).

**Diagnostics per snapshot (saved to `diagnostics.csv` and plotted):**
- Ledgers (mass, energy, angular momentum).
- Centre-of-mass orbit vs the Phase 1 point-mass orbit.
- Stream geometry: length along the orbit, width and thickness (principal axes of the particle distribution, or a local width measure along the stream).
- Maximum density and temperature; mass above 10⁵ K, 10⁶ K.
- dM/dε histogram (specific orbital energy of every particle), at the start, just before pericentre and just after. Before: the tidal spread above. After: the nozzle shock adds a spread; quantify it.
- Time-step level histogram (from Phase 4).
- **Hole resolution (D5, built: `sim/diagnostics.py`):** for particles at r < 60, the median and largest 2h/r and the share of neighbour pairs more than 90° apart around the hole, with a warning above 1%; the typical (median) and hottest-1% u of the gas near the hole (r < 300), compared with the gravity-only ceiling of review §2d *(pending: cloud size — the ceiling depends on it)*.
- Swallowed particles (D6): time, angular momentum at the start and at removal, u; any on their first approach listed separately (the alarm in 5.3).
- Shock marker: the viscous heating rate after reconstruction (Hiten, 2026-10-15; the switch's α is ~1 for all the cold gas and no longer marks shocks).

**Pictures for Hiten** (`outputs/checks/`):
- Six panels, x–y plane, particles coloured by temperature: start, halfway in, approaching (r ≈ 500), at pericentre (zoomed), just after (zoomed), fan of returning gas.
- An edge-on (x–z) view at pericentre showing the vertical squeeze.
- A quick matplotlib animation (MP4) of the whole phase: no lensing, just particles. It's a preview, not the final look.

---

## 5.3 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Glass not relaxed | density "breathing", particles jittering at t = 0 | relax longer; check density noise < 1% before carving |
| Edge particles flying off | a halo of fast particles from the cloud surface | normal: the edge expands into vacuum at ~c_s; check speeds are ~c_s, not more |
| Huge h for stray particles | slow neighbour searches | h_max cap; particles with few neighbours are effectively ballistic, fine. (A capped particle with too few neighbours used to crash the gradient pass; fixed, D11.) |
| Kernels span the hole | gas moving in opposite directions on either side of the hole interacts through the kernel and is braked into the hole (the Phase 4 1k plunge) | resolution near the hole: D5 warns; more particles or a smaller cloud (D3, D4); never trust anything near the hole while the warning is on |
| Viscosity heats a smooth squeeze | the cold stream warms 25–35× and its angular momentum is stirred on the way in, with no shock | velocity reconstruction (D2); check with the viscosity-off comparison of review §2c |
| Time step collapses at pericentre | run stalls | expected spike; measure; check v_sig and the nozzle compression are physical (compare with Bonnerot & Stone 2021 estimates of the compression factor) |
| Stream thinner than h | the stream becomes a line of particles; pressure meaningless | resolution limit: report effective resolution (particles per stream cross-section); 10k improves; mention in report |
| Centre of mass drifts from the point-mass orbit before pericentre | bug in gravity or coupling | with pressure negligible it must follow; compare ledger and orbit |
| Energy ledger jumps at pericentre | coupling or limiter bug | rerun Phase 4 test 4 on this setup |
| Particles reaching r_acc on the first pass | means a wrong orbit (r_p = 25 ≫ 4) | stop and investigate |

---

## 5.4 Physics questions this phase answers (for the report)

1. How thin does the stream get, and does the cloud's pressure matter during the approach?
2. How strong is the nozzle shock: temperature jump, fraction of gas shocked? (Converged published work finds very little real heating at pericentre, ~4 × 10⁻⁵ of the orbital energy, and that low-resolution SPH overestimates it; our runs can only bound it. Report the measured heating against the gravity-only ceiling of review §2d rather than as a measured shock strength.)
3. What is the energy spread dM/dε after pericentre, and how does it set the return times?

---

## Done when

- [ ] 10k (and the 50k) runs complete to t = 500,000 with flat ledgers; the 1k smoke test runs cleanly.
- [ ] Plots and preview animation reviewed by Hiten.
- [ ] Measured tidal energy spread agrees with the ±R/r_a² estimate (within ~30%; it's an estimate). *(pending: cloud size)*
- [ ] Results noted in `docs/results-log.md` (start this file now: one dated entry per meaningful result).
