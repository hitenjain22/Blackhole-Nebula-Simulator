# Phase 12: Science results and the technical report (in-depth)

**Goal:** turn the runs into defensible results: what converges with resolution and what doesn't, how the outcome depends on the physical choices (relativity, cooling, viscosity, orbit), how it compares with published work, and a technical report a physicist (or Hiten's mentor) can check.

**Ends with:** a resolution study from 1k to 50k particles; parameter studies; a comparison section; `docs/report.md` with ~15 figures; a compute-budget table.

**Primary sources for comparison:** Bonnerot et al. 2016 (disc formation, cooling dependence), Hayasaki, Stone & Loeb 2013 (circularization via precession), Bonnerot & Stone 2021 (review), Price et al. 2024 (GR SPH tidal disruption; "Eddington envelopes"), Steinberg & Stone 2024 (radiation-hydrodynamics TDE; check), Gillessen et al. 2012 and later G2 papers (the real cloud), Lodato & Price 2010 (SPH disk viscosity). Treat all comparisons as **qualitative or order-of-magnitude** unless the setups match: our cloud is not a star, and our orbit is not theirs.

---

## 12.1 Headline quantities (defined once, in `analysis/metrics.py`)

Every run produces the same table, computed by one tested function (no one-off analysis scripts: a number in the report must come from `metrics.py` or a script that calls it):

| Symbol | Definition |
|---|---|
| n_crash | pericentre passage at which the first stream–stream crash happens (mass in shocks at r ≈ r_SI above threshold) |
| t_crash | time of the first crash |
| ΔE_pass | orbital energy dissipated per passage (from the ledger) |
| t_circ | time when ≥ 50% of the bound, non-accreted mass has e < 0.3 and r < 200 |
| (H/r)_torus | mass-weighted rms(z)/r of the torus at t_circ + 2 ring orbits |
| r_torus | mass-weighted mean radius of the torus at the same time |
| Ṁ_peak | peak accretion rate (binned), in Ṁ_Edd |
| t_half | time when 50% of the cloud mass has been accreted |
| L_peak | peak luminosity, in L_Edd |
| η_rad | E_radiated / (M_accreted c²) |
| f_unb | unbound (escaped) mass fraction |
| Ledger drift | max \|ΔE\| / Σm\|H\| and \|ΔJ\|/J over the run |

---

## 12.2 Resolution study

- Baseline physics (GN, β = 230, α = 0.1, r_p = 25) at **1k, 2.5k, 5k, 10k, 20k, 50k** particles.
- Three seeds at 1k, 2.5k and 5k (cheap) to measure run-to-run scatter.
- Plot each headline quantity vs N (log axis) with seed error bars.
- Call a quantity **converged** if it changes by less than the seed scatter (or 10%) between 20k and 50k; otherwise report the trend and say it isn't converged. Don't extrapolate unless the trend is clean (then show a Richardson-style estimate as a secondary number).
- Expected: quantities set by large-scale dynamics (t_crash, r_torus, t_half) converge earlier; quantities set by thin structures (crash dissipation, f_unb, L_peak) converge last or not at all at 50k. That's normal at this resolution; research codes use 10⁵–10⁶ particles.

---

## 12.3 Parameter studies (10k where affordable, else 1k + 5k checks)

| Study | Values | Question |
|---|---|---|
| Gravity | GN vs Newtonian | how much does relativistic precession speed circularization? |
| Cooling β | 1, 10, 100, 230, ∞ | thin ring ↔ thick torus; light-curve shape |
| Viscosity α | 0.01, 0.03, 0.1, 0.3 | t_half ∝ α⁻¹? |
| Pericentre r_p | 15, 25, 40 | precession 43.17° / 24.04° / 14.48°; crash radius r_SI 451 / 1,664 / 3,403; circularization radius 31.19 / 50.89 / 80.43 (expected-values §2) |
| Cloud radius R | 50, 100, 200 | energy spread ∝ R: earlier crash for bigger clouds? |
| Spin (Phase 11) | a = 0, 0.9 aligned, 0.9 tilted | do streams miss? |

Each study: one figure (headline quantities vs parameter) + one picture row (same moment, same camera).

---

## 12.4 Comparisons

1. **Thin vs thick with cooling** vs Bonnerot et al. 2016: same qualitative trend expected (efficient cooling → thin ring at r_circ; inefficient → thick, extended torus).
2. **Precession-driven circularization** vs Hayasaki, Stone & Loeb 2013 and the review: crash location formula vs our measured r_SI.
3. **Drain** vs the 1D model (Phase 7) and the α scaling.
4. **Light curve** vs observed tidal-disruption flares: peak luminosity relative to Eddington and time scales (days to weeks for our compact torus). Qualitative.
5. **The real G2:** G2 passed pericentre in early 2014 at about 1,560 ± 70 Schwarzschild radii ≈ **3,100 M** (Plewa et al. 2017, checked online; Gillessen et al. 2012 had predicted ~3,100 R_s ≈ 6,200 M before later orbit fits brought it closer). That is over 100 times wider than our r_p = 25. Run a cheap 1k variant with r_p = 3,000 from the same apocentre to show the cloud is only stretched, never crashes into itself or forms a torus within the run. That explains why G2 didn't flare and makes our scenario choice (a G2-like cloud on a much closer orbit) clear.

For each comparison, write what matches, what doesn't, and why (different setup vs a problem in our code).

---

## 12.5 Error budget (what limits each result)

A table in the report: for each headline quantity, the dominant uncertainty: resolution (from 12.2), stochastic scatter (seeds), physics approximations (β-cooling instead of radiation transport, α viscosity instead of magnetic turbulence, GN timing differences near the hole of a few %, LTE EOS, no self-gravity, ISCO shift missing for spin), and numerical parameters (AV switch settings, η, C_cour). Never present a number without its main caveat.

---

## 12.6 The technical report (`docs/report.md`; optional PDF via pandoc)

1. Summary (one paragraph and the hero image).
2. Scenario and why (0.1 M☉ cloud, Sgr A*-like hole, orbit).
3. Methods: GN gravity and the canonical integrator; SPH (grad-h, switches); block time steps and the exact energy coupling; EOS and cooling; viscosity and the 1D model; the renderer (Schwarzschild/Kerr rays, transfer, colour).
4. **Verification:** the five layers (PLAN §9), with the key figures: orbit precession test, Sod, Gresho with/without the switch, Sedov limiter demo, hot-blob ledger (right vs wrong coupling), ring spreading vs 1D, shadow and Einstein-ring tests, redshift test.
5. Results: the stage-by-stage story with numbers (12.1), resolution study, parameter studies.
6. Comparisons (12.4).
7. Limitations and out of scope (PLAN §13; 12.5).
8. How it was built: the AI-assisted workflow (from AI_LOG.md; Phase 13).
9. References (from `reference/references.md`, only those actually used).

**Figures (~15):** stage montage; orbit rosette; Sod; Gresho; ledger flat-line + wrong coupling; crash zoom; eccentricity heat map; β grid; Newtonian vs GN; Ṁ(t) and L(t); SPH vs 1D drain; resolution study panel; true-colour vs multi-band vs cinematic; Kerr vs Schwarzschild render; energy budget.

---

## 12.7 Compute budget

Before launching the campaign, list every run with its estimated wall time (`nebula estimate`) and nights needed. A rough guide: 1k runs ≈ tens of minutes; 10k runs ≈ hours; 50k ≈ 1–2 nights each. The resolution study + parameter studies is likely 2–3 weeks of overnight slots. Prioritize: baseline 10k → resolution study → β → GN vs Newton → α → r_p → R → spin. Record actual times as you go.

---

## 12.8 What will go wrong, and the fix

| Problem | Fix |
|---|---|
| A headline quantity doesn't converge | report it as not converged, with the trend; that's honest science |
| Seeds disagree more than resolution levels | report scatter; increase N before claiming a trend |
| Analysis code differs between runs | one `metrics.py`, tested on a synthetic run with known answers |
| Over-claiming from comparisons | qualitative unless setups match; state the differences |
| Cherry-picked figures | every figure is generated by a script from a run folder; the script and run id are in the caption |
| Running out of nights | follow the priority list; cut from the bottom |

---

## Done when

- [ ] Resolution study complete through at least 20k (50k if affordable).
- [ ] Studies in 12.3 done in priority order (cut-list respected).
- [ ] `docs/report.md` written, every figure reproducible by a script, read by Hiten and his mentor.
- [ ] Compute-budget table filled with actual times; committed.
