# Phase 7 breakdown (APPROVED by Hiten 2026-10-21, with Claude's recommended answers)

Drafted 2026-10-21 while R6.1 v2 runs (CLAUDE.md workflow 5: re-read the phase file, check it
against the code and against what Phase 6 taught us, propose steps, wait for approval).

## What Phase 6 changed, and what that means for the Phase 7 plan as written

| Phase 7 plan says | What we now know | Proposal (for Hiten to decide) |
|---|---|---|
| R7.1 at **10k**; R7.3/R7.4 at **1k** | 1k is a smoke test only (D1); at 10k the first pass is dominated by numerical heating (unbound 11% vs 0.75% at 50k) | Science runs at 50k (cloud); 1k/10k only for quick checks of code paths. The α scan (R7.3) and r_acc check (R7.4) at 50k cost ~3 days each on the trial machine, so pick the essential ones (e.g. α = 0.03 and 0.3 besides the 0.1 of R7.1) |
| R7.1 starts from the cloud ("from the start until half the cloud has been swallowed") | R6.1 v2 already follows the cloud from the start to t = 3.8 × 10⁶ | Start R7.1 from an R6.1 v2 checkpoint once a torus exists (adds physical viscosity from there); saves ~2 days per run and keeps R6.1 as the inviscid reference |
| Ledger requirement |ΔE|/Σm|H| < 10⁻⁴ | After the crash the relevant scale is the energy the shocks process (Hiten, 2026-10-21) | Apply the same post-crash rule in Phase 7 |
| Ring-spreading test cost "40 min at 10k, 3–4 h at 50k at 2 × 10⁵ updates/s" | Measured speed: ~6,600 particle-steps/s on the Mac, ~32,000 on the cloud machine | 10k ring test on the Mac (~4 h), 50k on the cloud (~1 day); remeasure before committing |
| 1D test 3 uses `mpmath` | Not installed (no dependency yet) | Add `mpmath` as a dev/test dependency (small, pure Python; reason: independent high-precision check of the analytic ring solution) |
| Torus at r ≈ 51 | Hole-resolution (D5) and the over-heated near-hole gas matter for the drain | Report ⟨h⟩/H and D5 numbers for the torus; R7.4 (r_acc 3 vs 5) checks the inner boundary |

## Proposed steps (each: plan in 2–5 lines, tests, report, wait; CLAUDE.md workflow)

| Step | Builds | Tests / checks |
|---|---|---|
| 7.1 | `physics/disk1d.py`: finite-volume viscous diffusion in √r cells, Crank–Nicolson with a Rannacher start, zero-torque inner edge, zero-flux outer edge, ν(r) as a function or from arrays; Ṁ_acc(t) log | 1D tests 1–6 (expected-values §5: LBP masses and Σ values, angular momentum, the ive/mpmath check, steady state, self-similar drain 3%, mass ledger 10⁻¹²) |
| 7.2 | `analysis/disk_profile.py`: torus plane from total J, Σ(r), H/r(r), c_s(r) → ν(r) from a snapshot | hand-made rings and tilted rings with known profiles |
| 7.3 | Physical viscosity, Option A (Navier–Stokes shear, ν = α c_s²/Ω_K, two first derivatives) in `physics/` + Numba twin in `sim/sph.py`; viscous time-step criterion | reference vs Numba; a shear-flow test with the exact decay rate; energy ledger with viscosity; angular-momentum drift measured |
| 7.4 | Option B (Lodato & Price 2010 disc viscosity through the AV) as the cross-check / fallback | same tests; α_SS mapping reported |
| 7.5 | **Ring-spreading test** (Lodato & Price 2010): ring at r₀ = 50, H/r = 0.1, locally isothermal, Newtonian gravity, vs the 1D model; numerical floor α_num with physical viscosity off | L1 < 15% at 0.05/0.1/0.25 t_ν; α fit within 20%; α ≥ 3 α_num at production resolution; decides A vs B (DECISIONS) |
| 7.6 | Plunge-region diagnostics (v_r, density, sonic radius inside r = 10) | ISCO test-particle comparison; sonic point outside r_acc |
| 7.7 | Production: R7.1 (50k, from an R6.1 v2 torus, α = 0.1), R7.5/R7.6 (1D only, cheap), then the agreed subset of R7.3/R7.4 | ledgers, Ṁ(t), SPH vs 1D, α scaling, results-log entries, pictures for Hiten |

Steps 7.1–7.2 need no simulation runs and could proceed on the Mac while the cloud machine is busy.

## Questions for Hiten
1. Approve the step order above, and starting Phase 7 once Phase 6's results are reviewed?
2. Science runs at 50k (cloud) instead of 10k/1k, with a reduced α scan (which values)?
3. Start R7.1 from an R6.1 v2 checkpoint (once a torus exists) instead of from the cloud?
4. Add `mpmath` as a test dependency?

## Answers (Hiten approved 2026-10-21: "I agree with the phase 7 plans")
1. Step order approved; steps 7.1–7.2 (no simulation runs) start on the Mac while R6.1 v2 finishes on the cloud.
2. Science runs at 50k; α scan reduced to α = 0.03 and 0.3 besides R7.1's 0.1 (each ~3 days on the trial machine); r_acc check (R7.4) at 50k if affordable, else at 10k with the caveat.
3. R7.1 starts from an R6.1 v2 checkpoint once a torus exists.
4. `mpmath` added as a test dependency.
