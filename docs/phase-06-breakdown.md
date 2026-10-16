# Phase 6 step breakdown (written 2026-10-16 overnight; Hiten asleep, authorised to proceed)

CLAUDE.md workflow 5 asks for a breakdown and approval before a new phase. Hiten authorised work on Phase 6 overnight (but not Phase 7), so this breakdown is written down and followed; review it in the morning.

## Checked against what exists (after Phase 5 and the plunge review)

- Gravity (GN), coupling, block steps with limiter, velocity reconstruction (D2), swallowing at r_acc = 4 with ledgers (D6), the hole-resolution check (D5), run folders/checkpoints, Phase 5 diagnostics: built and tested.
- The EOS is ideal gas only; no cooling, no opacity, no escape radius, no per-particle orbit elements, no Newtonian control gravity.
- **On hold (Hiten's decision, 2026-10-15):** the Phase 6 cloud size, so all science runs (R6.1–R6.5), the crash-timing predictions and anything that depends on the cloud size. Nothing here assumes a size.
- **2026-10-16:** Hiten chose cloud radius 60 (DECISIONS, D4 update); crash timing in expected-values §2c. Steps 6.1–6.7 are built and all tests pass. End time t = 3.8 × 10⁶ (Hiten, 2026-10-16). The runs wait for his go-ahead.

## Steps (all before any long run, phase-06 §6.3)

| Step | Builds | Tests |
|---|---|---|
| 6.1 | Escape radius 50,000 in the removed-particle ledger (reason "escaped"); random glass rotation for seeded clouds (R6.5) | ledger exact through escapes; seeded clouds differ, unseeded one unchanged |
| 6.2 | `physics/eos.py`: gas + radiation EOS in SI (u, P, β_gas, Γ₁, c_s; T from (ρ, u) by Newton on log T with bisection on [1, 10¹³] K), code-unit wrappers through `units.py` | tests 1–3 (expected-values §9 points, limits, 200 × 200 round trip) |
| 6.3 | `physics/opacity.py`: κ_es, free–free (Kramers-type) κ_abs with its coefficient checked in a textbook, κ_eff = √(κ_abs(κ_abs + κ_es)) | hand-computed values; limits |
| 6.4 | `physics/cooling.py`: temperature floor u_floor(ρ) at 10⁴ K from the EOS; β-cooling relaxing exactly toward it; E_radiated in the ledger and a per-particle luminosity; operator-split in the middle of each particle's kick–drift–kick (block steps) | tests 6, 7, 9 |
| 6.5 | The general EOS inside SPH (pressure and sound speed per particle; a "radiation-limit" option P = ρu/3 for test 4) | test 4 (radiation-dominated shock tube, γ = 4/3 exact solution); test 5 (colliding slabs, the key check of shock capture with D2) |
| 6.6 | Phase 4's ledger test with cooling and the full EOS | test 8 |
| 6.7 | Diagnostics (§6.5): per-particle orbit elements from (H, ℓ) via the exact Schwarzschild turning points, eccentricity histogram, ring/torus measures, luminosity, Ṁ, temperature histogram, optical-depth fractions, shock marker = reconstructed viscous heating rate (Hiten 2026-10-15); a Newtonian point-mass gravity option for the control run R6.4 | each on a hand-made case with a known answer |
| 6.8 | Runs R6.1–R6.5 at radius 60 to t = 3.8 × 10⁶ (waiting for Hiten's go-ahead) | ledgers, lost-gas share; `loss_check.py` against a 50k run |

Each step: check the formulas against the sources (and open-source codes where they exist: Phantom's `eos_idealplusrad`, MESA's EOS notes for Γ₁), fast and reference versions where there is a kernel, full test suite before commit.
