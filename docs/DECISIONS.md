# Decisions

Agreed changes and choices not fixed by the plan, with the reason (CLAUDE.md rule 7). Newest last.

---

## 2026-10-14: Python 3.12 pinned (Step 0.1)

Pinned with `uv python pin 3.12` (`.python-version`). 3.12 is the newest Python with Intel-Mac (x86_64) wheels for every package we need, in particular Numba/llvmlite (below). Fallback if a later step fails: 3.11.

## 2026-10-14: Numba pinned to 0.62.x (Step 0.1)

Numba 0.63 and later depend on llvmlite ≥ 0.46, and llvmlite published no `macosx_*_x86_64` wheels after **0.45.1** (checked on PyPI's file lists). Without a wheel, uv tries to build llvmlite from source, which needs LLVM and fails on this Mac. So `pyproject.toml` requires `numba>=0.62.1,<0.63` (llvmlite 0.45.1).

Consequence: Numba 0.62 supports NumPy only up to 2.3, so uv resolved **NumPy 2.3.5** (not the latest 2.5). Never run `uv lock --upgrade` without checking this pin still holds.

Resolved versions at the time: numba 0.62.1, llvmlite 0.45.1, numpy 2.3.5, scipy 1.18.1, matplotlib 3.11.2, wgpu 0.32.0, imageio 2.38.1, imageio-ffmpeg 0.6.0, pytest 9.1.1, ruff 0.17.0 (exact versions are locked in `uv.lock`).

## 2026-10-14: Numba speed calibration (Step 0.3)

Stand-in neighbour sum (`nebula._bench.pairs`, `scripts/numba_bench.py`), ~55 neighbours per point, short burst (not sustained; Phase 10 measures throttling over 60 minutes):

| N | 1 thread | 2 threads | 4 threads | speed-up at 4 |
|---|---|---|---|---|
| 10,000 | 1.67 × 10⁷ pairs/s | 3.33 × 10⁷ | **4.33 × 10⁷** | 2.59× |
| 50,000 | 1.31 × 10⁷ | 2.53 × 10⁷ | **3.08 × 10⁷** | 2.35× |

Results were bit-identical for 1, 2 and 4 threads. 50k is slower per pair, probably because its data no longer fits in the CPU cache (not investigated). Phases 3–7 recalibrate their time estimates from the 4-thread figures; the real SPH rate is measured in Phase 3 (§3.4).

## 2026-10-14: GPU check (Step 0.4)

wgpu 0.32 on the Iris Plus (Metal) works. Check pattern (2,000 loop iterations per pixel) at 1280 × 720 in 256 × 256 tiles: **~150 ms per image, ≤ 12 ms per tile** (first tile ~0.3 s including shader compilation). GPU matches the float32 CPU reference to 1.5 × 10⁻⁶ (tolerance 10⁻⁵). No fallback needed; Phase 2 proceeds on the GPU path. Readback uses `queue.read_buffer` once per tile.

## 2026-10-14: "Relative error" of a vector means per vector (Step 1.2)

For vector results (velocities, accelerations), a relative tolerance is applied to the length of the error vector divided by the length of the vector (`check(..., per_vector=True)`), not to each x, y, z component separately. Component-wise relative error depends on how the coordinate axes happen to point: a component that is nearly zero makes a perfect result look bad. Evidence from the first run: v → p → v round trip 4.0 × 10⁻¹⁶ per vector but 7.9 × 10⁻¹⁴ on one component 3,000× smaller than its vector; Hamilton's-equations check 1.1 × 10⁻⁸ per vector, 1.0 × 10⁻⁵ on one component 2,700× smaller than its vector. The tolerance numbers themselves are unchanged. Scalars are still compared element by element.

## 2026-10-14: GN formulas checked against the paper (Step 1.2)

`physics/gravity.py` was checked against Tejeda & Rosswog 2013 (arXiv:1303.4068): the Lagrangian (Eq. 2.11; Cartesian form A.4) and the Cartesian acceleration (Eq. A.5) match the plan's formulas with r_g = GM/c² = 1. The independent finite-difference test was confirmed to catch three deliberate typos (a 3 → 2 coefficient, a sign flip, a wrong power).

## 2026-10-14: Orbit time-step parameter η_orb = 0.00125 (Step 1.4)

Requirement (phase-01 Step 1.4, test 4): over 3 radial periods of the cloud orbit (r_p = 25, r_a = 5,000) with quantized adaptive steps η r^(3/2), |ΔH/H| < 10⁻⁴. Tested as the **maximum** over the run (the leapfrog's energy error swings up at each pericentre and returns; expected-values §2 warns to test the maximum).

| η | steps (3 orbits) | max \|ΔH/H\| | end \|ΔH/H\| | radial period error | precession |
|---|---|---|---|---|---|
| 0.005 (plan's suggestion) | 10,295 | 1.05 × 10⁻³ ✗ | 4.2 × 10⁻⁵ | 2.6 × 10⁻⁵ ✗ | 24.0395° |
| 0.0025 | 20,577 | 2.6 × 10⁻⁴ ✗ | 5.5 × 10⁻⁶ | 2.3 × 10⁻⁶ | 24.0397° |
| **0.00125** | 41,155 | **6.4 × 10⁻⁵** ✓ | 5.1 × 10⁻⁷ | 5.6 × 10⁻⁷ | 24.0398° |

The plan's "η = 0.005 gave 4 × 10⁻⁵" was the end-of-run value; the maximum is 25× larger. Relative to |H| the error is amplified because the orbit is nearly parabolic (|H| = 2 × 10⁻⁴ while the potential at pericentre is 0.04). **Chosen: η_orb = 0.00125** (also the value Phase 4's ledger test already specifies). Consequence: ~13,700 orbit steps per radial orbit per particle instead of the ~3,400 the plan assumed; Phase 4–7 cost estimates must use this. Option to revisit in Phase 4 (via rule 7, with tests): time-symmetric step selection (Hut, Makino & McMillan 1995), which may allow a larger η. Fixed-step tests use η = 0.02 for the circular orbit (plan's reference value; max |ΔH/H| 5.6 × 10⁻⁹ over 1,000 orbits).
