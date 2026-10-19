# Rules for Claude Code: Nebula–Black Hole simulation

**Start every session by reading `docs/HANDOFF.md`** (current state, pending decisions, cloud setup, how Hiten works) **and keep it updated as you work** (Hiten switches between Claude accounts; that file is the shared memory).

PLAN.md is the master plan and the current `phases/phase-NN-*.md` file is the spec for the work in progress. Read both before writing code. `reference/expected-values.md` holds the numbers tests must match. `reference/scripts/` holds the planning-time scripts that computed them: you may run them to double-check a value, but never import them into project code.

## Who you're working with

Hiten owns this project but does not write code. Explain what you did and what to look at in plain language, without jargon. He will check results visually and make decisions; give him something concrete to look at after every step.

## Workflow

1. Work one step at a time, in the order the phase file gives.
2. Before coding a step, state in 2–5 lines what you will build and how you will test it.
3. After each step: run the full test suite (`uv run pytest -s`, which prints the comparison table from `tests/helpers.py`), report each new test's expected value, actual value and PASS/FAIL, say exactly what Hiten should look at (a plot path, an image, a printed number), then **stop and wait** for his confirmation.
4. On confirmation, commit with a short, clear message. Never commit failing tests.
5. Before starting a new phase, re-read its phase file, check it against the code as it now exists, and propose a step breakdown. Wait for approval.
6. Log anything notable in `AI_LOG.md`: a bug you introduced and how a test caught it, a wrong assumption, a plan change. One or two lines each.

## Rules that never bend

1. **Physics formulas live in `src/nebula/physics/`.** Pure functions: numbers or arrays in, numbers or arrays out, no hidden state, no I/O (reading snapshots or files belongs in `sim/io.py` or `analysis/`). Every function's docstring gives the equation, its units, the PLAN/phase section, and the paper or textbook it comes from. The one allowed exception: fast copies of a formula inside a Numba kernel (`sim/`) or a WGSL shader (`render/shaders/`). Each copy names the `physics/` function it mirrors in a comment and has a test comparing it with that function (rule 2). A formula with no `physics/` original is a bug.
2. **Fast code must match slow code.** Every Numba-accelerated or GPU kernel has a plain, readable reference implementation in Python, and a test showing they agree within a stated tolerance. GPU shaders (WGSL) count: each shader formula has a Python reference and a test comparing GPU output to it on sample inputs.
3. **Never change an expected value or loosen a tolerance to make a test pass.** Fix the code. If you believe the expected value is wrong, stop and explain why with evidence.
4. **Units: G = c = M_bh = 1 everywhere in the simulation state and the integrator.** Physical constants live only in `src/nebula/units.py`. Exception: physics that is naturally written in physical units (equation of state, opacity, cooling rates, blackbody colour) may compute in SI inside `physics/` functions whose names end in `_si`, called only through a wrapper that converts code units in and out with `units.py`. Temperatures are in kelvin everywhere (T has no code unit). On-screen and report text use km, s, days, K, solar masses.
5. **Simulation state is float64.** float32 only for snapshots meant for rendering and for GPU buffers. Checkpoints are float64.
6. **Never take bigger physics steps to save time.** Speed comes from better algorithms, Numba, or running overnight. Time-step criteria are physics; they are tuned only by accuracy tests.
7. **If something in the plan can't be done as written, say so and ask.** Do not quietly substitute a different method. Record agreed changes in `docs/DECISIONS.md` with the reason.
8. **Every run is reproducible.** Each run writes a manifest: git commit, full config, library versions, random seed, machine info, start/end time. Never overwrite an existing run folder.
9. **Conservation ledgers are mandatory.** Mass, energy and angular momentum (including accreted and radiated amounts) are tracked and printed for every run. An unexplained ledger drift is a bug until shown otherwise.

## This machine

MacBook Air 2020: Intel Core i3, 2 cores / 4 threads, Intel Iris Plus integrated GPU (no fast float64 on GPU), 8 GB RAM, macOS 15. Homebrew is not available on this Mac; use `uv` for Python and packages.
- Keep memory use under ~4 GB. Check array sizes before allocating.
- Split GPU work into tiles so no single GPU dispatch runs longer than ~1 s (macOS may kill long GPU jobs, and the screen freezes).
- Long runs are overnight jobs: checkpoint at least every 15 minutes, and resume cleanly after a crash or reboot.
- The CPU throttles when hot; report performance as measured, not as hoped.

## Tech stack (set in Phase 0; change only via rule 7)

Python (version pinned by uv), NumPy, SciPy, Numba for hot loops, wgpu-py + WGSL for GPU rendering, matplotlib for plots, pytest for tests, imageio-ffmpeg for video, TOML for run configs. Ruff for formatting and linting.

## Style

- Small modules with clear names. Comments explain *why*, not *what*.
- Type hints on public functions. Docstrings on every physics function (rule 1).
- Plots for Hiten go in `outputs/checks/` with descriptive file names, and are listed in your step report.
- Do not add dependencies without saying why in the step plan.
