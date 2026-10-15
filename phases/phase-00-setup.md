# Phase 0: Setup and platform check (detailed)

**Goal:** a reproducible Python environment on Hiten's Intel MacBook Air, and proof that every critical library works on this exact machine *before* any physics is written. Several libraries are dropping Intel-Mac support, so this phase is a real risk check, not a formality.

**Ends with:** a platform report, a Numba speed number, a GPU test image, a test video, and a passing (tiny) test suite, all committed and pushed.

**Prerequisites (already done by Hiten):** `git` (Apple Command Line Tools) and `uv 0.13.1` installed. VS Code with Python, Jupyter, Claude Code and Ruff extensions. An empty GitHub repository cloned locally, with the plan files copied in.

---

## Step 0.1: Project skeleton and environment

**Do:**
1. `uv init --package --name nebula` style layout matching PLAN.md §6 (`src/nebula/...`, `tests/`, `configs/`, `notebooks/`, `docs/`, `outputs/checks/`).
2. Pin Python: `uv python install 3.12` and `uv python pin 3.12`. (3.12 is the safest choice for Numba and wgpu wheels on Intel macOS; if a later step fails, try 3.11 before anything else.)
3. Add dependencies with `uv add`: `numpy scipy numba matplotlib wgpu imageio imageio-ffmpeg`. Dev group: `pytest ruff ipykernel`.
4. `.gitignore`: `.venv/ runs/ frames/ videos/ outputs/ __pycache__/ .ipynb_checkpoints/ *.npz` (but keep `outputs/checks/.gitkeep`; checks images are regenerated, not committed, except the few milestone images listed in each phase, which go to `docs/figures/`).
5. Create `AI_LOG.md` (header + first entry), `docs/DECISIONS.md` (header + "Python 3.12 pinned because…").
6. Ruff config in `pyproject.toml` (line length 100, sensible rule set: `E,F,I,B,UP,NPY`).
7. VS Code: `.vscode/settings.json` selecting `.venv/bin/python` as interpreter and pytest as the test runner.

**Check:** `uv sync` succeeds; `uv run python -c "import numpy, scipy, numba, wgpu, imageio_ffmpeg; print('ok')"` prints ok.

**What can go wrong:**
- *No wheel for Intel macOS* for some package/version: uv will try to build from source and fail (Numba/llvmlite need LLVM). Fix: pin an older version that still ships `macosx_*_x86_64` wheels (check PyPI "Download files" for the version). Record the pin and reason in DECISIONS.md. Try Python 3.11 if 3.12 has no compatible pair.
- *uv picks an ARM Python:* impossible on this Mac, but confirm `uv run python -c "import platform; print(platform.machine())"` prints `x86_64`.

**Hiten looks at:** the printed `ok`, and the file tree in VS Code.

---

## Step 0.2: Platform report

**Do:** `scripts/platform_report.py` writes `outputs/checks/platform.txt` with:
- Python version, every key package version, `platform.machine()`, macOS version.
- CPU model, physical/logical cores, RAM (via `os.sysconf` / `sysctl hw.memsize`).
- GPU: `wgpu.gpu.request_adapter_sync(power_preference="high-performance")` info (name, backend, adapter type) and limits: `max_buffer_size`, `max_storage_buffer_binding_size`, `max_texture_dimension_3d`, `max_compute_workgroup_size_x`, `max_compute_invocations_per_workgroup`, and supported features (look for `float32-filterable`, `timestamp-query`).

**Check:** adapter backend is **Metal** and the adapter is the Intel Iris GPU, not a software/CPU fallback. If it's a CPU adapter, stop and report.

**What can go wrong:** wgpu may fail to find an adapter when run from a non-GUI context; running from VS Code's terminal should work. If not, record it: overnight rendering must then run from a logged-in session (that's fine; document it).

**Hiten looks at:** `platform.txt`.

---

## Step 0.3: Numba check and speed calibration

**Do:**
1. `src/nebula/_bench/pairs.py`: a reference NumPy function and a Numba `@njit(parallel=True, cache=True, fastmath=False)` function computing, for N points, Σ_j exp(−|x_i − x_j|²) over all j within distance 2 (a stand-in for an SPH neighbour sum). Use a simple cell grid for the Numba version.
2. Test: Numba result equals reference to relative 1e-12 for N = 2,000 random points (seeded).
3. Benchmark: pair evaluations per second for N = 10k and 50k (uniform cube sized for ~60 neighbours each), with `NUMBA_NUM_THREADS` = 1, 2, 4. Save to `outputs/checks/numba_bench.txt`.

**Check:** correctness test passes; parallel speed-up is real (expect ~1.6–2.5× from 2 cores with hyper-threading). Record the pairs/second figure: Phase 3–7 time estimates are recalibrated from it.

**What can go wrong:**
- First call is slow (compilation, several seconds): use `cache=True`, and exclude the first call from timing.
- Race conditions: in parallel loops, each thread may only write to its own `i`. Never write to `j` inside a `prange` over `i`. (This rule carries into all SPH code.)
- `fastmath=True` changes results and breaks NaN checks: keep it off unless a test proves it harmless for that kernel.

**Hiten looks at:** the benchmark file; Claude explains what the number means for later run times.

---

## Step 0.4: GPU check

**Do:**
1. `src/nebula/render/device.py`: create device/queue; helper to run a compute shader over an image in **tiles** (e.g. 256×256 per dispatch), writing to a storage **buffer** of `vec4<f32>` (buffers avoid texture-format limitations), then read back to NumPy.
2. WGSL test shader: for each pixel compute an analytic pattern with real arithmetic in a loop (e.g. 2,000 iterations of a known recurrence such as the logistic map with parameters per pixel) so the result is checkable.
3. Python reference of the same computation in float32 (NumPy).
4. Test: GPU vs reference agree to ~1e-5 relative (float32; iterate fewer steps if chaos amplifies rounding: choose a non-chaotic parameter range).
5. Save `outputs/checks/gpu_test.png` (tone-mapped) and the time per tile and per 1280×720 image.

**Check:** agreement; no tile takes more than ~1 s; the screen does not freeze.

**What can go wrong:**
- *GPU watchdog / UI freeze* with large dispatches: make the tile smaller.
- *Readback stalls:* map buffers asynchronously or use `queue.read_buffer`; one readback per tile is fine at this scale.
- *wgpu-py API differences between versions:* pin the version that works in DECISIONS.md.
- **Fallback if wgpu-py cannot run:** (1) try an older wgpu-py; (2) the renderer is written CPU-side in Numba (slower: budget ~1–2 minutes per 1080p frame, still OK overnight for short videos); record the decision. Do not proceed to Phase 2 without a working path.

**Hiten looks at:** `gpu_test.png`.

---

## Step 0.5: Video check

**Do:** generate 90 frames of a moving test pattern (NumPy), write `outputs/checks/test.mp4` via `imageio` with the ffmpeg plugin from `imageio-ffmpeg`: H.264, `yuv420p`, CRF 18, 30 fps (these settings play everywhere, including QuickTime and phones).

**Check:** the file opens in QuickTime and plays smoothly.

**What can go wrong:** odd frame dimensions break `yuv420p` (both width and height must be even); colour shifts from wrong colour range: use `-pix_fmt yuv420p` and standard BT.709 tags.

**Hiten looks at:** the video.

---

## Step 0.6: Test helper and conventions

**Do:**
1. `tests/helpers.py`: `check(name, expected, actual, rtol=None, atol=None)` that prints a line `name | expected | actual | tol | PASS/FAIL` and asserts. Every physics test uses it, so `uv run pytest -s` prints the comparison table that CLAUDE.md requires.
2. `tests/test_smoke.py`: one trivial check to see the format.
3. `conftest.py`: fixed seed fixture; a `slow` marker for long tests (run with `-m slow`), so the default suite stays under ~1 minute.
4. Add `reference/expected-values.md` to the repo (from this plan) and a note that tests cite it by section.

**Check:** `uv run pytest -s` shows the table; `uv run ruff check` is clean.

---

## Step 0.7: First commit and push

Commit everything, push to GitHub. README gets a short "what this is" paragraph and "how to run the tests".

**Hiten looks at:** the repository on GitHub.

---

## Done when

- [ ] Environment reproducible from `uv sync` on a clean checkout.
- [ ] Platform report shows Metal + Iris GPU.
- [ ] Numba correctness and benchmark recorded.
- [ ] GPU test image correct and tiled.
- [ ] Test video plays.
- [ ] Test table output works; lint clean; pushed.
