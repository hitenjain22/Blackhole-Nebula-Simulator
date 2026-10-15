# Decisions

Agreed changes and choices not fixed by the plan, with the reason (CLAUDE.md rule 7). Newest last.

---

## 2026-10-14: Python 3.12 pinned (Step 0.1)

Pinned with `uv python pin 3.12` (`.python-version`). 3.12 is the newest Python with Intel-Mac (x86_64) wheels for every package we need, in particular Numba/llvmlite (below). Fallback if a later step fails: 3.11.

## 2026-10-14: Numba pinned to 0.62.x (Step 0.1)

Numba 0.63 and later depend on llvmlite ≥ 0.46, and llvmlite published no `macosx_*_x86_64` wheels after **0.45.1** (checked on PyPI's file lists). Without a wheel, uv tries to build llvmlite from source, which needs LLVM and fails on this Mac. So `pyproject.toml` requires `numba>=0.62.1,<0.63` (llvmlite 0.45.1).

Consequence: Numba 0.62 supports NumPy only up to 2.3, so uv resolved **NumPy 2.3.5** (not the latest 2.5). Never run `uv lock --upgrade` without checking this pin still holds.

Resolved versions at the time: numba 0.62.1, llvmlite 0.45.1, numpy 2.3.5, scipy 1.18.1, matplotlib 3.11.2, wgpu 0.32.0, imageio 2.38.1, imageio-ffmpeg 0.6.0, pytest 9.1.1, ruff 0.17.0 (exact versions are locked in `uv.lock`).
