# AI log

What Claude Code got wrong, assumed, or changed, and how it was caught. One or two lines per entry, newest last.

---

- **2026-10-14, Step 0.1 (environment).** Plain `uv add numba` picked the newest Numba (0.68), whose llvmlite 0.50 has no Intel-Mac build; uv tried to compile it and failed. The plan had predicted this (Phase 0, "What can go wrong"). Fixed by pinning `numba>=0.62.1,<0.63`, the last line with Intel-Mac wheels (checked on PyPI); recorded in `docs/DECISIONS.md`.
- **2026-10-14, Step 0.1 (commit).** First commit missed files: `runs/` in .gitignore also hid `configs/runs/` (fixed with root-anchored `/runs/`), and `reference/expected-values.md` was absent (the wrong one of two identical copies had been removed during cleanup; the survivor was moved back into `reference/`). Caught by listing the committed files before moving on.
